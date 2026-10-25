"""Conservative board-position leakage and distribution diagnostics."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from tinychess.data.storage import read_games


def audit(folder):
    sets = {threshold: defaultdict(set) for threshold in [0, 10]}
    distributions = {key: Counter() for key in ['game_split', 'position_split', 'ending', 'result', 'eco', 'time_control', 'phase', 'side_to_move', 'rating_bucket']}
    games = list(read_games(folder))
    ids = set()
    trajectories = Counter()
    for game in games:
        m = game['metadata']
        if m['game_id'] in ids:
            raise RuntimeError('Duplicate game ID; source/split leakage')
        ids.add(m['game_id'])
        trajectories[m['trajectory_sha256']] += 1
        for key, value in [('game_split', m['split']), ('ending', m['ending_semantics']), ('result', m['result']),
                           ('eco', m['eco']), ('time_control', m['time_control']),
                           ('rating_bucket', str(min(m['white_elo'], m['black_elo'])//200*200))]:
            distributions[key][value] += 1
    value_classes = Counter()
    mask_counts = Counter()
    samples = []
    for path in sorted(Path(folder).glob('*.positions.parquet')):
        columns = ['game_id','split','ply','position_hash','wdl_class','value_weight','training_eligible','fen','action','human_move_uci','repetition_count']
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=columns):
            for row in batch.to_pylist():
                split, key, ply = row['split'], row['position_hash'], row['ply']
                for threshold in [0, 10]:
                    if ply >= threshold:
                        sets[threshold][split].add(key)
                distributions['position_split'][split] += 1
                distributions['side_to_move'][row['fen'].split()[1]] += 1
                # Descriptive phase buckets, not a strategic teacher label.
                distributions['phase']['opening' if ply < 20 else 'middlegame' if ply < 80 else 'late'] += 1
                value_classes[str(row['wdl_class'])] += 1
                mask_counts['value_masked_rows'] += int(row['value_weight'] == 0)
                mask_counts['frozen_eval_excluded_rows'] += int(not row['training_eligible'])
                if len(samples) < 4 or (row['repetition_count'] >= 2 and len(samples) < 8):
                    samples.append({k: v.hex() if isinstance(v, bytes) else v for k, v in row.items()})
    overlap = {}
    for threshold in [0, 10]:
        train = sets[threshold]['train']
        entries = {}
        for split in ['validation', 'test']:
            keys = sets[threshold][split]
            entries[split] = dict(unique_positions=len(keys), shared_with_train=len(train & keys),
                                  unique_overlap_fraction=len(train & keys)/len(keys) if keys else 0)
        entries['train_unique_positions'] = len(train)
        entries['validation_test_shared'] = len(sets[threshold]['validation'] & sets[threshold]['test'])
        overlap[f'ply_ge_{threshold}'] = entries
    # A reproducible row-level position-disjoint validation/test subset manifest.
    disjoint = Path(folder) / 'position_disjoint_eval.jsonl'
    disjoint_count = Counter()
    with disjoint.open('w') as out:
        for path in sorted(Path(folder).glob('*.positions.parquet')):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=['game_id','split','ply','position_hash']):
                for row in batch.to_pylist():
                    if row['split'] != 'train' and row['position_hash'] not in sets[0]['train']:
                        out.write(json.dumps(dict(game_id=row['game_id'], split=row['split'], ply=row['ply']), sort_keys=True)+'\n')
                        disjoint_count[row['split']] += 1
    return dict(distributions={k: dict(sorted(v.items())) for k,v in distributions.items()},
                wdl_classes=dict(value_classes), masks=dict(mask_counts), unique_game_ids=len(ids), duplicate_game_ids=0,
                repeated_trajectory_groups=sum(v > 1 for v in trajectories.values()),
                overlap=overlap, disjoint_eval_rows=dict(disjoint_count), representative_rows=samples,
                utc_month_mismatch_games=sum(m['metadata'].get('utc_month_differs_from_source', False) for m in games),
                position_key='SHA256 of board, STM, castling, legally capturable EP; excludes clocks/history (conservative overlap)',
                opening_rebalance='Distribution recorded; no silent downsampling in this correctness smoke. Training sampler must cap opening/position frequency.')
