"""Read-only sample audit plus a stricter mutually disjoint held-out row index."""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path

import chess
import numpy as np
import pyarrow.parquet as pq

from tinychess.data.storage import read_games, unpack_features
from tinychess.data.human import RESULTS, classify_ending


def verify(folder, output):
    folder = Path(folder)
    manifest = json.loads((folder/'manifest.json').read_text())
    if (folder/'FAIL_STOP.json').exists():
        raise RuntimeError('Dataset failed')
    games = list(read_games(folder))
    by_id = {g['metadata']['game_id']:g for g in games}
    assert len(by_id) == len(games) == manifest['counts']['accepted_games']
    coverage = defaultdict(list)
    openings = Counter()
    for game in games:
        m = game['metadata']
        assert len(game['actions']) == m['plies'] <= 512
        assert m['outcome_white'] == RESULTS[m['result']]
        time_forfeit = classify_ending(chess.Board(), m['termination_header'], m['result']) == 'time_forfeit'
        assert m['value_weight'] == (0.0 if time_forfeit else 1.0)
        board = chess.Board(m['final_board_fen'])
        assert m['actual_checkmate'] == board.is_checkmate()
        if m['actual_checkmate']:
            assert m['final_board_reason'] == 'checkmate' and m['result'] == board.result()
        coverage[m['source_month']].append(m['utc_date']+'T'+m['utc_time'])
        openings[m['eco']] += 1
    keys = {t: defaultdict(set) for t in [0,10]}
    stats = Counter()
    examples = []
    for path in sorted(folder.glob('*.positions.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            for row in batch.to_pylist():
                m = by_id[row['game_id']]['metadata']
                assert row['action'] == by_id[row['game_id']]['actions'][row['ply']]
                assert row['color'] == row['ply'] % 2
                expected = RESULTS[m['result']] * (1 if row['color']==0 else -1)
                assert row['outcome_stm'] == expected
                assert row['wdl_class'] == (0 if expected>0 else 2 if expected<0 else 1)
                assert row['value_weight'] == m['value_weight'] and row['policy_weight']==1
                feature = unpack_features(row)
                assert feature['legal_mask'].shape == (4672,) and feature['legal_mask'][row['action']]
                assert np.isfinite(feature['planes']).all() and np.isfinite(feature['context']).all()
                board = chess.Board(row['fen'])
                assert row['position_hash'] == hashlib.sha256(' '.join(board.fen(en_passant='legal').split()[:4]).encode()).digest()
                move = chess.Move.from_uci(row['human_move_uci'])
                assert move in board.legal_moves
                stats['positions'] += 1
                stats['castling_moves'] += board.is_castling(move)
                stats['en_passant_moves'] += board.is_en_passant(move)
                stats['promotion_'+row['human_move_uci'][-1]] += int(move.promotion is not None) if move.promotion else 0
                stats['repeated_positions'] += row['repetition_count'] >= 2
                stats['value_masked_positions'] += row['value_weight'] == 0
                for t in [0,10]:
                    if row['ply'] >= t:
                        keys[t][row['split']].add(row['position_hash'])
                if len(examples)<3 and row['ply'] in [12,31,59]:
                    example = {k:v for k,v in row.items() if k not in ['planes','context','legal_mask','position_hash']}
                    example.update(game_metadata=m, position_hash=row['position_hash'].hex(),
                                   legal_action_indices=np.flatnonzero(feature['legal_mask']).tolist(),
                                   legal_mask_hex=row['legal_mask'].hex(), encoder=dict(planes_shape=[8,8,120],
                                   planes_sha256=hashlib.sha256(row['planes']).hexdigest(),context=feature['context'].tolist()))
                    examples.append(example)
    overlap = {}
    for t in [0,10]:
        overlap[str(t)] = {split:dict(unique=len(keys[t][split]), shared_with_train=len(keys[t][split]&keys[t]['train'])) for split in ['validation','test']}
    strict_path = folder/'strict_disjoint_eval.jsonl'
    strict = Counter()
    row_overlap = defaultdict(Counter)
    forbidden_by_split = {'validation': keys[0]['train'], 'test': keys[0]['train'] | keys[0]['validation']}
    with strict_path.open('w') as out:
        for path in sorted(folder.glob('*.positions.parquet')):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=['game_id','split','ply','position_hash']):
                for row in batch.to_pylist():
                    split, key = row['split'],row['position_hash']
                    if split == 'train':
                        continue
                    for t in [0,10]:
                        if row['ply'] >= t:
                            row_overlap[f'{split}_ply_ge_{t}']['rows'] += 1
                            row_overlap[f'{split}_ply_ge_{t}']['shared_rows'] += key in keys[t]['train']
                    forbidden = forbidden_by_split[split]
                    if key not in forbidden:
                        strict[split] += 1
                        out.write(json.dumps(dict(game_id=row['game_id'],split=split,ply=row['ply']),sort_keys=True)+'\n')
    report=dict(status='PASS', dataset_digest=manifest['dataset_digest'], all_rows_label_mask_metadata_checks=True,
                verified_games=len(games), stats={k:v for k,v in stats.items() if v},
                temporal_coverage={m:dict(first=min(times),last=max(times),distinct_utc_days=len({s[:10] for s in times})) for m,times in sorted(coverage.items())},
                unique_position_overlap=overlap,row_position_overlap={k:dict(v) for k,v in row_overlap.items()},
                strict_disjoint_eval_rows=dict(strict),strict_disjoint_eval_sha256=hashlib.sha256(strict_path.read_bytes()).hexdigest(),
                strict_disjoint_definition='validation excludes train; test excludes train AND validation; does not collapse legitimate repeated human moves',
                top_openings=openings.most_common(10),representative_records=examples,
                frozen_eval_note='No frozen puzzle/GPT match fixture manifest supplied; only configured fixtures can be excluded')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['representative_records']}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('reports/m1_sample_audit.json'))
    a=p.parse_args()
    verify(a.dataset,a.output)
