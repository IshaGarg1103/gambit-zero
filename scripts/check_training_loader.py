"""Read-only M2 Stage 1 A-C sample audit and bounded original-Parquet benchmark.

This never downloads source PGNs, rewrites Dataset A or builds a derived cache.
Run without --download to require verified local originals. Default sample:
three training shards spread over Jan-Jul, one August and one September shard.
Reports sample statistics separately from accepted full-data manifest counts.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import struct
import sys
import time

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tinychess.data.training_loader import (
    VerifiedStore, inventory, iter_records, iter_batches, parquet_batches,
    validate_game, GAME_SCHEMA, REPO, REVISION, CATALOG_SHA256, SPLITS, sha256,
)


def fingerprint(planes, context, mask, action, outcome, wdl, pw, vw, eligible, board_hash):
    return hashlib.sha256(planes + context + mask + struct.pack('<ibiff?', int(action),
                          int(outcome), int(wdl), float(pw), float(vw), bool(eligible)) + board_hash).digest()


def row_fingerprint(row):
    return fingerprint(row['planes'], row['context'], row['legal_mask'], row['action'],
                       row['outcome_stm'], row['wdl_class'], row['policy_weight'],
                       row['value_weight'], row['training_eligible'], row['position_hash'])


def run(args):
    root = Path(args.root)
    store = VerifiedStore(root, download=args.download, keep_downloads=True)
    existing = {p.resolve() for p in root.rglob('*.parquet')} if root.exists() else set()
    report = dict(status='RUNNING', repo=REPO, revision=REVISION, catalog_sha256=CATALOG_SHA256,
                  measured_utc=datetime.now(timezone.utc).isoformat(),
                  scope='five original shards; sample statistics only',
                  environment=dict(python=platform.python_version(), platform=platform.platform(),
                      cpu_count=psutil.cpu_count(), ram_bytes=psutil.virtual_memory().total),
                  parameters=dict(seed=args.seed, batch_size=args.batch_size,
                      chunk_rows=args.chunk_rows, shuffle_buffer=args.shuffle_buffer, repeats=args.repeats),
                  cache_implemented=False, splits={}, benchmark={})
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    acquired = []
    def save():
        temporary = destination.with_suffix('.json.partial')
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
        temporary.replace(destination)
    try:
        shards, catalog = inventory(store)
        report['full_data_manifest_counts_not_remeasured'] = catalog['counts']
        report['full_inventory_positions_bytes'] = sum(s.positions.size for s in shards)
        selected = []
        for split in SPLITS:
            group = [s for s in shards if s.split == split]
            selected.extend([group[i] for i in (0,len(group)//2,len(group)-1)] if split == 'train' else [group[0]])
        expected = sum(s.positions.size+s.games.size for s in selected)
        report['expected_sample_download_bytes'] = expected
        print(f'Original sample download: {expected:,} bytes; full position inventory: '
              f'{report["full_inventory_positions_bytes"]:,} bytes. No PGN archives.', flush=True)
        started = time.perf_counter()
        for s in selected:
            for spec in (s.positions, s.games):
                with store.file(spec) as p:
                    acquired.append((p, spec))
        report['acquisition_seconds'] = time.perf_counter()-started
        report['sample_download_bytes'] = store.downloaded_bytes
        report['sample_disk_bytes'] = sum(spec.size for _, spec in acquired)
        report['files'] = [dict(path=spec.path, bytes=spec.size, sha256=spec.sha256) for _,spec in acquired]
        save()
        references, orders = {}, {}
        per_channel_min = np.full(120, np.inf); per_channel_max = np.full(120, -np.inf)
        all_binary = np.ones(120, bool); negative_zero = np.zeros(120, bool)
        # Bounded exact unique sets: <=4,096 bit patterns/channel; if exceeded,
        # mark capped, never assert a complete inventory or binary compression.
        unique_bits = [set() for _ in range(120)]; capped = [False]*120
        for split in SPLITS:
            subset = [s for s in selected if s.split == split]
            stats = dict(rows=0, eligible_positions=0, opening_ply_lt_10=0,
                         value_zero_weight=0, wdl={}, game_statistics={})
            wdl, boards = Counter(), Counter()
            game_lengths, tc, ending = Counter(), Counter(), Counter()
            for s in subset:
                with store.file(s.games) as path:
                    for batch in parquet_batches(path, GAME_SCHEMA, args.chunk_rows):
                        for row in batch:
                            g, _ = validate_game(row, s)
                            game_lengths[g['plies']] += 1; tc[g['time_control']] += 1
                            ending[g['ending_semantics']] += 1
            for row in iter_records(store, subset, split, chunk_rows=args.chunk_rows):
                stats['rows'] += 1
                stats['eligible_positions'] += int(row['training_eligible'])
                stats['opening_ply_lt_10'] += int(row['ply'] < 10)
                stats['value_zero_weight'] += int(row['value_weight'] == 0)
                wdl[['win','draw','loss'][row['wdl_class']]] += 1
                boards[row['position_hash']] += 1
                key = (row['game_id'], row['ply'])
                if key in references:
                    raise ValueError('Cross-split duplicate game/ply')
                if row['training_eligible']:
                    references[key] = row_fingerprint(row)
                values = np.frombuffer(row['planes'], '<f4').reshape(64,120)
                per_channel_min = np.minimum(per_channel_min, values.min(axis=0))
                per_channel_max = np.maximum(per_channel_max, values.max(axis=0))
                all_binary &= np.all((values == 0) | (values == 1), axis=0)
                negative_zero |= np.any((values == 0) & np.signbit(values), axis=0)
                # Channels spatially constant or binary in normal chess have a
                # very small unique set. This is measured sample-only evidence.
                for channel in range(120):
                    if not capped[channel]:
                        unique_bits[channel].update(np.unique(values[:,channel].view(np.uint32)).tolist())
                        if len(unique_bits[channel]) > 4096:
                            unique_bits[channel] = set(); capped[channel] = True
            stats.update(wdl=dict(wdl), unique_board_hashes=len(boards),
                         repeated_board_identities=sum(n>1 for n in boards.values()),
                         repeat_occurrences_beyond_first=sum(boards.values())-len(boards),
                         max_board_frequency=max(boards.values(),default=0),
                         board_identity='SHA256(pieces, side to move, castling, legally capturable EP); excludes clocks/history',
                         game_statistics=dict(games=sum(game_lengths.values()), length_histogram=dict(game_lengths),
                                              time_controls=dict(tc), endings=dict(ending)))
            report['splits'][split] = stats
            print(f'{split}: exact original reference checked {stats["rows"]:,} rows', flush=True)
        report['channel_inventory_sample_only'] = [dict(channel=i, min=float(per_channel_min[i]),
            max=float(per_channel_max[i]), exact_binary_values=bool(all_binary[i]),
            negative_zero_observed=bool(negative_zero[i]), unique_float32_bits_count=None if capped[i] else len(unique_bits[i]),
            unique_count_capped=capped[i]) for i in range(120)]
        # Exact output audit outside throughput measurement; references contain
        # only SHA256/row identity, never whole decoded feature tensors.
        emitted = set()
        for split in SPLITS:
            subset = [s for s in selected if s.split == split]
            order = hashlib.sha256(); maxima=[]; unique_games=[]; rows=0
            for batch in iter_batches(store, subset, split, batch_size=args.batch_size, chunk_rows=args.chunk_rows,
                                      shuffle_buffer=args.shuffle_buffer, seed=args.seed):
                maxima.append(batch.max_positions_per_game); unique_games.append(len(set(batch.game_id)))
                for i, key in enumerate(zip(batch.game_id, batch.ply.tolist())):
                    actual = fingerprint(batch.planes[i].tobytes(), batch.context[i].tobytes(),
                        np.packbits(batch.legal_mask[i], bitorder='little').tobytes(),batch.action[i],
                        batch.outcome_stm[i], batch.wdl_class[i],batch.policy_weight[i],batch.value_weight[i],
                        batch.training_eligible[i],batch.position_hash[i])
                    if key in emitted or actual != references[key]:
                        raise ValueError(f'Duplicate or nonidentical decoded record: {key}')
                    emitted.add(key); order.update(json.dumps(key,separators=(',',':')).encode()+b'\n'); rows+=1
            report['splits'][split].update(decoded_rows_verified_byte_exact=rows,
                order_sha256=order.hexdigest(), max_positions_from_one_game_per_batch=max(maxima,default=0),
                min_unique_games_per_batch=min(unique_games,default=0),
                batch_count=len(maxima), last_batch_size=rows%args.batch_size or args.batch_size)
            orders[split]=order.hexdigest()
        if len(emitted) != len(references):
            raise ValueError('Incomplete eligible row accounting')
        for repeat in range(args.repeats):
            subset = [s for s in selected if s.split == 'train']
            start = time.perf_counter(); count=0; order=hashlib.sha256(); max_game=0
            for batch in iter_batches(store, subset, 'train', batch_size=args.batch_size, chunk_rows=args.chunk_rows,
                                      shuffle_buffer=args.shuffle_buffer, seed=args.seed):
                count += len(batch.action); max_game=max(max_game,batch.max_positions_per_game)
                for key in zip(batch.game_id,batch.ply.tolist()):
                    order.update(json.dumps(key,separators=(',',':')).encode()+b'\n')
            seconds=time.perf_counter()-start
            if order.hexdigest()!=orders['train']:
                raise ValueError('Same-seed shuffled order changed')
            report['benchmark'][str(repeat)]=dict(positions=count,seconds=seconds,positions_per_second=count/seconds,
                order_sha256=order.hexdigest(),max_positions_from_one_game_per_batch=max_game)
            print(f'Warm local original-Parquet loader {repeat+1}: {count/seconds:,.0f} positions/s',flush=True)
        report['same_seed_order_verified'] = True
        report['byte_exact_output_rows'] = len(emitted)
        report['cache_decision'] = ('DEFER: original Parquet is >=5K positions/s on every measured repeat'
            if min(x['positions_per_second'] for x in report['benchmark'].values())>=5000
            else 'DEFER: bounded correct original loader retained; optimization needs separate measurement/review')
        report['peak_process_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform=='darwin' else 1024)
        report['status'] = 'PASS_SAMPLE_SCOPE'
    except BaseException as exc:
        report['status']='FAIL'; report['error_type']=type(exc).__name__
        # Do not echo HTTP authorization headers or credentials from exceptions.
        raise
    finally:
        removed=0
        if not args.keep_sample:
            for path,spec in acquired:
                if path.resolve() not in existing and path.exists():
                    if sha256(path) != spec.sha256:
                        raise ValueError('Refusing to delete changed sample file')
                    path.unlink();removed+=1
        report['temporary_sample_files_removed']=removed
        report['kept_sample_requested']=args.keep_sample
        save()
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='data/processed/m2_stage1')
    p.add_argument('--output',default='data/processed/m2_stage1/validation.json')
    p.add_argument('--download',action='store_true')
    p.add_argument('--keep-sample',action='store_true')
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--chunk-rows',type=int,default=128)
    p.add_argument('--shuffle-buffer',type=int,default=16384)
    p.add_argument('--seed',type=int,default=20261030)
    p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    if args.repeats<1:
        p.error('--repeats must be positive')
    result=run(args)
    print(json.dumps({k:result[k] for k in ['status','byte_exact_output_rows','cache_decision','temporary_sample_files_removed']},indent=2))
