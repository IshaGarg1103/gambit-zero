"""Measure both physical layouts on the exact same accepted human games."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import resource
import sys
import time

import jax
import numpy as np
import pyarrow.parquet as pq

from tinychess.chess.pgx_env import init, step
from tinychess.chess.state_features import encode_jit
from tinychess.data.storage import read_games, unpack_features


def measure(folder, output, minibatch=64, seed=20261025):
    folder = Path(folder)
    if (folder / 'FAIL_STOP.json').exists():
        raise RuntimeError('Refuse to benchmark a failed dataset')
    started = time.perf_counter()
    games = list(read_games(folder))
    game_decode_seconds = time.perf_counter()-started
    index = json.loads((folder/'index.json').read_text())
    game_by_id = {g['metadata']['game_id']:g for g in games}
    cumulative = np.cumsum([r['plies'] for r in index])
    positions = int(cumulative[-1])
    rng = random.Random(seed)
    chosen = rng.sample(range(positions), min(minibatch, positions))
    requests = defaultdict(list)
    for global_row in chosen:
        i = int(np.searchsorted(cumulative, global_row, side='right'))
        offset = global_row - (int(cumulative[i-1]) if i else 0)
        requests[i].append(offset)
    # Both random-access measurements use warm filesystem caches, a preloaded
    # small game/index table and initialized JAX. Files are opened on demand.
    jax.block_until_ready(encode_jit(init(jax.random.PRNGKey(0))))
    materialized = {}
    started = time.perf_counter()
    for i, wanted in sorted(requests.items()):
        entry = index[i]
        rows = pq.ParquetFile(folder/entry['position_file']).read_row_group(entry['position_row_group']).to_pylist()
        for ply in wanted:
            materialized[(entry['game_id'],ply)] = unpack_features(rows[ply])
    material_random = time.perf_counter()-started
    replay_steps = 0
    started = time.perf_counter()
    for i, wanted in sorted(requests.items()):
        entry = index[i]
        actions = game_by_id[entry['game_id']]['actions']
        state = init(jax.random.PRNGKey(0))
        for ply in range(max(wanted)+1):
            if ply in wanted:
                actual = jax.device_get(encode_jit(state))
                expected = materialized[(entry['game_id'],ply)]
                for key in actual:
                    np.testing.assert_array_equal(actual[key], expected[key])
            if ply < max(wanted):
                state = step(state, actions[ply])
                replay_steps += 1
    replay_random = time.perf_counter()-started
    # Full sequential replay-derived features, not just action JSON decode.
    started = time.perf_counter()
    digest = hashlib.sha256()
    replayed = 0
    for game in games:
        state = init(jax.random.PRNGKey(0))
        for action in game['actions']:
            encoded = jax.device_get(encode_jit(state))
            digest.update(encoded['planes'].tobytes())
            digest.update(encoded['context'].tobytes())
            digest.update(np.packbits(encoded['legal_mask'], bitorder='little').tobytes())
            replayed += 1
            state = step(state, action)
        jax.block_until_ready(state)
    sequential_replay = time.perf_counter()-started
    started = time.perf_counter()
    read_digest = hashlib.sha256()
    decoded = 0
    for path in sorted(folder.glob('*.positions.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=['planes','context','legal_mask']):
            for row in batch.to_pylist():
                features = unpack_features(row)
                assert features['planes'].shape == (8,8,120)
                for key in ['planes','context','legal_mask']:
                    read_digest.update(row[key])
                decoded += 1
    sequential_materialized = time.perf_counter()-started
    assert decoded == replayed == positions
    assert digest.hexdigest() == read_digest.hexdigest(), 'Every replay-derived feature must match materialized encoding'
    game_bytes = sum(p.stat().st_size for p in folder.glob('*.games.parquet'))
    position_bytes = sum(p.stat().st_size for p in folder.glob('*.positions.parquet'))
    decision = 'materialized_position_parquet' if material_random < replay_random else 'compact_game_parquet_with_replay'
    report = dict(games=len(games), positions=positions,
                  compact_game_bytes=game_bytes, materialized_position_bytes=position_bytes,
                  compact_bytes_per_position=game_bytes/positions, materialized_bytes_per_position=position_bytes/positions,
                  game_metadata_and_actions_decode_seconds=game_decode_seconds,
                  sequential_replay_seconds=sequential_replay, replay_encoded_positions_per_second=replayed/sequential_replay,
                  sequential_materialized_decode_seconds=sequential_materialized,
                  materialized_positions_per_second=decoded/sequential_materialized,
                  random_minibatch_size=len(chosen), random_seed=seed,
                  distinct_games_in_minibatch=len(requests), random_replay_steps=replay_steps,
                  random_materialized_seconds=material_random, random_compact_replay_seconds=replay_random,
                  random_access_includes='warm filesystem; materialized row-group read+feature decode vs game replay+encode+equality check; small game/index tables preloaded',
                  feature_stream_sha256=digest.hexdigest(), all_features_equal=True,
                  selected_primary=decision, archive_representation='compact_game_parquet retained alongside materialized prototype',
                  peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                  devices=[str(d) for d in jax.devices()],
                  limitation='CPU measurements on 1000 games; no GPU dataloader, network forward, cold disk or scaling extrapolation validation')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('reports/m1_storage_benchmark.json'))
    p.add_argument('--minibatch', type=int, default=64)
    a = p.parse_args()
    if a.minibatch <= 0:
        p.error('positive minibatch required')
    measure(a.dataset, a.output, a.minibatch)
