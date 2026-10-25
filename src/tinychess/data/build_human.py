"""Bounded 100/1000 accepted-game extraction, with no 50K execution path."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
from importlib.metadata import version

import yaml

from tinychess.chess.action_codec import ACTION_MAP_VERSION
from tinychess.chess.pgx_env import RULES_POLICY_VERSION
from tinychess.chess.state_features import ENCODER_VERSION
from tinychess.data.human import replay_game, RejectedGame, split_for_month, SCHEMA_VERSION
from tinychess.data.sources import fetch_prefix, open_text, game_blocks, sha256_file
from tinychess.data.storage import MonthWriter
from tinychess.data.audits import audit
from tinychess.validation.parity import ParityError


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def build(config_path, output, stage=1000, offline=False):
    config = yaml.safe_load(Path(config_path).read_text())
    if stage not in [100, 1000]:
        raise ValueError('Only the 100/1000-game smoke gates are authorized')
    if config['schema'] != SCHEMA_VERSION:
        raise ValueError('Unsupported dataset schema')
    quotas = [spec[f'smoke{stage}'] for spec in config['months'].values()]
    if any(not isinstance(q, int) or q <= 0 for q in quotas) or sum(quotas) != stage:
        raise ValueError('Configured quotas must exactly match the authorized 100/1000-game gate')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    available = shutil.disk_usage(output).free
    if available < config['storage']['output_budget_bytes'] * 2:
        raise RuntimeError('Insufficient real free storage for bounded output budget')
    started = time.perf_counter()
    counters = Counter()
    rejections = Counter()
    by_month = {}
    provenance, index = [], []
    seen = set()
    current_writer = None
    try:
        with (output / 'rejected_games.jsonl').open('w') as rejects:
            for month, spec in sorted(config['months'].items()):
                if split_for_month(month) != spec['split']:
                    raise ValueError('Configured chronological split mismatch')
                if offline:
                    path = Path(config['source']['cache']) / f'{month}.pgn.zst.prefix'
                    meta = json.loads(path.with_suffix(path.suffix+'.json').read_text())
                    if meta['month'] != month or meta['url'] != config['source']['url_template'].format(month=month):
                        raise RuntimeError('Offline cached month/URL provenance mismatch')
                    if meta['prefix_bytes'] != config['source']['compressed_prefix_bytes']:
                        raise RuntimeError('Offline cached prefix budget mismatch')
                    if sha256_file(path) != meta['prefix_sha256']:
                        raise RuntimeError('Offline source checksum mismatch')
                else:
                    path, meta = fetch_prefix(month, config)
                provenance.append(meta)
                current_writer = MonthWriter(output, month)
                accepted, rows, checked = 0, 0, 0
                with open_text(path) as text:
                    for ordinal, (block, framing_error) in enumerate(game_blocks(text, config['storage']['max_game_text_bytes'], config['storage']['max_line_chars']), 1):
                        counters['encountered_games'] += 1
                        checked += 1
                        try:
                            if framing_error:
                                raise RejectedGame(framing_error)
                            replayed = replay_game(block, month, ordinal, meta['url'], config)
                            game_id = replayed.game['metadata']['game_id']
                            if game_id in seen:
                                raise RejectedGame('duplicate_game_id')
                            seen.add(game_id)
                        except RejectedGame as exc:
                            why = str(exc)
                            rejections[why] += 1
                            rejects.write(canonical(dict(month=month, source_ordinal=ordinal, reason=why,
                                                         raw_pgn_sha256=hashlib.sha256(block.encode()).hexdigest()))+'\n')
                            continue
                        current_writer.append(replayed)
                        accepted += 1
                        rows += len(replayed.positions)
                        counters['accepted_games'] += 1
                        counters['positions'] += len(replayed.positions)
                        counters['parity_positions'] += len(replayed.positions) + 1
                        counters['checkmate_games'] += replayed.game['metadata']['actual_checkmate']
                        counters['time_forfeit_games'] += replayed.game['metadata']['value_weight'] == 0
                        if accepted == spec[f'smoke{stage}']:
                            break
                if accepted != spec[f'smoke{stage}']:
                    raise RuntimeError(f'Bounded prefix exhausted for {month}: {accepted} accepted; no automatic larger download')
                current_writer.close()
                index.extend(current_writer.index)
                current_writer = None
                by_month[month] = dict(split=spec['split'], encountered=checked, accepted=accepted,
                                       rejected=checked-accepted, positions=rows)
                size = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
                if size > config['storage']['output_budget_bytes']:
                    raise RuntimeError('Output storage budget exceeded; stop before scaling')
                print(canonical(dict(month=month, **by_month[month])), flush=True)
        (output / 'index.json').write_text(canonical(index)+'\n')
        audit_result = audit(output)
        (output / 'audit.json').write_text(json.dumps(audit_result, sort_keys=True, indent=2)+'\n')
        checksums = {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()}
        manifest = dict(schema=config['schema'], stage=stage, config=config,
                        config_sha256=sha256_file(config_path), sources=provenance,
                        versions={p: version(p) for p in ['jax','jaxlib','pgx','mctx','flax','optax','chess','pyarrow','zstandard']},
                        rules_policy=RULES_POLICY_VERSION, action_map=ACTION_MAP_VERSION, encoder=ENCODER_VERSION,
                        counts=dict(counters), by_month=by_month, rejection_counts=dict(sorted(rejections.items())),
                        full_legal_action_parity_mismatches=0, checkmate_assertions=counters['checkmate_games'],
                        checksums=checksums, dataset_digest=hashlib.sha256(canonical(checksums).encode()).hexdigest(),
                        sampling_bias='Chronological prefixes only; not representative within each month',
                        frozen_evaluation_fixtures=len(config.get('frozen_evaluation_fens', [])),
                        source_full_archives_verified=False)
        (output / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2)+'\n')
        elapsed = time.perf_counter()-started
        measurement = dict(elapsed_seconds=elapsed, accepted_games_per_second=counters['accepted_games']/elapsed,
                           processed_games_per_second=counters['encountered_games']/elapsed,
                           positions_per_second=counters['positions']/elapsed,
                           peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                           python=platform.python_version(), output_bytes=sum(p.stat().st_size for p in output.iterdir()),
                           free_disk_bytes_before=available)
        (output / 'measurements.json').write_text(json.dumps(measurement, indent=2)+'\n')
        print(canonical(dict(status='PASS', counts=dict(counters), dataset_digest=manifest['dataset_digest'], measurements=measurement)), flush=True)
        return manifest
    except Exception as exc:
        (output / 'FAIL_STOP.json').write_text(json.dumps(dict(error=str(exc), type=type(exc).__name__, counts=dict(counters)), indent=2)+'\n')
        raise
    finally:
        if current_writer is not None:
            current_writer.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/data_human.yaml')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', type=int, choices=[100,1000], default=1000)
    p.add_argument('--offline', action='store_true')
    args = p.parse_args()
    build(args.config, args.output, args.stage, args.offline)
