"""Read-only, bounded audit of every published row against PGN/Pgx replay.

Downloads one game/position shard pair at a time without the Hub disk cache.
Does not upload, alter generation receipts, rescan archives, or touch live runs.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import time

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['HF_HUB_DISABLE_XET'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import pyarrow.parquet as pq
import yaml
import zstandard as zstd
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.utils import build_hf_headers, get_session
from tinychess.data.archive_scan import canonical
from tinychess.data.human import replay_game
from tinychess.data.hf_month import feature_sha, logical_game_sha
from tinychess.data.storage import GAME_SCHEMA, POSITION_SCHEMA


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_games(config, month):
    """Audit the frozen split quotas, including 7,500-game held-out months."""
    entry = config['months'][month]
    count = entry['accepted_games']
    require(type(count) is int and count > 0, 'Invalid configured month quota')
    require(entry['split'] == ('train' if month <= '2026-07' else
                              'validation' if month == '2026-08' else 'test'),
            'Month split configuration mismatch')
    return count


def compare_row(actual, expected):
    require(actual.keys() == expected.keys(), 'Position column set mismatch')
    for key in expected:
        require(actual[key] == expected[key],
                f"Row mismatch: game={expected['game_id']} ply={expected['ply']} field={key}")


def parquet_rows(path, schema):
    file = pq.ParquetFile(path)
    # Parquet canonicalizes list child names from Arrow's "item" to "element".
    # Compare semantic types/nullability and version metadata separately; no
    # action types, feature dimensions or version identifiers are waived.
    require(file.schema_arrow.equals(schema, check_metadata=False)
            and file.schema_arrow.metadata == schema.metadata, f'Schema mismatch: {path.name}')
    for batch in file.iter_batches(batch_size=64):
        yield from batch.to_pylist()


class RemoteReader:
    def __init__(self, repo, revision, root):
        self.repo, self.revision, self.root = repo, revision, root
        self.bytes = self.files = self.peak_disk = 0
        self.headers = build_hf_headers(token=True)

    def fetch(self, remote, expected=None):
        require(shutil.disk_usage(self.root).free > 20 * 2**30, '20 GiB disk reserve reached')
        path = self.root / Path(remote).name
        require(not path.exists(), 'Temporary file name collision')
        limit = expected['bytes'] if expected else 32 * 2**20
        require(limit <= 32 * 2**20, 'Remote object exceeds 32 MiB audit file bound')
        digest, count = hashlib.sha256(), 0
        url = hf_hub_url(self.repo, remote, repo_type='dataset', revision=self.revision)
        with get_session().stream('GET', url, headers=self.headers, follow_redirects=True, timeout=120) as response:
            response.raise_for_status()
            with path.open('wb') as output:
                for chunk in response.iter_bytes(chunk_size=2**20):
                    count += len(chunk)
                    require(count <= limit, 'Remote file exceeded declared byte bound')
                    digest.update(chunk)
                    output.write(chunk)
        if expected:
            require(count == expected['bytes'] and digest.hexdigest() == expected['sha256'],
                    f'Remote payload checksum/size mismatch: {remote}')
        self.files += 1
        self.bytes += count
        self.peak_disk = max(self.peak_disk, sum(p.stat().st_size for p in self.root.iterdir() if p.is_file()))
        return path

    def read_json(self, remote):
        p = self.fetch(remote)
        value = json.loads(p.read_text())
        p.unlink()
        return value


def audit(args):
    started = time.monotonic()
    report = dict(status='RUNNING', repo=args.repo, revision=args.revision,
                  started_utc=datetime.now(timezone.utc).isoformat(), months={}, games_checked=0,
                  position_rows_checked=0, parity_states_checked=0, mismatches=0,
                  source_archives_redownloaded=False, dataset_d_status='not_checked_deferred',
                  live_generation_modified=False)
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    def save():
        report.update(updated_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.monotonic()-started,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024))
        temporary = destination.with_suffix('.json.partial')
        temporary.write_text(json.dumps(report, sort_keys=True, indent=2)+'\n')
        temporary.replace(destination)
    save()
    try:
        config = yaml.safe_load(Path(args.config).read_text())
        require(config.get('dataset_d_policy') == 'deferred', 'This audit expects the recorded deferred Dataset D policy')
        config = dict(config, frozen_evaluation_fens=[])
        sources = {s['month']: s for s in json.loads(Path(config['source_manifest']).read_text())['sources']}
        seen = set()
        os.nice(10)  # Yield CPU scheduling priority to the active extraction.
        with tempfile.TemporaryDirectory(prefix='hf-row-audit-', dir=destination.parent) as folder:
            remote = RemoteReader(args.repo, args.revision, Path(folder))
            # Resolve/access-check this exact immutable repository revision once.
            info = HfApi().repo_info(args.repo, repo_type='dataset', revision=args.revision)
            require(info.sha == args.revision, 'Audit revision must be an exact immutable commit')
            for month in args.months:
                quota = expected_games(config, month)
                base = f'data/{args.run_id}'
                manifest = remote.read_json(f'{base}/manifests/{month}.json')
                repro = remote.read_json(f'{base}/manifests/{month}.reproduction.json')
                require(repro['status'] == 'REPRODUCTION_PASS' and repro['shard_digest'] == manifest['shard_digest'], 'Original reproduction receipt mismatch')
                require(manifest['accepted_games'] == quota and manifest['dataset_d_status'] == 'not_checked_deferred', 'Unexpected month count or evaluation policy')
                source = manifest['source']
                for key in ['month', 'url', 'archive_size_bytes', 'published_sha256']:
                    require(source[key] == sources[month][key], 'Frozen source identity mismatch: '+key)
                selection = remote.fetch(manifest['selection']['path_in_repo'], manifest['selection'])
                records, decoded_bytes = {}, 0
                with selection.open('rb') as compressed:
                    with zstd.ZstdDecompressor().stream_reader(compressed) as decoder:
                        with io.TextIOWrapper(decoder, encoding='utf-8') as lines:
                            for line in lines:
                                decoded_bytes += len(line.encode())
                                require(decoded_bytes <= 128*2**20, 'Selected PGN metadata exceeds memory allowance')
                                value = json.loads(line)
                                candidate = value['candidate']
                                require(candidate['game_id'] not in records, 'Duplicate selected game ID')
                                require(hashlib.sha256(candidate['pgn'].encode()).hexdigest() == candidate['raw_pgn_sha256'], 'Selected raw PGN hash mismatch')
                                records[candidate['game_id']] = value
                selection.unlink()
                require(len(records) == quota, 'Selection input count mismatch')
                result = dict(status='RUNNING', games=0, rows=0, parity_states=0, mismatches=0,
                              strong_games=0, time_forfeit_games=0, time_forfeit_rows=0,
                              days={}, wdl_rows={}, shard_pairs_checked=0)
                report['months'][month] = result
                days, outcomes, ids, shard_hashes = Counter(), Counter(), [], []
                for shard in manifest['shards']:
                    paths = {f['kind']: remote.fetch(f['path_in_repo'], f) for f in shard['files']}
                    rows = iter(parquet_rows(paths['positions'], POSITION_SCHEMA))
                    shard_ids, row_count = [], 0
                    for game_row in parquet_rows(paths['games'], GAME_SCHEMA):
                        game_id = game_row['game_id']
                        require(game_id not in seen and game_id in records, 'Duplicate or unselected game')
                        record = records.pop(game_id)
                        candidate = record['candidate']
                        replayed = replay_game(candidate['pgn'], month, candidate['source_ordinal'], source['url'], config)
                        expected = replayed.game
                        require(game_id == expected['metadata']['game_id'], 'Replayed game ID mismatch')
                        require(json.loads(game_row['metadata']) == expected['metadata'], 'Game metadata mismatch: '+game_id)
                        require(game_row['actions'] == expected['actions'], 'Action sequence mismatch: '+game_id)
                        require(logical_game_sha(replayed) == record['validation']['logical_sha256'], 'Logical replay digest mismatch')
                        require(feature_sha(replayed) == record['validation']['feature_sha256'], 'Feature replay digest mismatch')
                        for expected_row in replayed.positions:
                            actual = next(rows, None)
                            require(actual is not None, 'Missing materialized row')
                            compare_row(actual, expected_row)
                            row_count += 1
                            outcomes[str(actual['wdl_class'])] += 1
                        m = expected['metadata']
                        strong = min(m['white_elo'], m['black_elo']) >= 2000
                        result['strong_games'] += int(strong)
                        days[m['utc_date']] += 1
                        if m['value_weight'] == 0:
                            result['time_forfeit_games'] += 1
                            result['time_forfeit_rows'] += len(replayed.positions)
                        result['games'] += 1
                        result['rows'] += len(replayed.positions)
                        result['parity_states'] += len(replayed.positions)+1
                        report['games_checked'] += 1
                        report['position_rows_checked'] += len(replayed.positions)
                        report['parity_states_checked'] += len(replayed.positions)+1
                        seen.add(game_id); ids.append(game_id); shard_ids.append(game_id)
                        if report['games_checked'] % 100 == 0:
                            report.update(download_bytes=remote.bytes, downloaded_files=remote.files, peak_temporary_disk_bytes=remote.peak_disk)
                            save()
                        if args.max_games and report['games_checked'] >= args.max_games:
                            report['status'] = 'PARTIAL_DIAGNOSTIC'
                            save()
                            return report
                    require(next(rows, None) is None, 'Unexpected extra position rows')
                    require(row_count == shard['rows'] and len(shard_ids) == shard['games'], 'Shard row/game count mismatch')
                    require(hashlib.sha256(canonical(shard_ids).encode()).hexdigest() == shard['game_ids_sha256'], 'Shard game ordering mismatch')
                    for f in shard['files']:
                        shard_hashes.append(f['sha256'])
                    for path in paths.values():
                        path.unlink()
                    result['shard_pairs_checked'] += 1
                require(not records, 'Selected games absent from output shards')
                require(result['games'] == quota and result['rows'] == manifest['training_positions'], 'Month totals mismatch')
                require(result['strong_games']*5 >= result['games']*3, '60% strong-rating quota failed')
                require(hashlib.sha256(canonical(ids).encode()).hexdigest() == manifest['selected_ids_sha256'], 'Month selected ID digest mismatch')
                require(hashlib.sha256(canonical(shard_hashes).encode()).hexdigest() == manifest['shard_digest'], 'Month shard digest mismatch')
                result.update(status='PASS', days=dict(days), wdl_rows=dict(outcomes))
                report.update(download_bytes=remote.bytes, downloaded_files=remote.files, peak_temporary_disk_bytes=remote.peak_disk)
                save()
            report['status'] = 'PASS'
            report['checks'] = ['Full downloaded payload SHA256/size', 'Exact Parquet schemas and metadata',
                'All compact game metadata and actions', 'Every materialized field and feature byte against fresh PGN replay',
                'Full Pgx/python-chess per-ply legality, state/history/status/reward parity including final boards',
                'Filters, side-to-move WDL, timeout weights, horizon, order, counts and daily/rating coverage',
                'Unique game identities across the audited months', 'Original logical/feature/selected-ID/shard digests']
            save()
        return report
    except Exception as exc:
        report.update(status='FAIL', mismatches=1, error=dict(type=type(exc).__name__, message=str(exc)),
                      limitation='Failure aborts this audit; the running extraction has not been modified.')
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='ishagarg1103/gambit-zero')
    parser.add_argument('--revision', required=True)
    parser.add_argument('--run-id', default='human-v3-pilot-20261025')
    parser.add_argument('--months', nargs='+', default=['2026-01', '2026-02', '2026-03'])
    parser.add_argument('--config', default='configs/audit_human_50k.yaml')
    parser.add_argument('--output', default='reports/m1_hf_rows_audit.json')
    parser.add_argument('--max-games', type=int, default=0, help='Diagnostic limit only; never produces PASS')
    args = parser.parse_args()
    result = audit(args)
    print(json.dumps({key:result.get(key) for key in ['status','games_checked','position_rows_checked','parity_states_checked','elapsed_seconds','revision']}, indent=2))
