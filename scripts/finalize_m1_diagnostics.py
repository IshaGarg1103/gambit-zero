"""Audit published metadata and publish a position-disjoint evaluation index.

No archive scan, tensor regeneration or dataset replacement. Network execution
requires --execute. --publish additionally uploads immutable diagnostics only.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import time

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import pyarrow as pa
import pyarrow.parquet as pq

from tinychess.data.archive_scan import DiskGuard, canonical
from tinychess.data.closeout import CloseoutIndex, payload_sha
from tinychess.data.hf_upload import HubTransport, VerifiedUploader
from tinychess.data.storage import GAME_SCHEMA, POSITION_SCHEMA

PIN = 'd409fce8bd444c2d456ef4c343dbc9b5e25981eb'
INDEX_SCHEMA = pa.schema([
    ('game_id', pa.string()), ('split', pa.string()), ('ply', pa.uint16()),
    ('position_hash', pa.binary(32)), ('source_path', pa.string()),
    ('row_in_shard', pa.uint32()),
], metadata={b'schema': b'position-disjoint-index-v1', b'source_revision': PIN.encode(),
             b'selection': b'held-out board hash absent from all training plies',
             b'equivalence': b'pieces,STM,castling,legally-capturable-EP; excludes clocks/history'})
COLUMNS = ['game_id', 'source_month', 'split', 'ply', 'position_hash',
           'outcome_stm', 'wdl_class', 'value_weight', 'policy_weight', 'training_eligible']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parquet(payload, schema):
    file = pq.ParquetFile(pa.BufferReader(payload))
    require(file.schema_arrow.equals(schema, check_metadata=False)
            and file.schema_arrow.metadata == schema.metadata, 'Published Parquet schema mismatch')
    return file


def run(args):
    project = Path(__file__).resolve().parents[1]
    status = json.loads((project / 'reports/m1_50k_run_status.json').read_text())
    expected = json.loads((project / 'reports/m1_final_split_audit.json').read_text())['audit']
    require(status['status'] == 'COMPLETE' and status['final_revision'] == PIN,
            'Only the completed, pinned 50K pilot is supported')
    spool = project / 'data/processed/m1_closeout_spool'
    planned_bytes = sum(m['artifact_bytes'] for m in status['months'].values())
    # Includes a second held-out positions pass, metadata, upload and read-back.
    plan = dict(source_revision=PIN, free_disk_bytes=shutil.disk_usage(project).free,
                planned_peak_local_bytes=2**30, free_disk_reserve_bytes=20*2**30,
                network_payload_limit_bytes=5*2**30, original_artifact_bytes=planned_bytes,
                archive_bytes=0, downloads_cached=False, database_deleted_after_run=True,
                failed_unverified_uploads_retained=True)
    print(json.dumps(plan, indent=2), flush=True)
    require(plan['free_disk_bytes'] >= 21*2**30, 'Insufficient disk space for diagnostics plus reserve')
    if not args.execute:
        return dict(status='PREFLIGHT_ONLY', plan=plan)
    started = time.monotonic()
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='RUNNING', plan=plan, repo=args.repo, source_revision=PIN,
                  started_utc=datetime.now(timezone.utc).isoformat(), months={}, receipts=[],
                  dataset_d_status='not_checked_deferred', diagnostic_index_counts={},
                  original_dataset_modified=False)
    remote = HubTransport(args.repo, private=False)
    info = remote.api.repo_info(args.repo, repo_type='dataset', revision=PIN)
    require(info.sha == PIN and not info.private, 'Unexpected source revision or visibility')
    remote.deadline = time.monotonic() + 2*3600
    bytes_read = 0
    def fetch(receipt):
        nonlocal bytes_read
        require(receipt['bytes'] <= 32*2**20, 'Artifact exceeds bounded reader limit')
        require(bytes_read + receipt['bytes'] <= plan['network_payload_limit_bytes'], 'Diagnostic network limit exceeded')
        payload = b''.join(remote.iter_remote(receipt['path_in_repo'], PIN, receipt['bytes']))
        bytes_read += len(payload)
        require(len(payload) == receipt['bytes'] and payload_sha(payload) == receipt['sha256'],
                'Published artifact checksum/size mismatch')
        return payload
    def save():
        report.update(elapsed_seconds=time.monotonic()-started, downloaded_payload_bytes=bytes_read,
                      peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                      * (1 if sys.platform == 'darwin' else 1024))
        partial = destination.with_suffix('.json.partial')
        partial.write_text(json.dumps(report, sort_keys=True, indent=2)+'\n')
        partial.replace(destination)
    os.nice(10)
    uploader = None
    try:
        if args.publish:
            uploader = VerifiedUploader(args.repo, spool, spool_cap_bytes=128*2**20,
                                        deadline=remote.deadline, transfer_budget_bytes=256*2**20)
        with tempfile.TemporaryDirectory(prefix='m1-closeout-', dir=destination.parent) as folder:
            guard = DiskGuard(folder, 2**30, 20*2**30)
            index = CloseoutIndex(folder, PIN, guard)
            try:
                manifests = []
                for month, entry in sorted(status['months'].items()):
                    manifest = json.loads(fetch(entry['manifest_receipt']))
                    require(manifest['month'] == month and manifest['accepted_games'] == entry['target_games']
                            and manifest['training_positions'] == entry['positions']
                            and manifest['shard_digest'] == entry['shard_digest'], 'Month receipt mismatch')
                    manifests.append(manifest)
                    row_count = game_count = 0
                    ids, hashes = [], []
                    for shard in manifest['shards']:
                        guard.check(32*2**20)
                        files = {f['kind']: f for f in shard['files']}
                        games = {}
                        index.begin()
                        for batch in parquet(fetch(files['games']), GAME_SCHEMA).iter_batches(batch_size=128):
                            for game in batch.to_pylist():
                                metadata = index.add_game(game, month, manifest['split'])
                                require(not metadata.get('utc_month_differs_from_source', False), 'UTC month mismatch')
                                games[game['game_id']] = metadata
                        seen_plies = {}
                        rows_in_shard = 0
                        for batch in parquet(fetch(files['positions']), POSITION_SCHEMA).iter_batches(batch_size=1024, columns=COLUMNS):
                            rows = batch.to_pylist()
                            index.add_rows(rows, games, month, manifest['split'], seen_plies)
                            rows_in_shard += len(rows)
                        require(rows_in_shard == shard['rows'] and len(games) == shard['games'], 'Shard counts mismatch')
                        require(all(seen_plies.get(gid) == game['plies'] for gid, game in games.items()),
                                'Incomplete game position sequence')
                        require(payload_sha(canonical(list(games)).encode()) == shard['game_ids_sha256'], 'Shard order mismatch')
                        index.commit()
                        game_count += len(games)
                        row_count += rows_in_shard
                        ids.extend(games)
                        hashes.extend(f['sha256'] for f in shard['files'])
                    require(game_count == entry['target_games'] and row_count == entry['positions'], 'Month total mismatch')
                    require(payload_sha(canonical(ids).encode()) == manifest['selected_ids_sha256']
                            and payload_sha(canonical(hashes).encode()) == manifest['shard_digest'], 'Month digest mismatch')
                    report['months'][month] = dict(games=game_count, rows=row_count, status='PASS')
                    save()
                    print(f'{month}: {game_count:,} games / {row_count:,} rows checked', flush=True)
                result = index.result()
                for scope in ['all_plies', 'ply_ge_10', 'games_by_split', 'equivalence']:
                    require(result['overlap'][scope] == expected[scope], 'Published split audit disagrees: '+scope)
                report.update(result)
                counts, late_counts = Counter(), Counter()
                # Each source shard supplies a small index shard; never all tensors.
                for manifest in manifests:
                    if manifest['split'] == 'train':
                        continue
                    for shard in manifest['shards']:
                        receipt = next(f for f in shard['files'] if f['kind'] == 'positions')
                        rows = parquet(fetch(receipt), POSITION_SCHEMA).read(columns=['game_id', 'split', 'ply', 'position_hash']).to_pylist()
                        selected = list(index.disjoint_rows(rows, receipt['path_in_repo']))
                        counts[manifest['split']] += len(selected)
                        late_counts[manifest['split']] += sum(r['ply'] >= 10 for r in selected)
                        if uploader is not None:
                            buffer = pa.BufferOutputStream()
                            pq.write_table(pa.Table.from_pylist(selected, schema=INDEX_SCHEMA), buffer,
                                           compression='zstd', compression_level=3, row_group_size=1024)
                            encoded = buffer.getvalue().to_pybytes()
                            digest = payload_sha(encoded)
                            output = spool / (digest+'.parquet')
                            if output.exists():
                                require(payload_sha(output.read_bytes()) == digest, 'Existing unverified diagnostic artifact changed')
                            else:
                                output.write_bytes(encoded)
                            target = f'diagnostics/{PIN}/position-disjoint-v1/{manifest["split"]}/{digest}.parquet'
                            report['receipts'].append(uploader.publish(output, target))
                            save()
                report.update(diagnostic_index_counts=dict(counts), diagnostic_index_ply_ge_10_counts=dict(late_counts),
                              diagnostic_selection='held-out rows whose board hash is absent from ALL train plies; '
                              'validation and test may overlap each other; no tensors or labels changed',
                              index_disk_bytes_at_end=sum(p.stat().st_size for p in Path(folder).rglob('*') if p.is_file()))
                report['status'] = 'PASS'
                save()
            finally:
                index.close()
        if uploader is not None:
            payload = (json.dumps(report, sort_keys=True, indent=2)+'\n').encode()
            path = spool / ('summary-'+payload_sha(payload)+'.json')
            path.write_bytes(payload)
            receipt = uploader.publish(path, f'diagnostics/{PIN}/summary-{payload_sha(payload)}.json')
            report['summary_receipt'] = receipt
            report['published_revision'] = receipt['revision']
            # Keep the complete journal/receipts in the small report before cleanup.
            report['upload_journal'] = json.loads(uploader.journal.read_text())
            save()
            shutil.rmtree(spool)
        report['temporary_cleanup_completed'] = True
        save()
        return report
    except Exception as exc:
        report.update(status='FAIL', error_type=type(exc).__name__,
                      error='Closeout stopped; inspect the bounded validation failure. No training data changed.')
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='ishagarg1103/gambit-zero')
    parser.add_argument('--output', default='reports/m1_50k_diagnostics.json')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--publish', action='store_true')
    arguments = parser.parse_args()
    require(not arguments.publish or arguments.execute, '--publish requires --execute')
    result = run(arguments)
    print(json.dumps({k: result[k] for k in ['status', 'diagnostic_index_counts', 'published_revision'] if k in result}, indent=2))
