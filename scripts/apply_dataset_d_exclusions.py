"""Publish an auditable, lossless Dataset A derivative excluding frozen D boards.

The original immutable revision stays accessible. Only affected training shards
are rewritten; every retained field is compared with the original, and each
rewritten shard is independently filtered twice before upload. No chess features
or labels are regenerated. Default invocation is a network-free resource plan.
"""
import argparse
from collections import Counter
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import threading
import time

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from tinychess.data.archive_scan import DiskGuard, canonical
from tinychess.data.eval_filter import position_keys, prefetched, verify_retained_rows, write_filter
from tinychess.data.evidence import evidence_exists, read_json_evidence
from tinychess.data.hf_audit import AuditIndex, BANDS
from tinychess.data.hf_batch import publish_batch
from tinychess.data.hf_root import publish_root_atomic
from tinychess.data.hf_upload import HubTransport, VerifiedUploader

PIN = 'd409fce8bd444c2d456ef4c343dbc9b5e25981eb'
REPO = 'ishagarg1103/gambit-zero'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(args):
    diagnostic = Path('reports/m1_dataset_d_upload_diagnostic.json')
    if args.execute and evidence_exists(diagnostic):
        previous = read_json_evidence(diagnostic)
        resume = previous.get('resume_not_before_utc')
        if resume and datetime.now(timezone.utc) < datetime.fromisoformat(resume):
            raise SystemExit('Stopped before network access: Hub retry window ends '+resume)
    status = read_json_evidence('reports/m1_50k_run_status.json')
    freeze = read_json_evidence('reports/m1_dataset_d_freeze.json')
    diagnostics = read_json_evidence('reports/m1_50k_diagnostics.json')
    require(diagnostics['status'] == 'PASS' and diagnostics.get('temporary_cleanup_completed'),
            'Finish the original diagnostic upload before publishing the revised catalog')
    # A completed run must not reread gigabytes or create fresh receipt commits.
    # Rechecking the immutable remote inventory is a separate read-only command.
    completed_path = Path(args.output)
    verification_path = Path('reports/m1_dataset_d_publication_verification.json')
    if args.execute and evidence_exists(completed_path) and evidence_exists(verification_path):
        completed = read_json_evidence(completed_path)
        verification = read_json_evidence(verification_path)
        if (completed.get('status') == 'PASS' and completed.get('temporary_cleanup_completed')
                and completed.get('parent_revision') == PIN
                and completed['frozen_receipt']['sha256'] == freeze['frozen_receipt']['sha256']
                and verification.get('status') == 'PASS'
                and verification.get('revision') == completed.get('final_revision')):
            print(json.dumps(dict(status='ALREADY_VERIFIED', revision=completed['final_revision'],
                  counts=completed['counts'], evidence='saved completed publication and independent verification',
                  recheck_command='.venv/bin/python scripts/verify_m1_dataset_d_publication.py'), indent=2))
            return
    free = shutil.disk_usage('.').free
    # 1 GiB index + rollback upper bound, 128 MiB spool and 128 MiB margin.
    plan = dict(free_disk_bytes=free, planned_peak_local_bytes=1280*2**20,
                free_disk_reserve_bytes=20*2**30, source_archive_bytes=0,
                source_hub_download_limit_bytes=3*2**30,
                upload_and_readback_payload_limit_bytes=5*2**30,
                maximum_local_shard_copies=2, source_prefetch_workers=2,
                compressed_payload_memory_bound_bytes=128*2**20, original_revision=PIN)
    print(json.dumps(plan, indent=2), flush=True)
    require(free >= plan['free_disk_reserve_bytes']+plan['planned_peak_local_bytes'], 'Insufficient storage reserve')
    if not args.execute:
        return
    started = time.monotonic()
    destination = Path(args.output)
    spool = Path('data/processed/m1_dataset_d_filter_spool')
    remote = HubTransport(REPO, private=False)
    deadline = started + 2*3600
    remote.deadline = deadline
    uploaded = VerifiedUploader(REPO, spool, spool_cap_bytes=128*2**20,
                                deadline=deadline, transfer_budget_bytes=5*2**30)
    bytes_read = 0
    reserved_source_bytes = 0
    transfer_lock = threading.Lock()
    report = dict(status='RUNNING', plan=plan, parent_revision=PIN, months={},
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  frozen_receipt=freeze['frozen_receipt'], rewritten_shards=[],
                  original_data_preserved=True, tensors_regenerated=False)
    report['prior_attempt_reports'] = [str(path) for path in sorted(Path('reports').glob('m1_dataset_d_exclusions_attempt*.json'))]
    def save():
        report.update(elapsed_seconds=time.monotonic()-started,
                      source_downloaded_payload_bytes=bytes_read,
                      peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                      * (1 if sys.platform == 'darwin' else 1024))
        partial = destination.with_suffix('.json.partial')
        partial.write_text(json.dumps(report, sort_keys=True, indent=2)+'\n')
        partial.replace(destination)
    def fetch(receipt, revision=PIN):
        nonlocal bytes_read, reserved_source_bytes
        require(receipt['bytes'] <= 32*2**20, 'Object exceeds bounded read limit')
        with transfer_lock:
            require(reserved_source_bytes + receipt['bytes'] <= 3*2**30, 'Source Hub transfer limit exceeded')
            reserved_source_bytes += receipt['bytes']
        payload = b''.join(remote.iter_remote(receipt['path_in_repo'], revision, receipt['bytes']))
        with transfer_lock:
            bytes_read += len(payload)
        require(len(payload) == receipt['bytes'] and hashlib.sha256(payload).hexdigest() == receipt['sha256'],
                'Source artifact checksum mismatch')
        return payload
    def publish_json(value, prefix):
        payload = (json.dumps(value, sort_keys=True, indent=2)+'\n').encode()
        sha = hashlib.sha256(payload).hexdigest()
        path = spool / (sha+'.json')
        if path.exists():
            require(path.read_bytes() == payload, 'Existing unverified metadata changed')
        else:
            path.write_bytes(payload)
        return uploaded.publish(path, prefix+'-'+sha+'.json')
    save()
    frozen = json.loads(fetch(freeze['frozen_receipt'], freeze['frozen_receipt']['revision']))
    keys = position_keys(frozen['position_fens'])
    run_id = 'human-v3-pilot-20261025-dataset-d-v1'
    replacements, revised_months = {}, []
    all_matches = Counter({split: 0 for split in BANDS})
    late_matches = Counter({split: 0 for split in BANDS})
    matches_unique = {split: set() for split in BANDS}
    try:
        with tempfile.TemporaryDirectory(prefix='m1-eval-filter-', dir='reports') as folder:
            guard = DiskGuard(folder, 2**30, 20*2**30)
            index = AuditIndex(folder, freeze['frozen_receipt']['sha256'], guard)
            try:
                for month, entry in sorted(status['months'].items()):
                    original = json.loads(fetch(entry['manifest_receipt']))
                    require(original['month'] == month and original['accepted_games'] == entry['target_games'], 'Month identity mismatch')
                    manifest = copy.deepcopy(original)
                    removed, rows, games = 0, 0, 0
                    pending = []
                    def flush():
                        if not pending:
                            return
                        receipts = publish_batch(uploaded, remote, {record['remote']: record['local'] for record in pending})
                        for record in pending:
                            replacement = receipts[record['remote']]
                            parent = record['parent']
                            replacements[parent['path_in_repo']] = replacement['path_in_repo']
                            report['rewritten_shards'].append(dict(parent=parent, filtered=replacement,
                                excluded_rows=record['excluded'], retained_rows=record['retained'],
                                reproduction_sha256=record['sha'], every_retained_field_compared=True))
                            record['file'].update({k: replacement[k] for k in ['path_in_repo', 'bytes', 'sha256', 'revision', 'verification']})
                        pending.clear()
                        save()
                    def fetch_positions(shard):
                        return fetch(next(file for file in shard['files'] if file['kind'] == 'positions'))
                    for shard, payload in prefetched(manifest['shards'], fetch_positions, workers=2):
                        guard.check(64*2**20)
                        index.begin()
                        file = next(f for f in shard['files'] if f['kind'] == 'positions')
                        parquet = pq.ParquetFile(pa.BufferReader(payload))
                        t, l = BANDS[manifest['split']]
                        matched = shard_rows = 0
                        for batch in parquet.iter_batches(batch_size=1024, columns=['game_id', 'source_month', 'split', 'ply', 'position_hash']):
                            updates = []
                            for row in batch.to_pylist():
                                require(row['source_month'] == month and row['split'] == manifest['split'], 'Source position split mismatch')
                                if row['ply'] == 0:
                                    index.db.execute('INSERT INTO games VALUES(?,?,?,?)',
                                                     (row['game_id'], month, manifest['split'], 'inherited-verified-original'))
                                    games += 1
                                match = row['position_hash'] in keys
                                if match:
                                    all_matches[manifest['split']] += 1
                                    late_matches[manifest['split']] += int(row['ply'] >= 10)
                                    matches_unique[manifest['split']].add(row['position_hash'])
                                if match and manifest['split'] == 'train':
                                    matched += 1
                                else:
                                    updates.append((row['position_hash'], 1, int(row['ply'] >= 10)))
                                shard_rows += 1
                            index.db.executemany(f'INSERT INTO positions(k,{t},{l}) VALUES(?,?,?) '
                                                f'ON CONFLICT(k) DO UPDATE SET {t}={t}+1,{l}={l}+excluded.{l}', updates)
                        require(shard_rows == shard['rows'], 'Source shard count mismatch')
                        if matched:
                            first = spool/(file['sha256']+'.filtered.parquet')
                            second = spool/(file['sha256']+'.repeat.parquet')
                            if first.exists():
                                count = verify_retained_rows(payload, keys, first)
                                a = (count, hashlib.sha256(first.read_bytes()).hexdigest())
                            else:
                                a = write_filter(payload, keys, first)
                            require(first.stat().st_size <= 32*2**20, 'Filtered shard exceeds 32 MiB bound')
                            b = write_filter(payload, keys, second)
                            require(a == b and a[0] == shard_rows-matched, 'Independent filter reproduction failed')
                            require(verify_retained_rows(payload, keys, first) == a[0], 'Retained row comparison failed')
                            second.unlink()  # Reproducible disposable copy; first retained until remote verification.
                            pending.append(dict(local=first,
                                remote=f'data/{run_id}/positions/train/{month}/{a[1]}.positions.parquet',
                                parent=copy.deepcopy(file), file=file, excluded=matched, retained=a[0], sha=b[1]))
                            if len(pending) >= 8 or sum(record['local'].stat().st_size for record in pending) >= 32*2**20:
                                flush()
                            shard['rows'] -= matched
                        removed += matched
                        rows += shard_rows-matched
                        index.commit()
                    flush()
                    require(rows+removed == original['training_positions'] and games == original['accepted_games'], 'Month preservation/count mismatch')
                    manifest.update(training_positions=rows, dataset_d_status='checked_frozen_snapshot',
                                    dataset_d_sha256=freeze['frozen_receipt']['sha256'], dataset_d=freeze['frozen_receipt'],
                                    parent_manifest=entry['manifest_receipt'], parent_data_revision=PIN,
                                    postfilter=dict(excluded_rows=removed, accepted_game_identities_preserved=True,
                                        method='exact board-hash exclusion; all retained tensors/actions/labels copied byte-for-byte',
                                        reproduction='each changed shard filtered twice; unchanged source files reused by verified hash',
                                        compact_game_actions='complete original provenance; do not replay as unfiltered training'))
                    manifest['shard_digest'] = hashlib.sha256(canonical([f['sha256'] for s in manifest['shards'] for f in s['files']]).encode()).hexdigest()
                    receipt = publish_json(manifest, f'data/{run_id}/manifests/{month}')
                    revised_months.append((manifest, receipt))
                    report['months'][month] = dict(accepted_games=games, retained_positions=rows,
                                                  excluded_positions=removed, manifest_receipt=receipt,
                                                  shard_digest=manifest['shard_digest'])
                    save()
                    print(f'{month}: {rows:,} retained rows; {removed:,} exact evaluation matches excluded', flush=True)
                summary = index.summary(frozen['position_fens'])
                require(summary['games_by_split'] == dict(train=35000, validation=7500, test=7500), 'Game counts changed')
                for scope in ['all_plies', 'ply_ge_10']:
                    require(summary[scope]['train_dataset_d_shared_unique'] == 0, 'Evaluation leakage remains in training')
                report.update(postfilter_overlap=summary, original_dataset_d_matching_rows=dict(all_matches),
                              original_dataset_d_matching_rows_ply_ge_10=dict(late_matches),
                              original_dataset_d_matching_unique={split: len(values) for split, values in matches_unique.items()})
                from urllib.parse import urlparse
                source_ids = {urlparse(puzzle['source_row']['GameUrl']).path.split('/')[1]
                              for puzzle in frozen['puzzles']}
                overlaps = {split: [] for split in BANDS}
                for gid in sorted(source_ids):
                    row = index.db.execute('SELECT split FROM games WHERE id=?', (gid,)).fetchone()
                    if row:
                        overlaps[row[0]].append(gid)
                report['dataset_d_source_game_ids_shared'] = overlaps
                audit_receipt = publish_json(summary, f'audits/{run_id}')
            finally:
                index.close()
        # Read original root documentation from the pinned data revision. Its
        # checksums are recorded here; unchanged original revision remains usable.
        catalog_bytes = b''.join(remote.iter_remote('dataset_manifest.json', PIN, 2**20))
        card_bytes = b''.join(remote.iter_remote('README.md', PIN, 2**20))
        bytes_read += len(catalog_bytes)+len(card_bytes)
        catalog = json.loads(catalog_bytes)
        require(catalog['counts']['train']['games'] == 35000 and catalog['fully_reproduced'], 'Unexpected parent catalog')
        parent_run = catalog['run_id']
        parent_status = catalog['status']
        parent_train_positions = catalog['counts']['train']['positions']
        catalog.update(run_id=run_id, status='complete-50k-filtered-evaluation-v1',
                       dataset_d_status='checked_frozen_snapshot', dataset_d=freeze['frozen_receipt'],
                       parent_data_revision=PIN, audit=audit_receipt,
                       original_position_disjoint_diagnostics=diagnostics['summary_receipt'],
                       reproduction_method='original full Pgx/feature second builds plus byte-identical repeated exact filtering',
                       months=[dict(month=m['month'], accepted_games=m['accepted_games'], training_positions=m['training_positions'],
                                    shard_digest=m['shard_digest'], source=m['source'], dataset_d_sha256=m['dataset_d_sha256'],
                                    manifest_path=r['path_in_repo']) for m, r in revised_months])
        for split in BANDS:
            catalog['counts'][split]['positions'] = sum(m['training_positions'] for m, _ in revised_months if m['split'] == split)
        card = card_bytes.decode()
        _, metadata_text, body = card.split('---', 2)
        body = body.replace(f'Run: `{parent_run}`.', f'Run: `{run_id}`. Parent run: `{parent_run}`.', 1)
        body = body.replace(f'**Status: {parent_status}.**', f'**Status: {catalog["status"]}.**', 1)
        metadata = yaml.safe_load(metadata_text)
        for config in metadata['configs']:
            if config['config_name'] == 'human_v3':
                for group in config['data_files']:
                    group['path'] = [replacements.get(path, path) for path in group['path']]
        start = body.index('Dataset D is deferred and its overlap has NOT been checked.')
        end = body.index('\n## Upload, reproducibility and recovery', start)
        body = body[:start]+f'''Dataset D `{frozen['version']}` is frozen independently of overlap/model performance.
Exact matching training positions are physically excluded; accepted game identities
and held-out chronological data remain intact. Original data revision: `{PIN}`.
Puzzle solutions are evaluation only and are never Dataset A training labels.
Complete compact game actions remain original provenance: replaying them as
training requires the same frozen evaluation exclusions. Match pieces, side to
move, castling and legally capturable EP; clocks/history are ignored conservatively.
See `dataset_manifest.json` for the frozen suite receipt, revised monthly manifests,
postfilter overlap and the original position-disjoint diagnostic subset. That
original subset is conservative for the filtered training set, not claimed maximal.
\n'''+body[end:]
        body += '\n## Dataset D revision\n\n'+json.dumps(catalog['counts'], sort_keys=True)+'\n'
        body += '\nRewritten shards were filtered twice and every retained field/feature byte was compared with the original. '
        body += 'M1 final review and the separate GPU benchmark disposition remain required. No model training performed.\n'
        # Avoid preserving the original table's obsolete train position count.
        body = body.replace(f'| train | 35000 | {parent_train_positions} |', f'| train | 35000 | {catalog["counts"]["train"]["positions"]} |')
        payload = (json.dumps(catalog, sort_keys=True, indent=2)+'\n').encode()
        path = spool/('catalog-'+hashlib.sha256(payload).hexdigest()+'.json')
        path.write_bytes(payload)
        catalog_path = path
        payload = ('---\n'+yaml.safe_dump(metadata, sort_keys=False)+'---'+body).encode()
        path = spool/('card-'+hashlib.sha256(payload).hexdigest()+'.md')
        path.write_bytes(payload)
        root_receipts = publish_root_atomic(uploaded, remote,
            {'README.md': path, 'dataset_manifest.json': catalog_path},
            {'README.md': hashlib.sha256(card_bytes).hexdigest(),
             'dataset_manifest.json': hashlib.sha256(catalog_bytes).hexdigest()})
        catalog_receipt, card_receipt = root_receipts['dataset_manifest.json'], root_receipts['README.md']
        require(remote.head() == card_receipt['revision'], 'Repository changed before final verification')
        journal = json.loads(uploaded.journal.read_text())
        primary = {file['path_in_repo']: file['bytes'] for month, _ in revised_months
                   for shard in month['shards'] for file in shard['files']}
        report.update(status='PASS', dataset_d_status='checked_frozen_snapshot', counts=catalog['counts'],
                      audit_receipt=audit_receipt, catalog_receipt=catalog_receipt,
                      final_revision=card_receipt['revision'], final_url=f'https://huggingface.co/datasets/{REPO}',
                      parent_catalog_sha256=hashlib.sha256(catalog_bytes).hexdigest(),
                      parent_card_sha256=hashlib.sha256(card_bytes).hexdigest(),
                      referenced_primary_artifact_bytes=sum(primary.values()),
                      newly_published_artifact_bytes=sum(file['bytes'] for file in journal['files'].values())
                      +sum(file['bytes'] for file in root_receipts.values()),
                      root_publication=json.loads((uploaded.root/'root_publication.json').read_text()),
                      upload_journal=journal)
        save()
        shutil.rmtree(spool)
        report['temporary_cleanup_completed'] = True
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
    except Exception as exc:
        report.update(status='FAIL', error_type=type(exc).__name__,
                      error='Stopped; unverified output retained and original data revision preserved',
                      http_status=getattr(exc,'http_status',None),
                      retry_after_seconds=getattr(exc,'retry_after_seconds',None))
        if report['http_status'] == 429 and report['retry_after_seconds'] is not None:
            observed = datetime.now(timezone.utc)
            resume = observed+timedelta(seconds=report['retry_after_seconds']+60)
            diagnostic.write_text(json.dumps(dict(status='STOPPED', http_status=429,
                error_type=type(exc).__name__, retry_after_seconds=report['retry_after_seconds'],
                observed_utc=observed.isoformat(), resume_not_before_utc=resume.isoformat(),
                resume_not_before_ist=resume.astimezone(timezone(timedelta(hours=5,minutes=30))).isoformat()),
                sort_keys=True, indent=2)+'\n')
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', default='reports/m1_dataset_d_exclusions.json')
    run(parser.parse_args())
