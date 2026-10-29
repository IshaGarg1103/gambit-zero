"""Verify the final immutable Dataset A/D publication without caching tensors.

Changed shards were fully read back during upload; all original shards were
fully checked during filtering/earlier audits. This additional check compares
their final-revision LFS SHA256 IDs and fully reads the small manifests/card.
It does not pretend to download every tensor shard a second time.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import yaml

from tinychess.data.hf_upload import HubTransport
from tinychess.data.evidence import read_json_evidence
from tinychess.data.publication import check_card_paths, check_object


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run():
    started = time.monotonic()
    source = read_json_evidence('reports/m1_dataset_d_exclusions.json')
    require(source['status'] == 'PASS' and source['temporary_cleanup_completed'],
            'Exclusion publication must finish before the final revision check')
    original = read_json_evidence('reports/m1_50k_run_status.json')
    repo, revision = 'ishagarg1103/gambit-zero', source['final_revision']
    remote = HubTransport(repo, private=False)
    remote.deadline = time.monotonic()+1800
    info = remote.api.repo_info(repo, repo_type='dataset', revision=revision)
    require(info.sha == revision and not info.private, 'Unexpected final revision or visibility')
    report = dict(status='RUNNING', repo=repo, revision=revision,
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  verified_months=[], full_read_payload_bytes=0,
                  lfs_objects_checked=0, regular_objects_read_back=0,
                  file_metadata_only_for_large_tensors=True, download_cache_used=False)
    target = Path('reports/m1_dataset_d_publication_verification.json')
    def save():
        report['elapsed_seconds'] = time.monotonic()-started
        temporary = target.with_suffix('.json.partial')
        temporary.write_text(json.dumps(report, sort_keys=True, indent=2)+'\n')
        temporary.replace(target)
    def fetch(receipt):
        require(receipt['bytes'] <= 32*2**20, 'Publication object exceeds read bound')
        require(report['full_read_payload_bytes']+receipt['bytes'] <= 128*2**20,
                'Final small-object read-back exceeds 128 MiB budget')
        payload = b''.join(remote.iter_remote(receipt['path_in_repo'], revision, receipt['bytes']))
        require(len(payload) == receipt['bytes'] and hashlib.sha256(payload).hexdigest() == receipt['sha256'],
                'Final-revision object read-back mismatch')
        report['full_read_payload_bytes'] += len(payload)
        return payload
    try:
        catalog = json.loads(fetch(source['catalog_receipt']))
        require(catalog['parent_data_revision'] == source['parent_revision']
                and catalog['dataset_d_status'] == 'checked_frozen_snapshot', 'Final catalog evaluation status mismatch')
        require(catalog['counts'] == source['counts']
                and {s: v['games'] for s, v in catalog['counts'].items()} == dict(train=35000, validation=7500, test=7500),
                'Published split counts disagree')
        frozen_receipt = catalog['dataset_d']
        require(frozen_receipt['sha256'] == source['frozen_receipt']['sha256'], 'Frozen evaluation identity changed')
        frozen = json.loads(fetch(frozen_receipt))
        require(len(frozen['puzzle_ids']) == 1000 and len(frozen['opening_suite']) == 32, 'Unexpected frozen suite size')
        audit = json.loads(fetch(catalog['audit']))
        require(audit == source['postfilter_overlap'], 'Published overlap report changed')
        for scope in ['all_plies', 'ply_ge_10']:
            require(audit[scope]['train_dataset_d_shared_unique'] == 0, 'Dataset D remains in training')
        expected_files, expected_paths = {}, {kind: {} for kind in ['positions', 'games']}
        manifest_counts = {split: dict(games=0, positions=0) for split in catalog['counts']}
        require(len(catalog['months']) == 9, 'Incomplete month catalog')
        for entry in catalog['months']:
            month = entry['month']
            receipt = source['months'][month]['manifest_receipt']
            require(entry['manifest_path'] == receipt['path_in_repo'], 'Month manifest path mismatch')
            manifest = json.loads(fetch(receipt))
            parent = original['months'][month]
            require(manifest['month'] == month and manifest['split'] == parent['split']
                    and manifest['accepted_games'] == parent['target_games']
                    and manifest['source'] == parent['source'], 'Frozen source/game/split identity changed')
            require(manifest['dataset_d_sha256'] == frozen_receipt['sha256'], 'Month excludes a different evaluation suite')
            require(manifest['training_positions'] == entry['training_positions']
                    and manifest['shard_digest'] == entry['shard_digest'], 'Month totals/digest disagree with catalog')
            files = [file for shard in manifest['shards'] for file in shard['files']]
            from tinychess.data.archive_scan import canonical
            require(hashlib.sha256(canonical([f['sha256'] for f in files]).encode()).hexdigest() == manifest['shard_digest'],
                    'Monthly object digest mismatch')
            require(sum(s['rows'] for s in manifest['shards']) == manifest['training_positions']
                    and sum(s['games'] for s in manifest['shards']) == manifest['accepted_games'], 'Shard totals mismatch')
            excluded = parent['positions']-manifest['training_positions']
            require(excluded == manifest['postfilter']['excluded_rows']
                    and (manifest['split'] == 'train' or excluded == 0), 'Wrong exclusion split/count')
            for file in files:
                previous = expected_files.get(file['path_in_repo'])
                require(previous is None or (previous['sha256'], previous['bytes']) == (file['sha256'], file['bytes']),
                        'Conflicting reused artifact')
                expected_files[file['path_in_repo']] = file
                expected_paths[file['kind']].setdefault(manifest['split'], []).append(file['path_in_repo'])
            manifest_counts[manifest['split']]['games'] += manifest['accepted_games']
            manifest_counts[manifest['split']]['positions'] += manifest['training_positions']
            report['verified_months'].append(month)
        require(manifest_counts == catalog['counts'], 'Aggregated monthly counts mismatch')
        card_receipt = dict(source['root_publication']['files']['README.md'])
        card = fetch(card_receipt).decode()
        _, frontmatter, _ = card.split('---', 2)
        check_card_paths(yaml.safe_load(frontmatter), expected_paths)
        require(f'Run: `{catalog["run_id"]}`.' in card
                and f'**Status: {catalog["status"]}.**' in card, 'Card run/version is stale')
        for split, counts in catalog['counts'].items():
            require(f'| {split} | {counts["games"]} | {counts["positions"]} |' in card,
                    'Human-readable card split counts are stale')
        require('Dataset D is deferred and its overlap has NOT been checked.' not in card,
                'Stale evaluation disclaimer remains in revised card')
        names = sorted(expected_files)
        for start in range(0, len(names), 50):
            batch = names[start:start+50]
            info = remote.api.get_paths_info(repo, batch, repo_type='dataset', revision=revision)
            require({f.path for f in info} == set(batch), 'Final-revision shard missing')
            for file in info:
                receipt = expected_files[file.path]
                if check_object(file, receipt):
                    report['lfs_objects_checked'] += 1
                else:
                    fetch(receipt)
                    report['regular_objects_read_back'] += 1
            save()
        report.update(status='PASS', counts=catalog['counts'],
                      unique_primary_objects=len(expected_files),
                      primary_object_bytes=sum(f['bytes'] for f in expected_files.values()),
                      train_dataset_d_shared_unique=0,
                      card_and_catalog_same_atomic_revision=True,
                      original_revision_preserved=source['parent_revision'],
                      finished_utc=datetime.now(timezone.utc).isoformat())
        save()
        print(json.dumps({k: report[k] for k in ['status', 'revision', 'counts', 'unique_primary_objects']}, indent=2))
    except Exception as error:
        report.update(status='FAIL', error_type=type(error).__name__)
        save()
        raise


if __name__ == '__main__':
    run()
