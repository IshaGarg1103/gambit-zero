"""Offline integrity and dependency checks for the frozen PR #2 cleanup.

Run fetch_m1_evidence.py --verify first. This checks final results against the
original bundle; it never modifies evidence or contacts a dataset service.
"""
import ast
import csv
import hashlib
import json
from pathlib import Path
import posixpath
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from tinychess.data.evidence import EvidenceStore, rewrite_archived_links


def verify():
    store = EvidenceStore(ROOT, offline=True)
    with zipfile.ZipFile(store.archive()) as bundle:
        originals = json.loads(bundle.read('ARCHIVE_MANIFEST.json'))['files']
        index = {v['path']: dict(bytes=v['bytes'], sha256=v['sha256']) for v in originals}
        assert index == store.manifest['files'], 'Recovery indices disagree'
        assert len(originals) == 199
        assert len(bundle.namelist()) == 200
        for path, expected in index.items():
            payload = bundle.read(path)
            assert len(payload) == expected['bytes']
            assert hashlib.sha256(payload).hexdigest() == expected['sha256'], path
        acceptance = 'reports/m1_final_acceptance.json'
        assert (ROOT/acceptance).read_bytes() == bundle.read(acceptance), 'Acceptance results changed'
        pilot = 'reports/m1_50k_pilot.md'
        expected = rewrite_archived_links(bundle.read(pilot).decode(), pilot, root=ROOT)
        assert (ROOT/pilot).read_text() == expected, 'Scientific report changed beyond relocated links'
        canonical = json.loads((ROOT/'reports/m1_sources_50k.json').read_text())
        frozen = json.loads(bundle.read('reports/m1_50k_run_preflight.json'))
        identity = ['month', 'url', 'archive_size_bytes', 'published_sha256',
                    'full_archive_checksum_verified']
        assert [{k: s[k] for k in identity} for s in canonical['sources']] == [
            {k: s[k] for k in identity} for s in frozen['sources']], 'Frozen source identities changed'
    summary = json.loads((ROOT/'reports/m1_validation_summary.json').read_text())
    for path, digest in summary['protected_semantics'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    assert summary['dataset_revision'] == 'b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c'
    assert summary['total_games'] == 50000 and summary['total_retained_positions'] == 3793600
    assert summary['parity_mismatches'] == 0
    with (ROOT/'reports/pr2_file_audit.tsv').open() as f:
        audit = list(csv.DictReader(f, delimiter='\t'))
    removed = [v for v in audit if v['decision'] == 'remove']
    retained = [v for v in audit if v['decision'] == 'retain']
    scientific_files = 0
    for entry in retained:
        path = ROOT/entry['path']
        if (entry['original_sha256'] and entry['path'].startswith('reports/')
                and path.suffix in ['.json', '.txt', '.patch']):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['original_sha256'], entry['path']
            scientific_files += 1
    assert len(removed) == 101
    for entry in removed:
        assert not (ROOT/entry['path']).exists(), 'Recreated snapshot: '+entry['path']
        assert entry['original_sha256'] == index[entry['path']]['sha256']
    links = 0
    for entry in retained:
        path = ROOT/entry['path']
        assert path.is_file(), 'Missing retained file: '+entry['path']
        if path.suffix != '.md':
            continue
        for match in re.finditer(r'\]\(([^)]+)\)', path.read_text()):
            target = match[1]
            if ':' in target or target.startswith('#'):
                continue
            file, _, anchor = target.partition('#')
            resolved = (path.parent/file).resolve()
            assert resolved.exists(), f'{entry["path"]}: broken link {target}'
            if resolved.name == 'm1_evidence_archive.md' and anchor:
                assert f'id="{anchor}"' in resolved.read_text(), 'Missing archive anchor '+anchor
            links += 1
    modules = 0
    for folder in ['src', 'scripts', 'tests', 'benchmarks']:
        for path in (ROOT/folder).rglob('*.py'):
            if '.venv' not in path.parts:
                ast.parse(path.read_text(), filename=str(path))
                modules += 1
    import yaml
    for name in ['configs/data_human_50k_run.yaml', 'configs/data_human_50k_stream.yaml',
                 'configs/audit_human_50k.yaml']:
        config = yaml.safe_load((ROOT/name).read_text())
        assert (ROOT/config['source_manifest']).is_file(), name
    return dict(status='PASS', original_files_verified=len(originals), removed_files=len(removed),
                retained_pr_files=len(retained), markdown_links_checked=links,
                python_modules_parsed=modules, acceptance_json_byte_identical=True,
                original_scientific_artifacts_byte_identical=scientific_files,
                report_identical_except_archive_links=True, protected_semantics_unchanged=True,
                frozen_source_identities_unchanged=True,
                archive_revision=store.manifest['archive']['revision'],
                archive_sha256=store.manifest['archive']['sha256'])


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2, sort_keys=True))
