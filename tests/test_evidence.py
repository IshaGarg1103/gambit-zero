"""Checksummed offline evidence recovery without restoring generated reports."""
import hashlib
import io
import json
import stat
import zipfile

import pytest

from tinychess.data.evidence import EvidenceError, EvidenceStore, rewrite_archived_links


@pytest.fixture
def evidence(tmp_path):
    payload = b'{"status":"PASS","count":50000}\n'
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as bundle:
        bundle.writestr('reports/old.json', payload)
    packed = stream.getvalue()
    manifest = dict(archive=dict(bytes=len(packed), sha256=hashlib.sha256(packed).hexdigest()),
                    files={'reports/old.json': dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())})
    (tmp_path/'reports').mkdir()
    (tmp_path/'reports/m1_evidence_archive.json').write_text(json.dumps(manifest))
    cache = tmp_path/'data/processed/m1_evidence'/f"{manifest['archive']['sha256']}.zip"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(packed)
    return EvidenceStore(tmp_path, offline=True), cache


def test_offline_recovery_does_not_restore_reports(evidence):
    store, _ = evidence
    assert store.read_json('reports/old.json') == {'status': 'PASS', 'count': 50000}
    assert not (store.root/'reports/old.json').exists()


def test_local_runtime_results_take_precedence(evidence):
    store, _ = evidence
    (store.root/'reports/old.json').write_text('{"status":"RUNNING"}')
    assert store.read_json('reports/old.json') == {'status': 'RUNNING'}
    assert json.loads(store.read_archived_bytes('reports/old.json'))['status'] == 'PASS'


def test_corrupt_archive_rejected(evidence):
    store, cache = evidence
    payload = bytearray(cache.read_bytes())
    payload[-1] ^= 1
    cache.write_bytes(payload)
    with pytest.raises(EvidenceError, match='SHA256'):
        store.read_bytes('reports/old.json')


@pytest.mark.parametrize('field,value,message', [
    ('sha256', '0'*64, 'SHA256'),
    ('bytes', 1, 'size/type'),
    ('bytes', 9*2**20, 'decompression bound'),
])
def test_member_integrity_and_bounds(evidence, field, value, message):
    store, _ = evidence
    store.manifest['files']['reports/old.json'][field] = value
    with pytest.raises(EvidenceError, match=message):
        store.read_bytes('reports/old.json')


def test_missing_cache_offline_is_explicit(evidence):
    store, cache = evidence
    cache.unlink()
    with pytest.raises(FileNotFoundError, match='fetch_m1_evidence'):
        store.read_bytes('reports/old.json')


@pytest.mark.parametrize('path', ['../outside', '/etc/passwd'])
def test_path_escape_rejected(evidence, path):
    store, _ = evidence
    with pytest.raises(EvidenceError):
        store.read_bytes(path)


def test_archive_symlink_rejected(evidence):
    store, _ = evidence
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as bundle:
        info = zipfile.ZipInfo('reports/old.json')
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        bundle.writestr(info, b'/etc/passwd')
    payload = stream.getvalue()
    spec = store.manifest['archive']
    spec.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    store.manifest['files']['reports/old.json']['bytes'] = len(b'/etc/passwd')
    cache = store.root/'data/processed/m1_evidence'/f"{spec['sha256']}.zip"
    cache.write_bytes(payload)
    with pytest.raises(EvidenceError, match='size/type'):
        store.read_bytes('reports/old.json')


def test_missing_optional_and_required_evidence(evidence):
    store, _ = evidence
    assert store.read_json('reports/absent.json', required=False) is None
    with pytest.raises(FileNotFoundError):
        store.read_json('reports/absent.json')


def test_document_links_redirect_to_archive_catalog(evidence):
    store, _ = evidence
    text = '[old](old.json) [remote](https://example.org/test.json)'
    result = rewrite_archived_links(text, 'reports/report.md', root=store.root)
    assert result == '[old](m1_evidence_archive.md#reports-old-json) [remote](https://example.org/test.json)'
