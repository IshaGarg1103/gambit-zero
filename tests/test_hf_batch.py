"""Batching reduces commits without weakening verification or recovery."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tinychess.data.hf_batch import publish_batch
from tinychess.data.hf_upload import UploadError, VerifiedUploader, safe_hf_errors


class BatchRemote:
    def __init__(self):
        self.current = 'base'
        self.versions = {'base': {}}
        self.commits = 0
        self.lose_response = False
        self.corrupt = None
        self.api = SimpleNamespace(create_commit=self.commit)

    def head(self):
        return self.current

    def exists(self, name, revision):
        return name in self.versions[revision]

    def hash_remote(self, name, revision, maximum):
        value = self.versions[revision][name]
        assert len(value) <= maximum
        return ('0'*64 if self.corrupt == name else hashlib.sha256(value).hexdigest(), len(value))

    def commit(self, repo, *, operations, parent_commit, **kwargs):
        assert parent_commit == self.current
        self.commits += 1
        value = dict(self.versions[self.current])
        value.update({op.path_in_repo: Path(op.path_or_fileobj).read_bytes() for op in operations})
        self.current = str(self.commits)
        self.versions[self.current] = value
        if self.lose_response:
            self.lose_response = False
            raise OSError('lost response')
        return SimpleNamespace(oid=self.current)


def setup(tmp_path):
    uploader = VerifiedUploader('fixture/dataset', tmp_path, reserve_disk_bytes=2**20,
                                transfer_budget_bytes=16*2**20)
    files = {f'data/fixture/{name}': tmp_path/name for name in ['a', 'b']}
    for name, file in files.items():
        file.write_bytes(name.encode())
    return uploader, BatchRemote(), files


def test_batch_one_commit_checks_all_files_before_cleanup(tmp_path):
    uploader, remote, files = setup(tmp_path)
    receipts = publish_batch(uploader, remote, files)
    assert remote.commits == 1
    assert {entry['revision'] for entry in receipts.values()} == {'1'}
    assert all(entry['status'] == 'deleted' for entry in receipts.values())
    assert not any(path.exists() for path in files.values())


def test_batch_lost_response_recovers_without_duplicate_commit(tmp_path):
    uploader, remote, files = setup(tmp_path)
    remote.lose_response = True
    with pytest.raises(UploadError):
        publish_batch(uploader, remote, files)
    assert all(path.exists() for path in files.values())
    publish_batch(uploader, remote, files)
    assert remote.commits == 1


def test_batch_readback_failure_preserves_every_local_shard(tmp_path):
    uploader, remote, files = setup(tmp_path)
    remote.corrupt = 'data/fixture/b'
    with pytest.raises(UploadError, match='all local shards retained'):
        publish_batch(uploader, remote, files)
    assert all(path.exists() for path in files.values())


def test_batch_never_replaces_conflicting_immutable_bytes(tmp_path):
    uploader, remote, files = setup(tmp_path)
    remote.versions['base']['data/fixture/a'] = b'x'*len(files['data/fixture/a'].read_bytes())
    with pytest.raises(UploadError, match='Conflicting'):
        publish_batch(uploader, remote, files)
    assert remote.commits == 0
    assert all(path.exists() for path in files.values())


def test_batch_does_not_accept_mutable_root_documents(tmp_path):
    uploader, remote, files = setup(tmp_path)
    with pytest.raises(UploadError, match='immutable dataset'):
        publish_batch(uploader, remote, {'README.md': next(iter(files.values()))})


def test_safe_http_errors_expose_rate_limit_without_credentials():
    class FakeHTTPError(Exception):
        response = SimpleNamespace(status_code=429, headers={'retry-after': '3471'})
    @safe_hf_errors
    def request():
        raise FakeHTTPError('Authorization: Bearer fixture-secret')
    with pytest.raises(UploadError) as caught:
        request()
    assert caught.value.http_status == 429
    assert caught.value.retry_after_seconds == 3471
    assert 'fixture-secret' not in str(caught.value)
