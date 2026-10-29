"""Card/catalog publication must be atomic and recover from lost responses."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tinychess.data.hf_root import publish_root_atomic
from tinychess.data.hf_upload import UploadError, VerifiedUploader


class RootRemote:
    def __init__(self):
        self.revision = 'base'
        self.versions = {'base': {'README.md': b'old card', 'dataset_manifest.json': b'old catalog'}}
        self.commits = 0
        self.lose_response = False
        self.corrupt = None
        self.api = SimpleNamespace(create_commit=self.commit)

    def head(self):
        return self.revision

    def hash_remote(self, name, revision, maximum):
        payload = self.versions[revision][name]
        assert len(payload) <= maximum
        if self.corrupt == name and revision != 'base':
            return '0'*64, len(payload)
        return hashlib.sha256(payload).hexdigest(), len(payload)

    def commit(self, repo_id, *, operations, parent_commit, **kwargs):
        assert repo_id == 'fixture/dataset'
        assert parent_commit == self.revision
        assert {op.path_in_repo for op in operations} == {'README.md', 'dataset_manifest.json'}
        self.commits += 1
        self.revision = str(self.commits)
        self.versions[self.revision] = {op.path_in_repo: Path(op.path_or_fileobj).read_bytes() for op in operations}
        if self.lose_response:
            self.lose_response = False
            raise OSError('lost response')
        return SimpleNamespace(oid=self.revision)


def setup(tmp_path):
    uploader = VerifiedUploader('fixture/dataset', tmp_path, reserve_disk_bytes=2**20,
                                transfer_budget_bytes=16*2**20)
    files = {name: tmp_path/('local-'+name) for name in ['README.md', 'dataset_manifest.json']}
    for name, path in files.items():
        path.write_bytes(('new '+name).encode())
    remote = RootRemote()
    parents = {name: hashlib.sha256(value).hexdigest() for name, value in remote.versions['base'].items()}
    return uploader, remote, files, parents


def test_atomic_root_is_one_commit_verified_before_cleanup(tmp_path):
    uploader, remote, files, parents = setup(tmp_path)
    receipts = publish_root_atomic(uploader, remote, files, parents)
    assert remote.commits == 1
    assert {r['revision'] for r in receipts.values()} == {'1'}
    assert not any(path.exists() for path in files.values())


def test_lost_response_preserves_files_then_recovers_without_duplicate_commit(tmp_path):
    uploader, remote, files, parents = setup(tmp_path)
    remote.lose_response = True
    with pytest.raises(UploadError):
        publish_root_atomic(uploader, remote, files, parents)
    assert all(path.exists() for path in files.values())
    receipts = publish_root_atomic(uploader, remote, files, parents)
    assert remote.commits == 1
    assert {r['revision'] for r in receipts.values()} == {'1'}


def test_readback_failure_preserves_both_root_files(tmp_path):
    uploader, remote, files, parents = setup(tmp_path)
    remote.corrupt = 'dataset_manifest.json'
    with pytest.raises(UploadError, match='both local files retained'):
        publish_root_atomic(uploader, remote, files, parents)
    assert all(path.exists() for path in files.values())


def test_external_root_change_cannot_be_overwritten(tmp_path):
    uploader, remote, files, parents = setup(tmp_path)
    remote.versions['base']['README.md'] = b'external edit'
    with pytest.raises(UploadError, match='changed externally'):
        publish_root_atomic(uploader, remote, files, parents)
    assert remote.commits == 0
    assert all(path.exists() for path in files.values())
