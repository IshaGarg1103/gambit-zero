"""Atomically publish the dataset card/catalog; keep both until verified."""
import hashlib
from pathlib import Path

from tinychess.data.archive_scan import atomic_json
from tinychess.data.hf_upload import UploadError
from tinychess.data.hf_batch import commit_batch

from huggingface_hub import CommitOperationAdd


def publish_root_atomic(uploader, remote, files, parent_hashes):
    if set(files) != {'README.md', 'dataset_manifest.json'}:
        raise UploadError('Atomic publication is limited to the card and catalog')
    entries = {}
    for name, local in files.items():
        local = Path(local)
        if local.is_symlink() or not local.resolve().is_relative_to(uploader.root):
            raise UploadError('Root metadata must be inside the managed spool')
        if local.stat().st_size > 2**20:
            raise UploadError('Root metadata exceeds 1 MiB per-file bound')
        payload = local.read_bytes()
        entries[name] = dict(path_in_repo=name, local_name=local.name,
                             bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    journal = uploader.root/'root_publication.json'
    state = dict(status='sealed', files=entries, repo_id=uploader.repo_id)
    atomic_json(journal, state, uploader.guard)
    try:
        uploader.check_runtime()
        uploader.reserve_transfer(2*2**20 + 2*sum(e['bytes'] for e in entries.values()))
        parent = remote.head()
        current = {name: remote.hash_remote(name, parent, 2**20)[0] for name in entries}
        if current == {name: entry['sha256'] for name, entry in entries.items()}:
            revision = parent  # Recover a lost response without a second commit.
        else:
            if current != parent_hashes:
                raise UploadError('Root metadata changed externally; refusing to overwrite it')
            operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(files[name])) for name in sorted(files)]
            revision = commit_batch(remote, uploader.repo_id, operations, parent,
                                    message='Publish verified Dataset A evaluation exclusions')
        state.update(status='uploaded', revision=revision)
        atomic_json(journal, state, uploader.guard)
        for name, entry in entries.items():
            if remote.hash_remote(name, revision, entry['bytes']) != (entry['sha256'], entry['bytes']):
                raise UploadError('Atomic root read-back mismatch; both local files retained')
        state['status'] = 'verified'
        atomic_json(journal, state, uploader.guard)
        # Durable verification of BOTH files precedes deletion of EITHER.
        for name, local in files.items():
            if hashlib.sha256(Path(local).read_bytes()).hexdigest() != entries[name]['sha256']:
                raise UploadError('Root metadata changed locally during verification')
        for local in files.values():
            Path(local).unlink()
        state['status'] = 'deleted'
        atomic_json(journal, state, uploader.guard)
        return {name: dict(entry, revision=revision,
                           verification='atomic card/catalog commit; full streamed SHA256/size read-back')
                for name, entry in entries.items()}
    except Exception as exc:
        state.update(status='failed', error_type=type(exc).__name__)
        atomic_json(journal, state, uploader.guard)
        if isinstance(exc, UploadError):
            raise
        raise UploadError('Atomic root publication failed; credentials omitted; local files retained') from None
