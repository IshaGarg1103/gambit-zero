"""Small verified immutable upload batches, reducing Hub commit pressure."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath

from tinychess.data.archive_scan import atomic_json
from tinychess.data.hf_upload import UploadError, safe_hf_errors

# hf_upload sets no-Xet/no-progress environment before the SDK's constants load.
from huggingface_hub import CommitOperationAdd


@safe_hf_errors
def commit_batch(remote, repo_id, operations, parent, message='Add verified bounded dataset shard batch'):
    return remote.api.create_commit(repo_id, operations=operations, repo_type='dataset',
        revision='main', parent_commit=parent, num_threads=2,
        commit_message=message).oid


def publish_batch(uploader, remote, files):
    """One <=64 MiB / <=8-file commit; full read-back precedes any deletion.

    `files` maps immutable repository paths to managed local paths. Failed
    uploads remain sealed in the existing journal; a lost response is recovered
    by checking the immutable remote paths, not blindly creating another commit.
    """
    if not 1 <= len(files) <= 8:
        raise UploadError('A batch must contain one to eight immutable shards')
    entries = {}
    for name, local in files.items():
        path, local = PurePosixPath(name), Path(local)
        if path.is_absolute() or '..' in path.parts or '\\' in name or not name.startswith('data/'):
            raise UploadError('Batch paths must be immutable dataset artifacts')
        if local.is_symlink() or not local.resolve().is_relative_to(uploader.root):
            raise UploadError('Batch file must be in the managed spool')
        size = local.stat().st_size
        if size > 32*2**20:
            raise UploadError('Shard exceeds 32 MiB bound')
        entries[name] = dict(local_name=local.name, path_in_repo=name, bytes=size,
                             sha256=hashlib.sha256(local.read_bytes()).hexdigest(), status='sealed')
    if sum(entry['bytes'] for entry in entries.values()) > 64*2**20:
        raise UploadError('Batch exceeds 64 MiB bound')
    with uploader.lock:
        state = json.loads(uploader.journal.read_text()) if uploader.journal.exists() else dict(
            repo_id=uploader.repo_id, private=uploader.private, files={})
        if state['repo_id'] != uploader.repo_id or state['private'] is not uploader.private:
            raise UploadError('Batch journal repository/visibility mismatch')
        for name, entry in entries.items():
            previous = state['files'].get(name)
            if previous and previous['sha256'] != entry['sha256']:
                raise UploadError('Immutable batch path has conflicting local bytes')
            state['files'][name] = entry
        atomic_json(uploader.journal, state, uploader.guard)
        try:
            uploader.check_runtime()
            # Existing objects: two read-backs. Missing: upload + read-back.
            uploader.reserve_transfer(2*sum(entry['bytes'] for entry in entries.values()))
            parent, missing = remote.head(), []
            for name, entry in entries.items():
                if remote.exists(name, parent):
                    if remote.hash_remote(name, parent, entry['bytes']) != (entry['sha256'], entry['bytes']):
                        raise UploadError('Conflicting immutable remote batch path')
                else:
                    missing.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(files[name])))
            revision = commit_batch(remote, uploader.repo_id, missing, parent) if missing else parent
            for entry in entries.values():
                entry.update(status='uploaded', revision=revision)
            atomic_json(uploader.journal, state, uploader.guard)
            for name, entry in entries.items():
                if remote.hash_remote(name, revision, entry['bytes']) != (entry['sha256'], entry['bytes']):
                    raise UploadError('Batch read-back checksum mismatch; all local shards retained')
            for entry in entries.values():
                entry.update(status='verified', verified_utc=datetime.now(timezone.utc).isoformat(),
                             verification='bounded batch commit; full cache-free streamed SHA256/size read-back')
            atomic_json(uploader.journal, state, uploader.guard)
            for name, local in files.items():
                if hashlib.sha256(Path(local).read_bytes()).hexdigest() != entries[name]['sha256']:
                    raise UploadError('Local batch artifact changed during verification')
            for name, local in files.items():
                Path(local).unlink()
                entries[name]['status'] = 'deleted'
            atomic_json(uploader.journal, state, uploader.guard)
            return {name: dict(entry) for name, entry in entries.items()}
        except Exception as exc:
            for entry in entries.values():
                entry['last_error_type'] = type(exc).__name__
                entry['http_status'] = getattr(exc, 'http_status', None)
                entry['retry_after_seconds'] = getattr(exc, 'retry_after_seconds', None)
            atomic_json(uploader.journal, state, uploader.guard)
            if isinstance(exc, UploadError):
                raise
            raise UploadError('Batch upload failed; credentials omitted; local shards retained') from None
