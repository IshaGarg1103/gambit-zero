"""Checksum-verified, journaled HF uploads. Never remove an unverified shard."""
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path,PurePosixPath
import re
import time
from functools import wraps

# Must precede SDK import: HTTP/LFS upload + cache-free streamed read-back.
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS','1')

from filelock import FileLock
from huggingface_hub import HfApi,get_token,hf_hub_url,constants
from huggingface_hub.utils import get_session

from tinychess.data.archive_scan import DiskGuard,atomic_json
from tinychess.data.sources import sha256_file


class UploadError(RuntimeError):
    pass


def safe_hf_errors(fn):
    @wraps(fn)
    def wrapped(*args,**kwargs):
        try:return fn(*args,**kwargs)
        except UploadError:raise
        except Exception as exc:
            response=getattr(exc,'response',None)
            status=getattr(response,'status_code',None)
            status=status if type(status) is int and 100<=status<=599 else None
            retry=response.headers.get('retry-after','') if response is not None else ''
            retry=int(retry) if re.fullmatch(r'\d{1,8}',retry) else None
            error=UploadError('HF request failed ('+type(exc).__name__+
                (', HTTP '+str(status) if status is not None else '')+'); no credential details logged')
            error.http_status=status
            error.retry_after_seconds=retry
            raise error from None
    return wrapped


class HubTransport:
    """Only installed public SDK methods; no Hub download/cache helper."""
    def __init__(self,repo_id,private):
        if constants.ENDPOINT.rstrip('/') != 'https://huggingface.co':
            raise UploadError('Only the official Hugging Face endpoint is configured')
        if not constants.HF_HUB_DISABLE_XET:
            raise UploadError('Launch with HF_HUB_DISABLE_XET=1 to avoid SDK/Xet disk caches')
        token=get_token()
        if not token:
            raise UploadError('Authenticate locally using .venv/bin/hf auth login; never paste a token in chat')
        self.repo_id,self.private=repo_id,private
        self.api=HfApi(endpoint='https://huggingface.co',token=token)
        self._token=token  # Memory only: never journaled, formatted, or printed.
        self._ensured=False

    @safe_hf_errors
    def ensure_repo(self):
        if getattr(self,'_ensured',False):
            return self.head()
        self.api.create_repo(self.repo_id,repo_type='dataset',private=self.private,exist_ok=True)
        info=self.api.repo_info(self.repo_id,repo_type='dataset')
        if info.private is not self.private:
            raise UploadError('Existing repository visibility differs from requested visibility; no files uploaded')
        self._ensured=True
        return info.sha

    @safe_hf_errors
    def head(self):
        info=self.api.repo_info(self.repo_id,repo_type='dataset')
        if info.private is not self.private:
            raise UploadError('Repository visibility changed; upload stopped')
        return info.sha

    @safe_hf_errors
    def exists(self,path,revision):
        return self.api.file_exists(self.repo_id,path,repo_type='dataset',revision=revision)

    @safe_hf_errors
    def upload(self,local,path,parent):
        if getattr(self,'reserve_transfer',None):
            self.reserve_transfer(Path(local).stat().st_size)
        commit=self.api.upload_file(path_or_fileobj=str(local),path_in_repo=path,
            repo_id=self.repo_id,repo_type='dataset',revision='main',parent_commit=parent,
            commit_message='Add verified Gambit Zero dataset artifact')
        return commit.oid

    def iter_remote(self,path,revision,max_bytes):
        info=self.api.get_paths_info(self.repo_id,[path],repo_type='dataset',revision=revision)
        if len(info)!=1 or info[0].size > max_bytes:
            raise UploadError('Remote path missing or larger than the expected artifact')
        if getattr(self,'reserve_transfer',None):
            self.reserve_transfer(info[0].size)
        url=hf_hub_url(self.repo_id,path,repo_type='dataset',revision=revision,endpoint='https://huggingface.co')
        size=0
        # httpx2 (the pinned SDK backend) strips Authorization on cross-origin
        # redirects; signed storage URLs do not receive our bearer credential.
        with get_session().stream('GET',url,headers={'Authorization':'Bearer '+self._token,
                'Accept-Encoding':'identity'},follow_redirects=True,timeout=60) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes(chunk_size=65536):
                if getattr(self,'deadline',None) is not None and time.monotonic()>self.deadline:
                    raise UploadError('HF invocation runtime allowance exhausted; local file retained')
                size+=len(chunk)
                if size>max_bytes:
                    raise UploadError('Remote stream exceeds expected bytes')
                yield chunk

    @safe_hf_errors
    def hash_remote(self,path,revision,max_bytes):
        digest,size=hashlib.sha256(),0
        for chunk in self.iter_remote(path,revision,max_bytes):
            digest.update(chunk)
            size+=len(chunk)
        return digest.hexdigest(),size

    @safe_hf_errors
    def read_verified_small(self,receipt,cap_bytes=32*2**20):
        if receipt['bytes']>cap_bytes:
            raise UploadError('Selected-game metadata exceeds bounded read limit')
        value=b''.join(self.iter_remote(receipt['path_in_repo'],receipt['revision'],receipt['bytes']))
        if len(value)!=receipt['bytes'] or hashlib.sha256(value).hexdigest()!=receipt['sha256']:
            raise UploadError('Selected-game metadata read-back checksum mismatch')
        return value


class VerifiedUploader:
    def __init__(self,repo_id,root,*,private=False,transport=None,
                 spool_cap_bytes=256*2**20,reserve_disk_bytes=20*2**30,deadline=None,transfer_budget_bytes=None):
        if not re.fullmatch(r'[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*',repo_id):
            raise ValueError('Expected Hugging Face owner/dataset name')
        if type(private) is not bool:
            raise ValueError('Repository visibility must be an explicit boolean')
        self.repo_id,self.private=repo_id,private
        self.root=Path(root).resolve()
        self.guard=DiskGuard(self.root,spool_cap_bytes,reserve_disk_bytes)
        self.journal=self.root/'upload_journal.json'
        self.transport=transport
        self.deadline=deadline
        self.transfer_budget_bytes=transfer_budget_bytes
        self.lock=FileLock(str(self.journal)+'.lock',timeout=60)

    def _transport(self):
        if self.transport is None:
            self.transport=HubTransport(self.repo_id,self.private)
        self.transport.deadline=self.deadline
        self.transport.reserve_transfer=self.reserve_transfer if self.transfer_budget_bytes is not None else None
        return self.transport

    def reserve_transfer(self,size):
        import json
        path=self.root/'hub_transfer_budget.json'
        with FileLock(str(path)+'.lock',timeout=60):
            state=json.loads(path.read_text()) if path.exists() else {'reserved_payload_bytes':0}
            projected=state['reserved_payload_bytes']+size
            if projected>self.transfer_budget_bytes:
                raise UploadError('Hub logical payload allowance exhausted; local files retained')
            state['reserved_payload_bytes']=projected
            state['allowance_bytes']=self.transfer_budget_bytes
            atomic_json(path,state,self.guard)

    def check_runtime(self):
        if self.deadline is not None and time.monotonic()>self.deadline:
            raise UploadError('HF invocation runtime allowance exhausted; local artifacts retained')

    def publish(self,local,remote,*,delete_after_verify=True,allow_update=False):
        self.check_runtime()
        import json
        local=Path(local)
        if local.is_symlink():
            raise UploadError('Refusing to delete/upload a symlink')
        local=local.resolve()
        if not local.is_relative_to(self.root):
            raise UploadError('Only files in the managed temporary spool can be published/deleted')
        path=PurePosixPath(remote)
        if path.is_absolute() or '..' in path.parts or not remote or '\\' in remote:
            raise UploadError('Unsafe repository-relative path')
        if allow_update and remote not in ['README.md','dataset_manifest.json','source_provenance.json']:
            raise UploadError('Only explicit root metadata updates may replace remote content')
        with self.lock:
            state=json.loads(self.journal.read_text()) if self.journal.exists() else dict(repo_id=self.repo_id,private=self.private,files={})
            if state['repo_id']!=self.repo_id or state['private']!=self.private:
                raise UploadError('Upload journal repository/visibility mismatch')
            previous=state['files'].get(remote)
            if local.exists():
                size,sha=local.stat().st_size,sha256_file(local)
            elif previous and previous['status'] in ['verified','deleted']:
                size,sha=previous['bytes'],previous['sha256']
            else:
                raise UploadError('Local artifact missing without a verified receipt')
            if previous and previous['sha256']!=sha and not allow_update:
                raise UploadError('Immutable shard path has a conflicting local checksum')
            entry=dict(local_name=local.name,path_in_repo=remote,sha256=sha,bytes=size,status='sealed')
            if previous and previous['sha256']==sha and previous.get('revision'):
                entry['revision']=previous['revision']
            state['files'][remote]=entry
            atomic_json(self.journal,state,self.guard)
            try:
                transport=self._transport()
                transport.ensure_repo()
                head=transport.head()
                revision=entry.get('revision')
                if revision is None:
                    if transport.exists(remote,head):
                        try:
                            remote_sha,remote_size=transport.hash_remote(remote,head,size)
                        except UploadError:
                            if not allow_update:
                                raise
                            remote_sha,remote_size=None,None
                        if (remote_sha,remote_size)==(sha,size):
                            revision=head  # Lost-response recovery: no duplicate upload.
                        elif not allow_update:
                            raise UploadError('Remote immutable path contains conflicting bytes')
                    if revision is None:
                        if not local.exists():
                            raise UploadError('Unverified missing local artifact cannot be retried')
                        revision=transport.upload(local,remote,head)
                    entry.update(status='uploaded',revision=revision)
                    atomic_json(self.journal,state,self.guard)
                remote_sha,remote_size=transport.hash_remote(remote,revision,size)
                if (remote_sha,remote_size)!=(sha,size):
                    raise UploadError('Remote read-back SHA256/size mismatch; local artifact retained')
                entry.update(status='verified',revision=revision,
                             verified_utc=datetime.now(timezone.utc).isoformat(),
                             verification='full cache-free streamed read-back at immutable revision')
                atomic_json(self.journal,state,self.guard)  # Durable BEFORE unlink.
                if delete_after_verify and local.exists():
                    if sha256_file(local)!=sha:
                        raise UploadError('Local artifact changed during upload; retained')
                    local.unlink()
                    entry['status']='deleted'
                    atomic_json(self.journal,state,self.guard)
                return dict(entry)
            except Exception as exc:
                # SDK/HTTP exception text can include URLs/headers. Store and
                # expose only its class; never copy tokens into logs or reports.
                entry['last_error_type']=type(exc).__name__
                atomic_json(self.journal,state,self.guard)
                if isinstance(exc,UploadError):
                    raise
                raise UploadError('HF operation failed ('+type(exc).__name__+'); local file retained for retry') from None
