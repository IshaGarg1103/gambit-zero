"""Read local results or checksummed, archived PR evidence without extraction.

Frozen evidence is cached only under ignored data/processed/. Original filenames
are preserved inside the archive, not restored into tracked reports/. Local
runtime results take precedence so new executions can still report progress.
"""
import hashlib
import json
import os
import posixpath
from pathlib import Path, PurePosixPath
import re
import stat
import urllib.request
import zipfile

from filelock import FileLock


class EvidenceError(RuntimeError):
    pass


class EvidenceStore:
    def __init__(self, root=None, *, offline=None):
        self.root = Path(root or Path.cwd()).resolve()
        path = self.root/'reports/m1_evidence_archive.json'
        self.manifest = json.loads(path.read_text()) if path.exists() else None
        self.offline = os.environ.get('TINYCHESS_EVIDENCE_OFFLINE') == '1' if offline is None else offline

    def key(self, path):
        path = Path(path)
        if not path.is_absolute() and '..' in path.parts:
            raise EvidenceError('Evidence path traversal is forbidden')
        absolute = (self.root/path).resolve()
        if not absolute.is_relative_to(self.root):
            raise EvidenceError('Evidence path must remain inside the project')
        return absolute.relative_to(self.root).as_posix()

    def exists(self, path):
        key = self.key(path)
        return (self.root/key).is_file() or bool(self.manifest and key in self.manifest['files'])

    def archive(self):
        if self.manifest is None:
            raise FileNotFoundError('No local or archived evidence manifest is available')
        spec = self.manifest['archive']
        if not 0 < spec['bytes'] <= 16*2**20 or not re.fullmatch('[0-9a-f]{64}', spec['sha256']):
            raise EvidenceError('Invalid bounded archive descriptor')
        cache = self.root/'data/processed/m1_evidence'/f"{spec['sha256']}.zip"
        cache.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(cache)+'.lock', timeout=60):
            if not cache.exists():
                if self.offline:
                    raise FileNotFoundError('Archived evidence is not cached; run scripts/fetch_m1_evidence.py --verify')
                repo, revision, remote = spec['repo_id'], spec['revision'], spec['path_in_repo']
                if not re.fullmatch(r'[\w.-]+/[\w.-]+', repo) or not re.fullmatch('[0-9a-f]{40}', revision):
                    raise EvidenceError('Invalid immutable evidence repository/revision')
                if PurePosixPath(remote).is_absolute() or '..' in PurePosixPath(remote).parts:
                    raise EvidenceError('Unsafe archive repository path')
                url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{remote}'
                temporary = cache.with_suffix('.zip.partial')
                digest, size = hashlib.sha256(), 0
                try:
                    with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as out:
                        while chunk := response.read(65536):
                            size += len(chunk)
                            if size > spec['bytes']:
                                raise EvidenceError('Evidence archive exceeded its byte bound')
                            digest.update(chunk)
                            out.write(chunk)
                    if size != spec['bytes'] or digest.hexdigest() != spec['sha256']:
                        raise EvidenceError('Downloaded evidence archive SHA256/size mismatch')
                    temporary.replace(cache)
                finally:
                    temporary.unlink(missing_ok=True)
            if cache.stat().st_size != spec['bytes']:
                raise EvidenceError('Cached evidence archive SHA256/size mismatch')
            payload = cache.read_bytes()
            if len(payload) != spec['bytes'] or hashlib.sha256(payload).hexdigest() != spec['sha256']:
                raise EvidenceError('Cached evidence archive SHA256/size mismatch')
        return cache

    def read_bytes(self, path):
        key = self.key(path)
        local = self.root/key
        if local.is_file():
            if local.stat().st_size > 8*2**20:
                raise EvidenceError('Local evidence exceeds its read bound')
            return local.read_bytes()
        if self.manifest is None or key not in self.manifest['files']:
            raise FileNotFoundError(key)
        expected = self.manifest['files'][key]
        if not 0 <= expected['bytes'] <= 8*2**20:
            raise EvidenceError('Evidence entry exceeds its decompression bound')
        with zipfile.ZipFile(self.archive()) as archive:
            info = archive.getinfo(key)
            if info.file_size != expected['bytes'] or stat.S_ISLNK(info.external_attr >> 16):
                raise EvidenceError('Evidence member size/type mismatch')
            payload = archive.read(info)
        if hashlib.sha256(payload).hexdigest() != expected['sha256']:
            raise EvidenceError('Archived evidence entry SHA256 mismatch')
        return payload

    def read_json(self, path, *, required=True):
        if not self.exists(path):
            if required:
                raise FileNotFoundError(str(path))
            return None
        return json.loads(self.read_bytes(path))


def read_json_evidence(path, *, root=None, required=True):
    return EvidenceStore(root).read_json(path, required=required)


def evidence_exists(path, *, root=None):
    return EvidenceStore(root).exists(path)


def rewrite_archived_links(text, document, *, root=None):
    """Point Markdown links at the archive catalog when local evidence is gone."""
    store = EvidenceStore(root)
    if store.manifest is None:
        return text
    parent = PurePosixPath(document).parent.as_posix()
    def replace(match):
        target = match[1]
        if ':' in target or target.startswith('#'):
            return match[0]
        path = target.split('#', 1)[0]
        key = posixpath.normpath(posixpath.join(parent, path))
        if key in store.manifest['files'] and not (store.root/key).exists():
            anchor = re.sub('[^a-z0-9]+', '-', key.lower()).strip('-')
            catalog = posixpath.relpath('reports/m1_evidence_archive.md', parent)
            return ']('+catalog+'#'+anchor+')'
        return match[0]
    return re.sub(r'\]\(([^)]+)\)', replace, text)
