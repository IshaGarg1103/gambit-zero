"""Fetch/verify the small frozen evidence bundle; never restore tracked reports."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from tinychess.data.evidence import EvidenceStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--path', help='Original project-relative evidence filename')
    parser.add_argument('--output', type=Path, help='Optional export under ignored data/processed/')
    args = parser.parse_args()
    store = EvidenceStore(ROOT, offline=args.offline)
    archive = store.archive()
    if args.path:
        payload = store.read_bytes(args.path)
        if args.output:
            output = args.output.resolve()
            if not output.is_relative_to(ROOT/'data/processed'):
                parser.error('Export evidence only under ignored data/processed/')
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(payload)
        print(json.dumps(dict(path=args.path, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())))
    if args.verify or not args.path:
        count = 0
        # Read directly from the frozen bundle, including entries still retained
        # locally, so verification is independent of the current source tree.
        import zipfile
        with zipfile.ZipFile(archive) as bundle:
            for path, expected in store.manifest['files'].items():
                info = bundle.getinfo(path)
                if info.file_size != expected['bytes'] or info.file_size > 8*2**20:
                    raise ValueError('Archive member bounds mismatch: '+path)
                payload = bundle.read(info)
                if hashlib.sha256(payload).hexdigest() != expected['sha256']:
                    raise ValueError('Archive member checksum mismatch: '+path)
                count += 1
        print(json.dumps(dict(status='PASS', entries_verified=count,
                              archive_sha256=store.manifest['archive']['sha256'])))


if __name__ == '__main__':
    main()
