"""Package only accepted training/reproduction code and the verified small sample."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
ACCEPTED_SAMPLE_SHA256="cc00bda4d5feb8a980be6e32208712f50eb94bb197ae60d01aeaff6e47ced590"


def source_commit(root):
    git=root/'.git';head=(git/'HEAD').read_text().strip()
    if not head.startswith('ref: '):return head
    ref=head[5:];loose=git/ref
    if loose.exists():return loose.read_text().strip()
    for line in (git/'packed-refs').read_text().splitlines():
        if not line.startswith(('#','^')) and line.split()[1]==ref:return line.split()[0]
    raise RuntimeError('Cannot resolve source commit')


def build(sample,output,root=ROOT):
    sample=Path(sample);root=Path(root);output=Path(output)
    # Refuse changed data before packaging it. This reads no network or credentials.
    manifest=json.loads((sample/'manifest.json').read_text())
    if manifest.get('sample_sha256')!=ACCEPTED_SAMPLE_SHA256:
        raise ValueError('Use the accepted frozen sample')
    if hashlib.sha256((sample/'sample.npz').read_bytes()).hexdigest()!=manifest['sample_sha256']:
        raise ValueError('Sample checksum mismatch')
    paths=set()
    for folder in ['src','tests','licenses']:
        for p in (root/folder).rglob('*'):
            if any(x in ('__pycache__','.venv','build','dist') or x.endswith('.egg-info') for x in p.parts):continue
            if p.is_file() and (p.suffix in ('.py','.json','.yaml','.txt','.patch') or folder=='licenses'):paths.add(p)
    for name in ['pyproject.toml','uv.lock','.python-version','requirements-lock.txt','requirements-cuda-lock.txt','requirements-wandb.txt',
                 'configs/stage3a_fullbatch_accepted.json','docs/m2_stage3a_run.md',
                 'scripts/confirm_stage3a_fullbatch.py','scripts/freeze_overfit_sample.py',
                 'scripts/prepare_stage3a_bundle.py','scripts/report_fullbatch_confirmation.py']:
        p=root/name
        if not p.is_file():raise FileNotFoundError(name)
        paths.add(p)
    payloads={}
    for p in sorted(paths):
        if not p.resolve().is_relative_to(root.resolve()) or p.stat().st_size>2**20:raise ValueError('Unsafe/oversized source input')
        payloads[p.relative_to(root).as_posix()]=p.read_bytes()
    for name in ['sample.npz','manifest.json']:payloads['input/'+name]=(sample/name).read_bytes()
    if sum(map(len,payloads.values()))>40*2**20:raise ValueError('Bundle exceeds 40 MiB bound')
    source=dict(format='gambit-zero-accepted-stage3a-bundle-v1',source_base_commit=source_commit(root),
        source_note='Exact working-tree source snapshot, including any pending cleanup changes.',
        files={n:hashlib.sha256(b).hexdigest() for n,b in sorted(payloads.items())})
    payloads['STAGE3A_BUNDLE.json']=(json.dumps(source,indent=2,sort_keys=True)+'\n').encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        for name,b in sorted(payloads.items()):
            info=zipfile.ZipInfo('gambit-zero-stage3a/'+name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,b)
    return dict(zip=str(output.resolve()),bytes=output.stat().st_size,sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                source_manifest_sha256=hashlib.sha256(payloads['STAGE3A_BUNDLE.json']).hexdigest())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sample',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(build(a.sample,a.output),indent=2))
