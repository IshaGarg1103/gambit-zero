"""Exact Stage 3A source overlay plus the small derived real-position sample."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile
from prepare_molab_stage2b import build as source_bundle

ROOT=Path(__file__).resolve().parents[1]

def build(sample,output):
    sample=Path(sample);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        base=source_bundle(Path(tmp)/'source.zip')
        with zipfile.ZipFile(base['zip']) as z:
            payloads={n.split('/',1)[1]:z.read(n) for n in z.namelist() if not n.endswith('BLACKWELL_BUNDLE.json')}
            old=json.loads(z.read('gambit-zero-stage2b/BLACKWELL_BUNDLE.json'))
    payloads['requirements-wandb.txt']=(ROOT/'requirements-wandb.txt').read_bytes()
    for n in ('sample.npz','manifest.json'):payloads['input/'+n]=(sample/n).read_bytes()
    if sum(map(len,payloads.values()))>40*2**20:raise ValueError('Stage 3A input exceeds 40 MiB bound')
    manifest=dict(format='gambit-zero-stage3a-input-v1',source_base_commit=old['source_base_commit'],
        source_note='Exact working-tree overlay; training code not yet committed.',
        files={n:hashlib.sha256(b).hexdigest() for n,b in sorted(payloads.items())})
    payloads['STAGE3A_BUNDLE.json']=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        for n,b in sorted(payloads.items()):
            i=zipfile.ZipInfo('gambit-zero-stage3a/'+n,(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;z.writestr(i,b)
    return dict(zip=str(output.resolve()),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),bytes=output.stat().st_size,
                source_manifest_sha256=hashlib.sha256(payloads['STAGE3A_BUNDLE.json']).hexdigest())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sample',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(build(a.sample,a.output),indent=2))
