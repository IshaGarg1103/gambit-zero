"""Fresh isolated Stage 3A Molab job. Final export excludes checkpoint weights."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'data/processed/stage3a_run';out.mkdir(parents=True,exist_ok=True)
    if (out/'job.json').exists():raise RuntimeError('Use a fresh bundle extraction; job already exists')
    job=dict(status='RUNNING',commands=[],started_utc=datetime.now(timezone.utc).isoformat())
    env=dict(os.environ)
    for k in ['PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','JAX_PLATFORM_NAME']:env.pop(k,None)
    env.update(JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',JAX_DEFAULT_MATMUL_PRECISION='highest',
        CUDA_VISIBLE_DEVICES='0',UV_CACHE_DIR=str(ROOT/'.uv-cache'),UV_PYTHON_INSTALL_DIR=str(ROOT/'.uv-python'),
        HF_HUB_OFFLINE='1',TINYCHESS_EVIDENCE_OFFLINE='1',WANDB_ENTITY='ishagarg-research',WANDB_PROJECT='gambit-zero',
        WANDB_DISABLE_CODE='true')
    def run(cmd,name,extra=None):
        cmd=list(map(str,cmd));job['commands'].append(cmd)
        (out/'commands.json').write_text(json.dumps(job['commands'],indent=2)+'\n')
        with (out/name).open('w') as log:
            p=subprocess.Popen(cmd,cwd=ROOT,env=dict(env,**(extra or {})),stdout=log,stderr=subprocess.STDOUT)
            code=p.wait()
        if code:raise RuntimeError(f'{name} failed ({code}); inspect bounded job logs')
    try:
        m=json.loads((ROOT/'STAGE3A_BUNDLE.json').read_text())
        for n,h in m['files'].items():
            if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h:raise RuntimeError('Source hash mismatch: '+n)
        if not env.get('WANDB_API_KEY'):raise RuntimeError('WANDB_API_KEY must be inherited from Molab Secrets')
        run(['uv','sync','--project',ROOT,'--locked','--python','3.11.15','--extra','test','--extra','data','--extra','cuda'],'setup.log')
        python=ROOT/'.venv/bin/python'
        run(['uv','pip','install','--python',python,'-r','requirements-wandb.txt','--constraint','requirements-lock.txt'],'tracking-install.log')
        run(['uv','pip','check','--python',python],'package-check.log')
        run([python,'scripts/inspect_stack.py','--require-gpu','--output',out/'stack.json'],'stack.log')
        run([python,'scripts/inspect_blackwell.py','--output',out/'host.json'],'host.log')
        run([python,'-m','pytest','tests/test_stage3a.py','-q'],'focused-tests.log',{'JAX_PLATFORMS':'cpu'})
        run([python,'scripts/run_stage3a.py','--sample','input','--output',out/'training','--steps','3000','--batch','32','--lr','0.001','--wandb-mode','online'],'training.log')
        job['status']='PASS'
    except BaseException as exc:
        job.update(status='FAIL',failure=f'{type(exc).__name__}: {exc}')
        (out/'job-failure.txt').write_text(traceback.format_exc())
    finally:
        job['completed_utc']=datetime.now(timezone.utc).isoformat()
        (out/'job.json').write_text(json.dumps(job,indent=2,sort_keys=True)+'\n')
        files=[ROOT/'STAGE3A_BUNDLE.json',ROOT/'input/manifest.json']
        files+=[p for p in out.rglob('*') if p.is_file() and p.suffix in ('.json','.txt','.log')
                and 'wandb' not in p.relative_to(out).parts]
        # Checkpoints are uploaded as W&B artifacts by the trainer; metadata only here.
        target=ROOT.parent/'gambit-zero-stage3a-results.zip'
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
            for p in files:z.write(p,p.relative_to(ROOT).as_posix())
            z.writestr('RESULTS_MANIFEST.json',json.dumps(dict(format='gambit-zero-stage3a-results-v1',
                files={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2)+'\n')
        print('RESULTS ZIP:',target,flush=True)
        print('RESULTS SHA256:',hashlib.sha256(target.read_bytes()).hexdigest(),flush=True)
    return 0 if job['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
