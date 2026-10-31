"""Fresh Molab Stage 2B runner: isolated pinned CUDA env, six one-step cases."""
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback
import zipfile

ROOT=Path(__file__).resolve().parents[1]
# Notebook hosts may enable Python safe-path mode, excluding the script directory.
# Resolve sibling helpers explicitly before importing them in the fresh process.
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src'))
from tinychess.validation.stage2b_plan import planned_cases,DATASET_REVISION
from molab_m1_blackwell import verify_bundle
from report_stage2b import render


def package_results(root,result):
    root=Path(root);destination=root.parent/'gambit-zero-stage2b-results.zip'
    files=[p for p in (root/'reports/stage2b_blackwell').rglob('*') if p.is_file()]
    files += [root/'BLACKWELL_BUNDLE.json',root/'reports/m2_stage2b.md']
    files=[p for p in files if p.exists()]
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):archive.write(path,path.relative_to(root).as_posix())
        archive.writestr('RESULTS_MANIFEST.json',json.dumps(dict(format='gambit-zero-stage2b-results-v1',
            gate=result.get('gate','FAIL'),files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2)+'\n')
    print('RESULTS ZIP:',destination,flush=True)
    print('ZIP SHA256:',hashlib.sha256(destination.read_bytes()).hexdigest(),flush=True)
    return destination


def main():
    work=ROOT/'reports/stage2b_blackwell';work.mkdir(parents=True,exist_ok=True)
    report=dict(status='RUNNING',cases=[],dataset_revision=DATASET_REVISION,
        started_utc=datetime.now(timezone.utc).isoformat(),commands=[],protected_sources_unchanged=False)
    output=work/'run.json';env=dict(os.environ)
    for name in ['PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','JAX_PLATFORM_NAME']:env.pop(name,None)
    env.update(JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',
        JAX_DEFAULT_MATMUL_PRECISION='highest',
        CUDA_VISIBLE_DEVICES=env.get('CUDA_VISIBLE_DEVICES','0').split(',')[0],
        UV_CACHE_DIR=str(ROOT/'.uv-cache'),UV_PYTHON_INSTALL_DIR=str(ROOT/'.uv-python'),
        HF_HUB_OFFLINE='1',TINYCHESS_EVIDENCE_OFFLINE='1',PYTHONUNBUFFERED='1')
    def save():output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    def run(command,name,check=True):
        command=list(map(str,command));report['commands'].append(command)
        (work/'commands.json').write_text(json.dumps(report['commands'],indent=2)+'\n')
        print('RUN:', ' '.join(command),flush=True)
        with (work/name).open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            try:
                for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
                code=process.wait()
            except BaseException:
                process.terminate()
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                raise
        if code and check:raise RuntimeError(f'Command failed ({code}); inspect {name}')
        return code
    manifest=None;protected={}
    try:
        if output.exists():raise RuntimeError('Use the generated cell for a fresh extraction; do not mix runs')
        manifest=verify_bundle(ROOT)
        if manifest.get('format')!='gambit-zero-stage2b-input-v1':raise RuntimeError('Wrong input bundle type')
        protected={p:h for p,h in manifest['files'].items() if p!='reports/m2_stage2b.md'}
        report.update(source_base_commit=manifest['source_base_commit'],
            source_manifest_sha256=hashlib.sha256((ROOT/'BLACKWELL_BUNDLE.json').read_bytes()).hexdigest(),
            input_zip_sha256=env.get('STAGE2B_INPUT_ZIP_SHA256'))
        disk=shutil.disk_usage(ROOT);report['disk_preflight']=dict(free_bytes=disk.free,minimum_bytes=15*2**30)
        if disk.free<15*2**30:raise RuntimeError('Need 15 GiB free remote disk for the locked CUDA environment/caches')
        run(['nvidia-smi'],'nvidia-smi.log')
        try:uv=importlib.metadata.version('uv')
        except importlib.metadata.PackageNotFoundError:uv=None
        if uv!='0.10.9':run([sys.executable,'-m','pip','install','--no-cache-dir','uv==0.10.9'],'uv-install.log')
        run([sys.executable,'-m','uv','sync','--project',ROOT,'--locked','--python','3.11.15',
             '--extra','test','--extra','data','--extra','cuda'],'setup.log')
        python=ROOT/'.venv/bin/python'
        run([sys.executable,'-m','uv','pip','check','--python',python],'package-check.log')
        run([python,'scripts/inspect_stack.py','--require-gpu','--output',work/'stack.json'],'stack.log')
        run([python,'scripts/inspect_blackwell.py','--output',work/'host.json'],'host.log')
        report['host']=json.loads((work/'host.json').read_text());save()
        stopped=False;blocked_profiles=set()
        for case in planned_cases():
            name=f"{case['profile']}-{case['compute']}-b{case['batch']}"
            if stopped:
                report['cases'].append(dict(case,status='SKIPPED',reason='Stopped after prior worker failure'));continue
            if case['profile'] in blocked_profiles:
                report['cases'].append(dict(case,status='SKIPPED',reason='Profile stopped after earlier resource skip'));continue
            target=work/(name+'.json')
            code=run([python,'scripts/check_model_blackwell.py','--profile',case['profile'],'--compute',case['compute'],
                      '--batch',case['batch'],'--output',target],name+'.log',check=False)
            row=json.loads(target.read_text()) if target.exists() else dict(status='FAILED',reason='Worker exited without a result')
            row=dict(row,**case,worker_returncode=code)
            if code:row['status']='FAILED'
            report['cases'].append(row);save()
            stopped=row['status']=='FAILED'
            if row['status']=='SKIPPED':blocked_profiles.add(case['profile'])
        if stopped:raise RuntimeError('Model worker failed; later cases were not attempted')
        # Focused model/rules/codec checks only. No MCTS/search tests or loader benchmark.
        code=run([python,'-m','pytest','tests/test_transformer.py','tests/test_m2_stage0.py','tests/test_rules.py','-x','-q'],
                 'correctness.log',check=False)
        report['regression_returncode']=code
        matches=re.findall(r'(\d+) passed', (work/'correctness.log').read_text())
        report['regression_passed']=int(matches[-1]) if matches else 0
        if code:raise RuntimeError('Blackwell correctness checks failed')
        report['status']='COMPLETE'
    except BaseException as exc:
        report.update(status='FAILED',failure=f'{type(exc).__name__}: {exc}')
        (work/'failure.log').write_text(traceback.format_exc());print(report['failure'],flush=True)
    finally:
        report['protected_sources_unchanged']=bool(protected) and all((ROOT/p).is_file() and hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in protected.items())
        report['completed_utc']=datetime.now(timezone.utc).isoformat();save()
        report['gate']=render(ROOT)
        report.update(json.loads(output.read_text()));save();package_results(ROOT,report)
        print('STAGE 2B GATE:',report['gate'],flush=True)
    return 0 if report['gate']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
