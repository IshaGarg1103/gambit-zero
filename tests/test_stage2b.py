"""Stage 2B controls/packaging plus tiny CPU AdamW checks; no GPU claim."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import zipfile

import pytest

from tinychess.validation.model_blackwell import run_case,case_estimate,host_available
from tinychess.validation.stage2b_plan import planned_cases,EXPECTED_COUNTS
from tinychess.model.transformer import PROFILES

ROOT=Path(__file__).resolve().parents[1]
# Script imports are standard-library only and never trigger a notebook run.
sys.path.insert(0,str(ROOT/'scripts'))
from report_stage2b import assess,render
from prepare_molab_stage2b import build
from molab_stage2b import package_results
from molab_m1_blackwell import verify_bundle


def evidence():
    rows=[dict(c,status='PASS',gpu_measured=True,parameter_count=EXPECTED_COUNTS[c['profile']],
        parameter_updates=1,master_params_fp32=True,gradients_finite=True,optimizer_fp32=True,
        gradient_squared_norm=1.,parameter_delta_squared=0.1,jax_peak_bytes_in_use=2**30) for c in planned_cases()]
    return dict(status='COMPLETE',cases=rows,host={'status':'PASS'},protected_sources_unchanged=True,
                regression_returncode=0,regression_passed=1)


def test_plan_is_small_and_bounded():
    cases=planned_cases()
    assert len(cases)==6 and {r['batch'] for r in cases}=={4,32}
    assert all(r['compute']=='bfloat16' for r in cases if r['batch']==32)
    assert assess(evidence())==('PASS',{'25m':32,'100m':32},True)


@pytest.mark.parametrize('field,value',[
    ('parameter_updates',2),('parameter_updates',0),('master_params_fp32',False),
    ('gradients_finite',False),('optimizer_fp32',False),('gradient_squared_norm',0.),
    ('parameter_count',1),('gpu_measured',False),('status','CPU_TEST_ONLY'),
])
def test_false_pass_receipts_rejected(field,value):
    r=evidence();r['cases'][0][field]=value
    assert assess(r)[0]=='FAIL'


@pytest.mark.parametrize('field,value',[
    ('regression_returncode',1),('regression_passed',0),('protected_sources_unchanged',False),('status','FAILED'),
])
def test_failed_setup_tests_or_drift_reject_gate(field,value):
    r=evidence();r[field]=value;assert assess(r)[0]=='FAIL'


def test_skips_never_invent_measurements_or_recommendations():
    r=evidence();r['cases'][-2].update(status='SKIPPED',reason='budget')
    assert assess(r)==('PASS',{'25m':4,'100m':32},True)
    r=evidence();r['cases'][2]['status']='SKIPPED'
    assert assess(r)[0]=='CONDITIONAL PASS'
    r=evidence();r['cases'][0]['jax_peak_bytes_in_use']=None
    assert assess(r)==('CONDITIONAL PASS',{},False)
    r=evidence();r['cases'][0]['status']='SKIPPED'
    assert assess(r)[0]=='FAIL'


@pytest.mark.parametrize('compute',['float32','bfloat16'])
def test_tiny_single_update_is_finite_and_cannot_satisfy_gpu_gate(compute):
    result=run_case('tiny',compute,4,require_gpu=False)
    assert result['status']=='CPU_TEST_ONLY'
    assert result['parameter_updates']==1 and result['parameter_delta_squared']>0
    assert result['gradients_finite'] and result['optimizer_fp32']
    assert result['gradient_squared_norm']>0 and result['loss_after']==pytest.approx(float(result['loss_after']))
    assert set(result['phases'])=={'initialize','forward','backward','adamw_once','post_update_forward'}
    assert all(x['compile_seconds']>=0 and x['synchronized_execution_seconds']>=0 for x in result['phases'].values())


def test_gpu_worker_refuses_cpu_or_tiny_substitution():
    with pytest.raises(ValueError,match='CPU unit-test'):run_case('tiny','float32',4)
    assert case_estimate(PROFILES['100m'],100031308,32)>case_estimate(PROFILES['25m'],25622092,4)
    assert host_available()[0]>0


def test_source_zip_is_repeatable_complete_and_excludes_data(tmp_path):
    one=build(tmp_path/'one.zip');first=(tmp_path/'molab_stage2b_cell.py').read_text()
    two=build(tmp_path/'two.zip')
    assert one['sha256']==two['sha256']
    assert one['sha256'] in first
    compile(first,'molab_stage2b_cell.py','exec')
    with zipfile.ZipFile(one['zip']) as z:
        names=z.namelist()
        assert all(not x.endswith(('.parquet','.pgn','.ckpt','.safetensors')) for x in names)
        assert all('.egg-info/' not in x for x in names)
        assert all('/.venv/' not in x and '/.git/' not in x and '/data/' not in x.split('stage2b/',1)[-1][:5] for x in names if '/src/' not in x)
        m=json.loads(z.read('gambit-zero-stage2b/BLACKWELL_BUNDLE.json'))
        for path,sha in m['files'].items():assert hashlib.sha256(z.read('gambit-zero-stage2b/'+path)).hexdigest()==sha
        for file in ['src/tinychess/model/transformer.py','m2_revised_flow.md','scripts/molab_stage2b.py']:
            assert 'gambit-zero-stage2b/'+file in names
        z.extractall(tmp_path/'extracted')
    verify_bundle(tmp_path/'extracted/gambit-zero-stage2b')
    # Test the extracted payload, not imports from this checkout. Molab can omit
    # the script directory from sys.path (e.g. PYTHONSAFEPATH / isolated mode).
    extracted=tmp_path/'extracted/gambit-zero-stage2b'
    probe = subprocess.run([sys.executable, '-I', '-c',
        'import runpy,sys; '
        'm=runpy.run_path(sys.argv[1],run_name="bundle_import_probe"); '
        'm["verify_bundle"](m["ROOT"]); print("EXTRACTED_IMPORTS_OK")',
        str(extracted/'scripts/molab_stage2b.py')],
        cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert probe.returncode==0, probe.stdout+probe.stderr
    assert 'EXTRACTED_IMPORTS_OK' in probe.stdout


@pytest.mark.parametrize('status',['COMPLETE','FAILED'])
def test_report_and_results_archive_work_without_cuda(tmp_path,status):
    work=tmp_path/'project/reports/stage2b_blackwell';work.mkdir(parents=True)
    r=evidence();r['status']=status
    (work/'run.json').write_text(json.dumps(r))
    (work.parents[1]/'BLACKWELL_BUNDLE.json').write_text('{}')
    gate=render(work.parents[1]);r['gate']=gate
    destination=package_results(work.parents[1],r)
    assert gate==('PASS' if status=='COMPLETE' else 'FAIL')
    with zipfile.ZipFile(destination) as z:
        index=json.loads(z.read('RESULTS_MANIFEST.json'))
        for file,sha in index['files'].items():assert hashlib.sha256(z.read(file)).hexdigest()==sha


@pytest.mark.parametrize('logical,physical,expected',[(100*2**30,80*2**30,24*2**30),(10*2**30,5*2**30,2*2**30)])
def test_gpu_budget_obeys_both_telemetry_sources(monkeypatch,logical,physical,expected):
    import tinychess.validation.model_blackwell as worker
    class Device:
        def memory_stats(self):return dict(bytes_limit=logical,bytes_in_use=0)
    monkeypatch.setattr(worker,'physical_memory',lambda:dict(free_bytes=physical,used_max_bytes=0))
    assert worker.gpu_budget(Device())[0]==expected


def test_cgroup_memory_limit_is_not_ignored(monkeypatch):
    import tinychess.validation.model_blackwell as worker
    from types import SimpleNamespace
    monkeypatch.setattr(worker.psutil,'virtual_memory',lambda:SimpleNamespace(available=20*2**30))
    def read(path,*args,**kwargs):
        if str(path)=='/sys/fs/cgroup/memory.max':return str(4*2**30)
        if str(path)=='/sys/fs/cgroup/memory.current':return str(3*2**30)
        raise OSError('not present')
    monkeypatch.setattr(Path,'read_text',read)
    assert worker.host_available()[0]==2**30


def test_bootstrap_failure_still_exports_verified_diagnostics(tmp_path,monkeypatch):
    import molab_stage2b as runner
    root=tmp_path/'source';root.mkdir()
    p=root/'source.py';p.write_text('# untouched')
    (root/'BLACKWELL_BUNDLE.json').write_text(json.dumps(dict(format='gambit-zero-stage2b-input-v1',
        source_base_commit='test',files={'source.py':hashlib.sha256(p.read_bytes()).hexdigest()})))
    monkeypatch.setattr(runner,'ROOT',root)
    from types import SimpleNamespace
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda _:SimpleNamespace(free=20*2**30))
    class FailedProcess:
        stdout=iter(['expected unavailable GPU\n'])
        def __init__(self,*args,**kwargs):pass
        def wait(self,*args,**kwargs):return 1
    monkeypatch.setattr(runner.subprocess,'Popen',FailedProcess)
    assert runner.main()==1
    assert (tmp_path/'gambit-zero-stage2b-results.zip').exists()
    report=json.loads((root/'reports/stage2b_blackwell/run.json').read_text())
    assert report['gate']=='FAIL' and report['status']=='FAILED' and not report['cases']
