"""Accepted recipe, minimal source bundle and extracted-runner import checks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from tinychess.train.fullbatch_confirmation import load_accepted_config

ROOT=Path(__file__).resolve().parents[1]


def test_accepted_config_matches_immutable_recorded_measurements():
    cfg=load_accepted_config(ROOT/'configs/stage3a_fullbatch_accepted.json')
    results=ROOT/cfg['provenance']['results']
    assert hashlib.sha256(results.read_bytes()).hexdigest()==cfg['provenance']['results_sha256']
    run=json.loads(results.read_text())['run']
    assert cfg['result']['updates']==run['steps']==281
    assert cfg['result']['policy_accuracy']==run['final']['policy_accuracy']
    assert cfg['training_configuration']==run['configuration']
    assert cfg['executed_batch']['microbatch']*cfg['executed_batch']['accumulation_steps']==1024


@pytest.mark.parametrize('key,value',[('lr',.001),('effective_batch',32),('history_dropout',.1),('sample_sha256','bad')])
def test_runtime_config_refuses_obsolete_or_changed_recipe(tmp_path,key,value):
    cfg=json.loads((ROOT/'configs/stage3a_fullbatch_accepted.json').read_text())
    cfg['training_configuration'][key]=value
    p=tmp_path/'config.json';p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='drift'):load_accepted_config(p)


def test_minimal_bundle_repeats_and_imports_from_outside_checkout(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('accepted_packager',ROOT/'scripts/prepare_stage3a_bundle.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Small opaque fixture tests packaging/integrity only, not dataset correctness.
    sample=tmp_path/'sample';sample.mkdir();payload=b'packaging fixture'
    (sample/'sample.npz').write_bytes(payload)
    (sample/'manifest.json').write_text(json.dumps({'sample_sha256':'cc00bda4d5feb8a980be6e32208712f50eb94bb197ae60d01aeaff6e47ced590'}))
    with pytest.raises(ValueError,match='checksum'):module.build(sample,tmp_path/'bad.zip')
    # Override only the fixture checksum; production enforces the frozen sample.
    monkeypatch.setattr(module,'ACCEPTED_SAMPLE_SHA256',hashlib.sha256(payload).hexdigest())
    (sample/'manifest.json').write_text(json.dumps({'sample_sha256':hashlib.sha256(payload).hexdigest()}))
    one=module.build(sample,tmp_path/'one.zip',root=ROOT);two=module.build(sample,tmp_path/'two.zip',root=ROOT)
    assert one['sha256']==two['sha256']
    with zipfile.ZipFile(one['zip']) as z:
        names=z.namelist()
        assert not any('diagnos' in x or '/.venv/' in x or '/__pycache__/' in x or x.endswith('.msgpack') for x in names)
        assert not any('/reports/' in x for x in names)
        assert 'gambit-zero-stage3a/scripts/confirm_stage3a_fullbatch.py' in names
        manifest=json.loads(z.read('gambit-zero-stage3a/STAGE3A_BUNDLE.json'))
        for name,h in manifest['files'].items():assert hashlib.sha256(z.read('gambit-zero-stage3a/'+name)).hexdigest()==h
        z.extractall(tmp_path/'extracted')
    runner=tmp_path/'extracted/gambit-zero-stage3a/scripts/confirm_stage3a_fullbatch.py'
    # Isolated import tests path/bootstrap without importing W&B or running training.
    probe=subprocess.run([sys.executable,'-I','-c','import runpy,sys; runpy.run_path(sys.argv[1],run_name="import_probe")',str(runner)],cwd=tmp_path,capture_output=True,text=True,timeout=30)
    assert probe.returncode==0,probe.stdout+probe.stderr
