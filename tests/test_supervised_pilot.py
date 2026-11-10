"""Stage 3B preparation: weighted tails, schedules, recovery and launch guards."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tinychess.model.transformer import ChessTransformer, ModelConfig
from tinychess.model.objectives import validate_batch
from tinychess.train.overfit_core import fingerprint, save_checkpoint, restore_checkpoint
from tinychess.train.fullbatch_confirmation import direct_step
from tinychess.train.supervised import (PilotConfig, initial_state, padded_batch, train_step,
    evaluation_sums, evaluation_groups, Metrics, EarlyStop, schedule, kernel_decay_mask)
from tinychess.train.supervised_runner import replay_batches, run, checked_batches
from tinychess.train.tracking import Tracker

ROOT = Path(__file__).resolve().parents[1]


def batch(n=5):
    rng = np.random.default_rng(41)
    mask = np.zeros((n,4672),bool);mask[:,[0,73,146]] = True
    return dict(planes=rng.normal(size=(n,8,8,120)).astype(np.float32),
                context=np.zeros((n,8),np.float32),legal_mask=mask,
                action=(np.arange(n,dtype=np.int32)%3)*73,
                wdl_class=np.arange(n,dtype=np.int32)%3,
                policy_weight=np.ones(n,np.float32),value_weight=(np.arange(n)%3==0).astype(np.float32),
                training_eligible=np.ones(n,bool))


def state(b, cfg=None):
    cfg = cfg or PilotConfig()
    return initial_state(ChessTransformer(ModelConfig(1,16,4,32)), b, cfg, 3e-4, 1)


def assert_tree_close(a,b):
    for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
        np.testing.assert_allclose(x,y,rtol=3e-4,atol=3e-6)


def test_reviewed_config_counts_and_independent_initialization():
    cfg = PilotConfig.load(ROOT/'configs/stage3b_25m.json')
    assert cfg.batch_size == 1024 and cfg.steps_per_epoch == 2593
    assert cfg.train_positions%cfg.batch_size == 642
    assert cfg.validation_positions%cfg.microbatch == 4
    b = padded_batch(batch(),8)
    a = state(b);c = state(b)
    assert fingerprint(a.params) == fingerprint(c.params)
    assert fingerprint(a.opt_state) == fingerprint(c.opt_state)
    assert int(a.step) == 0


@pytest.mark.parametrize('changes',[
    dict(model='100m'),dict(history_dropout=.1),dict(dataset_revision='bad'),
    dict(shuffle_buffer=200000),dict(microbatch=64),dict(warmup_steps=999999),
    dict(learning_rates=(1e-4,1e-4,3e-4)),dict(learning_rates=(float('nan'),3e-4,6e-4)),
    dict(pilot_gpu_hours=13.),dict(weight_decay=-1.),dict(pilot_epochs=21)])
def test_config_refuses_out_of_scope(changes):
    with pytest.raises(ValueError):replace(PilotConfig(),**changes)


def test_warmup_cosine_endpoints_and_monotonic_decay():
    values=np.asarray(schedule(np.arange(30),.0003,10,30,.1))
    assert values[0] == pytest.approx(.00003)
    assert values[9] == pytest.approx(.0003)
    assert values[10] == pytest.approx(.0003)
    assert values[-1] == pytest.approx(.00003)
    assert np.all(np.diff(values[:10])>0) and np.all(np.diff(values[10:])<=0)
    assert float(schedule(100,.0003,10,30,.1)) == pytest.approx(.00003)


def test_padding_rejects_bad_rows_and_never_drops_tail():
    b=batch();p=padded_batch(b,8)
    assert len(p['action']) == 8
    np.testing.assert_array_equal(p['action'][:5],b['action'])
    assert float(p['policy_weight'].sum()) == 5
    assert np.all(np.asarray(p['policy_weight'][5:]) == 0)
    assert np.all(np.asarray(p['value_weight'][5:]) == 0)
    bad=dict(b,action=b['action'].copy());bad['action'][0]=4671
    with pytest.raises(ValueError):padded_batch(bad,8)
    with pytest.raises(ValueError):padded_batch(b,4)


def test_tail_accumulation_matches_direct_gradient_and_one_adamw_update():
    b=validate_batch(batch());p=padded_batch(batch(),8);s=state(b)
    a,am=direct_step(s,b)
    c,cm=train_step(s,p,microbatch=2)
    assert int(a.step) == int(c.step) == 1
    assert bool(cm['finite'])
    assert float(am['loss']) == pytest.approx(float(cm['loss']),rel=2e-6)
    assert float(cm['post_clip_norm']) <= 1.00001
    assert_tree_close(a,c)
    assert fingerprint(s.params) != fingerprint(c.params)


def test_zero_weight_forfeit_targets_do_not_change_update():
    b=batch();b['value_weight'][:]=0
    other=dict(b,wdl_class=(b['wdl_class']+1)%3)
    s=state(validate_batch(b))
    a,_=train_step(s,padded_batch(b,8),microbatch=2)
    c,_=train_step(s,padded_batch(other,8),microbatch=2)
    assert fingerprint(a.params) == fingerprint(c.params)
    assert fingerprint(a.opt_state) == fingerprint(c.opt_state)


def test_validation_uses_population_denominators_and_phase_partition():
    b=batch();s=state(validate_batch(b));total=Metrics();total.add(evaluation_sums(s,padded_batch(b,8)))
    pieces=Metrics()
    for lo,hi in [(0,3),(3,5)]:
        part={k:v[lo:hi] for k,v in b.items()}
        pieces.add(evaluation_sums(s,padded_batch(part,4)))
    for k,v in total.result().items():assert pieces.result()[k] == pytest.approx(v,abs=3e-6)
    groups=evaluation_groups(s,padded_batch(b,8),jnp.array([1,1,0,0,0,0,0,0],jnp.float32))
    assert float(groups[0]['policy_weight']) == 5
    for k in groups[0]:
        if k!='valid':assert float(groups[1][k]+groups[2][k]) == pytest.approx(float(groups[0][k]),abs=3e-6)
    assert total.result()['wdl_brier'] == pytest.approx(2/3,abs=1e-6)


def test_matrix_decay_mask_excludes_bias_and_norm():
    mask=kernel_decay_mask(dict(dense=dict(kernel=jnp.ones((2,3)),bias=jnp.ones(3)),
                                norm=dict(scale=jnp.ones(3)),position=jnp.ones((64,16))))
    assert mask == dict(dense=dict(kernel=True,bias=False),norm=dict(scale=False),position=False)


def test_checkpoint_restores_schedule_moments_and_next_update(tmp_path):
    b=padded_batch(batch(),8);s=state(b)
    s,_=train_step(s,b,microbatch=2)
    metadata=dict(stage='m2-stage3b',recipe_hash='fixture',cursor=dict(epoch=0,next_batch=1))
    save_checkpoint(tmp_path,s,metadata)
    restored,receipt=restore_checkpoint(tmp_path,state(b),dict(stage='m2-stage3b',recipe_hash='fixture'))
    assert receipt['cursor'] == dict(epoch=0,next_batch=1)
    a,_=train_step(s,b,microbatch=2);c,_=train_step(restored,b,microbatch=2)
    assert fingerprint(a.params) == fingerprint(c.params)
    assert fingerprint(a.opt_state) == fingerprint(c.opt_state)
    with pytest.raises(ValueError,match='metadata'):
        restore_checkpoint(tmp_path,state(b),dict(stage='m2-stage3a'))


def test_replay_cursor_exact_suffix_and_exhaustion_check():
    assert list(replay_batches(iter(range(7)),3)) == list(enumerate(range(7)))[3:]
    assert list(replay_batches(iter(range(7)),7)) == []
    assert list(replay_batches(iter([]),0)) == []
    with pytest.raises(ValueError):list(replay_batches(iter(range(7)),8))
    with pytest.raises(ValueError):list(replay_batches(iter([]),1))
    checked=[]
    list(replay_batches(iter(range(7)),3,lambda:checked.append(True)))
    assert len(checked) == 7


def test_patience_only_observes_complete_validation_results():
    e=EarlyStop()
    assert e.observe(3.) == (True,False)
    assert e.observe(3.1) == (False,False)
    assert e.observe(2.9) == (True,False)
    assert e.observe(2.9) == (False,False)
    assert e.observe(2.9) == (False,True)
    with pytest.raises(FloatingPointError):e.observe(float('nan'))


def test_test_split_never_opened():
    with pytest.raises(ValueError):checked_batches(None,[], 'test',PilotConfig())


def test_preflight_default_cannot_enter_execute(monkeypatch,tmp_path):
    import tinychess.train.supervised_runner as runner
    args=SimpleNamespace(config=ROOT/'configs/stage3b_25m.json',mode='sweep',lr_index=0,lr=None,
                         output=tmp_path/'output',store=tmp_path/'store',download=False,execute_approved=False)
    monkeypatch.setattr(runner,'VerifiedStore',lambda *a,**k:object())
    monkeypatch.setattr(runner,'inventory',lambda s:([],{}))
    monkeypatch.setattr(runner,'preflight',lambda *a:{'status':'PREFLIGHT_ONLY'})
    monkeypatch.setattr(runner,'_execute',lambda *a:pytest.fail('Training launched without approval flag'))
    assert run(args,ROOT)['status'] == 'PREFLIGHT_ONLY'


def test_tracking_supervised_groups_and_environment_only_key(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(init=lambda **k:calls.append(k) or SimpleNamespace(),
                                                          Settings=lambda **k:k))
    monkeypatch.setenv('WANDB_API_KEY','unit-test-fixture')
    for group in ('m2-lr-sweep','m2-supervised'):
        Tracker({'seed':1},tmp_path,group=group,name='unit',mode='online',tags=['stage3b'])
        assert calls[-1]['group'] == group
        assert 'unit-test-fixture' not in repr(calls[-1])
        assert calls[-1]['settings']['save_code'] is False


def test_cli_requires_explicit_launch_flag_and_mode(tmp_path):
    spec=importlib.util.spec_from_file_location('pilot_cli',ROOT/'scripts/run_supervised_pilot.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    args=module.parser().parse_args(['--mode','sweep','--lr-index','0','--output',str(tmp_path)])
    assert not args.execute_approved and not args.download
    with pytest.raises(SystemExit):module.parser().parse_args(['--output',str(tmp_path)])


@pytest.fixture
def tiny_runner(monkeypatch):
    """Synthetic orchestration test only: no 25M/GPU experiment or real data."""
    import tinychess.train.supervised_runner as runner
    cfg=replace(PilotConfig(),microbatch=2,accumulation_steps=2)
    # Reduced populations are a test fixture, not an accepted external config.
    for key,value in [('train_positions',7),('validation_positions',5),('warmup_steps',1)]:
        object.__setattr__(cfg,key,value)
    calls=[]
    class Tracking:
        url='offline-unit';id='offline-unit'
        def __init__(self,*a,**k):calls.append(k)
        def log(self,*a,**k):pass
        def checkpoint(self,*a,**k):return 'unit-checkpoint'
        def finish(self,*a,**k):pass
    monkeypatch.setattr(runner,'Tracker',Tracking)
    monkeypatch.setattr(runner,'hardware_metadata',lambda:{'unit_fixture':True})
    monkeypatch.setattr(runner,'blackwell_device',lambda:SimpleNamespace(device_kind='unit-fixture',memory_stats=lambda:{}))
    monkeypatch.setattr(runner,'gpu_budget',lambda d:(2**40,{}))
    monkeypatch.setattr(runner,'PhysicalSampler',lambda:SimpleNamespace(start=lambda:None,stop=lambda:None,
        peak=0,samples=0,errors=0))
    monkeypatch.setattr(runner,'ChessTransformer',lambda c:ChessTransformer(ModelConfig(1,16,4,32)))
    # Count guard is independently tested on real 25M shapes in test_transformer.
    monkeypatch.setattr(runner,'parameter_count',lambda p:25622092)
    def batches(store,shards,split,config,epoch=0):
        count=config.train_positions if split=='train' else config.validation_positions
        size=config.batch_size if split=='train' else config.microbatch
        all_rows=batch(count)
        for lo in range(0,count,size):
            hi=min(lo+size,count);n=hi-lo
            yield SimpleNamespace(**{k:v[lo:hi] for k,v in all_rows.items()},
                ply=np.arange(lo,hi)+8,game_id=tuple(str(i) for i in range(lo,hi)),
                source_month=('2026-01' if split=='train' else '2026-08',)*n,
                split=split,max_positions_per_game=1)
    monkeypatch.setattr(runner,'checked_batches',batches)
    return runner,cfg,calls,batches


def test_tiny_runner_full_epoch_and_exact_mid_epoch_replay(tiny_runner,tmp_path,monkeypatch):
    runner,cfg,calls,batches=tiny_runner
    args=SimpleNamespace(mode='sweep',wandb_mode='offline',resume=False)
    store=SimpleNamespace(downloaded_bytes=0)
    direct=tmp_path/'direct';direct.mkdir()
    reference=runner._execute(args,ROOT,cfg,3e-4,1,1.,direct,store,[],{})
    assert reference['status']=='COMPLETED' and reference['final_step']==2
    assert reference['epochs'][0]['validation']['rows']==5
    assert calls[0]['group']=='m2-lr-sweep'
    interrupted=tmp_path/'interrupted';interrupted.mkdir();fail=[True]
    def interrupt(store,shards,split,config,epoch=0):
        for i,b in enumerate(batches(store,shards,split,config,epoch)):
            if split=='train' and i==1 and fail[0]:
                fail[0]=False;raise runner.BudgetStop('unit interruption')
            yield b
    monkeypatch.setattr(runner,'checked_batches',interrupt)
    first=runner._execute(args,ROOT,cfg,3e-4,1,1.,interrupted,store,[],{})
    assert first['status']=='BUDGET_STOP' and first['final_step']==1
    receipt=json.loads((interrupted/'latest/checkpoint.json').read_text())
    assert receipt['cursor']==dict(epoch=0,next_batch=1)
    assert receipt['early_best'] is None
    args.resume=True
    resumed=runner._execute(args,ROOT,cfg,3e-4,1,1.,interrupted,store,[],{})
    assert resumed['status']=='COMPLETED' and resumed['final_step']==2
    a=json.loads((direct/'latest/checkpoint.json').read_text())
    b=json.loads((interrupted/'latest/checkpoint.json').read_text())
    assert a['sha256']==b['sha256']
    assert a['cursor']==b['cursor']==dict(epoch=1,next_batch=0)
    assert resumed['epochs'][0]['validation']['policy_loss']==reference['epochs'][0]['validation']['policy_loss']


def test_resume_metadata_failure_preserves_existing_checkpoint(tiny_runner,tmp_path):
    runner,cfg,_,_=tiny_runner
    args=SimpleNamespace(mode='sweep',wandb_mode='offline',resume=False)
    runner._execute(args,ROOT,cfg,3e-4,1,1.,tmp_path,SimpleNamespace(downloaded_bytes=0),[],{})
    before=(tmp_path/'latest/checkpoint.json').read_bytes();receipt=json.loads(before)
    payload=tmp_path/'latest'/receipt['file'];digest_before=payload.read_bytes()
    args.resume=True
    with pytest.raises(ValueError,match='metadata'):
        runner._execute(args,ROOT,cfg,1e-4,1,1.,tmp_path,SimpleNamespace(downloaded_bytes=0),[],{})
    assert (tmp_path/'latest/checkpoint.json').read_bytes()==before
    assert payload.read_bytes()==digest_before


def test_fresh_run_refuses_existing_weights_without_deleting_them(tiny_runner,tmp_path):
    runner,cfg,_,_=tiny_runner
    (tmp_path/'latest').mkdir();old=tmp_path/'latest/old.msgpack';old.write_bytes(b'keep')
    args=SimpleNamespace(mode='sweep',wandb_mode='offline',resume=False)
    with pytest.raises(ValueError,match='Existing experiment'):
        runner._execute(args,ROOT,cfg,3e-4,1,1.,tmp_path,SimpleNamespace(downloaded_bytes=0),[],{})
    assert old.read_bytes()==b'keep'
    assert not (tmp_path/'run_guard.json').exists()
