"""Stage 3A bounded training/checkpoint/tracking controls, with a tiny CPU model."""
import hashlib
import json
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tinychess.model.transformer import ChessTransformer,ModelConfig
from tinychess.model.objectives import validate_batch
from tinychess.train.overfit_core import (create_state,learning_rate,save_checkpoint,restore_checkpoint)
from tinychess.train.fullbatch_confirmation import direct_step as train_step,evaluate_rows as evaluate
from tinychess.train.overfit_sample import choose_rows,rank,logical_hash,FIELDS
from tinychess.train.tracking import Tracker


def batch():
    mask=np.zeros((4,4672),bool);mask[:,[0,73,146]]=True
    return validate_batch(dict(planes=np.ones((4,8,8,120),np.float32),context=np.zeros((4,8),np.float32),
        legal_mask=mask,action=np.array([0,73,146,0],np.int32),wdl_class=np.array([0,1,2,0],np.int32),
        policy_weight=np.ones(4,np.float32),value_weight=np.array([1,1,1,0],np.float32),training_eligible=np.ones(4,bool)))


def state():return create_state(ChessTransformer(ModelConfig(1,16,4,32)),jax.random.PRNGKey(3),batch())


def equal(a,b):
    return all(np.array_equal(x,y) for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)))


def test_initial_loss_exact_uniform_and_master_fp32():
    s=state();m=evaluate(s,batch())
    assert float(m['policy_loss'])==pytest.approx(np.log(3),abs=1e-6)
    assert float(m['wdl_loss'])==pytest.approx(np.log(3),abs=1e-6)
    assert all(x.dtype==jnp.float32 for x in jax.tree_util.tree_leaves(s.params))


def test_weight_zero_labels_do_not_affect_loss_or_gradient():
    s=state();b=batch();other=dict(b,wdl_class=b['wdl_class'].at[-1].set(2))
    a,ma=train_step(s,b);c,mc=train_step(s,other)
    assert equal(a,c);assert float(ma['value_loss'])==float(mc['value_loss'])


def test_nonfinite_update_rejected():
    s=state();b=batch();b=dict(b,planes=b['planes'].at[0,0,0,0].set(jnp.nan))
    nxt,m=train_step(s,b)
    assert not bool(m['finite']) and int(nxt.step)==0 and equal(s,nxt)


def test_checkpoint_exact_resume_next_update_and_corruption(tmp_path):
    s,m=train_step(state(),batch());assert bool(m['finite'])
    receipt=save_checkpoint(tmp_path,s,{'sample':'abc','seed':7})
    restored,_=restore_checkpoint(tmp_path,state(),{'sample':'abc','seed':7})
    assert equal(s,restored)
    x,_=train_step(s,batch());y,_=train_step(restored,batch());assert equal(x,y)
    with pytest.raises(ValueError,match='metadata'):restore_checkpoint(tmp_path,state(),{'sample':'wrong'})
    (tmp_path/receipt['file']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):restore_checkpoint(tmp_path,state(),{'sample':'abc'})


def test_checkpoint_atomic_retains_one_and_no_partial(tmp_path):
    s=state();save_checkpoint(tmp_path,s,{})
    s,_=train_step(s,batch());save_checkpoint(tmp_path,s,{})
    assert len(list(tmp_path.glob('*.msgpack')))==1 and not list(tmp_path.glob('*.partial'))


def test_accepted_learning_rate_schedule():
    assert float(learning_rate(0))==pytest.approx(3e-6)
    assert float(learning_rate(99))==pytest.approx(3e-4)
    assert float(learning_rate(1999))==pytest.approx(3e-4)


def test_sample_selection_deterministic_one_per_game_and_duplicates_reported():
    records=[dict(game_id=str(i),ply=p,training_eligible=True,planes=bytes([i]),context=b'x',
                  action=1,wdl_class=0,value_weight=1.) for i in range(8) for p in range(4)]
    candidates=sorted({r['game_id'] for r in records},key=lambda g:rank(1,'game',g))
    a,ma=choose_rows(iter(records),candidates,4,1)
    b,mb=choose_rows(iter(reversed(records)),candidates,4,1)
    assert a==b and ma==mb and len({r['game_id'] for r in a})==4
    records[0]['training_eligible']=False
    a,_=choose_rows(iter(records),candidates,4,1)
    assert all(r['training_eligible'] for r in a)
    allsame=[dict(r,planes=b'a') for r in records]
    with pytest.raises(ValueError,match='Not enough'):choose_rows(iter(allsame),candidates,4,1)


def test_tracking_offline_no_key_and_explicit_account(monkeypatch,tmp_path):
    calls=[]
    class Run:
        url='offline';id='id'
        def log(self,*a,**k):calls.append(('log',a,k))
        def finish(self,*a,**k):calls.append(('finish',a,k))
    def init(**kwargs):calls.append(('init',kwargs));return Run()
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(init=init,Settings=lambda **x:x))
    monkeypatch.delenv('WANDB_API_KEY',raising=False)
    t=Tracker({'seed':1},tmp_path,mode='offline');t.log({'loss':1.},0);t.finish(True)
    assert calls[0][1]['entity']=='ishagarg-research' and calls[0][1]['mode']=='offline'
    assert calls[0][1]['settings']['disable_code']
    with pytest.raises(RuntimeError,match='environment'):Tracker({},tmp_path,mode='online')
    assert Tracker({},tmp_path,mode='disabled').url is None


def test_tracking_checkpoint_only_explicit_files(monkeypatch,tmp_path):
    attached=[]
    class Artifact:
        def __init__(self,name,**kwargs):self.name=name
        def add_file(self,path):attached.append(path)
    class Run:
        def log_artifact(self,a):assert a.name=='stage3a-fullbatch-confirmation-checkpoint'
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(Artifact=Artifact))
    t=Tracker({},tmp_path,mode='disabled');t.run=Run()
    assert t.checkpoint(tmp_path,dict(file='state.msgpack',step=5,sha256='abc'))=='stage3a-fullbatch-confirmation-checkpoint'
    assert attached==[str(tmp_path/'state.msgpack'),str(tmp_path/'checkpoint.json')]


def test_loss_hash_detects_target_and_weight_change():
    b={k:np.asarray(v) for k,v in batch().items()};ids=[dict(game_id=str(i)) for i in range(4)]
    h=logical_hash(b,ids)
    b['value_weight']=b['value_weight'].copy();b['value_weight'][0]=0.
    assert logical_hash(b,ids)!=h


def test_duplicate_candidates_are_counted_not_silently_changed():
    records=[dict(game_id=str(i),ply=10,training_eligible=True,planes=bytes([i//2]),context=b'x',
                  action=i,wdl_class=0,value_weight=1.) for i in range(8)]
    rows,m=choose_rows(iter(records),list(map(str,range(8))),4,7)
    assert len(rows)==4 and m['duplicate_inputs_excluded']==3 and m['conflicting_duplicate_inputs_excluded']==3
