"""Global loss weighting, one clip/update and exact analysis bucket boundaries."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tinychess.model.transformer import ChessTransformer,ModelConfig
from tinychess.model.objectives import validate_batch
from tinychess.train.overfit_core import create_state
from tinychess.train.fullbatch_confirmation import complete_gradient,accumulated_gradient,apply_full_gradient,strata


def sample():
    rng=np.random.default_rng(12);n=8;mask=np.zeros((n,4672),bool);mask[:,[0,73,146]]=True
    return validate_batch(dict(planes=rng.normal(size=(n,8,8,120)).astype(np.float32),context=np.zeros((n,8),np.float32),
        legal_mask=mask,action=np.arange(n,dtype=np.int32)%3*73,wdl_class=np.arange(n,dtype=np.int32)%3,
        policy_weight=np.ones(n,np.float32),value_weight=np.array([0,0,0,1,1,1,0,1],np.float32),training_eligible=np.ones(n,bool)))


def test_accumulation_uses_global_value_weights_not_mean_of_means():
    b=sample();model=ChessTransformer(ModelConfig(1,16,4,32));s=create_state(model,jax.random.PRNGKey(4),b,.0003)
    s=s.replace(params=model.init(jax.random.PRNGKey(4),b['planes'],b['context'])['params'])
    direct,dm=jax.jit(complete_gradient)(s,b)
    accum,am=jax.jit(lambda state,batch:accumulated_gradient(state,batch,2))(s,b)
    assert float(dm['loss'])==pytest.approx(float(am['loss']),rel=2e-6)
    for x,y in zip(jax.tree_util.tree_leaves(direct),jax.tree_util.tree_leaves(accum)):
        np.testing.assert_allclose(x,y,rtol=2e-4,atol=2e-6)
    nxt,m=jax.jit(apply_full_gradient)(s,accum,am)
    assert int(nxt.step)==1 and bool(m['finite']) and float(m['post_clip_norm'])<=1.00001


def test_zero_value_weight_global_and_unequal_policy_weights():
    b=sample();b=dict(b,value_weight=jnp.zeros(8),policy_weight=jnp.arange(1,9,dtype=jnp.float32))
    model=ChessTransformer(ModelConfig(1,16,4,32));s=create_state(model,jax.random.PRNGKey(4),b)
    direct,dm=jax.jit(complete_gradient)(s,b);accum,am=jax.jit(lambda s,b:accumulated_gradient(s,b,2))(s,b)
    assert float(dm['loss'])==pytest.approx(float(am['loss']),abs=1e-6)
    for x,y in zip(jax.tree_util.tree_leaves(direct),jax.tree_util.tree_leaves(accum)):
        np.testing.assert_allclose(x,y,rtol=2e-4,atol=2e-6)


def test_bucket_boundaries_and_missed_ids():
    plies=[0,10,11,30,31,60,61,100];legal=[1,10,11,20,21,30,40,41]
    ids=[dict(game_id=str(i),ply=p,position_hash='same' if i<2 else str(i)) for i,p in enumerate(plies)]
    a=np.zeros(8,int);pred=np.array([0,1,0,1,0,1,0,1])
    r=strata(ids,a,pred,legal)
    assert [v['rows'] for v in r['ply_buckets'].values()]==[2,2,2,2]
    assert [v['rows'] for v in r['legal_move_buckets'].values()]==[2,2,2,1,1]
    assert r['board_hash_groups']['shared']['rows']==2
    assert [x['game_id'] for x in r['missed_rows']]==['1','3','5','7']
    assert r['accuracy']==.5


def test_analysis_rejects_unpaired_rows():
    with pytest.raises(ValueError):strata([],np.zeros(1),[],[])
