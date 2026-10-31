"""Independent model-output mapping and instrumented-step parity."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tinychess.model.transformer import ChessTransformer,ModelConfig
from tinychess.model.objectives import supervised_loss,validate_batch
from tinychess.train.overfit_core import create_state,train_step,evaluate,indices
from tinychess.train.overfit_diagnosis import diagnostic_step,policy_token_probe


def batch():
    rng=np.random.default_rng(5);mask=np.zeros((4,4672),bool);mask[:,[0,73,146]]=True
    return validate_batch(dict(planes=rng.normal(size=(4,8,8,120)).astype(np.float32),context=np.zeros((4,8),np.float32),
        legal_mask=mask,action=np.array([0,73,146,0],np.int32),wdl_class=np.array([0,1,2,0],np.int32),
        policy_weight=np.ones(4,np.float32),value_weight=np.array([1,1,1,0],np.float32),training_eligible=np.ones(4,bool)))


def test_actual_model_head_token_perturbation_all_sources():
    b=batch();model=ChessTransformer(ModelConfig(1,16,4,32))
    s=create_state(model,jax.random.PRNGKey(1),b)
    result=policy_token_probe(model,s.params,b['planes'],b['context'])
    assert result['checked_logits']==4672 and result['passed']


def test_instrumentation_matches_original_step_and_reaches_backbone():
    b=batch();model=ChessTransformer(ModelConfig(1,16,4,32));s=create_state(model,jax.random.PRNGKey(1),b)
    original,om=train_step(s,b);new,nm=diagnostic_step(s,b)
    for a,c in zip(jax.tree_util.tree_leaves(original),jax.tree_util.tree_leaves(new)):
        np.testing.assert_allclose(a,c,rtol=1e-5,atol=1e-7)
    assert float(nm['gradient_groups']['backbone'])==0 # expected at zero output-head init only
    assert float(nm['update_dot_gradient'])<0
    assert float(nm['clipped_grad_norm'])<=1.00001
    new,m=diagnostic_step(new,b)
    assert all(float(v)>0 for v in m['gradient_groups'].values())
    before=evaluate(s,b)
    for k in ['loss','policy_loss','wdl_loss']:
        assert float(before[k])==pytest.approx(float(nm[k]),abs=1e-6)
    assert int(new.step)==2


def test_shuffling_preserves_every_field_pair_and_weighted_loss():
    b=batch();ix=indices(4,0,4,4);shuffled={k:v[ix] for k,v in b.items()}
    for k in b:np.testing.assert_array_equal(shuffled[k],np.asarray(b[k])[ix])
    model=ChessTransformer(ModelConfig(1,16,4,32));s=create_state(model,jax.random.PRNGKey(1),b)
    original=evaluate(s,b);other=evaluate(s,shuffled)
    assert float(original['loss'])==pytest.approx(float(other['loss']),abs=1e-6)
    out=model.apply({'params':s.params},b['planes'],b['context'])
    valuegrad=jax.grad(lambda v:supervised_loss(dict(out,wdl_logits=v),b)[0])(out['wdl_logits'])
    np.testing.assert_array_equal(valuegrad[-1],0.)


def test_first_adam_update_matches_independent_formula_and_lr():
    import optax
    from tinychess.train.overfit_core import learning_rate
    b=batch();model=ChessTransformer(ModelConfig(1,16,4,32));s=create_state(model,jax.random.PRNGKey(9),b)
    # Nonzero heads: this check must not be blind to the policy mapping or trunk.
    s=s.replace(params=model.init(jax.random.PRNGKey(9),b['planes'],b['context'])['params'])
    # Prescribed gradients isolate Adam arithmetic from eager/JIT AD rounding
    # in mathematically-zero attention-key bias gradients.
    g=jax.tree_util.tree_map(lambda x:jnp.full_like(x,.01),s.params)
    cg,_=optax.clip_by_global_norm(1.).update(g,optax.EmptyState())
    nxt=jax.jit(lambda state,grad:state.apply_gradients(grads=grad))(s,g)
    rate=float(learning_rate(0,1e-3))
    for p,c,q in zip(jax.tree_util.tree_leaves(s.params),jax.tree_util.tree_leaves(cg),jax.tree_util.tree_leaves(nxt.params)):
        expected=np.asarray(p)-rate*np.asarray(c)/(np.abs(np.asarray(c))+1e-8)
        np.testing.assert_allclose(q,expected,rtol=2e-6,atol=1e-7)
    assert all(np.all(np.asarray(q)<=np.asarray(p)) for p,q in zip(jax.tree_util.tree_leaves(s.params),jax.tree_util.tree_leaves(nxt.params)))
    assert int(nxt.step)==1


def test_weighted_losses_match_independent_nonuniform_numpy_reference():
    b=batch();model=ChessTransformer(ModelConfig(1,16,4,32))
    params=model.init(jax.random.PRNGKey(12),b['planes'],b['context'])
    out=model.apply(params,b['planes'],b['context']);_,m=supervised_loss(out,b)
    p=np.asarray(out['policy_logits'],np.float64);v=np.asarray(out['wdl_logits'],np.float64)
    mask=np.asarray(b['legal_mask']);pw=np.asarray(b['policy_weight']);vw=np.asarray(b['value_weight'])
    p=np.where(mask,p,-np.inf);p=p-np.max(p,-1,keepdims=True);v=v-np.max(v,-1,keepdims=True)
    p=p-np.log(np.exp(p).sum(-1,keepdims=True));v=v-np.log(np.exp(v).sum(-1,keepdims=True))
    p_loss=np.sum(-p[np.arange(4),np.asarray(b['action'])]*pw)/pw.sum()
    v_loss=np.sum(-v[np.arange(4),np.asarray(b['wdl_class'])]*vw)/vw.sum()
    assert float(m['policy_loss'])==pytest.approx(p_loss,abs=1e-6)
    assert float(m['value_loss'])==pytest.approx(v_loss,abs=1e-6)
