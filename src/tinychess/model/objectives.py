"""FP32 legal policy/WDL functions and checked loader boundary; no training loop.

Compiled kernels return explicit validity flags and NaN loss on invalid targets.
Call validate_batch at the host boundary before JIT/grad. Terminal inference rows
are handled with an all-zero policy plus has_legal_actions=False, never uniform.
"""
import jax
import jax.numpy as jnp
import numpy as np


class InvalidBatch(ValueError):
    pass


def masked_policy(logits,legal_mask):
    if logits.ndim!=2 or logits.shape[-1]!=4672 or legal_mask.shape!=logits.shape:
        raise ValueError('Policy logits and mask must be (B,4672)')
    if legal_mask.dtype!=jnp.bool_:
        raise ValueError('Legal masks must be boolean')
    logits=logits.astype(jnp.float32)
    has_legal=jnp.any(legal_mask,axis=-1)
    maximum=jnp.max(jnp.where(legal_mask,logits,jnp.finfo(jnp.float32).min),axis=-1,keepdims=True)
    maximum=jnp.where(has_legal[:,None],maximum,0.)
    # Center by the *legal* maximum before finite masking: even legal logits
    # below -1e9 must not lose mass to the illegal sentinel.
    masked=jnp.where(legal_mask,logits-maximum,jnp.float32(-1e9))
    probabilities=jnp.where(legal_mask,jax.nn.softmax(masked,axis=-1),0.)
    probabilities=probabilities/jnp.where(has_legal[:,None],probabilities.sum(-1,keepdims=True),1.)
    return dict(logits=masked,probabilities=probabilities,has_legal_actions=has_legal)


def policy_cross_entropy(logits,legal_mask,actions):
    policy=masked_policy(logits,legal_mask)
    if actions.shape!=(logits.shape[0],) or not jnp.issubdtype(actions.dtype,jnp.integer):
        raise ValueError('Actions must be a batch of integer indices')
    bounded=jnp.clip(actions,0,4671)
    legal_target=jnp.take_along_axis(legal_mask,bounded[:,None],axis=-1)[:,0]
    valid=policy['has_legal_actions'] & (actions>=0) & (actions<4672) & legal_target
    # Crucially: cross-entropy uses the legally masked fp32 logits.
    logp=jax.nn.log_softmax(policy['logits'],axis=-1)
    nll=-jnp.take_along_axis(logp,bounded[:,None],axis=-1)[:,0]
    return jnp.where(valid,nll,jnp.nan),valid


def wdl_predictions(logits):
    if logits.ndim!=2 or logits.shape[-1]!=3:
        raise ValueError('WDL logits must be (B,3) in win,draw,loss order')
    probabilities=jax.nn.softmax(logits.astype(jnp.float32),axis=-1)
    return dict(probabilities=probabilities,value=probabilities[:,0]-probabilities[:,2])


def supervised_loss(outputs,batch):
    """Loss/gradient smoke primitive; weighted means, zero loss if no value weight.

    This performs no parameter updates. Invalid masks/targets/weights yield an
    invalid flag and NaN total, so compiled callers cannot quietly use bad data.
    """
    policy,valid=policy_cross_entropy(outputs['policy_logits'],batch['legal_mask'],batch['action'])
    logits=outputs['wdl_logits'].astype(jnp.float32)
    n=outputs['policy_logits'].shape[0]
    if logits.shape!=(n,3) or any(batch[k].shape!=(n,) for k in (
            'wdl_class','policy_weight','value_weight','training_eligible')):
        raise ValueError('Loss expects WDL (B,3) and one-dimensional batch labels/weights')
    if not jnp.issubdtype(batch['wdl_class'].dtype,jnp.integer) or batch['training_eligible'].dtype!=jnp.bool_:
        raise ValueError('WDL classes must be integers and eligibility boolean')
    classes=batch['wdl_class'];bounded=jnp.clip(classes,0,2)
    value=-jnp.take_along_axis(jax.nn.log_softmax(logits,axis=-1),bounded[:,None],axis=-1)[:,0]
    eligible=batch['training_eligible']
    pw=batch['policy_weight'].astype(jnp.float32);vw=batch['value_weight'].astype(jnp.float32)
    valid=valid & (classes>=0) & (classes<3) & eligible
    valid=valid & jnp.isfinite(pw) & jnp.isfinite(vw) & (pw>=0) & (vw>=0)
    p_sum=jnp.sum(pw,dtype=jnp.float32);v_sum=jnp.sum(vw,dtype=jnp.float32)
    p_loss=jnp.sum(policy*pw,dtype=jnp.float32)/jnp.where(p_sum>0,p_sum,1.)
    v_loss=jnp.sum(value*vw,dtype=jnp.float32)/jnp.where(v_sum>0,v_sum,1.)
    okay=jnp.all(valid) & jnp.isfinite(p_loss) & jnp.isfinite(v_loss)
    total=jnp.where(okay,p_loss+v_loss,jnp.nan)
    return total,dict(policy_loss=p_loss,value_loss=v_loss,valid=okay)


def validate_batch(batch):
    """Host-only checked conversion from Stage 1 TrainingBatch or a mapping.

    Never call this in the model/search hot path. Does not alter features,
    perspective, eligibility or weights. There is no python-chess dependency.
    """
    names=('planes','context','legal_mask','action','wdl_class','policy_weight','value_weight','training_eligible')
    raw={k:np.asarray(getattr(batch,k) if hasattr(batch,k) else batch[k]) for k in names}
    n=raw['planes'].shape[0]
    shapes=dict(planes=(n,8,8,120),context=(n,8),legal_mask=(n,4672))
    for name,value in raw.items():
        if value.shape!=shapes.get(name,(n,)):
            raise InvalidBatch(f'Wrong {name} shape')
    for name in ('planes','context','policy_weight','value_weight'):
        if raw[name].dtype!=np.float32 or not np.isfinite(raw[name]).all():
            raise InvalidBatch(f'{name} must be finite float32')
    if raw['legal_mask'].dtype!=bool or raw['training_eligible'].dtype!=bool:
        raise InvalidBatch('Masks/eligibility must be boolean')
    if n==0 or not raw['training_eligible'].all() or not raw['legal_mask'].any(axis=-1).all():
        raise InvalidBatch('Empty, ineligible or all-illegal training example')
    actions=raw['action'];classes=raw['wdl_class']
    if not np.issubdtype(actions.dtype,np.integer) or not ((actions>=0)&(actions<4672)).all():
        raise InvalidBatch('Invalid action indices')
    if not raw['legal_mask'][np.arange(n),actions].all():
        raise InvalidBatch('Human action is illegal')
    if not np.issubdtype(classes.dtype,np.integer) or not ((classes>=0)&(classes<3)).all():
        raise InvalidBatch('Invalid WDL classes')
    if (raw['policy_weight']<0).any() or (raw['value_weight']<0).any():
        raise InvalidBatch('Negative loss weights')
    return {k:jnp.asarray(v) for k,v in raw.items()}
