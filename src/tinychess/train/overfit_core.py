"""Minimal JIT training, deterministic minibatches and atomic checksum checkpoints."""
import hashlib
import json
import os
from pathlib import Path

from flax import serialization
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from tinychess.model.objectives import supervised_loss,masked_policy


def learning_rate(step, peak=1e-3, warmup=100):
    return jnp.float32(peak)*jnp.minimum((jnp.asarray(step,jnp.float32)+1)/warmup,1.)


def create_state(model, key, batch, peak=1e-3):
    params=model.init(key,batch['planes'],batch['context'])['params']
    # Run-specific initialization only: uniform policy/WDL for the sanity gate.
    for name in ('policy_head','value_head'):
        params[name]=jax.tree_util.tree_map(jnp.zeros_like,params[name])
    tx=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(
        lambda s:learning_rate(s,peak),b1=.9,b2=.95,eps=1e-8,weight_decay=0.,mu_dtype=jnp.float32))
    return TrainState.create(apply_fn=model.apply,params=params,tx=tx)


def metric_sums(outputs,batch):
    loss,m=supervised_loss(outputs,batch)
    policy=masked_policy(outputs['policy_logits'],batch['legal_mask'])
    pw=batch['policy_weight'];vw=batch['value_weight']
    return dict(loss=loss,valid=m['valid'],policy_loss=m['policy_loss'],wdl_loss=m['value_loss'],
        policy_correct=jnp.sum((jnp.argmax(policy['logits'],-1)==batch['action'])*pw),
        policy_weight=jnp.sum(pw),wdl_correct=jnp.sum((jnp.argmax(outputs['wdl_logits'],-1)==batch['wdl_class'])*vw),
        value_weight=jnp.sum(vw))


def finite_tree(tree):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(tree)]))


@jax.jit
def train_step(state,batch):
    def objective(params):
        m=metric_sums(state.apply_fn({'params':params},batch['planes'],batch['context']),batch)
        return m['loss'],m
    (_,m),grad=jax.value_and_grad(objective,has_aux=True)(state.params)
    candidate=state.apply_gradients(grads=grad)
    okay=m['valid'] & finite_tree(grad) & finite_tree(candidate.params) & finite_tree(candidate.opt_state)
    # An invalid update never commits even under a compiled caller.
    state=jax.lax.cond(okay,lambda:candidate,lambda:state)
    return state,dict(m,finite=okay,grad_norm=optax.global_norm(grad))


@jax.jit
def evaluate(state,batch):
    return metric_sums(state.apply_fn({'params':state.params},batch['planes'],batch['context']),batch)


def indices(seed,step,size,batch_size):
    if size%batch_size:raise ValueError('Frozen overfit sample must divide microbatch exactly')
    per_epoch=size//batch_size;epoch,slot=divmod(step,per_epoch)
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,epoch])))
    return rng.permutation(size)[slot*batch_size:(slot+1)*batch_size]


def save_checkpoint(directory,state,metadata):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    payload=serialization.to_bytes(jax.device_get(state))
    digest=hashlib.sha256(payload).hexdigest();name=f'step-{int(state.step):06d}-{digest[:16]}.msgpack'
    target=directory/name;temporary=directory/(name+'.partial')
    with temporary.open('wb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    temporary.replace(target)
    receipt=dict(metadata,format='gambit-zero-overfit-checkpoint-v1',step=int(state.step),file=name,
                 sha256=digest,bytes=len(payload))
    temp=directory/'checkpoint.json.partial';temp.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    temp.replace(directory/'checkpoint.json')
    # Keep only the current complete checkpoint, after pointer publication.
    for old in directory.glob('step-*.msgpack'):
        if old!=target:old.unlink()
    return receipt


def restore_checkpoint(directory,template,expected):
    directory=Path(directory);r=json.loads((directory/'checkpoint.json').read_text())
    if r.get('format')!='gambit-zero-overfit-checkpoint-v1':raise ValueError('Wrong checkpoint format')
    if Path(r['file']).name!=r['file']:raise ValueError('Unsafe checkpoint path')
    for k,v in expected.items():
        if r.get(k)!=v:raise ValueError('Checkpoint metadata mismatch: '+k)
    payload=(directory/r['file']).read_bytes()
    if len(payload)!=r['bytes'] or hashlib.sha256(payload).hexdigest()!=r['sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    state=serialization.from_bytes(template,payload)
    if int(state.step)!=r['step']:raise ValueError('Checkpoint step mismatch')
    return state,r


def verdict(initial,final,steps):
    if steps>3000 or not final.get('finite',True):return 'FAIL'
    if (final['policy_accuracy']>=.99 and final['policy_loss']<=.05
        and final['policy_loss']<initial['policy_loss']
        and final['wdl_accuracy']>=.95 and final['wdl_loss']<initial['wdl_loss']):return 'PASS'
    return 'FAIL'
