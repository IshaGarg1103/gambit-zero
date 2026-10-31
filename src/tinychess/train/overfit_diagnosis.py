"""Instrumentation around the unchanged Stage 3A model/loss/optimizer."""
import copy
import hashlib

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from tinychess.train.overfit_core import metric_sums,finite_tree


def group_norms(tree):
    groups={'backbone':[], 'attention':[], 'policy_head':[], 'wdl_head':[]}
    for name,value in tree.items():
        if name=='policy_head':groups['policy_head'].extend(jax.tree_util.tree_leaves(value))
        elif name in ('value_head','value_hidden'):groups['wdl_head'].extend(jax.tree_util.tree_leaves(value))
        else:
            groups['backbone'].extend(jax.tree_util.tree_leaves(value))
            if name.startswith('block_'):
                groups['attention'].extend(jax.tree_util.tree_leaves(value['attention']))
    return {k:optax.global_norm(v) for k,v in groups.items()}


@jax.jit
def diagnostic_step(state,batch):
    def objective(params):
        m=metric_sums(state.apply_fn({'params':params},batch['planes'],batch['context']),batch)
        return m['loss'],m
    (_,m),grad=jax.value_and_grad(objective,has_aux=True)(state.params)
    clipped,_=optax.clip_by_global_norm(1.).update(grad,optax.EmptyState())
    candidate=state.apply_gradients(grads=grad)
    okay=m['valid'] & finite_tree(grad) & finite_tree(candidate.params) & finite_tree(candidate.opt_state)
    next_state=jax.lax.cond(okay,lambda:candidate,lambda:state)
    delta=jax.tree_util.tree_map(lambda a,b:a-b,next_state.params,state.params)
    return next_state,dict(m,finite=okay,grad_norm=optax.global_norm(grad),
        clipped_grad_norm=optax.global_norm(clipped),gradient_groups=group_norms(grad),
        update_groups=group_norms(delta),update_dot_gradient=sum(
            jnp.vdot(d,g).real for d,g in zip(jax.tree_util.tree_leaves(delta),jax.tree_util.tree_leaves(grad))))


def policy_token_probe(model,params,planes,context):
    """Perturb one post-backbone square token INSIDE Dense(policy_head).

    Input-board perturbations would mix through attention and are unsuitable for
    this test. Nonzero, known output weights make the test independent of zero
    initialization. All 64 squares and 73 action planes are checked.
    """
    p=copy.copy(params);head=dict(params['policy_head'])
    head['kernel']=jnp.zeros_like(head['kernel']).at[0,:].set(jnp.arange(1,74,dtype=jnp.float32))
    head['bias']=jnp.zeros_like(head['bias']);p['policy_head']=head
    def forward(p,idx,amount):
        def intercept(next_fun,args,kwargs,ctx):
            if ctx.module.name=='policy_head' and ctx.method_name=='__call__':
                return next_fun(args[0].at[:,idx,0].add(amount),**kwargs)
            return next_fun(*args,**kwargs)
        with nn.intercept_methods(intercept):return model.apply({'params':p},planes[:1],context[:1])
    call=jax.jit(forward);baseline=call(p,jnp.int32(0),jnp.float32(0.))
    max_error=0.
    for idx in range(64):
        out=call(p,jnp.int32(idx),jnp.float32(1.))
        delta=np.asarray(out['policy_logits']-baseline['policy_logits'])[0]
        row,col=divmod(idx,8);source=col*8+7-row
        expected=np.zeros(4672,np.float32);expected[source*73:(source+1)*73]=np.arange(1,74)
        error=float(np.max(np.abs(delta-expected)));max_error=max(max_error,error)
        if error>1e-3:raise AssertionError(f'Model policy token mapping mismatch row={row} col={col} source={source} error={error}')
        np.testing.assert_array_equal(out['wdl_logits'],baseline['wdl_logits'])
    return dict(squares=64,action_planes=73,checked_logits=4672,max_absolute_error=max_error,passed=True)


def fingerprint(tree):
    h=hashlib.sha256()
    for x in jax.tree_util.tree_leaves(tree):
        x=np.asarray(x);h.update(str(x.shape).encode());h.update(x.dtype.str.encode());h.update(x.tobytes())
    return h.hexdigest()


def accuracy(m):
    return dict(policy_loss=float(m['policy_loss']),wdl_loss=float(m['wdl_loss']),
        policy_accuracy=float(m['policy_correct']/max(float(m['policy_weight']),1.)),
        wdl_accuracy=float(m['wdl_correct']/max(float(m['value_weight']),1.)))
