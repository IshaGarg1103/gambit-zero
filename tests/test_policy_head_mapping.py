"""Pointwise policy-head routing inside the actual model, using nonzero weights."""
import copy
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from tinychess.model.transformer import ChessTransformer,ModelConfig

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



def test_policy_head_all_source_tokens_and_planes():
    model=ChessTransformer(ModelConfig(1,16,4,32))
    planes=jnp.ones((1,8,8,120),jnp.float32);context=jnp.zeros((1,8),jnp.float32)
    params=model.init(jax.random.PRNGKey(1),planes,context)["params"]
    r=policy_token_probe(model,params,planes,context)
    assert r["passed"] and r["checked_logits"]==4672
