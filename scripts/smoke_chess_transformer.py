"""Bounded Stage 2A init/inference smoke; reduced gradient check, no updates.

One profile/precision per fresh process. Uses two already verified local human
positions when --data-root is given; never downloads data or builds checkpoints.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
import resource
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import psutil

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tinychess.model.transformer import ChessTransformer,ModelConfig,PROFILES,parameter_count
from tinychess.model.objectives import validate_batch,masked_policy,wdl_predictions,supervised_loss


def run(args):
    config=replace(PROFILES[args.profile],compute=args.compute)
    model=ChessTransformer(config)
    shapes=jax.eval_shape(model.init,jax.random.PRNGKey(args.seed),
        jax.ShapeDtypeStruct((2,8,8,120),jnp.float32),jax.ShapeDtypeStruct((2,8),jnp.float32))['params']
    count=parameter_count(shapes)
    available=psutil.virtual_memory().available
    estimate=count*4*4+512*2**20
    if available<estimate+2**30:
        raise RuntimeError(f'Insufficient host RAM: estimate {estimate:,} plus 1 GiB reserve; {available:,} available')
    if args.data_root:
        from tinychess.data.training_loader import VerifiedStore,inventory,iter_batches
        store=VerifiedStore(args.data_root,download=False)
        shards,_=inventory(store)
        iterator=iter_batches(store,[s for s in shards if s.split=='train'][:1],'train',shuffle=False,batch_size=2)
        try:
            original=next(iterator)
            batch=validate_batch(original)
            source=dict(kind='pinned-human_v3-local',game_ids=original.game_id,plies=original.ply.tolist(),
                        downloaded_bytes=store.downloaded_bytes)
        finally:iterator.close()
    else:
        mask=np.zeros((2,4672),bool);mask[:,[13,201]]=True
        batch=validate_batch(dict(planes=np.zeros((2,8,8,120),np.float32),context=np.zeros((2,8),np.float32),
            legal_mask=mask,action=np.array([13,201],np.int32),wdl_class=np.array([0,2],np.int32),
            policy_weight=np.ones(2,np.float32),value_weight=np.array([1.,0.],np.float32),
            training_eligible=np.ones(2,bool)))
        source=dict(kind='synthetic-only')
    start=time.monotonic()
    variables=jax.jit(model.init)(jax.random.PRNGKey(args.seed),batch['planes'],batch['context'])
    jax.block_until_ready(variables)
    initialized=time.monotonic()-start
    params=variables['params'];assert parameter_count(params)==count
    assert all(x.dtype==jnp.float32 and np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(params))
    start=time.monotonic()
    outputs=jax.jit(model.apply)(variables,batch['planes'],batch['context']);jax.block_until_ready(outputs)
    forward=time.monotonic()-start
    assert outputs['policy_logits'].shape==(2,4672) and outputs['wdl_logits'].shape==(2,3)
    assert all(x.dtype==jnp.float32 and np.isfinite(np.asarray(x)).all() for x in outputs.values())
    policy=masked_policy(outputs['policy_logits'],batch['legal_mask']);wdl=wdl_predictions(outputs['wdl_logits'])
    assert np.asarray(policy['has_legal_actions']).all()
    assert np.all(np.asarray(policy['probabilities'])[~np.asarray(batch['legal_mask'])]==0)
    np.testing.assert_allclose(np.asarray(policy['probabilities']).sum(-1),1.,atol=1e-6)
    value,metrics=supervised_loss(outputs,batch);assert bool(metrics['valid']) and np.isfinite(value)
    full_backward=dict(tested=False,requested=args.full_gradient)
    if args.full_gradient:
        full_estimate=count*4*6+512*2**20
        # Include parameters, gradients/compiler/temporary allowance and reserve.
        # This is an optional single backward, never an optimizer update.
        if available<full_estimate+2**30:
            full_backward['skip_reason']='Insufficient initial available host RAM for conservative full-backward estimate'
        elif psutil.virtual_memory().available<count*4*3+256*2**20+2**30:
            full_backward['skip_reason']='Insufficient remaining host RAM after initialization/forward'
        else:
            def full_objective(p):return supervised_loss(model.apply({'params':p},batch['planes'],batch['context']),batch)[0]
            start=time.monotonic()
            full_loss,full_grad=jax.jit(jax.value_and_grad(full_objective))(params)
            jax.block_until_ready(full_grad)
            full_seconds=time.monotonic()-start
            assert np.isfinite(full_loss) and all(x.dtype==jnp.float32 and np.isfinite(np.asarray(x)).all()
                for x in jax.tree_util.tree_leaves(full_grad))
            full_backward.update(tested=True,finite=True,seconds=full_seconds,loss=float(full_loss))
            del full_grad
            jax.clear_caches()
    # Reduced-model gradients are checked in either case, using the same ops.
    tiny=ChessTransformer(ModelConfig(2,32,4,64,args.compute))
    tiny_params=tiny.init(jax.random.PRNGKey(args.seed),batch['planes'],batch['context'])['params']
    def objective(p):return supervised_loss(tiny.apply({'params':p},batch['planes'],batch['context']),batch)[0]
    start=time.monotonic();tiny_loss,grad=jax.jit(jax.value_and_grad(objective))(tiny_params);jax.block_until_ready(grad)
    backward=time.monotonic()-start
    assert np.isfinite(tiny_loss) and all(x.dtype==jnp.float32 and np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(grad))
    return dict(status='PASS',profile=args.profile,configuration=config.metadata(),parameter_count=count,
        parameter_bytes=sum(x.size*x.dtype.itemsize for x in jax.tree_util.tree_leaves(params)),
        top_level_parameter_counts={k:parameter_count(v) for k,v in params.items()},
        input_source=source,seed=args.seed,devices=[str(x) for x in jax.devices()],
        python=platform.python_version(),platform=platform.platform(),jax_version=jax.__version__,
        host_available_bytes_before_init=available,estimated_working_bytes=estimate,
        initialized_seconds=initialized,compiled_first_forward_seconds=forward,
        output_shapes={k:list(v.shape) for k,v in outputs.items()},policy_loss=float(metrics['policy_loss']),
        value_loss=float(metrics['value_loss']),search_values=np.asarray(wdl['value']).tolist(),
        reduced_gradient=dict(parameters=parameter_count(tiny_params),seconds=backward,finite=True),
        full_profile_backward_tested=full_backward['tested'],full_profile_backward=full_backward,parameter_updates=0,
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=list(PROFILES),required=True)
    p.add_argument('--compute',choices=['float32','bfloat16'],default='float32')
    p.add_argument('--seed',type=int,default=20261030)
    p.add_argument('--data-root')
    p.add_argument('--full-gradient',action='store_true',help='One full-profile backward only if host RAM permits')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    result=run(args);target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ['status','profile','parameter_count','initialized_seconds','compiled_first_forward_seconds','peak_process_rss_bytes']},indent=2))
