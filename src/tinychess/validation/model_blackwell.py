"""Bounded, isolated model/AdamW validation for Stage 2B; never a trainer."""
from dataclasses import replace
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import psutil

from tinychess.model.transformer import ChessTransformer,ModelConfig,PROFILES,parameter_count
from tinychess.model.objectives import validate_batch,masked_policy,wdl_predictions,supervised_loss
from tinychess.validation.stage2b_plan import EXPECTED_COUNTS,planned_cases


class UnsafeAllocation(RuntimeError):
    pass


def host_available():
    available=psutil.virtual_memory().available
    sources={'psutil_available':available}
    for limit,current in [('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'),
                          ('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes')]:
        try:
            value=Path(limit).read_text().strip()
            if value!='max':
                free=max(0,int(value)-int(Path(current).read_text()))
                available=min(available,free);sources[limit]=free
        except (OSError,ValueError):pass
    return available,sources


def physical_memory():
    result=subprocess.run(['nvidia-smi','--query-gpu=memory.free,memory.used',
                           '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
    rows=[tuple(int(x.strip())*2**20 for x in line.split(',')) for line in result.stdout.splitlines() if line.strip()]
    if not rows:raise RuntimeError('Physical GPU telemetry unavailable')
    return dict(free_bytes=min(x[0] for x in rows),used_max_bytes=max(x[1] for x in rows))


class PhysicalSampler:
    """Device-wide sampled use, including other processes; JAX peak is separate."""
    def __init__(self):
        self.peak=0;self.samples=0;self.errors=0;self.done=threading.Event()
    def start(self):
        def poll():
            while not self.done.is_set():
                try:self.peak=max(self.peak,physical_memory()['used_max_bytes']);self.samples+=1
                except (OSError,ValueError,subprocess.SubprocessError):self.errors+=1
                self.done.wait(0.2)
        self.thread=threading.Thread(target=poll,daemon=True);self.thread.start()
    def stop(self):self.done.set();self.thread.join(timeout=12)


def case_estimate(config,count,batch):
    return count*4*12+batch*config.layers*65*config.width*4*24+512*2**20


def gpu_budget(device):
    stats=device.memory_stats() or {}
    if not stats.get('bytes_limit'):raise RuntimeError('JAX GPU capacity telemetry unavailable')
    free=min(max(0,stats['bytes_limit']-stats.get('bytes_in_use',0)),physical_memory()['free_bytes'])
    return int(min(free*0.4,24*2**30)),stats


def synthetic_batch(size):
    """Synthetic-only labels on scripted, canonically encoded Pgx positions.

    No PGNs, human records, datasets, search, or external rules engine calls.
    """
    from tinychess.chess.pgx_env import init,step
    from tinychess.chess.action_codec import uci_to_action
    from tinychess.chess.state_features import encode_jit
    state=init(jax.random.PRNGKey(9));features=[]
    for move in [None,'e2e4','e7e5','g1f3']:
        if move is not None:state=step(state,uci_to_action(state,move))
        features.append(jax.device_get(encode_jit(state)))
    items=[features[i%len(features)] for i in range(size)]
    masks=np.stack([x['legal_mask'] for x in items])
    return validate_batch(dict(planes=np.stack([x['planes'] for x in items]).astype(np.float32),
        context=np.stack([x['context'] for x in items]).astype(np.float32),legal_mask=masks,
        action=np.array([np.flatnonzero(m)[0] for m in masks],np.int32),
        wdl_class=np.arange(size,dtype=np.int32)%3,policy_weight=np.ones(size,np.float32),
        value_weight=(np.arange(size)%4!=0).astype(np.float32),training_eligible=np.ones(size,bool)))


def tree_finite(tree):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(tree)]))


def floating_fp32(tree):
    return all(x.dtype==jnp.float32 for x in jax.tree_util.tree_leaves(tree)
               if jnp.issubdtype(x.dtype,jnp.floating))


def run_case(profile,compute,batch_size,*,require_gpu=True,seed=20261030):
    if profile=='tiny':
        if require_gpu:raise ValueError('Tiny is a CPU unit-test profile only')
        config=ModelConfig(1,16,4,32,compute)
    else:config=replace(PROFILES[profile],compute=compute)
    if batch_size not in (4,32) and require_gpu:raise ValueError('Only bounded Stage 2B batches 4 and 32 are allowed')
    model=ChessTransformer(config);device=jax.devices()[0];gpu=device.platform=='gpu'
    if require_gpu and (not gpu or '6000' not in device.device_kind or 'Blackwell' not in device.device_kind):
        raise RuntimeError('RTX PRO 6000 Blackwell required; CPU results cannot satisfy the gate')
    shape_p=jax.ShapeDtypeStruct((batch_size,8,8,120),jnp.float32)
    shape_c=jax.ShapeDtypeStruct((batch_size,8),jnp.float32)
    shape_params=jax.eval_shape(model.init,jax.random.PRNGKey(seed),shape_p,shape_c)['params']
    count=parameter_count(shape_params)
    if profile in EXPECTED_COUNTS and count!=EXPECTED_COUNTS[profile]:raise RuntimeError('Model count drift')
    available,host_sources=host_available()
    estimated=case_estimate(config,count,batch_size)
    row=dict(profile=profile,compute=compute,batch=batch_size,status='RUNNING',gpu_measured=gpu,
        expected_parameter_count=count,configuration=config.metadata(),seed=seed,parameter_updates=0,
        input_source='synthetic labels; repeated four scripted canonical Pgx positions',
        default_matmul_precision=str(jax.config.jax_default_matmul_precision),
        host_available_sources=host_sources,phases={},estimated_device_bytes=estimated,
        packages={p:importlib.metadata.version(p) for p in ['jax','jaxlib','flax','optax']})
    sampler=None
    try:
        if gpu:
            budget,stats=gpu_budget(device);row['memory_budget_bytes']=budget
            if estimated>budget:
                row.update(status='SKIPPED',reason='Conservative model/optimizer/activation estimate exceeds GPU budget');return row
        if available<count*4*6+1024*2**20:
            row.update(status='SKIPPED',reason='Host RAM/container allowance is insufficient');return row
        if gpu:sampler=PhysicalSampler();sampler.start()
        batch=synthetic_batch(batch_size)
        def execute(name,fn,*args):
            started=time.perf_counter();lowered=jax.jit(fn).lower(*args)
            trace=time.perf_counter()-started;started=time.perf_counter();compiled=lowered.compile()
            compilation=time.perf_counter()-started;analysis=compiled.memory_analysis()
            if gpu:
                if analysis is None:raise UnsafeAllocation('Compiled memory analysis unavailable; execution skipped')
                current=device.memory_stats() or {};fresh_budget,_=gpu_budget(device)
                buffers=analysis.argument_size_in_bytes+analysis.output_size_in_bytes+analysis.temp_size_in_bytes
                required=2*buffers+current.get('bytes_in_use',0)+512*2**20
                if required>min(row['memory_budget_bytes'],fresh_budget):
                    raise UnsafeAllocation(f'{name}: compiled buffers plus reserve exceed refreshed GPU budget')
            started=time.perf_counter();result=compiled(*args);jax.block_until_ready(result)
            row['phases'][name]=dict(trace_seconds=trace,compile_seconds=compilation,
                synchronized_execution_seconds=time.perf_counter()-started,
                compiled_buffers=None if analysis is None else dict(arguments=analysis.argument_size_in_bytes,
                    outputs=analysis.output_size_in_bytes,temporaries=analysis.temp_size_in_bytes))
            return result
        params=execute('initialize',lambda key,p,c:model.init(key,p,c)['params'],jax.random.PRNGKey(seed),batch['planes'],batch['context'])
        row['parameter_count']=parameter_count(params)
        if row['parameter_count']!=count:raise RuntimeError('Initialized parameter count differs from traced count')
        if not floating_fp32(params) or not bool(jax.jit(tree_finite)(params)):raise RuntimeError('Invalid fp32 master parameters')
        forward=lambda p,b:model.apply({'params':p},b['planes'],b['context'])
        outputs=execute('forward',forward,params,batch)
        if outputs['policy_logits'].shape!=(batch_size,4672) or outputs['wdl_logits'].shape!=(batch_size,3) or any(x.dtype!=jnp.float32 for x in outputs.values()):
            raise RuntimeError('Incorrect fp32 output shape/dtype')
        policy=masked_policy(outputs['policy_logits'],batch['legal_mask']);wdl=wdl_predictions(outputs['wdl_logits'])
        if not bool(tree_finite(outputs)) or not np.all(np.asarray(policy['has_legal_actions'])):raise RuntimeError('Invalid forward')
        np.testing.assert_allclose(np.asarray(policy['probabilities']).sum(-1),1.,atol=1e-6)
        if np.any(np.asarray(policy['probabilities'])[~np.asarray(batch['legal_mask'])]!=0):raise RuntimeError('Illegal policy probability')
        if np.any(np.abs(np.asarray(wdl['value']))>1):raise RuntimeError('Invalid WDL scalar')
        tx=optax.adamw(learning_rate=1e-4,b1=0.9,b2=0.95,eps=1e-8,mu_dtype=jnp.float32,weight_decay=0.01)
        opt_state=tx.init(params);jax.block_until_ready(opt_state)
        if int(opt_state[0].count)!=0 or not floating_fp32(opt_state):raise RuntimeError('Invalid AdamW initial state')
        def objective(p,b):return supervised_loss(forward(p,b),b)
        ((loss,metrics),gradient)=execute('backward',jax.value_and_grad(objective,has_aux=True),params,batch)
        if not bool(metrics['valid']) or not np.isfinite(float(loss)) or not floating_fp32(gradient) or not bool(jax.jit(tree_finite)(gradient)):
            raise RuntimeError('Invalid loss or gradients')
        grad_squared=jax.jit(lambda g:sum(jnp.sum(jnp.square(x),dtype=jnp.float32)
            for x in jax.tree_util.tree_leaves(g)))(gradient)
        if not np.isfinite(float(grad_squared)) or float(grad_squared)<=0:raise RuntimeError('No finite nonzero gradient signal')
        def update(p,s,g):
            updates,new_state=tx.update(g,s,p)
            return optax.apply_updates(p,updates),new_state
        updated,new_state=execute('adamw_once',update,params,opt_state,gradient)
        row['parameter_updates']=int(new_state[0].count)
        if row['parameter_updates']!=1:raise RuntimeError('AdamW must execute exactly once')
        delta=jax.jit(lambda a,b:sum(jnp.sum(jnp.square(x-y),dtype=jnp.float32) for x,y in
                                   zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b))))(updated,params)
        if not np.isfinite(float(delta)) or float(delta)<=0 or not floating_fp32(updated) or not floating_fp32(new_state) or not bool(jax.jit(tree_finite)((updated,new_state))):
            raise RuntimeError('Parameters did not update finitely in fp32')
        after=execute('post_update_forward',forward,updated,batch)
        post_loss,post_metrics=supervised_loss(after,batch)
        if not bool(post_metrics['valid']) or not np.isfinite(float(post_loss)) or not bool(jax.jit(tree_finite)(after)):raise RuntimeError('Nonfinite output/loss after update')
        row.update(status='PASS' if gpu else 'CPU_TEST_ONLY',parameter_count=count,master_params_fp32=True,
            gradients_finite=True,optimizer_fp32=True,parameter_delta_squared=float(delta),loss_before=float(loss),
            gradient_squared_norm=float(grad_squared),
            loss_after=float(post_loss),policy_loss=float(metrics['policy_loss']),value_loss=float(metrics['value_loss']),
            gpu_kind=device.device_kind,optimizer=dict(name='AdamW',learning_rate=1e-4,b1=0.9,b2=0.95,weight_decay=0.01),
            output_shapes={k:list(x.shape) for k,x in outputs.items()},output_dtypes={k:str(x.dtype) for k,x in outputs.items()},
            update_note='Exactly one step on a fresh initialization; no loss-decrease requirement or tuning')
    except UnsafeAllocation as exc:
        row.update(status='SKIPPED',reason=str(exc))
    except Exception as exc:
        row.update(status='FAILED',error_type=type(exc).__name__,reason=str(exc))
    finally:
        if sampler:
            sampler.stop();row['physical_sampled_peak_used_bytes']=sampler.peak
            row['physical_samples']=sampler.samples;row['physical_sample_errors']=sampler.errors
        stats=device.memory_stats() or {}
        row['jax_peak_bytes_in_use']=stats.get('peak_bytes_in_use')
        row['jax_memory_stats']=stats
        row['peak_host_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    return row
