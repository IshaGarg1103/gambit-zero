"""One explicitly bounded Stage 3A diagnostic; orchestrator chooses A/B/C/D."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

from tinychess.model.transformer import ChessTransformer,PROFILES,parameter_count
from tinychess.model.objectives import validate_batch,supervised_loss,masked_policy
from tinychess.train.overfit_sample import load_sample
from tinychess.train.overfit_core import create_state,evaluate,indices,learning_rate,verdict
from tinychess.train.overfit_diagnosis import diagnostic_step,policy_token_probe,fingerprint,accuracy
from tinychess.validation.model_blackwell import PhysicalSampler,host_available,gpu_budget,case_estimate,floating_fp32


def main(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise RuntimeError('Diagnostic output exists; use a new path')
    limits={'A':(100,120),'B':(500,180),'C':(500,180),'D':(3000,240)}
    if not 0<a.steps<=limits[a.stage][0] or not 0<a.seconds<=limits[a.stage][1]:raise ValueError('Diagnostic budget exceeds frozen limits')
    if a.lr not in (1e-3,3e-4,1e-4):raise ValueError('Unapproved LR')
    if a.compute!=('float32' if a.stage=='C' else 'bfloat16'):raise ValueError('Unapproved compute mode for stage')
    arrays,manifest=load_sample(a.sample)
    device=jax.devices()[0]
    if device.platform!='gpu' or 'Blackwell' not in device.device_kind:raise RuntimeError('Blackwell required')
    config=replace(PROFILES['25m'],compute=a.compute);model=ChessTransformer(config)
    batch_size=1 if a.stage=='A' else 32
    if case_estimate(config,25622092,batch_size)>gpu_budget(device)[0] or host_available()[0]<3*2**30:
        raise RuntimeError('Unsafe resources')
    selected=np.array([np.flatnonzero((arrays['value_weight']>0)&(arrays['legal_mask'].sum(-1)>1))[0]]) if a.stage=='A' else np.arange(32)
    def batch(ix):return validate_batch({k:v[ix] for k,v in arrays.items()})
    fixed=batch(selected)
    config_meta=dict(stage=a.stage,lr=a.lr,steps_budget=a.steps,training_seconds_budget=a.seconds,
        compute=a.compute,model_parameters=25622092,seed=20261031,grad_clip=1.,warmup=100,weight_decay=0.,
        sample_logical_sha256=manifest['logical_sha256'],sample_sha256=manifest['sample_sha256'],
        subset_indices=selected.tolist() if a.stage!='D' else list(range(1024)),history_dropout=0.,
        initialization='original Stage 3A zero output heads',
        diagnostic_source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),ROOT/'src/tinychess/train/overfit_diagnosis.py',ROOT/'src/tinychess/train/overfit_core.py',ROOT/'src/tinychess/model/transformer.py']})
    result=dict(config=config_meta,status='RUNNING',learning_curve=[],probes=[],all_steps=[],audit={})
    r=wandb.init(entity='ishagarg-research',project='gambit-zero',group='m2-overfit-diagnostics',
        name=f"diagnostic-{a.stage}-{a.compute}-lr{a.lr:g}",job_type='stage3a-failure-diagnosis',
        config=config_meta,mode='online',dir=str(out),settings=wandb.Settings(disable_git=True,disable_code=True,save_code=False,console='off'))
    result['wandb_url']=r.url
    sampler=PhysicalSampler();sampler.start();wall=time.perf_counter()
    def eval_population(s):
        if a.stage!='D':return accuracy(jax.device_get(evaluate(s,fixed)))
        sums=dict(policy_loss=0.,wdl_loss=0.,policy_correct=0.,wdl_correct=0.,policy_weight=0.,value_weight=0.)
        for start in range(0,1024,32):
            m=jax.device_get(evaluate(s,batch(np.arange(start,start+32))))
            for key in ('policy_loss','wdl_loss'):
                sums[key]+=float(m[key])*float(m['policy_weight' if key=='policy_loss' else 'value_weight'])
            for key in ('policy_correct','wdl_correct','policy_weight','value_weight'):sums[key]+=float(m[key])
        return dict(policy_loss=sums['policy_loss']/sums['policy_weight'],wdl_loss=sums['wdl_loss']/sums['value_weight'],
            policy_accuracy=sums['policy_correct']/sums['policy_weight'],wdl_accuracy=sums['wdl_correct']/sums['value_weight'])
    def passed(m):
        if a.stage=='A':return m['policy_accuracy']==1. and m['wdl_accuracy']==1. and m['policy_loss']<.05 and m['wdl_loss']<.05
        if a.stage in ('B','C'):return m['policy_accuracy']>=.99 and m['wdl_accuracy']>=.95 and m['policy_loss']<=.05 and m['wdl_loss']<=.05
        return verdict(initial,m,int(state.step))=='PASS'
    try:
        t=time.perf_counter();state=jax.jit(lambda key:create_state(model,key,fixed,a.lr))(jax.random.PRNGKey(20261031));jax.block_until_ready(state)
        result['init_seconds']=time.perf_counter()-t
        assert parameter_count(state.params)==25622092 and floating_fp32(state.params) and floating_fp32(state.opt_state)
        result['initial_parameter_sha256']=fingerprint(state.params)
        if a.stage=='A':result['audit']['model_token_mapping']=policy_token_probe(model,state.params,fixed['planes'],fixed['context'])
        initial=eval_population(state);result['initial']=initial
        result['learning_curve'].append(dict(step=0,**initial));r.log({'eval/'+k:v for k,v in initial.items()},step=0)
        t=time.perf_counter();compiled=diagnostic_step.lower(state,fixed).compile();result['compile_seconds']=time.perf_counter()-t
        numerical_checks=[]
        analysis=compiled.memory_analysis()
        assert analysis is not None
        buffers=analysis.argument_size_in_bytes+analysis.output_size_in_bytes+analysis.temp_size_in_bytes
        assert 2*buffers+(device.memory_stats() or {}).get('bytes_in_use',0)+512*2**20<=gpu_budget(device)[0]
        start=time.perf_counter();pairing_max=0.;step_loss_max=0.
        for step in range(a.steps):
            if time.perf_counter()-start>a.seconds:
                result['budget_stop']='training time budget';break
            ix=indices(20261031,step,1024,32) if a.stage=='D' else selected
            b=batch(ix)
            # Same index vector applies to every field; compare original bytes.
            if step in (0,1,99) or (step+1)%100==0:
                for name in arrays:np.testing.assert_array_equal(np.asarray(b[name]),arrays[name][ix])
                before=jax.device_get(evaluate(state,b))
                old_parameter_hash=fingerprint(state.params)
            nxt,m=compiled(state,b);jax.block_until_ready((nxt,m));m=jax.device_get(m)
            if not bool(m['finite']):raise FloatingPointError('Nonfinite/rejected diagnostic update')
            if int(nxt.step)!=step+1:raise RuntimeError('Optimizer step did not persist')
            if step in (0,1,99) or (step+1)%100==0:
                for name in ('loss','policy_loss','wdl_loss'):
                    step_loss_max=max(step_loss_max,abs(float(m[name])-float(before[name])))
                if fingerprint(state.params)!=old_parameter_hash:raise RuntimeError('Input parameters mutated during compiled update')
                if step_loss_max>1e-4 and not numerical_checks:
                    # Read-only fp32 loss/AD comparison on identical fp32 masters.
                    # This is not a fp32 training run and applies no update.
                    reference=ChessTransformer(replace(config,compute='float32'))
                    def f(p):return supervised_loss(reference.apply({'params':p},b['planes'],b['context']),b)[0]
                    plain=jax.jit(f)(state.params)
                    ad,_=jax.jit(jax.value_and_grad(f))(state.params)
                    numerical_checks.append(dict(parameter_step=step,
                        bf16_evaluation_loss=float(before['loss']),bf16_ad_loss=float(m['loss']),
                        bf16_loss_difference=abs(float(before['loss'])-float(m['loss'])),
                        fp32_evaluation_loss=float(plain),fp32_ad_loss=float(ad),
                        fp32_loss_difference=abs(float(plain)-float(ad)),
                        input_parameter_fingerprint_unchanged=True))
                    if abs(float(plain)-float(ad))>1e-5:raise RuntimeError('Same-step loss discrepancy persists in fp32')
                if abs(float(m['loss'])-float(before['loss']))>max(.005,.01*abs(float(before['loss']))):
                    raise RuntimeError('Train/eval loss discrepancy exceeds bf16 numerical bound')
                counters=[int(x) for x in jax.tree_util.tree_leaves(nxt.opt_state) if np.ndim(x)==0 and jnp.issubdtype(x.dtype,jnp.integer)]
                assert counters and all(c==step+1 for c in counters)
                result['probes'].append(dict(parameter_step=step,optimizer_counters_after=counters,
                    gradient_groups={k:float(v) for k,v in m['gradient_groups'].items()},
                    update_groups={k:float(v) for k,v in m['update_groups'].items()},
                    update_dot_gradient=float(m['update_dot_gradient'])))
                if step==0:assert float(m['update_dot_gradient'])<0
            state=nxt
            item=dict(step=step+1,grad_norm=float(m['grad_norm']),clipped_grad_norm=float(m['clipped_grad_norm']),lr=float(learning_rate(step,a.lr)))
            result['all_steps'].append(item)
            if not all(np.isfinite(v) for v in item.values()):raise FloatingPointError('Nonfinite gradient norm')
            if (step+1)%10==0:r.log(item,step=step+1)
            interval=10 if a.stage=='A' else (100 if a.stage=='D' else 50)
            if (step+1)%interval==0 or step+1==a.steps:
                metrics=eval_population(state);row=dict(step=step+1,**metrics)
                result['learning_curve'].append(row);r.log({'eval/'+k:v for k,v in metrics.items()},step=step+1)
                print(json.dumps(row),flush=True)
                if passed(metrics):break
        result.update(final=eval_population(state),steps=int(state.step),training_seconds=time.perf_counter()-start,
            initial_parameter_contract='fp32 master/optimizer; original zero heads; no dropout',
            final_parameter_sha256=fingerprint(state.params))
        result['status']='PASS' if passed(result['final']) else 'FAIL'
        result['audit'].update(same_step_train_eval_max_absolute_error=step_loss_max,numerical_loss_checks=numerical_checks,
            paired_fields_verified=True,state_counters_persist=True,
            clip_frequency=sum(x['grad_norm']>1 for x in result['all_steps'])/len(result['all_steps']),
            grad_norm_min=min(x['grad_norm'] for x in result['all_steps']),grad_norm_max=max(x['grad_norm'] for x in result['all_steps']),
            post_clip_norm_max=max(x['clipped_grad_norm'] for x in result['all_steps']))
        r.log({'gate/pass':int(result['status']=='PASS')},step=result['steps'])
    except Exception as exc:
        result.update(status='ERROR',failure=type(exc).__name__+': '+str(exc))
        raise
    finally:
        sampler.stop();r.finish(exit_code=0 if result['status']=='PASS' else 1)
        result.update(total_wall_seconds=time.perf_counter()-wall,jax_memory_stats=device.memory_stats(),sampled_device_peak_bytes=sampler.peak,
            host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        (out/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=list('ABCD'),required=True);p.add_argument('--lr',type=float,required=True)
    p.add_argument('--steps',type=int,required=True);p.add_argument('--seconds',type=int,required=True)
    p.add_argument('--compute',choices=['bfloat16','float32'],default='bfloat16');p.add_argument('--sample',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=main(a);raise SystemExit(0 if r['status']=='PASS' else 1)
