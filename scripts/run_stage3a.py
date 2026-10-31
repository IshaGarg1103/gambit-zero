"""Bounded 25M/1024-real-position overfit only; no pilots, LR sweeps or self-play."""
import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import jax
import jax.numpy as jnp
import numpy as np
from tinychess.model.transformer import ChessTransformer,PROFILES,parameter_count
from tinychess.model.objectives import validate_batch
from tinychess.train.overfit_sample import load_sample
from tinychess.train.overfit_core import (create_state,train_step,evaluate,indices,
    save_checkpoint,restore_checkpoint,verdict,learning_rate)
from tinychess.train.tracking import Tracker
from tinychess.validation.model_blackwell import host_available,gpu_budget,case_estimate,PhysicalSampler,floating_fp32
from tinychess.data.training_loader import REVISION


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise RuntimeError('Use a fresh output or explicit --resume')
    if not 1<=args.steps<=3000:raise ValueError('Stage 3A is bounded at 3000 updates')
    if args.batch!=32 or args.lr not in (1e-3,3e-4):raise ValueError('Only approved microbatch/LR allowed')
    arrays,manifest=load_sample(args.sample)
    if len(arrays['action'])!=1024:raise ValueError('Production run requires exactly 1024 frozen real positions')
    device=jax.devices()[0]
    if device.platform!='gpu' or 'Blackwell' not in device.device_kind or '6000' not in device.device_kind:
        raise RuntimeError('Production overfit requires RTX PRO 6000 Blackwell')
    config=replace(PROFILES['25m'],compute='bfloat16');model=ChessTransformer(config)
    budget,_=gpu_budget(device);available,host_sources=host_available()
    if case_estimate(config,25622092,32)>budget or available<3*2**30:
        raise RuntimeError('Insufficient conservative GPU/host memory headroom')
    source_manifest=json.loads((ROOT/'STAGE3A_BUNDLE.json').read_text())
    for n,h in source_manifest['files'].items():
        if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h:raise RuntimeError('Source changed: '+n)
    source_sha=hashlib.sha256((ROOT/'STAGE3A_BUNDLE.json').read_bytes()).hexdigest()
    metadata=dict(sample_logical_sha256=manifest['logical_sha256'],dataset_revision=REVISION,
        source_manifest_sha256=source_sha,seed=args.seed,lr=args.lr,microbatch=32,
        source_base_commit=source_manifest['source_base_commit'],sample_positions=1024,sample_games=manifest['games'],
        model='25m',compute='bfloat16',history_dropout=0.,warmup=100,weight_decay=0.,grad_clip=1.,
        initialization='random backbone; zero policy_head and value_head output projections')
    result=dict(status='RUNNING',config=metadata,model_parameters=25622092,
        gpu=device.device_kind,host_memory_sources=host_sources,sample_manifest=manifest,
        source_base_commit=source_manifest['source_base_commit'],phases={},learning_curve=[],
        versions={p:importlib.metadata.version(p) for p in ('jax','jaxlib','flax','optax','wandb')})
    sampler=PhysicalSampler();sampler.start();tracker=None;t0=time.perf_counter()
    def batch(ix):return validate_batch({k:v[ix] for k,v in arrays.items()})
    small=batch(np.arange(32))
    def evaluate_all(state):
        totals=dict(policy_loss=0.,wdl_loss=0.,policy_correct=0.,wdl_correct=0.,policy_weight=0.,value_weight=0.)
        for start in range(0,1024,32):
            m=jax.device_get(evaluate(state,batch(np.arange(start,start+32))))
            if not bool(m['valid']) or not all(np.isfinite(v) for v in m.values()):raise FloatingPointError('Nonfinite/invalid evaluation')
            totals['policy_loss']+=float(m['policy_loss'])*float(m['policy_weight'])
            totals['wdl_loss']+=float(m['wdl_loss'])*float(m['value_weight'])
            for k in ('policy_correct','wdl_correct','policy_weight','value_weight'):totals[k]+=float(m[k])
        if totals['value_weight']<=0:raise RuntimeError('No value-weighted examples')
        return dict(policy_loss=totals['policy_loss']/totals['policy_weight'],
            wdl_loss=totals['wdl_loss']/totals['value_weight'],
            policy_accuracy=totals['policy_correct']/totals['policy_weight'],
            wdl_accuracy=totals['wdl_correct']/totals['value_weight'],finite=True)
    try:
        t=time.perf_counter();state=jax.jit(lambda key:create_state(model,key,small,args.lr))(jax.random.PRNGKey(args.seed));jax.block_until_ready(state)
        result['phases']['initialize_compile_and_execute_seconds']=time.perf_counter()-t
        if parameter_count(state.params)!=25622092 or not floating_fp32(state.params) or not floating_fp32(state.opt_state):raise RuntimeError('Model count/precision mismatch')
        initial=evaluate_all(state)
        baseline=float(np.mean(np.log(arrays['legal_mask'].sum(-1))))
        result['initial']=dict(initial,uniform_policy_baseline=baseline,uniform_wdl_baseline=float(np.log(3.)))
        if abs(initial['policy_loss']-baseline)>1e-4 or abs(initial['wdl_loss']-np.log(3))>1e-4:
            raise RuntimeError('Initial uniform-loss sanity gate failed')
        if args.resume:
            state,receipt=restore_checkpoint(args.resume,state,metadata)
            result['resume_from_step']=int(state.step)
        else:receipt={}
        tracker=Tracker(metadata,out/'wandb',mode=args.wandb_mode,run_id=receipt.get('wandb_run_id'))
        result['wandb_url']=tracker.url;result['wandb_id']=tracker.id
        result['learning_curve'].append(dict(step=int(state.step),**evaluate_all(state)))
        tracker.log({'eval/'+k:v for k,v in result['learning_curve'][-1].items() if k!='step'},int(state.step))
        t=time.perf_counter();compiled=train_step.lower(state,small).compile()
        result['phases']['training_compile_seconds']=time.perf_counter()-t
        mem=compiled.memory_analysis()
        if mem is None:raise RuntimeError('Training memory analysis unavailable')
        buffers=mem.argument_size_in_bytes+mem.output_size_in_bytes+mem.temp_size_in_bytes
        result['compiled_buffer_bytes']=buffers
        if 2*buffers+(device.memory_stats() or {}).get('bytes_in_use',0)+512*2**20>gpu_budget(device)[0]:
            raise RuntimeError('Compiled training buffers exceed safe memory budget')
        train_start=time.perf_counter();exec_seconds=0.;checkpoints=[]
        checkpoint_metadata=dict(metadata,wandb_run_id=tracker.id)
        for step in range(int(state.step),args.steps):
            b=batch(indices(args.seed,step,1024,32));t=time.perf_counter()
            state,m=compiled(state,b);jax.block_until_ready((state,m));exec_seconds+=time.perf_counter()-t
            m=jax.device_get(m)
            if not bool(m['finite']) or int(state.step)!=step+1:raise FloatingPointError('Nonfinite or rejected optimizer update')
            if (step+1)%10==0:
                tracker.log(dict(train_policy_loss=float(m['policy_loss']),train_wdl_loss=float(m['wdl_loss']),
                    train_policy_accuracy=float(m['policy_correct']/m['policy_weight']),
                    train_wdl_accuracy=float(m['wdl_correct']/max(float(m['value_weight']),1.)),
                    grad_norm=float(m['grad_norm']),learning_rate=float(learning_rate(step,args.lr))),step+1)
            if (step+1)%100==0 or step+1==args.steps:
                metrics=evaluate_all(state);row=dict(step=step+1,seconds=time.perf_counter()-train_start,**metrics)
                result['learning_curve'].append(row);tracker.log({'eval/'+k:v for k,v in metrics.items()},step+1)
                print(json.dumps(row),flush=True)
                (out/'progress.json').write_text(json.dumps(row,indent=2)+'\n')
                if (step+1)%500==0 or verdict(initial,metrics,step+1)=='PASS' or step+1==args.steps:
                    receipt=save_checkpoint(out/'checkpoint',state,checkpoint_metadata)
                    restored,_=restore_checkpoint(out/'checkpoint',state,metadata)
                    equal=all(np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(jax.tree_util.tree_leaves(state),jax.tree_util.tree_leaves(restored)))
                    if not equal or evaluate_all(restored)!=metrics:raise RuntimeError('Checkpoint roundtrip mismatch')
                    # Continue from the restored optimizer/step state, not the original.
                    state=jax.tree_util.tree_map(jnp.asarray,restored)
                    checkpoints.append(dict(step=step+1,sha256=receipt['sha256'],bytes=receipt['bytes'],exact_tree_equal=True,metrics_equal=True))
                if verdict(initial,metrics,step+1)=='PASS':break
        result.update(final=evaluate_all(state),steps=int(state.step),checkpoint_checks=checkpoints,
            synchronized_training_execution_seconds=exec_seconds,training_wall_seconds=time.perf_counter()-train_start)
        result['status']=verdict(initial,result['final'],result['steps'])
        result['checkpoint_resumed_and_continued']=any(c['step']<result['steps'] for c in checkpoints)
        if not checkpoints or not floating_fp32(state.params) or not floating_fp32(state.opt_state):raise RuntimeError('Missing checkpoint/precision verification')
        result['checkpoint_artifact']=tracker.checkpoint(out/'checkpoint',receipt)
        tracker.log({'gate/pass':int(result['status']=='PASS')},result['steps'])
    except BaseException as exc:
        result.update(status='FAIL',failure=type(exc).__name__+': '+str(exc))
        (out/'failure.txt').write_text(traceback.format_exc())
    finally:
        sampler.stop()
        if tracker:tracker.finish(result['status']=='PASS')
        result.update(total_wall_seconds=time.perf_counter()-t0,jax_memory_stats=device.memory_stats(),
            sampled_device_peak_bytes=sampler.peak,physical_samples=sampler.samples,physical_sample_errors=sampler.errors,
            peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            protected_sources_unchanged=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in source_manifest['files'].items()))
        if not result['protected_sources_unchanged']:result['status']='FAIL'
        (out/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('STAGE3A',result['status'],result.get('final'),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sample',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=20261031);p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--batch',type=int,default=32);p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--wandb-mode',choices=['online','offline','disabled'],default='online');p.add_argument('--resume')
    a=p.parse_args();raise SystemExit(0 if run(a)['status']=='PASS' else 1)
