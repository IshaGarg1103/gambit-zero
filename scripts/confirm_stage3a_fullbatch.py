"""Exactly one 25M full-batch overfit confirmation, at most 2000 updates."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import jax
import jax.numpy as jnp
import numpy as np
import wandb

from tinychess.model.transformer import ChessTransformer,PROFILES,parameter_count
from tinychess.model.objectives import validate_batch
from tinychess.train.overfit_core import create_state,learning_rate,save_checkpoint,restore_checkpoint
from tinychess.train.overfit_diagnosis import fingerprint
from tinychess.train.overfit_sample import load_sample
from tinychess.train.fullbatch_confirmation import direct_step,accumulation_step,evaluate_rows,strata
from tinychess.validation.model_blackwell import PhysicalSampler,gpu_budget,host_available,floating_fp32


def main():
    output=ROOT/'data/processed/stage3a_fullbatch_confirmation';output.mkdir(parents=True,exist_ok=True)
    # Atomic guard prevents accidental second confirmation on notebook rerun.
    with (output/'single_run_guard.json').open('x') as f:f.write('{"max_updates":2000,"runs":1}\n')
    if not os.environ.get('WANDB_API_KEY'):raise RuntimeError('Use WANDB_API_KEY environment only')
    arrays,manifest=load_sample(ROOT/'input')
    if len(arrays['action'])!=1024:raise RuntimeError('Wrong frozen population')
    if manifest['sample_sha256']!='cc00bda4d5feb8a980be6e32208712f50eb94bb197ae60d01aeaff6e47ced590':raise RuntimeError('Frozen sample changed')
    device=jax.devices()[0]
    if device.platform!='gpu' or 'Blackwell' not in device.device_kind:raise RuntimeError('Blackwell required')
    if host_available()[0]<4*2**30:raise RuntimeError('Insufficient host headroom')
    model=ChessTransformer(replace(PROFILES['25m'],compute='bfloat16'))
    batch=validate_batch(arrays);small=jax.tree_util.tree_map(lambda x:x[:32],batch)
    metadata=dict(model='25m',parameters=25622092,compute='bfloat16',master_dtype='float32',effective_batch=1024,
        lr=.0003,warmup=100,beta1=.9,beta2=.95,epsilon=1e-8,weight_decay=0.,clip_norm=1.,seed=20261031,
        history_dropout=0.,max_updates=2000,sample_sha256=manifest['sample_sha256'],sample_logical_sha256=manifest['logical_sha256'],
        policy_gate_accuracy=.99,policy_gate_ce=.05,initialization='original random backbone / zero output heads',
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'src/tinychess/train/fullbatch_confirmation.py',ROOT/'src/tinychess/train/overfit_core.py',ROOT/'src/tinychess/model/transformer.py']})
    result=dict(status='RUNNING',configuration=metadata,learning_curve=[],gradient_norms=[],phases={},D_checkpoint_available=False,
        D_checkpoint_note='Prior D stored metrics only; no local .msgpack or W&B checkpoint artifact was retained. No rerun to reconstruct it.')
    sampler=PhysicalSampler();sampler.start();run=None;started=time.perf_counter()
    def metrics(m):
        return dict(policy_accuracy=float(m['policy_correct']/m['policy_weight']),policy_loss=float(m['policy_loss']),
            wdl_accuracy=float(m['wdl_correct']/m['value_weight']),wdl_loss=float(m['wdl_loss']))
    try:
        t=time.perf_counter();state=jax.jit(lambda key:create_state(model,key,small,.0003))(jax.random.PRNGKey(20261031));jax.block_until_ready(state)
        result['phases']['init_seconds']=time.perf_counter()-t
        assert parameter_count(state.params)==25622092 and floating_fp32(state.params) and floating_fp32(state.opt_state)
        result['initial_parameter_sha256']=fingerprint(state.params)
        # Lower/compile each possible execution path before executing ANY update.
        t=time.perf_counter();compiled=direct_step.lower(state,batch).compile()
        result['phases']['direct_trace_compile_seconds']=time.perf_counter()-t
        analysis=compiled.memory_analysis();budget,_=gpu_budget(device)
        safe=analysis is not None
        if safe:
            buffers=analysis.argument_size_in_bytes+analysis.output_size_in_bytes+analysis.temp_size_in_bytes
            result['direct_compiled_buffers']=dict(arguments=analysis.argument_size_in_bytes,outputs=analysis.output_size_in_bytes,temporaries=analysis.temp_size_in_bytes)
            safe=(2*buffers+(device.memory_stats() or {}).get('bytes_in_use',0)+512*2**20<=budget)
        result['execution_mode']='direct-1024' if safe else '32x32-global-normalized-accumulation'
        if not safe:
            del compiled
            t=time.perf_counter();compiled=accumulation_step.lower(state,batch).compile()
            result['phases']['accumulation_trace_compile_seconds']=time.perf_counter()-t
            a=compiled.memory_analysis()
            if a is None or 2*(a.argument_size_in_bytes+a.output_size_in_bytes+a.temp_size_in_bytes)+(device.memory_stats() or {}).get('bytes_in_use',0)+512*2**20>gpu_budget(device)[0]:
                raise RuntimeError('No safe full-batch execution path; no update executed')
        initial=evaluate_rows(state,batch);jax.block_until_ready(initial)
        result['initial']=metrics(initial);result['learning_curve'].append(dict(step=0,**result['initial']))
        run=wandb.init(entity='ishagarg-research',project='gambit-zero',group='m2-overfit-diagnostics',
            name='stage3a-fullbatch-1024-confirmation',job_type='single-fullbatch-confirmation',
            mode='online',dir=str(output),config=dict(metadata,execution_mode=result['execution_mode']),
            settings=wandb.Settings(disable_git=True,disable_code=True,save_code=False,console='off'))
        result['wandb_url']=run.url;result['wandb_id']=run.id
        run.log({'eval/'+k:v for k,v in result['initial'].items()},step=0)
        begin=time.perf_counter();update_seconds=0.;clip_count=0
        for step in range(2000):
            t=time.perf_counter();state,m=compiled(state,batch);jax.block_until_ready((state,m));update_seconds+=time.perf_counter()-t
            m=jax.device_get(m)
            if not bool(m['finite']) or int(state.step)!=step+1:raise FloatingPointError('Nonfinite/rejected update')
            norm=float(m['grad_norm']);post=float(m['post_clip_norm'])
            if not np.isfinite(norm) or not np.isfinite(post):raise FloatingPointError('Nonfinite gradient norm')
            clip_count+=int(norm>1.)
            result['gradient_norms'].append(dict(step=step+1,before=norm,after=post))
            evaluation=evaluate_rows(state,batch);jax.block_until_ready(evaluation)
            mm=metrics(evaluation)
            if not bool(evaluation['valid']) or not all(np.isfinite(v) for v in mm.values()):raise FloatingPointError('Nonfinite evaluation')
            row=dict(step=step+1,**mm);result['learning_curve'].append(row)
            run.log(dict(**{'eval/'+k:v for k,v in mm.items()},grad_norm=norm,post_clip_norm=post,
                         learning_rate=float(learning_rate(step,.0003))),step=step+1)
            if (step+1)%10==0:
                (output/'progress.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
            if mm['policy_accuracy']>=.99 and mm['policy_loss']<=.05:
                result['status']='PASS';break
        else:result['status']='FAIL'
        result.update(final=mm,steps=int(state.step),clip_active_count=clip_count,clip_active_fraction=clip_count/int(state.step),
            training_wall_seconds=time.perf_counter()-begin,synchronized_update_seconds=update_seconds,
            examples_seen=int(state.step)*1024)
        prediction=np.asarray(evaluation['policy_prediction']);result['policy_predictions']=prediction.tolist()
        result['stratification']=strata(manifest['identities'],arrays['action'],prediction,arrays['legal_mask'].sum(-1))
        receipt=save_checkpoint(output/'checkpoint',state,dict(metadata,wandb_run_id=run.id))
        restored,_=restore_checkpoint(output/'checkpoint',state,metadata)
        assert fingerprint(restored.params)==fingerprint(state.params)
        restored_metrics=metrics(evaluate_rows(restored,batch));assert restored_metrics==mm
        result['checkpoint']=dict(receipt,restored_metrics_equal=True)
        artifact=wandb.Artifact('stage3a-fullbatch-confirmation-checkpoint',type='checkpoint',metadata={'stage3a_gate':result['status'],'step':result['steps']})
        artifact.add_file(str(output/'checkpoint'/receipt['file']));artifact.add_file(str(output/'checkpoint/checkpoint.json'))
        run.log_artifact(artifact)
        run.log({'gate/pass':int(result['status']=='PASS'),'clip/active_fraction':result['clip_active_fraction']},step=result['steps'])
    except BaseException as exc:
        result.update(status='ERROR',failure=type(exc).__name__+': '+str(exc));raise
    finally:
        sampler.stop()
        if run:run.finish(exit_code=0 if result['status']=='PASS' else 1)
        result.update(total_wall_seconds=time.perf_counter()-started,jax_memory_stats=device.memory_stats(),
            physical_sampled_peak_bytes=sampler.peak,host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            sample_unchanged=hashlib.sha256((ROOT/'input/sample.npz').read_bytes()).hexdigest()==manifest['sample_sha256'])
        (output/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        print('FULLBATCH CONFIRMATION',result['status'],result.get('final'),flush=True)
    return result

if __name__=='__main__':raise SystemExit(0 if main()['status']=='PASS' else 1)
