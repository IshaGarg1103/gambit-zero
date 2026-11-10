"""Approval-gated Stage 3B orchestration; original Parquet, fresh 25M starts."""
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import shutil
import sys
import subprocess
import time

from filelock import FileLock
import jax
import jax.numpy as jnp
import numpy as np

from tinychess.data.training_loader import VerifiedStore, inventory, iter_batches, shuffle_memory_plan, REVISION
from tinychess.model.transformer import ChessTransformer, PROFILES, parameter_count
from tinychess.train.overfit_core import save_checkpoint, restore_checkpoint, fingerprint
from tinychess.train.tracking import Tracker
from tinychess.train.supervised import (PilotConfig, initial_state, padded_batch, train_step,
                                       evaluation_sums, evaluation_groups, Metrics, EarlyStop, digest, schedule)
from tinychess.validation.model_blackwell import (host_available, gpu_budget, PhysicalSampler, floating_fp32)


class BudgetStop(RuntimeError):
    pass


def check_budget(start, hours):
    if time.monotonic() - start >= hours * 3600:
        raise BudgetStop('Approved wall-time limit reached')


def preflight(config, store, shards, catalog, output):
    """Only metadata/space checks; never downloads position shards or initializes GPU."""
    selected = [s for s in shards if s.split in ('train', 'validation')]
    for split, count in [('train', config.train_positions), ('validation', config.validation_positions)]:
        if (catalog['counts'][split]['positions'] != count
                or sum(s.rows for s in selected if s.split == split) != count):
            raise ValueError('Frozen split count mismatch: ' + split)
    files = [f for s in selected for f in (s.positions, s.games)]
    missing = sum(f.size for f in files if not (store.root / f.path).exists())
    output = Path(output);output.mkdir(parents=True, exist_ok=True)
    reserve = config.disk_reserve_gib * 2**30
    # Latest + best + atomic checkpoint copies/serialization/W&B staging.
    work = 3 * 2**30
    estimate = shuffle_memory_plan(config.train_positions, config.shuffle_buffer, config.batch_size)
    available, sources = host_available()
    checks = dict(dataset_disk=shutil.disk_usage(store.root).free >= reserve + missing + work,
                  checkpoint_disk=shutil.disk_usage(output).free >= reserve + work,
                  host_ram=available >= estimate + 3*2**30)
    return dict(status='PREFLIGHT_ONLY' if all(checks.values()) else 'PREFLIGHT_BLOCKED',
                resource_checks=checks, dataset_revision=REVISION, train_rows=config.train_positions,
                validation_rows=config.validation_positions, batch_size=config.batch_size,
                microbatch=config.microbatch, accumulation_steps=config.accumulation_steps,
                shuffle_buffer=config.shuffle_buffer, estimated_shuffle_bytes=estimate,
                host_available_bytes=available, host_memory_sources=sources,
                missing_parquet_bytes=missing, selected_parquet_bytes=sum(f.size for f in files),
                disk_reserve_bytes=reserve, checkpoint_work_allowance_bytes=work,
                store_free_bytes=shutil.disk_usage(store.root).free,
                output_free_bytes=shutil.disk_usage(output).free, steps_per_epoch=config.steps_per_epoch,
                validation_batches=int(np.ceil(config.validation_positions / config.microbatch)),
                original_files_deleted=0, data_downloads_enabled=store.download)


def source_hashes(root):
    paths = ['src/tinychess/train/supervised.py','src/tinychess/train/supervised_runner.py',
             'src/tinychess/train/fullbatch_confirmation.py','src/tinychess/train/overfit_core.py',
             'src/tinychess/train/tracking.py','src/tinychess/model/transformer.py',
             'src/tinychess/model/objectives.py','src/tinychess/data/training_loader.py',
             'scripts/run_supervised_pilot.py','requirements-lock.txt','requirements-cuda-lock.txt','requirements-wandb.txt']
    return {p:hashlib.sha256((Path(root)/p).read_bytes()).hexdigest() for p in paths}


def checked_batches(store, shards, split, config, epoch=0):
    """Loader checks every row; no train/validation/test mixing and no drop_last."""
    selected = [s for s in shards if s.split == split]
    if split not in ('train', 'validation') or not selected:
        raise ValueError('Only nonempty train and validation are allowed')
    size = config.batch_size if split == 'train' else config.microbatch
    return iter_batches(store, selected, split, batch_size=size, shuffle=split == 'train',
                        shuffle_buffer=config.shuffle_buffer, seed=config.seed, epoch=epoch)


def replay_batches(batches, next_batch, check=None):
    """Deterministic recovery: re-read/re-shuffle prefix; never restore by byte offset."""
    if type(next_batch) is not int or next_batch < 0: raise ValueError('Invalid recovery cursor')
    count = 0
    for i, batch in enumerate(batches):
        count = i+1
        if check: check()
        if i >= next_batch: yield i, batch
    if next_batch > count:
        raise ValueError('Recovery cursor exceeds epoch')


def validation(state, store, shards, config, start, hours):
    all_rows, early, late = Metrics(), Metrics(), Metrics()
    rows = 0; begin = time.monotonic()
    for batch in checked_batches(store, shards, 'validation', config):
        check_budget(start, hours)
        n = len(batch.action); rows += n
        if batch.split != 'validation' or any(m != '2026-08' for m in batch.source_month):
            raise ValueError('Validation split contamination')
        padded = padded_batch(batch, config.microbatch)
        keep = jnp.pad(jnp.asarray(batch.ply < 10, jnp.float32), (0, config.microbatch-n))
        values = evaluation_groups(state, padded, keep);jax.block_until_ready(values)
        for bucket, value in zip((all_rows, early, late), values): bucket.add(value)
    if rows != config.validation_positions: raise ValueError('Incomplete validation population')
    return dict(**all_rows.result(), rows=rows, seconds=time.monotonic()-begin,
                ply_lt10=early.result() if early.sums.get('policy_weight', 0) else None,
                ply_ge10=late.result() if late.sums.get('policy_weight', 0) else None)


def run(args, root):
    cfg = PilotConfig.load(args.config)
    if args.mode == 'sweep':
        if args.lr_index is None or args.lr is not None: raise ValueError('Sweep requires exactly --lr-index')
        lr = cfg.learning_rates[args.lr_index]; epochs = cfg.sweep_epochs; hours = cfg.sweep_gpu_hours
    else:
        if args.lr not in cfg.learning_rates or args.lr_index is not None:
            raise ValueError('Pilot requires an explicitly selected reviewed --lr')
        lr = args.lr; epochs = cfg.pilot_epochs; hours = cfg.pilot_gpu_hours
    output = Path(args.output);output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / 'worker.lock'), timeout=1):
        store = VerifiedStore(args.store, download=args.download, keep_downloads=True,
                              reserve_bytes=cfg.disk_reserve_gib*2**30)
        shards, catalog = inventory(store)
        prep = preflight(cfg, store, shards, catalog, output)
        (output/'preflight.json').write_text(json.dumps(prep, indent=2, sort_keys=True)+'\n')
        if not args.execute_approved or prep['status'] == 'PREFLIGHT_BLOCKED': return prep
        return _execute(args, root, cfg, lr, epochs, hours, output, store, shards, prep)


def hardware_metadata():
    result = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True,
                            text=True, timeout=10)
    expected = dict(jax='0.7.2',jaxlib='0.7.2',flax='0.12.0',optax='0.2.6',pgx='2.6.0',mctx='0.0.6')
    versions = {name:importlib.metadata.version(name) for name in expected}
    if versions != expected: raise RuntimeError('Installed core dependencies do not match locked versions')
    return dict(nvidia_smi=result.stdout.strip(), packages=versions)


def timed_batches(batches, result):
    iterator = iter(batches)
    while True:
        t = time.monotonic()
        try: batch = next(iterator)
        except StopIteration:
            result['loader_wait_seconds'] += time.monotonic()-t
            return
        result['loader_wait_seconds'] += time.monotonic()-t
        yield batch


def blackwell_device():
    device = jax.devices()[0]
    if device.platform != 'gpu' or 'Blackwell' not in device.device_kind:
        raise RuntimeError('Approved execution requires the Blackwell GPU host')
    return device


def _execute(args, root, cfg, lr, epochs, hours, output, store, shards, prep):
    device = blackwell_device()
    if args.wandb_mode == 'disabled': raise ValueError('Approved experiments require online/offline W&B')
    import os
    if args.wandb_mode == 'online' and not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError('WANDB_API_KEY must be supplied via environment')
    metadata = dict(stage='m2-stage3b',dataset_revision=REVISION,
                    recipe_hash=digest(dict(config=asdict(cfg), lr=lr, mode=args.mode)),
                    source_hashes=source_hashes(root), model='25m', parameters=25622092,
                    learning_rate=lr, mode=args.mode, seed=cfg.seed)
    if args.resume and not (output/'latest/checkpoint.json').is_file():
        raise ValueError('Resume requires a Stage 3B latest checkpoint')
    if not args.resume:
        if any((output/name).exists() for name in ('latest','best','result.json')):
            raise ValueError('Existing experiment output; use a fresh directory or explicit resume')
        # Guard remains on interruption; an explicit resume uses a checked checkpoint.
        with (output/'run_guard.json').open('x') as f: json.dump(metadata, f)
    start = time.monotonic(); tracker = None; state = None
    early = EarlyStop(); cursor = dict(epoch=0, next_batch=0); run_id = None
    result = dict(status='RUNNING', metadata=metadata, config=asdict(cfg), lr=lr,
                  mode=args.mode, epochs=[], preflight=prep, compilation_seconds=0., loader_wait_seconds=0.,
                  synchronized_update_seconds=0., positions_applied_this_session=0,
                  clip_active_this_session=0, updates_this_session=0,
                  environment=dict(python=platform.python_version(), platform=platform.platform(),
                                   jax=jax.__version__, jaxlib=__import__('jaxlib').__version__,
                                   gpu=device.device_kind), measured_new_session=True)
    session_start = start; checkpoint_allowed = not args.resume
    sampler = PhysicalSampler();sampler.start()
    def checkpoint(kind):
        if state is None or not checkpoint_allowed: return None
        return save_checkpoint(output/kind, state, dict(metadata, cursor=dict(cursor),
            early_best=early.best if np.isfinite(early.best) else None, early_stale=early.stale, elapsed_wall_seconds=time.monotonic()-start,
            epoch_history=result['epochs'], wandb_run_id=tracker.id if tracker else run_id))
    try:
        result['hardware'] = hardware_metadata()
        model = ChessTransformer(replace(PROFILES['25m'], compute=cfg.compute))
        initial = dict(planes=jnp.zeros((cfg.microbatch,8,8,120),jnp.float32),
                       context=jnp.zeros((cfg.microbatch,8),jnp.float32))
        init_start = time.monotonic()
        state = jax.jit(lambda: initial_state(model, initial, cfg, lr, epochs))()
        jax.block_until_ready(state)
        result['initialization_seconds'] = time.monotonic()-init_start
        if parameter_count(state.params) != 25622092 or not floating_fp32(state.params) or not floating_fp32(state.opt_state):
            raise RuntimeError('Parameter count/precision drift')
        result['fresh_initial_parameter_sha256'] = fingerprint(state.params)
        if args.resume:
            state, receipt = restore_checkpoint(output/'latest', state, metadata)
            cursor = receipt['cursor']; early = EarlyStop(receipt['early_best'] if receipt['early_best'] is not None else float('inf'), receipt['early_stale'])
            run_id = receipt['wandb_run_id']
            if (type(cursor.get('epoch')) is not int or not 0 <= cursor['epoch'] <= epochs
                    or not 0 <= cursor['next_batch'] <= cfg.steps_per_epoch
                    or int(state.step) != cursor['epoch']*cfg.steps_per_epoch+cursor['next_batch']):
                raise ValueError('Checkpoint cursor/optimizer step mismatch')
            if cursor['epoch'] >= epochs or early.stale >= cfg.patience_epochs:
                raise ValueError('Run already finished; no resume needed')
            start -= receipt['elapsed_wall_seconds']
            result['epochs'] = receipt['epoch_history']
            checkpoint_allowed = True
            check_budget(start, hours)
        tracker = Tracker(dict(metadata, config=asdict(cfg), lr=lr), output/'wandb', mode=args.wandb_mode,
                          run_id=run_id, group='m2-lr-sweep' if args.mode == 'sweep' else 'm2-supervised',
                          name=f'stage3b-25m-{args.mode}-lr{lr:g}',job_type='supervised-pilot',
                          tags=['stage3b','25m','history-dropout-0'],
                          artifact_name=f'stage3b-25m-{args.mode}-lr{lr:g}-checkpoint')
        result['wandb_url'] = tracker.url
        compiled = None; stop = False
        while cursor['epoch'] < epochs and not stop:
            epoch = cursor['epoch']; epoch_rows = cursor['next_batch']*cfg.batch_size
            epoch_rows = min(epoch_rows, cfg.train_positions)
            iterator = timed_batches(checked_batches(store, shards, 'train', cfg, epoch), result)
            batch_count = cursor['next_batch']
            for i, batch in replay_batches(iterator, cursor['next_batch'], lambda: check_budget(start, hours)):
                check_budget(start, hours)
                if batch.split != 'train' or any(m not in {f'2026-0{k}' for k in range(1,8)} for m in batch.source_month):
                    raise ValueError('Training split contamination')
                n = len(batch.action); padded = padded_batch(batch, cfg.batch_size)
                if compiled is None:
                    t = time.monotonic(); compiled = train_step.lower(state, padded, microbatch=cfg.microbatch).compile()
                    result['compilation_seconds'] += time.monotonic()-t
                    mem = compiled.memory_analysis()
                    if mem is None: raise RuntimeError('Missing compiled memory analysis')
                    buffers = mem.argument_size_in_bytes+mem.output_size_in_bytes+mem.temp_size_in_bytes
                    if 2*buffers+(device.memory_stats() or {}).get('bytes_in_use',0)+512*2**20 > gpu_budget(device)[0]:
                        raise RuntimeError('Unsafe compiled batch memory; no update applied')
                    result['compiled_buffer_bytes'] = buffers
                    if int(state.step) == 0:
                        initial_metrics = Metrics();initial_metrics.add(evaluation_sums(state,padded))
                        initial_values = initial_metrics.result()
                        pw = np.asarray(padded['policy_weight'])
                        expected = float(np.sum(np.log(np.asarray(padded['legal_mask']).sum(-1))*pw)/pw.sum())
                        if not np.isclose(initial_values['policy_loss'],expected,atol=1e-4):
                            raise ValueError('Initial policy loss does not match uniform legal policy')
                        if initial_values['wdl_loss'] is not None and not np.isclose(initial_values['wdl_loss'],np.log(3),atol=1e-4):
                            raise ValueError('Initial WDL loss does not match ln(3)')
                        result['initial_batch_metrics'] = dict(initial_values,expected_policy_loss=expected)
                        tracker.log({'initial/'+k:v for k,v in initial_values.items() if v is not None},0)
                check_budget(start, hours)
                t = time.monotonic(); state, values = compiled(state, padded);jax.block_until_ready((state,values))
                elapsed = time.monotonic()-t; values = jax.device_get(values)
                if not bool(values['finite']): raise FloatingPointError('Rejected nonfinite update')
                cursor = dict(epoch=epoch,next_batch=i+1); epoch_rows += n; batch_count = i+1
                result['synchronized_update_seconds'] += elapsed
                result['positions_applied_this_session'] += n; result['updates_this_session'] += 1
                result['clip_active_this_session'] += int(float(values['grad_norm']) > 1.)
                step = int(state.step)
                if step % cfg.log_every_steps == 0:
                    metrics = Metrics();metrics.add(evaluation_sums(state,padded))
                    tracker.log(dict(**{'train_batch/'+k:v for k,v in metrics.result().items() if v is not None},
                        grad_norm=float(values['grad_norm']), post_clip_norm=float(values['post_clip_norm']),
                        learning_rate=float(schedule(step-1,lr,cfg.warmup_steps,cfg.steps_per_epoch*epochs,cfg.min_lr_ratio)),
                        distinct_games=len(set(batch.game_id)), distinct_months=len(set(batch.source_month)),
                        max_positions_one_game=batch.max_positions_per_game, update_seconds=elapsed),step)
                    print(json.dumps(dict(step=step,epoch=epoch,rows=epoch_rows,loss=float(values['loss']))),flush=True)
                if step % cfg.checkpoint_every_steps == 0: checkpoint('latest')
            if epoch_rows != cfg.train_positions or batch_count != cfg.steps_per_epoch:
                raise ValueError('Incomplete eligible training epoch')
            val = validation(state,store,shards,cfg,start,hours)
            improved, stop = early.observe(val['policy_loss'],patience=cfg.patience_epochs,min_delta=cfg.min_delta)
            result['epochs'].append(dict(epoch=epoch,validation=val,improved=improved))
            logged = {'validation/'+k:v for k,v in val.items() if isinstance(v,(int,float))}
            for group in ('ply_lt10','ply_ge10'):
                if val[group]: logged.update({'validation/'+group+'/'+k:v for k,v in val[group].items() if v is not None})
            tracker.log(logged,int(state.step))
            cursor = dict(epoch=epoch+1,next_batch=0)
            if improved:
                receipt = checkpoint('best'); tracker.checkpoint(output/'best', receipt)
            checkpoint('latest')
            check_budget(start,hours)
        result['status'] = 'EARLY_STOPPED' if stop else 'COMPLETED'
    except BudgetStop:
        result['status'] = 'BUDGET_STOP';checkpoint('latest')
    except BaseException as exc:
        result['status'] = 'ERROR'; result['error_type'] = type(exc).__name__
        # Deliberately exclude exception bodies/HTTP URLs and credentials from evidence.
        if state is not None: checkpoint('latest')
        raise
    finally:
        sampler.stop()
        if tracker:
            try: tracker.finish(result['status'] in ('COMPLETED','EARLY_STOPPED'))
            except Exception as exc:
                result['tracking_finish_error_type'] = type(exc).__name__
                result['status'] = 'TRACKING_ERROR'
        result.update(wall_seconds=time.monotonic()-start, session_wall_seconds=time.monotonic()-session_start, final_step=int(state.step) if state is not None else 0,
                      cursor=cursor, physical_sampled_peak_bytes=sampler.peak,
                      physical_samples=sampler.samples, physical_telemetry_errors=sampler.errors,
                      jax_memory_stats=device.memory_stats(),
                      host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                      downloaded_bytes_this_session=store.downloaded_bytes,
                      synchronized_positions_per_second=(result['positions_applied_this_session']/result['synchronized_update_seconds']
                          if result['synchronized_update_seconds'] else None),
                      end_to_end_positions_per_second=result['positions_applied_this_session']/max(time.monotonic()-session_start,1e-9))
        (output/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result
