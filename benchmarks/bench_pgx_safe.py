"""Memory-budgeted independent-step benchmark; never attempt an unsafe batch."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

# Set before importing JAX/Pgx, including when launched outside the notebook.
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
import psutil

from tinychess.chess.pgx_env import ENV, RULES_POLICY_VERSION


def memory_budget(available, gpu, budget_mib=None):
    default = min(max(0, available) * (0.35 if gpu else 0.2), (16 if gpu else 2) * 2**30)
    return int(default if budget_mib is None else min(default, budget_mib * 2**20))


def physical_gpu_free():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                text=True, capture_output=True, check=True, timeout=10)
        # Conservative across devices, including CUDA visibility remapping.
        return min(int(line.strip()) * 2**20 for line in result.stdout.splitlines() if line.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def save_report(output, report):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix+'.partial')
    temporary.write_text(json.dumps(report, indent=2)+'\n')
    temporary.replace(output)


def benchmark(output, batches=(128, 1024, 4096, 8192), repeats=30, budget_mib=None, require_gpu=False):
    if not batches or min(batches) <= 0 or repeats <= 0 or (budget_mib is not None and budget_mib <= 0):
        raise ValueError('positive counts/budget required')
    device = jax.devices()[0]
    if require_gpu and device.platform != 'gpu':
        raise RuntimeError('GPU required; CPU results cannot satisfy the Blackwell gate')
    gpu = device.platform == 'gpu'
    stats = device.memory_stats() or {}
    available = max(0, stats.get('bytes_limit', 0) - stats.get('bytes_in_use', 0)) if gpu else psutil.virtual_memory().available
    physical_free = physical_gpu_free() if gpu else None
    if gpu:
        available = min(available, physical_free)
    budget = memory_budget(available, gpu, budget_mib)
    rows = []
    report = dict(format='pgx-safe-benchmark-v2', rules_policy=RULES_POLICY_VERSION,
                  devices=[dict(kind=d.device_kind, platform=d.platform, id=d.id) for d in jax.devices()],
                  jax_version=jax.__version__, repeats_requested=repeats,
                  mode='independent resident initial-state steps with synchronized dispatch',
                  memory_budget_bytes=budget, physical_gpu_free_bytes=physical_free,
                  rows=rows, gpu_measured=False, status='RUNNING')
    save_report(output, report)
    try:
        single = jax.jit(ENV.init)(jax.random.PRNGKey(0))
        jax.block_until_ready(single)
    except Exception as exc:
        report.update(status='FAILED', initialization_error=dict(type=type(exc).__name__, message=str(exc)))
        save_report(output, report)
        raise
    for size in batches:
        row = dict(batch=size, memory_budget_bytes=budget, repeats=repeats)
        rows.append(row)
        compiled = lowered = fn = None
        try:
            # Conservative preflight BEFORE large allocation or compilation.
            estimate = size * 2**20
            if estimate > budget * 0.8:
                row.update(status='SKIPPED', reason='conservative 1 MiB/state preflight exceeds 80% of memory budget',
                           preflight_bytes=estimate)
                continue
            shapes = jax.tree.map(lambda x: jax.ShapeDtypeStruct((size,) + x.shape, x.dtype), single)
            action_shape = jax.ShapeDtypeStruct((size,), jnp.int32)
            fn = jax.jit(jax.vmap(ENV.step))
            started = time.perf_counter()
            lowered = fn.lower(shapes, action_shape)
            row['trace_seconds'] = time.perf_counter()-started
            started = time.perf_counter()
            compiled = lowered.compile()
            row['compile_seconds'] = time.perf_counter()-started
            analysis = compiled.memory_analysis()
            if analysis is None:
                row.update(status='SKIPPED', reason='compiled memory analysis unavailable; no unsafe execution attempted')
                continue
            compiled_bytes = analysis.argument_size_in_bytes + analysis.output_size_in_bytes + analysis.temp_size_in_bytes
            required = compiled_bytes * 2 + 256 * 2**20
            row.update(compiled_buffer_bytes=compiled_bytes, required_bytes=required)
            if gpu:
                # Capacity may have changed during compilation; never rely only on startup telemetry.
                row['physical_gpu_free_before_execution_bytes'] = physical_gpu_free()
                current = device.memory_stats() or {}
                logical_free = max(0, current.get('bytes_limit', 0)-current.get('bytes_in_use', 0))
                row['execution_budget_bytes'] = memory_budget(
                    min(row['physical_gpu_free_before_execution_bytes'], logical_free), True, budget_mib)
            else:
                row['execution_budget_bytes'] = memory_budget(psutil.virtual_memory().available, False, budget_mib)
            if required > min(budget, row['execution_budget_bytes']):
                row.update(status='SKIPPED', reason='2x compiled buffers plus 256 MiB reserve exceeds current budget')
                continue
            states = jax.tree.map(lambda x: jnp.broadcast_to(x, (size,) + x.shape), single)
            actions = jnp.argmax(states.legal_action_mask, axis=-1).astype(jnp.int32)
            jax.block_until_ready((states, actions))
            started = time.perf_counter()
            for _ in range(3):
                jax.block_until_ready(compiled(states, actions))
            row['warmup_seconds'] = time.perf_counter()-started
            started = time.perf_counter()
            for _ in range(repeats):
                child = jax.block_until_ready(compiled(states, actions))
            elapsed = time.perf_counter()-started
            # This is deliberately repeated independent first-ply stepping, not a trajectory rollout.
            if not bool(jnp.all(child._x.step_count == 1)):
                raise RuntimeError('Benchmark transition correctness failed')
            if bool(jnp.any(child.terminated | child.truncated)):
                raise RuntimeError('Initial legal transitions unexpectedly ended the game')
            current = device.memory_stats() or {}
            row.update(status='MEASURED', warm_seconds=elapsed, steps_per_second=size*repeats/elapsed,
                       transition_check='PASS', peak_device_bytes=current.get('peak_bytes_in_use'),
                       device_memory_note='JAX allocator process cumulative high-water mark, including compilation and warmup; not total physical VRAM' if gpu else 'CPU backend: device peak unavailable',
                       peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024))
            del states, actions, child
        except Exception as exc:
            row.update(status='FAILED', error_type=type(exc).__name__, reason=str(exc))
        finally:
            compiled = lowered = fn = None
            report['gpu_measured'] = gpu and any(r.get('status') == 'MEASURED' for r in rows)
            save_report(output, report)
            print(json.dumps(row), flush=True)
        if row['status'] == 'FAILED':
            for remaining in batches[len(rows):]:
                rows.append(dict(batch=remaining, status='SKIPPED', reason='stopped after earlier batch failure'))
            break
    report['status'] = 'FAILED' if any(r['status'] == 'FAILED' for r in rows) else 'COMPLETE'
    save_report(output, report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('reports/m0_large_batch_cpu.json'))
    p.add_argument('--batches', nargs='+', type=int, default=[128, 1024, 4096, 8192])
    p.add_argument('--repeats', type=int, default=30)
    p.add_argument('--budget-mib', type=int)
    p.add_argument('--require-gpu', action='store_true')
    a = p.parse_args()
    if min(a.batches) <= 0 or a.repeats <= 0 or (a.budget_mib is not None and a.budget_mib <= 0):
        p.error('positive counts/budget required')
    result = benchmark(a.output, tuple(a.batches), a.repeats, a.budget_mib, a.require_gpu)
    if result['status'] == 'FAILED':
        raise SystemExit(1)
