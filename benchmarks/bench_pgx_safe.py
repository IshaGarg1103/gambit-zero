"""Memory-budgeted independent-step benchmark; never attempt an unsafe batch."""
import argparse
import os
import subprocess
import json
from pathlib import Path
import resource
import sys
import time

# Disable large default GPU reservations before importing JAX/Pgx.
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
import psutil

from tinychess.chess.pgx_env import ENV, RULES_POLICY_VERSION


def benchmark(output, batches=(128, 1024, 4096, 8192), repeats=30, budget_mib=None):
    device = jax.devices()[0]
    stats = device.memory_stats() or {}
    available = (stats.get('bytes_limit', 0) - stats.get('bytes_in_use', 0)) if device.platform == 'gpu' else psutil.virtual_memory().available
    physical_free = None
    if device.platform == 'gpu':
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                    text=True, capture_output=True, check=True, timeout=10)
            # A minimum across devices is conservative even if CUDA visibility
            # remaps the selected GPU. Do not guess unsupported device APIs.
            physical_free = min(int(line.strip()) * 2**20 for line in result.stdout.splitlines() if line.strip())
            available = min(available, physical_free)
        except (OSError, ValueError, subprocess.SubprocessError):
            available = 0  # Explicitly skip if current physical capacity is unknown.
    default_budget = min(available * 0.35, 16 * 2**30) if device.platform == 'gpu' else min(available * 0.2, 2 * 2**30)
    budget = int(default_budget if budget_mib is None else min(default_budget, budget_mib * 2**20))
    single = jax.jit(ENV.init)(jax.random.PRNGKey(0))
    jax.block_until_ready(single)
    rows = []
    for size in batches:
        row = dict(batch=size, memory_budget_bytes=budget)
        # Conservative preflight BEFORE large array allocation or compilation.
        estimate = size * 2**20
        if estimate > budget * 0.8:
            row.update(status='SKIPPED', reason='conservative 1 MiB/state preflight exceeds 80% of memory budget',
                       preflight_bytes=estimate)
            rows.append(row)
            print(json.dumps(row), flush=True)
            continue
        shapes = jax.tree.map(lambda x: jax.ShapeDtypeStruct((size,) + x.shape, x.dtype), single)
        action_shape = jax.ShapeDtypeStruct((size,), jnp.int32)
        fn = jax.jit(jax.vmap(ENV.step))
        started = time.perf_counter()
        lowered = fn.lower(shapes, action_shape)
        trace_s = time.perf_counter()-started
        started = time.perf_counter()
        compiled = lowered.compile()
        compile_s = time.perf_counter()-started
        analysis = compiled.memory_analysis()
        if analysis is None:
            row.update(status='SKIPPED', reason='compiled memory analysis unavailable; no unsafe execution attempted')
            rows.append(row)
            continue
        compiled_bytes = (analysis.argument_size_in_bytes + analysis.output_size_in_bytes + analysis.temp_size_in_bytes)
        required = compiled_bytes * 2 + 256 * 2**20
        if required > budget:
            row.update(status='SKIPPED', reason='2x compiled buffers plus 256 MiB reserve exceeds budget',
                       compiled_bytes=compiled_bytes, required_bytes=required,
                       trace_seconds=trace_s, compile_seconds=compile_s)
            rows.append(row)
            print(json.dumps(row), flush=True)
            continue
        states = jax.tree.map(lambda x: jnp.broadcast_to(x, (size,) + x.shape), single)
        actions = jnp.argmax(states.legal_action_mask, axis=-1).astype(jnp.int32)
        jax.block_until_ready((states, actions))
        started = time.perf_counter()
        for _ in range(3):
            jax.block_until_ready(compiled(states, actions))
        warmup = time.perf_counter()-started
        started = time.perf_counter()
        for _ in range(repeats):
            child = jax.block_until_ready(compiled(states, actions))
        elapsed = time.perf_counter()-started
        assert bool(jnp.all(child._x.step_count == 1))
        current = device.memory_stats() or {}
        row.update(status='MEASURED', trace_seconds=trace_s, compile_seconds=compile_s,
                   warmup_seconds=warmup, repeats=repeats, warm_seconds=elapsed,
                   steps_per_second=size*repeats/elapsed, compiled_buffer_bytes=compiled_bytes,
                   peak_device_bytes=current.get('peak_bytes_in_use'),
                   device_memory_note='process cumulative high-water mark; not per-batch allocation delta' if current else 'CPU backend: device peak unavailable',
                   peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024))
        rows.append(row)
        print(json.dumps(row), flush=True)
        del states, actions, child, compiled
    report = dict(rules_policy=RULES_POLICY_VERSION, devices=[str(d) for d in jax.devices()],
                  mode='independent resident initial-state steps with synchronized dispatch',
                  memory_budget_bytes=budget, physical_gpu_free_bytes=physical_free, rows=rows,
                  gpu_measured=device.platform == 'gpu' and any(r['status']=='MEASURED' for r in rows))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('reports/m0_large_batch_cpu.json'))
    p.add_argument('--batches', nargs='+', type=int, default=[128, 1024, 4096, 8192])
    p.add_argument('--repeats', type=int, default=30)
    p.add_argument('--budget-mib', type=int)
    a = p.parse_args()
    if min(a.batches) <= 0 or a.repeats <= 0 or (a.budget_mib is not None and a.budget_mib <= 0):
        p.error('positive counts/budget required')
    benchmark(a.output, tuple(a.batches), a.repeats, a.budget_mib)
