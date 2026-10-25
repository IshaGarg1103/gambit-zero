"""Resident initial-position step microbenchmark; not complete game throughput.

Repeated independent one-ply transitions avoid draw/terminal no-op inflation.
Times include synchronized host dispatch; initialization and action selection
are outside timing. Compile and warmup are reported independently.
"""
import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from tinychess.chess.pgx_env import ENV, RULES_POLICY_VERSION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("reports/pgx_batching.json"))
    args = parser.parse_args()
    if min(args.batches) < 1 or args.repeats < 1:
        parser.error("positive batch sizes and repeats required")
    rows = []
    for size in args.batches:
        states = jax.jit(jax.vmap(ENV.init))(jax.random.split(jax.random.PRNGKey(0), size))
        actions = jnp.argmax(states.legal_action_mask, axis=-1).astype(jnp.int32)
        jax.block_until_ready((states, actions))
        fn = jax.jit(jax.vmap(ENV.step))
        start = time.perf_counter()
        lowered = fn.lower(states, actions)
        trace_s = time.perf_counter() - start
        start = time.perf_counter()
        compiled = lowered.compile()
        compile_s = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(3):
            jax.block_until_ready(compiled(states, actions))
        warmup_s = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(args.repeats):
            result = jax.block_until_ready(compiled(states, actions))
        elapsed = time.perf_counter() - start
        assert bool(jnp.all(result._x.step_count == 1))
        row = dict(batch=size, trace_seconds=trace_s, compile_seconds=compile_s,
                   warmup_seconds=warmup_s, repeats=args.repeats, elapsed_seconds=elapsed,
                   steps_per_second=size * args.repeats / elapsed)
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = dict(devices=[str(d) for d in jax.devices()], jax=jax.__version__,
                  rules_policy=RULES_POLICY_VERSION, mode="resident_initial_position_independent_steps",
                  includes="step + legal mask + history + observation + host dispatch/synchronization",
                  excludes="init, action selection, network, search, trajectory resets", measurements=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
