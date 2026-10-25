"""Capture actual host metadata, source hashes, and inspected API contracts."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import jax
import mctx
import pgx
import pgx.chess
import pgx.core
import pgx.experimental.chess
import pgx._src.games.chess

from tinychess.chess.action_codec import ACTION_MAP_VERSION
from tinychess.chess.pgx_env import RULES_POLICY_VERSION


def command(args):
    if not shutil.which(args[0]):
        return "unavailable"
    p = subprocess.run(args, text=True, capture_output=True, check=False)
    return (p.stdout + p.stderr).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/stack_cpu.json"))
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    devices = jax.devices()
    if args.require_gpu and not any(d.platform == "gpu" for d in devices):
        raise SystemExit("GPU required; refusing to report CPU results as Blackwell validation")
    functions = [pgx.make("chess").init, pgx.make("chess").step, mctx.RootFnOutput,
                 mctx.RecurrentFnOutput, mctx.muzero_policy, mctx.gumbel_muzero_policy]
    modules = [pgx.core, pgx.chess, pgx.experimental.chess, pgx._src.games.chess,
               inspect.getmodule(mctx.muzero_policy)]
    disk = shutil.disk_usage(Path.cwd())
    import psutil
    report = dict(
        python=platform.python_version(), platform=platform.platform(), machine=platform.machine(),
        cpu=platform.processor(), logical_cpu_count=os.cpu_count(),
        ram_bytes=psutil.virtual_memory().total, free_disk_bytes=disk.free,
        devices=[dict(platform=d.platform, kind=d.device_kind, id=d.id) for d in devices],
        device_memory=[d.memory_stats() for d in devices],
        packages={p: importlib.metadata.version(p) for p in
                  ["jax", "jaxlib", "flax", "optax", "pgx", "mctx", "python-chess", "chess", "pytest"]},
        pgx_chess_version=pgx.make("chess").version, rules_policy=RULES_POLICY_VERSION,
        action_codec=ACTION_MAP_VERSION,
        api_signatures={f.__qualname__: str(inspect.signature(f)) for f in functions},
        source_sha256={m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in modules},
        nvidia_smi=command(["nvidia-smi"]), nvcc=command(["nvcc", "--version"]),
        compiler=command(["cc", "--version"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
