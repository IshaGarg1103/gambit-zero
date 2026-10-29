"""Require the requested GPU and capture its actual software/host fingerprint."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess

import jax
from jax.extend import backend
import psutil

ROOT = Path(__file__).resolve().parents[1]
DATASET_REVISION = 'b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c'


def command(args):
    try:
        p = subprocess.run(args, text=True, capture_output=True, timeout=20)
        return dict(returncode=p.returncode, stdout=p.stdout.strip(), stderr=p.stderr.strip())
    except (OSError, subprocess.SubprocessError) as exc:
        return dict(returncode=None, error=str(exc))


def inspect_host(output):
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != 'gpu':
        raise RuntimeError(f'Expected one visible CUDA GPU, got {devices}')
    device = devices[0]
    if '6000' not in device.device_kind or 'Blackwell' not in device.device_kind:
        raise RuntimeError(f'Expected RTX PRO 6000 Blackwell, got {device.device_kind}')
    query = command(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version,uuid,pci.bus_id',
                     '--format=csv,noheader,nounits'])
    if query['returncode'] != 0:
        raise RuntimeError('Cannot verify physical GPU memory: '+json.dumps(query))
    physical = []
    for row in csv.reader(query['stdout'].splitlines()):
        name, total, free, driver, uuid, pci = (v.strip() for v in row)
        physical.append(dict(name=name, total_bytes=int(total)*2**20, free_bytes=int(free)*2**20,
                             driver_version=driver, uuid=uuid, pci_bus_id=pci))
    matching = [d for d in physical if '6000' in d['name'] and 'Blackwell' in d['name']]
    if not matching or min(d['total_bytes'] for d in matching) < 90*2**30:
        raise RuntimeError('Physical telemetry does not establish the expected approximately 96 GB GPU')
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()
                if d.metadata['Name'].lower().replace('_', '-').startswith(('jax', 'nvidia-'))}
    for name in ['pgx', 'mctx', 'flax', 'optax', 'pytest', 'python-chess', 'chess', 'psutil']:
        packages[name] = importlib.metadata.version(name)
    for name, expected in dict(jax='0.7.2', jaxlib='0.7.2', pgx='2.6.0', mctx='0.0.6').items():
        if importlib.metadata.version(name) != expected:
            raise RuntimeError(f'Pinned dependency mismatch: {name}')
    summary = json.loads((ROOT/'reports/m1_validation_summary.json').read_text())
    assert summary['dataset_revision'] == DATASET_REVISION
    for name, digest in summary['protected_semantics'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Protected rules/encoder/schema drift: '+name)
    report = dict(status='PASS', captured_utc=datetime.now(timezone.utc).isoformat(),
                  gpu_kind=device.device_kind, jax_device_id=device.id, physical_gpus=physical,
                  physical_mapping_note='All physical GPUs listed; CUDA visibility may remap IDs. Memory budgets use the conservative minimum.',
                  jax_memory_stats=device.memory_stats(),
                  backend_platform_version=backend.get_backend().platform_version,
                  backend_runtime_type=backend.get_backend().runtime_type,
                  nvidia_smi=command(['nvidia-smi']), system_nvcc=command(['nvcc', '--version']),
                  cuda_note='nvidia-smi CUDA is driver capability; system nvcc may differ from the locked pip CUDA runtime. All are recorded separately.',
                  packages=packages, python=platform.python_version(), platform=platform.platform(),
                  machine=platform.machine(), logical_cpus=os.cpu_count(),
                  ram_total_bytes=psutil.virtual_memory().total,
                  ram_available_bytes=psutil.virtual_memory().available,
                  dataset_revision=DATASET_REVISION,
                  source_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((ROOT/'src').rglob('*.py'))},
                  dependency_lock_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                          for name in ['uv.lock', 'requirements-cuda-lock.txt']},
                  environment={name: os.environ.get(name) for name in
                               ['CUDA_VISIBLE_DEVICES', 'JAX_PLATFORMS', 'XLA_PYTHON_CLIENT_PREALLOCATE']})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(status='PASS', gpu=report['gpu_kind'], physical_gpus=physical,
                          backend_platform_version=report['backend_platform_version'])), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    inspect_host(p.parse_args().output)
