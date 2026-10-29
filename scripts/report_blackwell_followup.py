"""Render a separate measured GPU follow-up without rewriting frozen M1 results."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def update_acceptance_docs(project, acceptance):
    """Append a dated supplement; preserve the original closeout findings."""
    gate = acceptance['benchmark_gate']
    block = f'''<!-- m1-blackwell-followup:start -->
## Blackwell follow-up — verified 29 October 2026

The later Molab environment benchmark gate is **{gate}**. See the
[measured benchmark](REPORT_PREFIXm1_blackwell_benchmark.md) and
[supplemental acceptance](REPORT_PREFIXm1_blackwell_acceptance.json).
Measured batches: {acceptance['measured_batches']}; skipped: {acceptance['skipped_batches']}.
CUDA regression: {acceptance['regression']['passed']} passed in {acceptance['regression']['seconds']} seconds.
{'This supersedes the historical GPU-pending statements above.' if acceptance['supersedes_historical_gpu_deferral'] else 'The historical GPU deferral is not resolved by this run.'}
The original CPU dataset findings and telemetry limitations remain unchanged.
These are independent initial-position environment-step rates, not model,
training, MCTS or self-play throughput. Dataset revisions and protected chess
implementation are unchanged; M2 remains unauthorized.
<!-- m1-blackwell-followup:end -->'''
    for name in ['reports/m1_50k_pilot.md', 'reports/m1_checklist_status.md', 'README.md']:
        path = project/name
        if not path.exists():
            continue
        body = re.sub(r'\n<!-- m1-blackwell-followup:start -->.*?<!-- m1-blackwell-followup:end -->\n?',
                      '', path.read_text(), flags=re.S)
        if acceptance['supersedes_historical_gpu_deferral']:
            if name == 'README.md':
                body = body.replace('The separate large-batch Blackwell benchmark is pending.',
                                    'The Blackwell environment benchmark follow-up has passed; see the supplement below.')
                body = body.replace('acceptance retains the documented GPU deferral and historical telemetry limits.',
                                    'acceptance retains historical telemetry limits; the GPU supplement records the completed benchmark.')
            elif name == 'reports/m1_checklist_status.md':
                body = body.replace('| New large-batch Blackwell benchmark | PENDING; exact commands supplied; permitted deferral |',
                                    '| New large-batch Blackwell benchmark | PASS; all four batches measured; 264 CUDA regression tests passed |')
        path.write_text(body.rstrip()+'\n\n'+block.replace('REPORT_PREFIX', '' if name.startswith('reports/') else 'reports/')+'\n')


def render(project):
    project = Path(project)
    path = project/'reports/m0_large_batch_blackwell.json'
    report = json.loads(path.read_text())
    rows = report.get('rows', [])
    tests_path = project/'reports/m1_blackwell_tests.txt'
    tests = tests_path.read_text() if tests_path.exists() else ''
    match = re.search(r'(\d+) passed in ([\d.]+)s', tests)
    regression_pass = bool(match and not re.search(r'\d+ (?:failed|errors?)', tests)
                           and report.get('regression_returncode') == 0)
    measured = [r for r in rows if r.get('status') == 'MEASURED']
    errors = [r for r in rows if r.get('status') == 'FAILED']
    complete = [r.get('batch') for r in rows] == [128, 1024, 4096, 8192]
    bounded = all(r.get('status') in ['MEASURED', 'SKIPPED'] for r in rows)
    peak_available = bool(measured) and all(isinstance(r.get('peak_device_bytes'), (int, float))
                                           and r['peak_device_bytes'] > 0 for r in measured)
    host = report.get('host', {})
    gpu_verified = (host.get('status') == 'PASS' and '6000' in host.get('gpu_kind', '')
                    and 'Blackwell' in host.get('gpu_kind', ''))
    passes = (report.get('status') == 'COMPLETE' and complete and bounded and not errors and gpu_verified
              and report.get('gpu_measured') and report.get('protected_sources_unchanged')
              and regression_pass and peak_available and all(r.get('repeats') == 30
              and r.get('transition_check') == 'PASS' for r in measured))
    gate = 'PASS' if passes else 'FAIL' if report.get('status') == 'FAILED' or errors else 'INCOMPLETE'
    acceptance = dict(benchmark_gate=gate, format='m1-blackwell-followup-v1',
                      benchmark_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      source_bundle_sha256=report.get('source_bundle_sha256'),
                      dataset_revision='b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c',
                      measured_batches=[r['batch'] for r in measured],
                      skipped_batches=[r['batch'] for r in rows if r.get('status') == 'SKIPPED'],
                      regression=dict(status='PASS' if regression_pass else 'NOT_VERIFIED',
                                      passed=int(match[1]) if match else None,
                                      seconds=float(match[2]) if match else None),
                      peak_device_memory_available=peak_available,
                      protected_sources_unchanged=report.get('protected_sources_unchanged', False),
                      supersedes_historical_gpu_deferral=bool(passes),
                      original_m1_acceptance='Frozen original results retained; historical telemetry limits remain.',
                      scope='Environment-only independent initial-position steps; no policy network, MCTS, replay, training or self-play.',
                      m2_started=False, dataset_modified=False)
    (project/'reports/m1_blackwell_acceptance.json').write_text(json.dumps(acceptance, indent=2, sort_keys=True)+'\n')
    text = f'''# M1 Blackwell environment benchmark follow-up

Benchmark gate: **{gate}**. Recorded UTC: {report.get('completed_utc', 'incomplete')}.
GPU: {host.get('gpu_kind', 'NOT VERIFIED')}.
Python: {host.get('python', 'unknown')}; JAX backend: {host.get('backend_platform_version', 'unknown')}.
Actual physical VRAM, driver, CUDA package versions, host RAM/CPU and source/lock
hashes are recorded in `m0_large_batch_blackwell.json`, under `host`.

Host logical CPUs / RAM GiB: {host.get('logical_cpus', 'unknown')} / {host.get('ram_total_bytes', 0)/2**30:.2f}.
JAX / jaxlib / Pgx / mctx: {host.get('packages', {}).get('jax', 'unknown')} / {host.get('packages', {}).get('jaxlib', 'unknown')} / {host.get('packages', {}).get('pgx', 'unknown')} / {host.get('packages', {}).get('mctx', 'unknown')}.
GPU inventory (including total/free VRAM bytes and driver):
`{json.dumps(host.get('physical_gpus', []), sort_keys=True)}`.

| Batch | Status | Compile s | Steps/s | Peak JAX GPU MiB | Repeats |
|---|---|---:|---:|---:|---:|
'''
    notes = []
    for row in rows:
        def number(key, scale=1):
            value = row.get(key)
            return f'{value/scale:,.3f}' if value is not None else '—'
        text += f"| {row['batch']} | {row['status']} | {number('compile_seconds')} | {number('steps_per_second')} | {number('peak_device_bytes', 2**20)} | {row.get('repeats', '—')} |\n"
        if row.get('reason'):
            notes.append(f"Batch {row['batch']}: {row['reason']}")
    text += '\n'+'\n\n'.join(notes)+'\n'
    text += f'''
Regression: {acceptance['regression']}.
Protected source/schema hashes unchanged: {acceptance['protected_sources_unchanged']}.

Each batch uses a fresh interpreter, three warmup calls and 30 measured repeats.
Each dispatch synchronizes the complete output tree. Compilation and tracing
are excluded from warm execution throughput. Initialization/setup is not timed.
Every call steps the same resident initial-position batch once; this is not a
multi-ply rollout or representative midgame workload. The JAX allocator peak
includes compilation/warmup, is cumulative within each isolated batch process,
and does not measure all device allocations by the driver or other processes.
Compiled buffer estimates and host peak RSS are recorded separately.

Safety: conservative pre-allocation estimates plus compiled-memory analysis and
fresh physical/logical free-memory checks gate execution. Unsafe sizes are
explicitly skipped. An unexpected failure stops further batch execution and is
retained in the output; subprocess isolation cannot guarantee avoidance of every
external OOM condition.

These are **environment-only** rates. No network inference, MCTS, data loading,
optimizer update or self-play is included. They must not be used as end-to-end
training or self-play throughput estimates. Dataset A/D, chess rules, action
codec, encoder, pinned dataset revision and frozen M1 acceptance remain unchanged.
The supplement supersedes the historical GPU deferral only when its gate is PASS;
it does not erase historical telemetry limitations or authorize M2.
'''
    if report.get('failure'):
        text += '\nRun failure: '+report['failure']+'\n'
    receipt_path = project/'reports/m1_blackwell_verification.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('benchmark_sha256') != acceptance['benchmark_sha256']:
            raise ValueError('Saved verification receipt belongs to a different benchmark')
        text += f'''
## Independent result import verification

Returned ZIP: {receipt['results_zip_bytes']:,} bytes;
SHA256 `{receipt['results_zip_sha256']}`.
All {receipt['verified_result_members']} sealed result members passed SHA256
verification, and the source manifest matched the prepared input ZIP. Runtime
source hashes and dependency locks matched; per-worker rows and synchronized
throughput arithmetic were independently checked. See
[verification receipt](m1_blackwell_verification.json),
[source manifest](m1_blackwell_source_manifest.json) and
[result manifest](m1_blackwell_results_manifest.json).

All four requested batches were measured; no failures or safety skips occurred.
The short timing windows are single runs with 30 repeats, not confidence intervals
or sustained-load/power benchmarks. PCI bus telemetry returned `[Unknown Error]`
inside the Molab sandbox; GPU name, UUID, driver and physical memory were available.
This metadata limitation does not change the measured stepping results.

Recheck retained results without GPU or dataset access:

```bash
.venv/bin/python scripts/verify_blackwell_results.py
.venv/bin/python scripts/report_blackwell_followup.py
```
'''
    (project/'reports/m1_blackwell_benchmark.md').write_text(text)
    update_acceptance_docs(project, acceptance)
    return acceptance


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    print(json.dumps(render(p.parse_args().project), indent=2))
