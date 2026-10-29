"""Isolated Molab GPU benchmark runner; no Git or dataset service operations.

Execute as a script, not inside the notebook kernel. Always produces a results
ZIP, including failure diagnostics. Input must be the prepared checksummed bundle.
"""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BATCHES = [128, 1024, 4096, 8192]


def verify_bundle(root):
    root = Path(root)
    manifest = json.loads((root/'BLACKWELL_BUNDLE.json').read_text())
    for name, digest in manifest['files'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise RuntimeError('Missing or unsafe bundle member: '+name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Input bundle file checksum mismatch: '+name)
    return manifest


def source_hashes(root):
    paths = list((root/'src').rglob('*.py'))
    paths += [root/'configs/human_v3_schema.json', root/'reports/m1_final_acceptance.json',
              root/'reports/m1_dataset_d_freeze.json', root/'configs/dataset_d.yaml']
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.exists()}


def run_command(command, env, log_name, check=True):
    print('\nRUN:', ' '.join(map(str, command)), flush=True)
    log_path = ROOT/'reports/blackwell_followup'/log_name
    with log_path.open('w') as log:
        process = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if check and code:
        raise RuntimeError(f'Command exited {code}; see {log_path}')
    return code


def package_results(root, result):
    destination = root.parent/'gambit-zero-m1-blackwell-results.zip'
    paths = [p for p in (root/'reports/blackwell_followup').rglob('*') if p.is_file()]
    paths += [root/'reports'/name for name in ['m0_large_batch_blackwell.json',
              'm1_blackwell_benchmark.md', 'm1_blackwell_acceptance.json', 'm1_blackwell_tests.txt']
              if (root/'reports'/name).exists()]
    paths += [root/'BLACKWELL_BUNDLE.json']
    index = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(paths):
            bundle.write(path, path.relative_to(root).as_posix())
        bundle.writestr('RESULTS_MANIFEST.json', json.dumps(dict(status=result['status'], files=index), indent=2)+'\n')
    print(f'\nRESULTS ZIP: {destination}', flush=True)
    print('ZIP SHA256:', hashlib.sha256(destination.read_bytes()).hexdigest(), flush=True)
    return destination


def main():
    reports = ROOT/'reports'
    work = reports/'blackwell_followup'
    work.mkdir(parents=True, exist_ok=True)
    if (reports/'m0_large_batch_blackwell.json').exists():
        raise RuntimeError('Use a fresh extraction directory; refusing to mix previous results')
    result = dict(format='m1-blackwell-progressive-v1', status='RUNNING', rows=[],
                  batches_requested=BATCHES, repeats_requested=30, gpu_measured=False,
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  mode='independent resident initial-state steps with synchronized dispatch',
                  dataset_revision='b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c')
    output = reports/'m0_large_batch_blackwell.json'
    before = source_hashes(ROOT)
    env = dict(os.environ)
    for name in ['PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'JAX_PLATFORM_NAME']:
        env.pop(name, None)
    env.update(JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false',
               CUDA_VISIBLE_DEVICES=env.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0],
               UV_CACHE_DIR=str(ROOT/'.uv-cache'), UV_PYTHON_INSTALL_DIR=str(ROOT/'.uv-python'),
               PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TINYCHESS_EVIDENCE_OFFLINE='1')
    try:
        manifest = verify_bundle(ROOT)
        result['source_manifest_sha256'] = hashlib.sha256((ROOT/'BLACKWELL_BUNDLE.json').read_bytes()).hexdigest()
        result['source_bundle_sha256'] = env.get('BLACKWELL_INPUT_ZIP_SHA256')
        result['source_base_commit'] = manifest['source_base_commit']
        result['source_tag'] = manifest['source_tag']
        run_command(['nvidia-smi'], env, 'nvidia-smi.log')
        try:
            uv_version = importlib.metadata.version('uv')
        except importlib.metadata.PackageNotFoundError:
            uv_version = None
        if uv_version != '0.10.9':
            run_command([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'uv==0.10.9'], env, 'uv-install.log')
        run_command([sys.executable, '-m', 'uv', 'sync', '--project', str(ROOT), '--locked',
                     '--python', '3.11.15', '--extra', 'test', '--extra', 'data', '--extra', 'cuda'],
                    env, 'dependency-install.log')
        python = ROOT/'.venv/bin/python'
        run_command([python, 'scripts/inspect_stack.py', '--require-gpu', '--output', str(work/'stack.json')], env, 'stack.log')
        run_command([python, 'scripts/inspect_blackwell.py', '--output', str(work/'host.json')], env, 'host.log')
        result['host'] = json.loads((work/'host.json').read_text())
        result['rules_policy'] = json.loads((work/'stack.json').read_text())['rules_policy']
        for batch in BATCHES:
            row_path = work/f'batch-{batch}.json'
            code = run_command([python, 'benchmarks/bench_pgx_safe.py', '--require-gpu',
                                '--batches', str(batch), '--repeats', '30', '--output', str(row_path)],
                               env, f'batch-{batch}.log', check=False)
            partial = json.loads(row_path.read_text()) if row_path.exists() else {}
            rows = partial.get('rows', [])
            row = rows[0] if rows else dict(batch=batch, status='FAILED', reason='worker exited without a batch result')
            if code or partial.get('status') == 'FAILED':
                row.update(status='FAILED', returncode=code,
                           reason=row.get('reason', 'worker failed; inspect its saved log'))
            row['worker_process_isolated'] = True
            result['rows'].append(row)
            output.write_text(json.dumps(result, indent=2)+'\n')
            if row['status'] == 'FAILED':
                for remaining in BATCHES[len(result['rows']):]:
                    result['rows'].append(dict(batch=remaining, status='SKIPPED', reason='stopped after earlier worker failure'))
                raise RuntimeError(f'Batch {batch} failed; larger batches were not attempted')
        # One fresh CUDA process runs the complete suite, including rules/history,
        # action codec, encoder, mctx value signs and both-color mate checks.
        code = run_command([python, '-m', 'pytest', '-q'], env, 'regression.log', check=False)
        (reports/'m1_blackwell_tests.txt').write_bytes((work/'regression.log').read_bytes())
        result['regression_returncode'] = code
        if code:
            raise RuntimeError('GPU regression failed')
        result['status'] = 'COMPLETE'
    except BaseException as exc:
        result.update(status='FAILED', failure=f'{type(exc).__name__}: {exc}')
        (work/'failure.log').write_text(traceback.format_exc())
        print(result['failure'], flush=True)
    finally:
        result['protected_sources_unchanged'] = before == source_hashes(ROOT)
        if not result['protected_sources_unchanged']:
            result.update(status='FAILED', failure='Protected source/configuration drift during the run')
        result['gpu_measured'] = any(row.get('status') == 'MEASURED' for row in result['rows'])
        result['completed_utc'] = datetime.now(timezone.utc).isoformat()
        output.write_text(json.dumps(result, indent=2)+'\n')
        # Standard-library renderer also works if CUDA setup itself failed.
        import runpy
        acceptance = runpy.run_path(str(ROOT/'scripts/report_blackwell_followup.py'))['render'](ROOT)
        package_results(ROOT, result)
        print('BENCHMARK GATE:', acceptance['benchmark_gate'], flush=True)
    return 0 if acceptance['benchmark_gate'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
