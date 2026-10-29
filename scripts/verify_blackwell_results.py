"""Verify imported Molab GPU evidence, without GPU execution or dataset access."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def verify(project, archive=None):
    project = Path(project)
    reports = project/'reports'
    receipt = json.loads((reports/'m1_blackwell_verification.json').read_text())
    manifest = json.loads((reports/'m1_blackwell_results_manifest.json').read_text())
    source_bytes = (reports/'m1_blackwell_source_manifest.json').read_bytes()
    source = json.loads(source_bytes)
    for name in ['reports/m0_large_batch_blackwell.json', 'reports/m1_blackwell_acceptance.json',
                 'reports/m1_blackwell_tests.txt']:
        assert sha((project/name).read_bytes()) == manifest['files'][name], name
    assert sha(source_bytes) == manifest['files']['BLACKWELL_BUNDLE.json']
    if archive:
        payload = Path(archive).read_bytes()
        assert len(payload) == receipt['results_zip_bytes']
        assert sha(payload) == receipt['results_zip_sha256']
        with zipfile.ZipFile(archive) as bundle:
            assert len(bundle.namelist()) == len(manifest['files'])+1
            assert json.loads(bundle.read('RESULTS_MANIFEST.json')) == manifest
            for name, expected in manifest['files'].items():
                assert sha(bundle.read(name)) == expected, name
    report_bytes = (reports/'m0_large_batch_blackwell.json').read_bytes()
    report = json.loads(report_bytes)
    acceptance = json.loads((reports/'m1_blackwell_acceptance.json').read_text())
    assert sha(report_bytes) == acceptance['benchmark_sha256']
    assert sha(report_bytes) == receipt['benchmark_sha256']
    assert sha(source_bytes) == report['source_manifest_sha256']
    assert report['source_bundle_sha256'] == receipt['input_zip_sha256']
    assert report['status'] == 'COMPLETE' and report['gpu_measured']
    assert acceptance['benchmark_gate'] == 'PASS' and report['regression_returncode'] == 0
    assert report['host']['status'] == 'PASS'
    assert '6000' in report['host']['gpu_kind'] and 'Blackwell' in report['host']['gpu_kind']
    assert report['protected_sources_unchanged']
    assert report['dataset_revision'] == acceptance['dataset_revision'] == 'b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c'
    for name, expected in report['host']['source_sha256'].items():
        assert sha((project/name).read_bytes()) == expected, name
    # Documentation/rendering helpers can evolve after measurement; actual
    # benchmark/environment code and locks must remain the measured versions.
    for name in ['benchmarks/bench_pgx_safe.py', 'uv.lock', 'requirements-cuda-lock.txt']:
        assert sha((project/name).read_bytes()) == source['files'][name], name
    summary = json.loads((reports/'m1_validation_summary.json').read_text())
    for name, expected in summary['protected_semantics'].items():
        assert sha((project/name).read_bytes()) == expected, name
    assert [r['batch'] for r in report['rows']] == [128, 1024, 4096, 8192]
    for row in report['rows']:
        assert row['status'] == 'MEASURED' and row['repeats'] == 30 and row['transition_check'] == 'PASS'
        assert row['worker_process_isolated'] and row['peak_device_bytes'] > 0
        assert abs(row['steps_per_second']-row['batch']*30/row['warm_seconds']) < 1e-6
        assert row['required_bytes'] <= row['memory_budget_bytes']
    tests = (reports/'m1_blackwell_tests.txt').read_text()
    assert re.search(r'264 passed in 383\.84s', tests) and not re.search(r'\d+ (failed|errors?)', tests)
    return dict(status='PASS', gpu=report['host']['gpu_kind'], measured_batches=[r['batch'] for r in report['rows']],
                gpu_tests_passed=264, input_and_result_manifests_verified=True,
                raw_zip_verified=archive is not None, measured_source_and_locks_unchanged=True,
                dataset_accessed=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--zip', type=Path, help='Optional original returned ZIP for full member verification')
    args = p.parse_args()
    print(json.dumps(verify(ROOT, args.zip), indent=2))
