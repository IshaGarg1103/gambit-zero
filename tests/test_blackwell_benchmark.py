"""Benchmark safety/provenance tests; never need a remote GPU or dataset."""
import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = runpy.run_path(str(ROOT/'benchmarks/bench_pgx_safe.py'))
RENDER = runpy.run_path(str(ROOT/'scripts/report_blackwell_followup.py'))['render']
RUNNER = runpy.run_path(str(ROOT/'scripts/molab_m1_blackwell.py'))


def test_memory_budget_is_conservative_and_user_limit_cannot_raise_it():
    budget = BENCH['memory_budget']
    assert budget(96*2**30, True) == 16*2**30
    assert budget(0, True) == 0
    assert budget(100*2**20, True, 10000) == 35*2**20
    assert budget(96*2**30, True, 256) == 256*2**20


def test_insufficient_memory_skips_without_batch_compilation(tmp_path, monkeypatch):
    f = BENCH['benchmark']
    monkeypatch.setitem(f.__globals__, 'memory_budget', lambda *args: 0)
    result = f(tmp_path/'result.json', batches=(128, 8192), repeats=30)
    assert [r['status'] for r in result['rows']] == ['SKIPPED', 'SKIPPED']
    assert all('compile_seconds' not in r for r in result['rows'])
    assert json.loads((tmp_path/'result.json').read_text()) == result


def test_required_gpu_refuses_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(BENCH['jax'], 'devices', lambda: [SimpleNamespace(platform='cpu')])
    with pytest.raises(RuntimeError, match='GPU required'):
        BENCH['benchmark'](tmp_path/'result.json', require_gpu=True)


def test_compile_failure_is_saved_and_stops_larger_batches(tmp_path, monkeypatch):
    f = BENCH['benchmark']
    original_jit = BENCH['jax'].jit
    class BrokenCompiler:
        def lower(self, *args):
            raise RuntimeError('injected compiler failure')
    monkeypatch.setitem(f.__globals__, 'memory_budget', lambda *args: 2**30)
    monkeypatch.setattr(BENCH['jax'], 'jit', lambda func: original_jit(func) if func == BENCH['ENV'].init else BrokenCompiler())
    result = f(tmp_path/'result.json', batches=(128, 1024), repeats=30)
    assert result['status'] == 'FAILED'
    assert result['rows'][0]['error_type'] == 'RuntimeError'
    assert result['rows'][1]['status'] == 'SKIPPED'
    assert json.loads((tmp_path/'result.json').read_text())['status'] == 'FAILED'


def test_small_benchmark_checks_real_transitions(tmp_path):
    result = BENCH['benchmark'](tmp_path/'result.json', batches=(2,), repeats=2)
    row = result['rows'][0]
    assert row['status'] == 'MEASURED'
    assert row['transition_check'] == 'PASS'
    assert row['steps_per_second'] > 0
    assert row['compile_seconds'] >= 0
    assert row['warm_seconds'] > 0


def synthetic_report(tmp_path, measured=True):
    reports = tmp_path/'reports'
    reports.mkdir()
    rows = [dict(batch=b, status='MEASURED' if measured else 'SKIPPED', repeats=30,
                 peak_device_bytes=123456, transition_check='PASS') for b in [128, 1024, 4096, 8192]]
    report = dict(status='COMPLETE', rows=rows, gpu_measured=measured, protected_sources_unchanged=True,
                  regression_returncode=0, host=dict(status='PASS', gpu_kind='RTX PRO 6000 Blackwell'))
    (reports/'m0_large_batch_blackwell.json').write_text(json.dumps(report))
    (reports/'m1_blackwell_tests.txt').write_text('253 passed in 50.00s\n')
    return report


def test_report_does_not_pass_an_all_skipped_run(tmp_path):
    synthetic_report(tmp_path, measured=False)
    assert RENDER(tmp_path)['benchmark_gate'] == 'INCOMPLETE'


def test_report_accepts_verified_run_and_preserves_historical_acceptance(tmp_path):
    synthetic_report(tmp_path)
    original = tmp_path/'reports/m1_final_acceptance.json'
    original.write_text('{"frozen":"historical"}\n')
    assert RENDER(tmp_path)['benchmark_gate'] == 'PASS'
    assert original.read_text() == '{"frozen":"historical"}\n'
    assert 'environment-only' in (tmp_path/'reports/m1_blackwell_benchmark.md').read_text()


@pytest.mark.parametrize('field,value', [('regression_returncode', 1), ('protected_sources_unchanged', False)])
def test_report_rejects_failed_regression_or_source_drift(tmp_path, field, value):
    report = synthetic_report(tmp_path)
    report[field] = value
    (tmp_path/'reports/m0_large_batch_blackwell.json').write_text(json.dumps(report))
    assert RENDER(tmp_path)['benchmark_gate'] != 'PASS'


def test_bundle_checksum_rejects_modified_source(tmp_path):
    (tmp_path/'source.py').write_bytes(b'original')
    (tmp_path/'BLACKWELL_BUNDLE.json').write_text(json.dumps(dict(files={'source.py':hashlib.sha256(b'original').hexdigest()})))
    RUNNER['verify_bundle'](tmp_path)
    (tmp_path/'source.py').write_bytes(b'modified')
    with pytest.raises(RuntimeError, match='checksum mismatch'):
        RUNNER['verify_bundle'](tmp_path)


def test_failure_results_are_exported_with_checksums(tmp_path):
    work = tmp_path/'reports/blackwell_followup'
    work.mkdir(parents=True)
    (work/'failure.log').write_text('synthetic failure fixture')
    (tmp_path/'BLACKWELL_BUNDLE.json').write_text('{}')
    archive = RUNNER['package_results'](tmp_path, dict(status='FAILED'))
    with zipfile.ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read('RESULTS_MANIFEST.json'))
        for path, digest in manifest['files'].items():
            assert hashlib.sha256(bundle.read(path)).hexdigest() == digest
        assert manifest['status'] == 'FAILED'


def test_measured_supplement_is_idempotent_and_preserves_historical_body(tmp_path):
    synthetic_report(tmp_path)
    pilot = tmp_path/'reports/m1_50k_pilot.md'
    pilot.write_text('# Historical report\nOriginal scientific counts.\n')
    RENDER(tmp_path)
    first = pilot.read_bytes()
    RENDER(tmp_path)
    assert pilot.read_bytes() == first
    assert first.startswith(b'# Historical report\nOriginal scientific counts.\n')
    assert first.count(b'<!-- m1-blackwell-followup:start -->') == 1
    assert b'supersedes the historical GPU-pending' in first


def test_incomplete_supplement_does_not_resolve_gpu_deferral(tmp_path):
    synthetic_report(tmp_path, measured=False)
    pilot = tmp_path/'reports/m1_50k_pilot.md'
    pilot.write_text('# Historical report\n')
    RENDER(tmp_path)
    assert 'historical GPU deferral is not resolved' in pilot.read_text()


def test_stale_verification_receipt_is_rejected(tmp_path):
    synthetic_report(tmp_path)
    (tmp_path/'reports/m1_blackwell_verification.json').write_text(json.dumps(dict(benchmark_sha256='0'*64)))
    with pytest.raises(ValueError, match='different benchmark'):
        RENDER(tmp_path)
