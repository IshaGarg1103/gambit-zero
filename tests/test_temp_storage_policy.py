"""Output budget waivers must preserve actual disk-space safeguards."""
import pytest
from tinychess.data.archive_scan import DiskGuard, ScanError


def test_default_still_enforces_output_budget(tmp_path, monkeypatch):
    monkeypatch.delenv('TINYCHESS_ENFORCE_TEMP_STORAGE_CAPS', raising=False)
    guard = DiskGuard(tmp_path, 1, 1)
    with pytest.raises(ScanError, match='Managed disk budget'):
        guard.check(2)


def test_explicit_waiver_allows_growth_above_output_budget(tmp_path, monkeypatch):
    monkeypatch.setenv('TINYCHESS_ENFORCE_TEMP_STORAGE_CAPS', '0')
    (tmp_path / 'output').write_bytes(b'12345')
    guard = DiskGuard(tmp_path, 1, 1)
    assert guard.check()['managed_bytes'] == 5
    guard.check(100)


def test_waiver_cannot_bypass_free_disk_reserve(tmp_path, monkeypatch):
    monkeypatch.setenv('TINYCHESS_ENFORCE_TEMP_STORAGE_CAPS', '0')
    with pytest.raises(ScanError, match='Free-disk reserve'):
        DiskGuard(tmp_path, 1, 2**100)
