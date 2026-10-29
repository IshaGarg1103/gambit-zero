"""Bounded source diagnostics must not pretend a prefix verifies an archive."""
import io
from pathlib import Path
import runpy

import pytest
import zstandard as zstd


def module():
    return runpy.run_path(str(Path(__file__).parents[1]/'scripts/preflight_m1_50k.py'))


@pytest.mark.parametrize('size,rate', [(1,0), (1,-1), (-1,1)])
def test_invalid_projection_inputs(size, rate):
    with pytest.raises(ValueError):
        module()['estimated_seconds'](size, rate)


def test_network_projection_is_explicit_estimate():
    assert module()['estimated_seconds'](1000, 10) == 100


def test_frame_boundary_is_observed_not_invented(tmp_path):
    pgn = ('[Event "Rated Blitz game"]\n[WhiteElo "2000"]\n[BlackElo "2100"]\n'
           '[TimeControl "300+0"]\n[Result "1-0"]\n\n1. e4 e5 1-0\n\n').encode()
    first = zstd.ZstdCompressor().compress(pgn)
    path = tmp_path/'sample.pgn.zst.prefix'
    path.write_bytes(first+zstd.ZstdCompressor().compress(pgn))
    result = module()['inspect_cached_prefix'](path)
    assert result['complete_first_frame_seen']
    assert result['first_frame_end_compressed_offset'] == len(first)
    assert result['first_frame_decompressed_bytes'] == len(pgn)
    assert result['header_strong_candidates'] == 2
    assert not result['archive_wide_frame_structure_verified']
    assert not result['representative_of_full_month']


def test_short_probe_is_not_reported_as_success(monkeypatch):
    functions = module()
    class Response(io.BytesIO):
        status = 206
        headers = {'Content-Range': 'bytes 0-2/100000000'}
    monkeypatch.setitem(functions['throughput_probe'].__globals__, 'request',
                        lambda *a, **kw: Response(b'abc'))
    with pytest.raises(RuntimeError, match='Short throughput probe'):
        functions['throughput_probe']('https://example.invalid')


def test_range_response_must_start_at_zero(monkeypatch):
    functions = module()
    class Response(io.BytesIO):
        status = 206
        headers = {'Content-Range': 'bytes 100-102/100000000'}
    monkeypatch.setitem(functions['throughput_probe'].__globals__, 'request',
                        lambda *a, **kw: Response(b'abc'))
    with pytest.raises(RuntimeError, match='offset-zero'):
        functions['throughput_probe']('https://example.invalid')
