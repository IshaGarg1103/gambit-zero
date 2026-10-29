"""HTTP continuation must preserve the one original stream and full SHA gate."""
import hashlib
import http.client
import io
import json
import struct

import pytest
import zstandard as zstd

from tinychess.data.archive_scan import PZSTD_MAGIC, ScanError, scan_month
from tinychess.data.archive_http import ContinuousArchiveResponse


class Response(io.BytesIO):
    def __init__(self, payload, total, *, start=0, etag='"frozen"'):
        super().__init__(payload)
        self.status = 206 if start else 200
        self.headers = {'Content-Length': str(total-start), 'ETag': etag}
        if start:
            self.headers['Content-Range'] = f'bytes {start}-{total-1}/{total}'


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch):
    monkeypatch.setattr('tinychess.data.archive_http.time.sleep', lambda _: None)


def fixture():
    text = ''.join(f'[Event "Rated Rapid game"]\n[Site "https://lichess.org/{day:08}"]\n'
                   '[WhiteElo "2100"]\n[BlackElo "2200"]\n[TimeControl "300+0"]\n'
                   f'[UTCDate "2026.01.{day:02}"]\n[Termination "Normal"]\n'
                   '[Result "1-0"]\n\n1. e4 e5 1-0\n\n' for day in range(1,32)).encode()
    raw = b''
    for start in range(0, len(text), 251):
        frame = zstd.ZstdCompressor(write_checksum=True).compress(text[start:start+251])
        raw += struct.pack('<III', PZSTD_MAGIC, 4, len(frame)) + frame
    source = dict(month='2026-01', url='https://example.invalid/archive',
                  archive_size_bytes=len(raw), published_sha256=hashlib.sha256(raw).hexdigest(),
                  headers={'ETag':'"frozen"'})
    return raw, source


@pytest.mark.parametrize('cut', [7, 19, 437])
def test_reconnect_mid_header_or_frame_matches_uninterrupted_selection(tmp_path, cut):
    raw, source = fixture()
    calls = []
    def interrupted(request, **kwargs):
        headers = {k.lower():v for k,v in request.header_items()}
        calls.append(headers)
        if len(calls) == 1:
            return Response(raw[:cut], len(raw))
        assert headers['range'] == f'bytes={cut}-'
        assert headers['if-range'] == '"frozen"'
        return Response(raw[cut:], len(raw), start=cut)
    kwargs = dict(quota=31, seed=42, network_budget_bytes=len(raw), max_seconds=60, reserve_disk_bytes=1)
    resumed = scan_month(source, tmp_path/'resumed', opener=interrupted, **kwargs)
    baseline = scan_month(source, tmp_path/'baseline', opener=lambda *a,**kw:Response(raw,len(raw)), **kwargs)
    assert len(calls) == 2
    assert resumed['selection'] == baseline['selection']
    assert resumed['compression_frame_index'] == baseline['compression_frame_index']
    assert resumed['archive']['sha256'] == source['published_sha256']
    assert resumed['archive']['bytes'] == len(raw)
    assert resumed['scan_measurements']['http_reconnects'] == 1
    assert resumed['scan_measurements']['http_reconnection_events'][0]['offset'] == cut
    assert not list((tmp_path/'resumed').glob('*.zst'))


@pytest.mark.parametrize('damage', ['status200', 'wrong_start', 'wrong_end', 'wrong_total', 'wrong_length', 'changed_etag', 'encoding'])
def test_bad_suffix_is_never_accepted_or_checksum_verified(tmp_path, damage):
    raw, source = fixture()
    calls = []
    def opener(request, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return Response(raw[:7], len(raw))
        response = Response(raw[7:], len(raw), start=7)
        if damage == 'status200': response.status = 200
        if damage == 'wrong_start': response.headers['Content-Range'] = f'bytes 8-{len(raw)-1}/{len(raw)}'
        if damage == 'wrong_end': response.headers['Content-Range'] = f'bytes 7-{len(raw)-2}/{len(raw)}'
        if damage == 'wrong_total': response.headers['Content-Range'] = f'bytes 7-{len(raw)-1}/{len(raw)+1}'
        if damage == 'wrong_length': response.headers['Content-Length'] = str(len(raw))
        if damage == 'changed_etag': response.headers['ETag'] = '"changed"'
        if damage == 'encoding': response.headers['Content-Encoding'] = 'gzip'
        return response
    with pytest.raises(ScanError):
        scan_month(source,tmp_path,31,0,network_budget_bytes=len(raw),max_seconds=60,reserve_disk_bytes=1,opener=opener)
    assert len(calls) == 2
    assert not (tmp_path/'2026-01.selection.json').exists()
    assert json.loads((tmp_path/'2026-01.status.json').read_text())['full_archive_checksum_verified'] is False


def test_full_sha_still_required_after_valid_reconnection(tmp_path):
    raw, source = fixture()
    source['published_sha256'] = '0'*64
    calls = []
    def opener(request, **kwargs):
        calls.append(1)
        return Response(raw[:7],len(raw)) if len(calls)==1 else Response(raw[7:],len(raw),start=7)
    with pytest.raises(ScanError,match='SHA256 mismatch'):
        scan_month(source,tmp_path,31,0,network_budget_bytes=len(raw),max_seconds=60,reserve_disk_bytes=1,opener=opener)
    assert not (tmp_path/'2026-01.selection.json').exists()


def test_timeouts_reconnect_from_last_delivered_byte():
    raw, source = fixture()
    class TimedOut(Response):
        def read(self,size):
            if self.tell() >= 7: raise TimeoutError('test timeout')
            return super().read(min(size,7))
    calls=[]
    def opener(request,**kwargs):
        calls.append(dict(request.header_items()))
        return TimedOut(raw,len(raw)) if len(calls)==1 else Response(raw[7:],len(raw),start=7)
    with ContinuousArchiveResponse(source,opener=opener,max_seconds=60) as stream:
        chunks=[]
        while True:
            part=stream.read(7)
            if not part: break
            chunks.append(part)
        assert b''.join(chunks)==raw and stream.reconnects==1


def test_partial_bytes_from_incomplete_read_are_not_dropped_or_duplicated():
    raw, source = fixture()
    class Partial(Response):
        def read(self,size): raise http.client.IncompleteRead(raw[:7],5)
    calls=[]
    def opener(request,**kwargs):
        calls.append(1)
        return Partial(raw,len(raw)) if len(calls)==1 else Response(raw[7:],len(raw),start=7)
    with ContinuousArchiveResponse(source,opener=opener,max_seconds=60) as stream:
        chunks=[]
        while True:
            part=stream.read(12)
            if not part:break
            chunks.append(part)
        assert b''.join(chunks)==raw and stream.reconnects==1


def test_retry_limit_and_runtime_are_hard_stops():
    raw, source=fixture();calls=[]
    def opener(*args,**kwargs):calls.append(1);return Response(b'',len(raw))
    with ContinuousArchiveResponse(source,opener=opener,max_seconds=60,max_reconnects=2) as stream:
        with pytest.raises(ScanError,match='limit exhausted'):stream.read(12)
    assert len(calls)==3
    with ContinuousArchiveResponse(source,opener=opener,max_seconds=-1) as stream:
        with pytest.raises(ScanError,match='Runtime budget'):stream.read(12)
    assert len(calls)==3


def test_weak_validator_cannot_authorize_suffix():
    raw, source=fixture();source['headers']['ETag']='W/"weak"'
    with ContinuousArchiveResponse(source,opener=lambda *a,**k:Response(raw[:7],len(raw),etag='W/"weak"'),max_seconds=60) as stream:
        assert stream.read(12)==raw[:7]
        with pytest.raises(ScanError,match='Strong unchanged'):stream.read(12)
