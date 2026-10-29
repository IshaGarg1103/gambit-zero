import hashlib
import io
import json
from pathlib import Path
import random
import struct

import pytest
import zstandard as zstd

from tinychess.data.archive_scan import (ALGORITHM, DaySampler, DiskGuard, HeaderRejected,
    PZSTD_MAGIC, PzstdStream, ScanError, Shortfall, canonical, daily_quotas,
    header_candidate, priority, scan_month)


def pgn(day, index, rating=2100, **updates):
    headers = dict(Event='Rated Rapid game',Site=f'https://lichess.org/{day:02}{index:06}',
                   WhiteElo=str(rating),BlackElo=str(rating+100),TimeControl='300+0',
                   UTCDate=f'2026.01.{day:02}',Termination='Normal',Result='1-0')
    headers.update(updates)
    return ''.join(f'[{k} "{v}"]\n' for k,v in headers.items())+'\n1. e4 e5 1-0\n\n'


def compressed(text, chunk_size=4096):
    result = b''
    for i in range(0,len(text),chunk_size):
        frame = zstd.ZstdCompressor(write_checksum=True).compress(text[i:i+chunk_size])
        result += struct.pack('<III',PZSTD_MAGIC,4,len(frame))+frame
    return result


def decode(raw, expected=None, digest=None, **kwargs):
    decoder = PzstdStream(io.BytesIO(raw),expected or len(raw),digest or hashlib.sha256(raw).hexdigest(),**kwargs)
    with io.BufferedReader(decoder,buffer_size=73) as stream:
        value = stream.read()
    return value,decoder


def test_bounded_multiframe_decode_and_full_compressed_checksum():
    text = (pgn(1,0)+'; unicode: café\n').encode()*30
    raw = compressed(text,chunk_size=251)
    value,decoder = decode(raw)
    assert value == text and decoder.frames > 1
    assert decoder.verified and decoder.compressed_bytes == len(raw)
    assert len(decoder.frame_index) == decoder.frames
    for item in decoder.frame_index:
        start = item['compressed_frame_offset']
        frame = raw[start:start+item['compressed_frame_bytes']]
        assert hashlib.sha256(frame).hexdigest() == item['compressed_frame_sha256']
        assert len(zstd.ZstdDecompressor().decompress(frame)) == item['decoded_bytes']


@pytest.mark.parametrize('mutation', ['checksum','short_body','short_frame','extra_bytes','unknown_format'])
def test_bad_archive_never_verified(mutation):
    raw = compressed(pgn(1,0).encode())
    original_size = len(raw)
    digest = hashlib.sha256(raw).hexdigest()
    if mutation == 'checksum':
        digest = '0'*64
    elif mutation == 'short_body':
        raw = raw[:-2]
    elif mutation == 'short_frame':
        raw = raw[:-2]
        raw = raw[:8]+struct.pack('<I',len(raw)-12)+raw[12:]
        original_size,digest = len(raw),hashlib.sha256(raw).hexdigest()
    elif mutation == 'extra_bytes':
        raw += b'x'
    else:
        raw = b'X'+raw[1:]
    with pytest.raises((ScanError,zstd.ZstdError)):
        decode(raw,original_size,digest)


def test_frame_and_memory_budgets_fail_before_large_decode():
    raw = compressed(b'x'*10000)
    with pytest.raises(ScanError,match='Compressed frame'):
        decode(raw,max_compressed_frame=1)
    with pytest.raises(ScanError,match='bounded buffer|window'):
        decode(raw,max_decoded_frame=100)


@pytest.mark.parametrize('updates,reason', [
    ({'WhiteElo':'1799'},'rating_below'),({'WhiteElo':'?'},'invalid_rating'),
    ({'TimeControl':'299+1000'},'time_control_below'),({'TimeControl':'300+?'},'invalid_time'),
    ({'UTCDate':'2026.02.01'},'month_mismatch'),({'UTCDate':'2026.01.32'},'invalid_utc'),
    ({'Rated':'false'},'not_rated'),({'Variant':'Chess960'},'variant'),
    ({'Termination':'Rules infraction'},'excluded'),({'Result':'*'},'invalid_result'),
])
def test_header_prefilter_boundaries(updates,reason):
    with pytest.raises(HeaderRejected,match=reason):
        header_candidate(pgn(1,0,**updates),'2026-01')


def test_missing_duplicate_headers_and_stable_identity():
    text = pgn(1,0,rating=1800,TimeControl='300+0')
    assert header_candidate(text,'2026-01')['rating_stratum'] == 'regular'
    assert header_candidate(pgn(1,0,rating=2000),'2026-01')['rating_stratum'] == 'strong'
    assert header_candidate(text,'2026-01')['game_id'] == '01000000'
    for bad in [text.replace('[WhiteElo "1800"]\n',''),text.replace('\n\n','\n[Result "1-0"]\n\n',1)]:
        with pytest.raises(HeaderRejected):
            header_candidate(bad,'2026-01')
    missing_site = pgn(1,0,Site='')
    assert header_candidate(missing_site,'2026-01')['game_id'] == hashlib.sha256(missing_site.encode()).hexdigest()


def test_exact_calendar_quotas_and_unambiguous_priority():
    assert len(daily_quotas('2026-02',5000)) == 28
    assert sum(daily_quotas('2026-01',5000).values()) == 5000
    assert max(daily_quotas('2026-01',5000).values())-min(daily_quotas('2026-01',5000).values()) == 1
    assert priority(12,'2026-01','345') != priority(1,'2026-01','2345')
    with pytest.raises(ValueError): daily_quotas('2026-01',10)


def sample(order):
    sampler = DaySampler('2026-01',310,20261025,reserve_factor=2)
    for ordinal,text in order:
        sampler.offer(text,ordinal)
    return sampler


def fixtures():
    return list(enumerate([pgn(day,index,rating=2100 if index<16 else 1900)
                          for day in range(1,32) for index in range(28)],1))


def test_month_wide_determinism_rating_quotas_and_reserves():
    source = fixtures()
    first = sample(source)
    random.Random(42).shuffle(source)
    second = sample(source)
    assert first.snapshot() == second.snapshot()
    chosen,rejections,coverage = first.select(lambda record: record['game_id'])
    assert len(chosen) == 310 and not rejections
    assert len({r['game_id'] for r,value in chosen}) == 310
    assert len(coverage) == 31
    assert all(row['strong'] == 6 and row['accepted'] == 10 for row in coverage.values())
    assert first.candidate_bytes <= first.max_candidate_bytes


def test_validation_rejections_use_reserves_but_parity_is_fatal():
    from tinychess.data.human import RejectedGame
    from tinychess.validation.parity import ParityError
    sampler = sample(fixtures())
    failed = {min(sampler.heaps[(day,'strong')],key=lambda e:e.key).record['game_id'] for day in sampler.quotas}
    def validate(record):
        if record['game_id'] in failed:
            raise RejectedGame('fixture_parse_error')
        return record['game_id']
    selected,rejects,_ = sampler.select(validate)
    assert len(selected) == 310 and rejects == {'fixture_parse_error':31}
    calls = []
    def fatal(record):
        calls.append(record['game_id'])
        raise ParityError('fixture mismatch')
    with pytest.raises(ParityError): sampler.select(fatal)
    assert len(calls) == 1


def test_shortages_not_redistributed_or_called_success():
    sampler = DaySampler('2026-01',31,0)
    sampler.offer(pgn(1,0),1)
    with pytest.raises(Shortfall): sampler.select(lambda r:r['game_id'])
    memory = DaySampler('2026-01',31,0,max_candidate_bytes=1)
    with pytest.raises(ScanError,match='memory'): memory.offer(pgn(1,0),1)


def test_deterministic_rank_ties(monkeypatch):
    import tinychess.data.archive_scan as module
    monkeypatch.setattr(module,'priority',lambda *a:'0'*64)
    source = [(i,pgn(1,i)) for i in range(20)]
    a,b = DaySampler('2026-01',31,0,reserve_factor=1),DaySampler('2026-01',31,0,reserve_factor=1)
    for ordinal,text in source: a.offer(text,ordinal)
    for ordinal,text in reversed(source): b.offer(text,ordinal)
    assert a.snapshot()['candidates'] == b.snapshot()['candidates']
    assert a.snapshot()['candidates'][0]['game_id'] == '01000000'


class Response(io.BytesIO):
    status = 200
    def __init__(self,raw):
        super().__init__(raw)
        self.headers = {'Content-Length':str(len(raw))}


def test_scan_month_checkpoints_only_verified_stream_and_reuses_without_network(tmp_path):
    raw = compressed(''.join(pgn(day,index) for day in range(1,32) for index in range(2)).encode())
    source = dict(month='2026-01',url='https://example.invalid/archive',archive_size_bytes=len(raw),published_sha256=hashlib.sha256(raw).hexdigest())
    calls = []
    def opener(*a,**kw):
        calls.append(1)
        return Response(raw)
    kwargs = dict(network_budget_bytes=len(raw),max_seconds=60,reserve_disk_bytes=1,opener=opener)
    value = scan_month(source,tmp_path,31,0,**kwargs)
    assert value['archive']['full_archive_checksum_verified']
    assert value['kind'] == 'bounded_candidate_manifest_not_random_access_index'
    assert len(calls) == 1
    assert scan_month(source,tmp_path,31,0,**kwargs) == value
    assert len(calls) == 1
    assert all(p.suffix=='.json' for p in tmp_path.iterdir())
    path = tmp_path/'2026-01.selection.json'
    path.write_text(path.read_text().replace('utc-day-rating','tampered-day-rating'))
    with pytest.raises(ScanError,match='integrity'): scan_month(source,tmp_path,31,0,**kwargs)


def test_interrupted_month_restart_and_real_replay_validation(tmp_path):
    from tinychess.data.human import replay_game
    import yaml
    config = yaml.safe_load((Path(__file__).parents[1]/'configs/data_human.yaml').read_text())
    raw = compressed(''.join(pgn(day,0) for day in range(1,32)).encode())
    source = dict(month='2026-01',url='https://example.invalid/archive',archive_size_bytes=len(raw),published_sha256=hashlib.sha256(raw).hexdigest())
    def interrupted(*a,**kw):
        r = Response(raw[:-5]);r.headers['Content-Length'] = str(len(raw));return r
    kwargs = dict(network_budget_bytes=len(raw),max_seconds=60,reserve_disk_bytes=1)
    with pytest.raises(ScanError): scan_month(source,tmp_path,31,0,opener=interrupted,**kwargs)
    assert not (tmp_path/'2026-01.selection.json').exists()
    assert json.loads((tmp_path/'2026-01.status.json').read_text())['status'] == 'INCOMPLETE'
    value = scan_month(source,tmp_path,31,0,opener=lambda *a,**kw:Response(raw),**kwargs)
    assert value['archive']['full_archive_checksum_verified']
    sampler = DaySampler('2026-01',31,0)
    for candidate in value['selection']['candidates']: sampler.offer(candidate['pgn'],candidate['source_ordinal'])
    accepted,rejections,coverage = sampler.select(lambda r:replay_game(r['pgn'],'2026-01',r['source_ordinal'],source['url'],config))
    assert len(accepted) == 31 and not rejections and len(coverage) == 31
    assert sum(len(game.positions) for r,game in accepted) == 62


def test_resource_gates_precede_network_and_disk_reserve(tmp_path):
    source = dict(month='2026-01',url='https://example.invalid/archive',archive_size_bytes=100,published_sha256='0'*64)
    def forbidden(*a,**kw): pytest.fail('network opened before budget gate')
    with pytest.raises(ScanError,match='budgets'):
        scan_month(source,tmp_path,31,0,network_budget_bytes=None,max_seconds=None,opener=forbidden)
    with pytest.raises(ScanError,match='reserve'):
        DiskGuard(tmp_path,1000,2**100)
    guard = DiskGuard(tmp_path,100,1)
    with pytest.raises(ScanError,match='disk budget'): guard.check(101)


def test_default_cli_is_plan_only_and_execution_stays_disabled():
    import runpy
    import yaml
    root = Path(__file__).parents[1]
    module = runpy.run_path(str(root/'scripts/scan_m1_archives.py'))
    config = yaml.safe_load((root/'configs/data_human_50k_stream.yaml').read_text())
    result = module['plan'](config)
    assert result['mode'] == 'PLAN_ONLY_NO_NETWORK'
    assert result['full_archive_cache_bytes'] == result['decompressed_archive_disk_bytes'] == 0
    assert result['peak_additional_disk_budget_bytes'] == 2684354560
    assert result['destination']=='huggingface'
    assert not result['complete_local_dataset_required']
    assert not config['execution_enabled']
    with pytest.raises(ScanError,match='disabled'):
        module['execute'](config)


def test_one_month_session_never_selects_other_archive_sources():
    import runpy
    root=Path(__file__).parents[1]
    select=runpy.run_path(str(root/'scripts/scan_m1_archives.py'))['session_sources']
    sources=[dict(month=f'2026-{i:02}') for i in range(1,10)]
    config=dict(months={s['month']:{} for s in sources},session_months=['2026-01'])
    assert select(config,sources)==[dict(month='2026-01')]
    for invalid in [[],['2026-01','2026-02'],['2025-01']]:
        config['session_months']=invalid
        with pytest.raises(ScanError,match='exactly one'):select(config,sources)
