"""Offline end-to-end synthetic month scan, selection and corrected-Pgx replay."""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import time

import yaml
import zstandard as zstd
from tinychess.data.archive_scan import DaySampler,PZSTD_MAGIC,canonical,scan_month
from tinychess.data.human import replay_game


def main():
    started=time.perf_counter()
    games=[]
    for day in range(1,32):
        for index in range(28):
            rating=2100 if index<16 else 1900
            result=['1-0','0-1','1/2-1/2'][index%3]
            term='Time forfeit' if index%5==0 else 'Normal'
            headers=dict(Event='Rated Rapid game',Site=f'https://lichess.org/{day:02}{index:06}',
                         WhiteElo=str(rating),BlackElo=str(rating+100),TimeControl='300+0',
                         UTCDate=f'2026.01.{day:02}',Termination=term,Result=result)
            games.append(''.join(f'[{k} "{v}"]\n' for k,v in headers.items())+f'\n1. e4 e5 {result}\n\n')
    text=''.join(games).encode()
    raw=b''
    for i in range(0,len(text),4096):
        frame=zstd.ZstdCompressor(write_checksum=True).compress(text[i:i+4096])
        raw+=struct.pack('<III',PZSTD_MAGIC,4,len(frame))+frame
    source=dict(month='2026-01',url='https://example.invalid/synthetic-only',archive_size_bytes=len(raw),published_sha256=hashlib.sha256(raw).hexdigest())
    calls=[]
    class Response(io.BytesIO):
        status=200
        headers={'Content-Length':str(len(raw))}
    def opener(*args,**kwargs):
        calls.append(1)
        return Response(raw)
    with tempfile.TemporaryDirectory(prefix='gambit-zero-stream-fixture-') as tmp:
        kwargs=dict(network_budget_bytes=len(raw),max_seconds=60,reserve_disk_bytes=1,opener=opener)
        first=scan_month(source,tmp,310,20261025,**kwargs)
        second=scan_month(source,tmp,310,20261025,**kwargs)
        assert first==second and len(calls)==1
        sampler=DaySampler('2026-01',310,20261025)
        for r in first['selection']['candidates']:
            sampler.offer(r['pgn'],r['source_ordinal'])
        config=yaml.safe_load(Path('configs/data_human_50k_stream.yaml').read_text())
        selected,rejections,coverage=sampler.select(lambda r:replay_game(r['pgn'],'2026-01',r['source_ordinal'],source['url'],config))
        classes=Counter()
        masks=Counter()
        for candidate,game in selected:
            outcome={'1-0':1,'0-1':-1,'1/2-1/2':0}[game.game['metadata']['result']]
            for row in game.positions:
                assert row['outcome_stm']==outcome*(1 if row['color']==0 else -1)
                assert row['value_weight']==(0 if game.game['metadata']['termination_header']=='Time forfeit' else 1)
                classes[row['wdl_class']]+=1
                masks[str(row['value_weight'])]+=1
        cache_bytes=sum(p.stat().st_size for p in Path(tmp).rglob('*') if p.is_file())
        report=dict(status='PASS_SYNTHETIC_FIXTURE_ONLY',source_game_blocks=len(games),selected_games=len(selected),
            days_covered=len(coverage),coverage=coverage,selected_strong_games=sum(r['min_rating']>=2000 for r,g in selected),
            replayed_premove_positions=sum(len(g.positions) for r,g in selected),
            parity_states_including_final=sum(len(g.positions)+1 for r,g in selected),
            full_compressed_fixture_verified=first['archive']['full_archive_checksum_verified'],
            fixture_compressed_bytes=len(raw),fixture_decoded_bytes=len(text),
            selection_digest=first['selection_digest'],
            selected_ids_sha256=hashlib.sha256(canonical([r['game_id'] for r,g in selected]).encode()).hexdigest(),
            checkpoint_reuse_identical=first==second,fixture_transport_opens=len(calls),real_network_bytes=0,
            candidate_cache_bytes=cache_bytes,raw_archive_files_written=0,decoded_archive_files_written=0,
            full_real_archives_verified=0,rejections=rejections,wdl_class_counts=dict(classes),value_weight_counts=dict(masks),
            elapsed_seconds=time.perf_counter()-started,temporary_fixture_cache_removed_on_exit=True)
    Path('reports/m1_streaming_fixture_smoke.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='coverage'}))


if __name__=='__main__': main()
