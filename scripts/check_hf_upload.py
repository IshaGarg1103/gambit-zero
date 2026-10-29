"""Small synthetic shard smoke. Default OFFLINE; --upload uses authenticated Hub.

No Lichess network scan occurs, and no complete real monthly archive is saved.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile

import chess
import chess.pgn
import yaml
import zstandard as zstd

from tinychess.data.archive_scan import DiskGuard,PZSTD_MAGIC,scan_month
from tinychess.data.hf_audit import AuditIndex
from tinychess.data.hf_month import publish_month,reproduce_month
from tinychess.data.hf_upload import UploadError,VerifiedUploader
from tinychess.data.hf_card import publish_docs


class OfflineTransport:
    """Memory-only fault-free Hub double. Never reported as real remote evidence."""
    def __init__(self):
        self.revision='0'*40
        self.snapshots={self.revision:{}}
        self.files={}
        self.uploads=0
    def ensure_repo(self):return self.revision
    def head(self):return self.revision
    def exists(self,path,revision):return path in self.snapshots[revision]
    def upload(self,local,path,parent):
        if parent!=self.revision:raise UploadError('Concurrent update in offline fixture')
        self.uploads+=1
        self.files[path]=Path(local).read_bytes()
        self.revision=f'{self.uploads:040x}'
        self.snapshots[self.revision]=dict(self.files)
        return self.revision
    def hash_remote(self,path,revision,max_bytes):
        payload=self.snapshots[revision][path]
        if len(payload)>max_bytes:raise UploadError('Offline remote size exceeds bound')
        return hashlib.sha256(payload).hexdigest(),len(payload)
    def read_verified_small(self,receipt,cap_bytes=32*2**20):
        data=self.snapshots[receipt['revision']][receipt['path_in_repo']]
        if len(data)>cap_bytes or hashlib.sha256(data).hexdigest()!=receipt['sha256']:
            raise UploadError('Offline selection checksum mismatch')
        return data


def run(root,config,live):
    root=Path(root)
    # This is deliberately a synthetic month, not an official source snapshot.
    texts=[]
    for day in range(1,32):
        result=['1-0','0-1','1/2-1/2'][day%3]
        moves=['e2e4','e7e5']
        if day==1:
            moves=['f2f3','e7e5','g2g4','d8h4'];result='0-1'
        elif day==2:
            moves=['e2e4','e7e5','f1c4','b8c6','d1h5','g8f6','h5f7'];result='1-0'
        game=chess.pgn.Game()
        game.headers.update(Event='Rated Rapid game',Site=f'https://example.invalid/synthetic/hf{day:06}',
            WhiteElo='2100',BlackElo='2200',TimeControl='300+0',UTCDate=f'2026.01.{day:02}',
            Termination='Time forfeit' if day%5==0 else 'Normal',Result=result)
        node=game
        for uci in moves:node=node.add_variation(chess.Move.from_uci(uci))
        texts.append(game.accept(chess.pgn.StringExporter(headers=True,variations=False,comments=False))+'\n\n')
    frame=zstd.ZstdCompressor(write_checksum=True).compress(''.join(texts).encode())
    raw=struct.pack('<III',PZSTD_MAGIC,4,len(frame))+frame
    source=dict(month='2026-01',url='https://example.invalid/synthetic-hf-smoke-not-a-lichess-archive',
                archive_size_bytes=len(raw),published_sha256=hashlib.sha256(raw).hexdigest())
    class Response(io.BytesIO):
        status=200
        headers={'Content-Length':str(len(raw))}
    scan_month(source,root/'candidates',31,20261025,network_budget_bytes=len(raw),max_seconds=60,
               reserve_disk_bytes=20*2**30,opener=lambda *a,**kw:Response(raw))
    frozen=root/'frozen-smoke-d.json'
    frozen.write_text(json.dumps(dict(kind='synthetic_fixture',version='hf-smoke-d-v1',position_fens=[chess.STARTING_FEN])))
    data_config=yaml.safe_load(Path('configs/data_human_50k_stream.yaml').read_text())
    data_config['months']={'2026-01':dict(split='train',accepted_games=31)}
    uploader=VerifiedUploader(config['repo_id'],root/'spool',private=config['private'],
        transport=None if live else OfflineTransport(),spool_cap_bytes=config['spool_cap_bytes'],
        reserve_disk_bytes=config['free_disk_reserve_bytes'])
    audit=AuditIndex(root/'audit','hf-synthetic-smoke-v1',DiskGuard(root/'audit',config['audit_cap_bytes'],config['free_disk_reserve_bytes']))
    try:
        result=publish_month(root/'candidates/2026-01.selection.json',data_config,frozen,uploader,audit,
            run_id='hf-smoke-v1',synthetic=True,max_file_bytes=config['max_shard_bytes'],max_games=128)
        reproduction=reproduce_month(result['complete'],data_config,frozen,uploader,run_id='hf-smoke-v1',synthetic=True)
        catalog=publish_docs(uploader,[result],run_id='hf-smoke-v1',synthetic=True,fully_reproduced=True)
        Path('reports/m1_hf_dataset_card.md').write_text((uploader.root/'README.md').read_text())
        journal=json.loads(uploader.journal.read_text())
        revision=catalog['revision']
        for receipt in journal['files'].values():
            sha,size=uploader._transport().hash_remote(receipt['path_in_repo'],revision,receipt['bytes'])
            if (sha,size)!=(receipt['sha256'],receipt['bytes']):
                raise UploadError('Final catalog revision does not contain the verified artifact set')
        parquet_bytes=sum(file['bytes'] for shard in result['complete']['shards'] for file in shard['files'])
        report=dict(status='PASS_LIVE_HF_SMOKE' if live else 'PASS_OFFLINE_TRANSPORT_ONLY',
            repo_url=catalog['url'] if live else None,configured_repo_url=catalog['url'],
            visibility='private' if config['private'] else 'public',
            revision=revision if live else None,offline_fixture_revision=None if live else revision,
            verified_artifact_bytes=sum(v['bytes'] for v in journal['files'].values()),
            verified_artifacts=len(journal['files']),parquet_bytes=parquet_bytes,
            synthetic_counts=catalog['manifest']['counts'],real_50k_counts=dict(train=0,validation=0,test=0),
            parity_states=result['complete']['parity_states'],parity_mismatches=0,
            frozen_fixture_excluded_rows=result['complete']['dataset_d_excluded_positions'],
            second_feature_build=reproduction['status'],shard_digest=result['shard_digest'],
            verification='actual SDK HTTPS upload and streamed remote SHA256' if live else 'in-memory Hub double; NOT an actual Hugging Face upload',
            remaining_local_parquet_files=len(list((root/'spool').glob('*.parquet'))),
            remaining_local_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
            full_archive_scans_started=0,real_source_checksums_verified=0,
            tokens_logged=False)
    finally:audit.close()
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/hf_dataset.yaml')
    parser.add_argument('--upload',action='store_true')
    parser.add_argument('--output',default='reports/m1_hf_smoke.json')
    args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text())
    try:
        if args.upload:
            # Detect missing auth before generating any disk artifacts.
            from tinychess.data.hf_upload import HubTransport
            HubTransport(config['repo_id'],config['private'])
            report=run('data/processed/m1_hf_live_smoke',config,True)
        else:
            with tempfile.TemporaryDirectory(prefix='gambit-zero-hf-offline-') as root:
                report=run(root,config,False)
                report['offline_temporary_files_removed']=True
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    except UploadError as exc:
        # Our controlled UploadError messages contain no token or HTTP headers.
        blocked_auth=str(exc).startswith('Authenticate locally')
        Path(args.output).write_text(json.dumps(dict(
            status='BLOCKED_AUTHENTICATION' if blocked_auth else 'LIVE_HF_CHECK_NOT_COMPLETED',
            configured_repo_url='https://huggingface.co/datasets/'+config['repo_id'],
            repo_url=None,revision=None,uploaded_bytes=None,verification_completed=False,
            reason=str(exc),full_archive_scans_started=0,
            local_unverified_artifacts_retained=True),indent=2)+'\n')
        print(str(exc))
        raise SystemExit(1)
