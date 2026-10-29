import copy
import hashlib
import io
import json
from pathlib import Path
import struct

import chess
import pytest
import yaml
import zstandard as zstd

from tinychess.data.archive_scan import DiskGuard,PZSTD_MAGIC,ScanError,canonical,scan_month
from tinychess.data.hf_audit import AuditIndex
from tinychess.data.hf_month import publish_month,reproduce_month
from tinychess.data.hf_shards import ShardOutbox
from tinychess.data.hf_upload import UploadError,VerifiedUploader


class FakeHub:
    def __init__(self,private=False):
        self.private=private
        self.files={}
        self.snapshots={'0'*40:{}}
        self.revision='0'*40
        self.uploads=0
        self.fail_upload=False
        self.fail_verify=False
        self.lost_response=False
        self.on_ensure=None

    def ensure_repo(self):
        if self.on_ensure: self.on_ensure()
        return self.revision

    def head(self): return self.revision
    def exists(self,path,revision): return path in self.snapshots[revision]

    def upload(self,local,path,parent):
        assert parent==self.revision
        if self.fail_upload: raise RuntimeError('hf_TEST_SECRET must never be exposed')
        self.uploads+=1
        self.files[path]=Path(local).read_bytes()
        self.revision=f'{self.uploads:040x}'
        self.snapshots[self.revision]=dict(self.files)
        if self.lost_response:
            self.lost_response=False
            raise RuntimeError('response lost hf_TEST_SECRET')
        return self.revision

    def hash_remote(self,path,revision,max_bytes):
        payload=self.snapshots[revision][path]
        if len(payload)>max_bytes: raise UploadError('Remote larger than expected')
        return ('0'*64 if self.fail_verify else hashlib.sha256(payload).hexdigest()),len(payload)

    def read_verified_small(self,receipt,cap_bytes=32*2**20):
        payload=self.snapshots[receipt['revision']][receipt['path_in_repo']]
        assert len(payload)<=cap_bytes
        assert hashlib.sha256(payload).hexdigest()==receipt['sha256']
        return payload


def uploader(tmp_path,hub):
    return VerifiedUploader('fixture/gambit-zero',tmp_path/'spool',transport=hub,reserve_disk_bytes=1)


def test_upload_readback_before_delete_and_retry_idempotence(tmp_path):
    hub=FakeHub(); up=uploader(tmp_path,hub)
    local=up.root/'small.parquet';local.write_bytes(b'valid-small-test')
    receipt=up.publish(local,'smoke/small.parquet')
    assert not local.exists() and receipt['status']=='deleted'
    assert receipt['sha256']==hashlib.sha256(b'valid-small-test').hexdigest()
    assert up.publish(local,'smoke/small.parquet')['sha256']==receipt['sha256']
    assert hub.uploads==1
    local.write_bytes(b'valid-small-test')
    up.publish(local,'smoke/small.parquet')
    assert hub.uploads==1 and not local.exists()


@pytest.mark.parametrize('failure',['upload','verify','lost_response'])
def test_failed_upload_never_deletes_and_resume_is_safe(tmp_path,failure):
    hub=FakeHub();up=uploader(tmp_path,hub)
    local=up.root/'small.parquet';local.write_bytes(b'test-shard')
    setattr(hub,{'upload':'fail_upload','verify':'fail_verify','lost_response':'lost_response'}[failure],True)
    with pytest.raises(UploadError) as exc: up.publish(local,'smoke/small.parquet')
    assert 'hf_TEST_SECRET' not in str(exc.value)
    assert local.exists()
    assert 'hf_TEST_SECRET' not in up.journal.read_text()
    hub.fail_upload=hub.fail_verify=False
    up.publish(local,'smoke/small.parquet')
    assert not local.exists() and hub.uploads==1


def test_conflicts_and_non_spool_files_cannot_be_deleted(tmp_path):
    hub=FakeHub();up=uploader(tmp_path,hub)
    outside=tmp_path/'outside';outside.write_bytes(b'keep')
    with pytest.raises(UploadError,match='managed'):up.publish(outside,'smoke/keep')
    local=up.root/'test';local.write_bytes(b'original')
    up.publish(local,'smoke/immutable')
    local.write_bytes(b'changed')
    with pytest.raises(UploadError,match='conflicting'):up.publish(local,'smoke/immutable')
    assert local.read_bytes()==b'changed'
    with pytest.raises(UploadError,match='Unsafe'):up.publish(local,'../outside')


def test_real_installed_sdk_contract_and_no_xet_cache():
    import inspect
    from huggingface_hub import HfApi,constants
    from huggingface_hub.utils import get_session
    for name in ['create_repo','repo_info','file_exists','upload_file','get_paths_info']:
        assert 'repo_type' in inspect.signature(getattr(HfApi,name)).parameters
    assert 'parent_commit' in inspect.signature(HfApi.upload_file).parameters
    assert {'operations','repo_type','parent_commit'} <= set(inspect.signature(HfApi.create_commit).parameters)
    assert 'follow_redirects' in inspect.signature(get_session().stream).parameters
    assert constants.HF_HUB_DISABLE_XET


def month_fixture(tmp_path):
    texts=[]
    for day in range(1,32):
        headers=dict(Event='Rated Rapid game',Site=f'https://lichess.org/{day:02}000000',
            WhiteElo='2100',BlackElo='2200',TimeControl='300+0',UTCDate=f'2026.01.{day:02}',
            Result='1-0',Termination='Time forfeit' if day%5==0 else 'Normal')
        texts.append(''.join(f'[{k} "{v}"]\n' for k,v in headers.items())+'\n1. e4 e5 1-0\n\n')
    frame=zstd.ZstdCompressor(write_checksum=True).compress(''.join(texts).encode())
    raw=struct.pack('<III',PZSTD_MAGIC,4,len(frame))+frame
    source=dict(month='2026-01',url='https://example.invalid/synthetic',archive_size_bytes=len(raw),published_sha256=hashlib.sha256(raw).hexdigest())
    class Response(io.BytesIO):
        status=200
        headers={'Content-Length':str(len(raw))}
    root=tmp_path/'candidates'
    scan_month(source,root,31,20261025,network_budget_bytes=len(raw),max_seconds=60,
        reserve_disk_bytes=1,opener=lambda *a,**kw:Response(raw))
    config=yaml.safe_load((Path(__file__).parents[1]/'configs/data_human_50k_stream.yaml').read_text())
    config['months']={'2026-01':dict(split='train',accepted_games=31)}
    frozen=tmp_path/'frozen.json'
    frozen.write_text(json.dumps(dict(kind='synthetic_fixture',version='test-d-v1',position_fens=[chess.STARTING_FEN])))
    guard=DiskGuard(tmp_path/'audit',2**30,1)
    audit=AuditIndex(tmp_path/'audit','synthetic-run',guard)
    return root/'2026-01.selection.json',config,frozen,audit


def test_month_gate_shards_cloud_reproduction_and_exact_exclusions(tmp_path):
    import pyarrow.parquet as pq
    checkpoint,config,frozen,audit=month_fixture(tmp_path)
    hub=FakeHub();up=uploader(tmp_path,hub)
    # Nothing can reach upload until ALL 31 games pass month correctness gate.
    hub.on_ensure=lambda: pytest.fail('upload preceded month validation') if audit.summary()['games_by_split'].get('train')!=31 else None
    result=publish_month(checkpoint,config,frozen,up,audit,run_id='fixture',synthetic=True,max_file_bytes=2*2**20,max_games=4)
    complete=result['complete']
    assert complete['accepted_games']==31 and complete['dataset_d_excluded_positions']==31
    assert complete['training_positions']==31 and complete['parity_states']==93
    assert complete['parity_mismatches']==0 and complete['strong_games']==31
    assert not list(up.root.glob('*.parquet'))
    assert len(complete['shards'])==8
    for path,payload in hub.files.items():
        if path.endswith('.positions.parquet'):
            rows=pq.read_table(io.BytesIO(payload)).to_pylist()
            assert all(row['ply']==1 and row['training_eligible'] for row in rows)
    uploads=hub.uploads
    reproduced=reproduce_month(complete,config,frozen,up,run_id='fixture',synthetic=True)
    assert reproduced['status']=='REPRODUCTION_PASS'
    assert reproduced['shard_digest']==result['shard_digest']
    assert hub.uploads==uploads+1  # only reproduction manifest, identical shards reused
    assert audit.summary()['all_plies']['train_rows']==31
    # Repeating the entire month is also idempotent, no audit double count.
    partial=up.root/'fixture-2026-01-00000.positions.parquet.partial'
    partial.write_bytes(b'crashed before sealing or uploading')
    again=publish_month(checkpoint,config,frozen,up,audit,run_id='fixture',synthetic=True,max_file_bytes=2*2**20,max_games=4)
    assert again['complete']==complete and again['shard_digest']==result['shard_digest']
    assert not partial.exists()
    audit.close()


def test_unverified_source_and_missing_dataset_d_block_all_uploads(tmp_path):
    checkpoint,config,frozen,audit=month_fixture(tmp_path)
    hub=FakeHub();up=uploader(tmp_path,hub)
    value=json.loads(checkpoint.read_text());value.pop('checkpoint_sha256')
    value['archive']['full_archive_checksum_verified']=False
    value['checkpoint_sha256']=hashlib.sha256(canonical(value).encode()).hexdigest()
    checkpoint.write_text(canonical(value))
    with pytest.raises(ScanError,match='original compressed'):
        publish_month(checkpoint,config,frozen,up,audit,run_id='fixture',synthetic=True)
    assert hub.uploads==0
    audit.close()


def test_parity_failure_aborts_month_before_upload(tmp_path,monkeypatch):
    from tinychess.validation.parity import ParityError
    import tinychess.data.hf_month as module
    checkpoint,config,frozen,audit=month_fixture(tmp_path)
    hub=FakeHub();up=uploader(tmp_path,hub)
    def fail(*a,**kw):raise ParityError('injected rules mismatch')
    monkeypatch.setattr(module,'replay_game',fail)
    with pytest.raises(ParityError):
        publish_month(checkpoint,config,frozen,up,audit,run_id='fixture',synthetic=True)
    assert hub.uploads==0 and audit.summary()['games_by_split']=={}
    audit.close()


def test_visibility_is_checked_with_real_sdk_adapter_before_any_upload(monkeypatch):
    from types import SimpleNamespace
    from tinychess.data.hf_upload import HubTransport
    adapter=object.__new__(HubTransport)
    adapter.repo_id='fixture/gambit-zero';adapter.private=True
    class Api:
        def create_repo(self,*a,**kw): assert kw['private'] is True
        def repo_info(self,*a,**kw):return SimpleNamespace(private=False,sha='0'*40)
    adapter.api=Api()
    with pytest.raises(UploadError,match='visibility'):adapter.ensure_repo()


def test_reproduction_feature_mismatch_stops_before_new_upload(tmp_path,monkeypatch):
    import tinychess.data.hf_month as module
    checkpoint,config,frozen,audit=month_fixture(tmp_path)
    hub=FakeHub();up=uploader(tmp_path,hub)
    result=publish_month(checkpoint,config,frozen,up,audit,run_id='fixture',synthetic=True)
    old=hub.uploads
    monkeypatch.setattr(module,'feature_sha',lambda game:'0'*64)
    with pytest.raises(ScanError,match='logical game'):
        reproduce_month(result['complete'],config,frozen,up,run_id='fixture',synthetic=True)
    assert hub.uploads==old
    audit.close()


def test_compact_exact_overlap_audit_and_split_isolation(tmp_path):
    from types import SimpleNamespace
    from tinychess.validation.parity import board_identity
    key=hashlib.sha256(board_identity(chess.Board()).encode()).digest()
    audit=AuditIndex(tmp_path/'audit','test-signature',DiskGuard(tmp_path/'audit',2**30,1))
    def game(split,month,identity):
        return SimpleNamespace(game=dict(metadata=dict(game_id=identity,source_month=month,split=split)),
            positions=[dict(position_hash=key,ply=12,training_eligible=split!='train'),
                       dict(position_hash=b'x'*32,ply=10,training_eligible=True)])
    audit.begin()
    for split,month,id in [('train','2026-01','a'),('validation','2026-08','b'),('test','2026-09','c')]:
        audit.add(game(split,month,id),'same-logical-'+id)
    audit.commit()
    report=audit.summary([chess.STARTING_FEN])
    assert report['all_plies']['all_three_shared']==1
    assert report['ply_ge_10']['train_validation_shared']==1
    assert report['all_plies']['train_dataset_d_shared_unique']==0
    assert report['all_plies']['validation_dataset_d_shared_unique']==1
    audit.begin()
    with pytest.raises(ScanError,match='split/content'):
        audit.add(game('test','2026-09','a'),'different')
    audit.rollback();audit.close()


def test_sdk_exception_details_and_expired_runtime_never_escape(tmp_path):
    from tinychess.data.hf_upload import HubTransport
    adapter=object.__new__(HubTransport)
    adapter.repo_id='fixture/gambit-zero';adapter.private=False
    class Api:
        def create_repo(self,*a,**kw):raise RuntimeError('hf_TEST_SECRET')
    adapter.api=Api()
    with pytest.raises(UploadError) as exc:adapter.ensure_repo()
    assert 'hf_TEST_SECRET' not in str(exc.value)
    hub=FakeHub();up=uploader(tmp_path,hub);up.deadline=0
    file=up.root/'keep';file.write_bytes(b'local')
    with pytest.raises(UploadError,match='runtime'):up.publish(file,'smoke/keep')
    assert file.exists() and hub.uploads==0


def test_month_upload_failure_retains_sealed_game_shard_and_blocks_generation(tmp_path):
    checkpoint,config,frozen,audit=month_fixture(tmp_path)
    hub=FakeHub();up=uploader(tmp_path,hub)
    original=hub.upload
    def fail_games(local,path,parent):
        if path.endswith('.games.parquet'):raise RuntimeError('upload interrupted')
        return original(local,path,parent)
    hub.upload=fail_games
    with pytest.raises(UploadError):
        publish_month(checkpoint,config,frozen,up,audit,run_id='fixture',synthetic=True)
    assert list(up.root.glob('*.games.parquet'))
    assert not (up.root/'fixture-2026-01.complete.json').exists()
    assert checkpoint.exists()
    hub.upload=original
    result=publish_month(checkpoint,config,frozen,up,audit,run_id='fixture',synthetic=True)
    assert result['complete']['accepted_games']==31
    assert not list(up.root.glob('*.parquet'))
    audit.close()


def test_dataset_a_deferred_policy_reads_no_evaluation_artifact(monkeypatch):
    import tinychess.data.hf_month as module
    def forbidden(*a,**kw):pytest.fail('Dataset D must not be accessed in deferred Dataset A run')
    monkeypatch.setattr(module,'read_frozen_d',forbidden)
    value,digest=module.run_evaluation_inputs({'dataset_d_policy':'deferred'},None,synthetic=False)
    assert value=={'position_fens':[]} and digest is None


def test_source_provenance_progress_can_update_without_overwriting_shards(tmp_path):
    hub=FakeHub();up=uploader(tmp_path,hub)
    file=up.root/'source_provenance.json'
    file.write_text('{"verified": false}')
    up.publish(file,file.name,delete_after_verify=False,allow_update=True)
    file.write_text('{"verified": true}')
    up.publish(file,file.name,delete_after_verify=False,allow_update=True)
    assert hub.files[file.name]==b'{"verified": true}'
    with pytest.raises(UploadError,match='Only explicit'):
        up.publish(file,'data/immutable.parquet',allow_update=True)


def test_hub_payload_allowance_is_durable_and_bounded(tmp_path):
    hub=FakeHub();up=uploader(tmp_path,hub);up.transfer_budget_bytes=10
    up.reserve_transfer(6)
    with pytest.raises(UploadError,match='payload allowance'):up.reserve_transfer(5)
    assert json.loads((up.root/'hub_transfer_budget.json').read_text())['reserved_payload_bytes']==6
