"""Two-pass accepted-month gate -> bounded HF outbox, no month-sized features."""
import hashlib
import io
import json
from pathlib import Path
import re
import time

import chess
import zstandard as zstd

from tinychess.chess.action_codec import ACTION_MAP_VERSION
from tinychess.chess.pgx_env import RULES_POLICY_VERSION
from tinychess.chess.state_features import ENCODER_VERSION
from tinychess.data.archive_scan import ALGORITHM,DaySampler,ScanError,canonical,atomic_json
from tinychess.data.human import replay_game,split_for_month
from tinychess.data.hf_shards import ShardOutbox,validate_rows
from tinychess.data.sources import sha256_file
from tinychess.validation.parity import board_identity


def logical_game_sha(game):
    # Logical metadata and tensor digests are checked separately across builds.
    rows=[{k:(v.hex() if isinstance(v,bytes) else v) for k,v in row.items()
           if k not in ['planes','context','legal_mask']} for row in game.positions]
    return hashlib.sha256(canonical(dict(game=game.game,rows=rows)).encode()).hexdigest()


def stable_receipt(value):
    return {k:value[k] for k in ['path_in_repo','sha256','bytes','revision','verification']}


def feature_sha(game):
    digest=hashlib.sha256()
    for row in game.positions:
        for key in ['planes','context','legal_mask']:
            digest.update(row[key])
    return digest.hexdigest()


def read_frozen_d(path,*,synthetic):
    if Path(path).stat().st_size>32*2**20:
        raise ScanError('Frozen evaluation manifest exceeds bounded 32 MiB metadata limit')
    value=json.loads(Path(path).read_text())
    if not value.get('version') or not value.get('position_fens'):
        raise ScanError('Frozen Dataset D needs a version and exact nonempty position list')
    required='synthetic_fixture' if synthetic else 'dataset_d_frozen'
    if value.get('kind')!=required:
        raise ScanError('Dataset D fixture/production kind mismatch')
    if not synthetic and (not value.get('puzzle_ids') or not value.get('opening_suite') or not value.get('sources')):
        raise ScanError('Production Dataset D requires puzzle IDs, opening suite and source provenance')
    for source in value.get('sources',[]):
        if not source.get('url') or not source.get('snapshot_date') or not re.fullmatch('[0-9a-f]{64}',source.get('sha256','')):
            raise ScanError('Invalid frozen evaluation source provenance')
    for fen in value['position_fens']:
        if not chess.Board(fen).is_valid():
            raise ScanError('Invalid frozen Dataset D position')
    return value,sha256_file(path)


def run_evaluation_inputs(config,path,*,synthetic):
    if not synthetic and config.get('dataset_d_policy')=='deferred':
        return {'position_fens':[]},None
    return read_frozen_d(path,synthetic=synthetic)


def publish_month(checkpoint_path,config,frozen_d_path,uploader,audit,*,run_id,synthetic=False,
                  max_file_bytes=32*2**20,max_games=128):
    started=time.perf_counter()
    timing={'parse_replay_parity_encode_seconds':0.0,'upload_verify_seconds':0.0,'candidate_validation_attempts':0}
    if not re.fullmatch('[A-Za-z0-9_-]+',run_id):
        raise ScanError('Unsafe run identifier')
    checkpoint_path=Path(checkpoint_path)
    checkpoint=json.loads(checkpoint_path.read_text())
    saved=checkpoint.pop('checkpoint_sha256',None)
    if saved!=hashlib.sha256(canonical(checkpoint).encode()).hexdigest():
        raise ScanError('Source candidate checkpoint checksum mismatch')
    identity,archive=checkpoint['identity'],checkpoint['archive']
    if (archive['full_archive_checksum_verified'] is not True or archive['sha256']!=identity['published_sha256']
            or archive['bytes']!=identity['archive_size_bytes']):
        raise ScanError('Month cannot upload before complete original compressed-source verification')
    selection=checkpoint['selection']
    if (identity['algorithm']!=ALGORITHM or selection['algorithm']!=ALGORITHM
            or selection['month']!=identity['month'] or selection['seed']!=identity['seed']
            or selection['reserve_factor']!=config['sampling']['reserve_factor']):
        raise ScanError('Frozen sampling algorithm/month/seed/reserves mismatch')
    if checkpoint['selection_digest']!=hashlib.sha256(canonical(selection).encode()).hexdigest():
        raise ScanError('Selection artifact digest mismatch')
    month=identity['month']
    if not synthetic:
        sources=json.loads(Path(config['source_manifest']).read_text())['sources']
        frozen_source=next((s for s in sources if s['month']==month),None)
        if not frozen_source or any(identity[k]!=frozen_source[k] for k in ['url','archive_size_bytes','published_sha256']):
            raise ScanError('Production checkpoint does not match the frozen official monthly source')
    quota=config['months'][month]['accepted_games']
    if identity['quota']!=quota or identity['seed']!=config['seed']:
        raise ScanError('Selection config/seed/quota mismatch')
    if not synthetic and quota!=(5000 if month<='2026-07' else 7500):
        raise ScanError('Production monthly quotas remain exactly 5K/7.5K')
    if config['months'][month]['split']!=split_for_month(month):
        raise ScanError('Chronological split assignment mismatch')
    if config['filters']['min_base_seconds']!=300 or config['filters']['min_rating']!=1800:
        raise ScanError('50K base-time/rating rules must remain 300 seconds / 1800 Elo')
    frozen,d_sha=run_evaluation_inputs(config,frozen_d_path,synthetic=synthetic)
    replay_config=dict(config,frozen_evaluation_fens=frozen['position_fens'])
    sampler=DaySampler(month,quota,config['seed'],reserve_factor=selection['reserve_factor'])
    for record in selection['candidates']:
        if hashlib.sha256(record['pgn'].encode()).hexdigest()!=record['raw_pgn_sha256']:
            raise ScanError('Cached candidate PGN checksum mismatch')
        sampler.offer(record['pgn'],record['source_ordinal'])
    def validate(candidate):
        uploader.check_runtime()
        tick=time.perf_counter()
        timing['candidate_validation_attempts']+=1
        try:
            replayed=replay_game(candidate['pgn'],month,candidate['source_ordinal'],identity['url'],replay_config)
        finally:
            timing['parse_replay_parity_encode_seconds']+=time.perf_counter()-tick
        validate_rows(replayed,month,materialized=True)
        m=replayed.game['metadata']
        if m['utc_date'].replace('.','-')!=candidate['day'] or m['game_id']!=candidate['game_id']:
            raise ScanError('Fully parsed identity/day differs from header selection')
        sha=logical_game_sha(replayed)
        audit.add(replayed,sha)
        return dict(logical_sha256=sha,feature_sha256=feature_sha(replayed),metadata=m,
            rows=len(replayed.positions),excluded=sum(not row['training_eligible'] for row in replayed.positions) if m['split']=='train' else 0)
    # ALL source/parity/outcome/exclusion/rating checks finish before ANY upload.
    audit.begin()
    try:
        selected,rejections,coverage=sampler.select(validate)
        if len(selected)!=quota or sum(min(v['metadata']['white_elo'],v['metadata']['black_elo'])>=2000 for c,v in selected)*5 < quota*3:
            raise ScanError('Accepted monthly count or 60% rating quota failed')
        if sum(v['rows']-v['excluded'] for c,v in selected)<=0:
            raise ScanError('Frozen evaluation exclusions leave no usable training records')
        audit.commit()
    except Exception:
        audit.rollback()
        raise
    selected_sha=hashlib.sha256(canonical([c['game_id'] for c,v in selected]).encode()).hexdigest()
    timing['validation_elapsed_seconds']=time.perf_counter()-started
    approval=dict(stage='VALIDATED_MONTH',kind='synthetic_fixture' if synthetic else 'human_dataset',month=month,
        split=split_for_month(month),accepted_games=quota,
        strong_games=sum(min(v['metadata']['white_elo'],v['metadata']['black_elo'])>=2000 for c,v in selected),
        pre_exclusion_positions=sum(v['rows'] for c,v in selected),
        dataset_d_excluded_positions=sum(v['excluded'] for c,v in selected) if d_sha else None,
        dataset_d_status='checked_frozen_fixture' if synthetic else 'not_checked_deferred' if d_sha is None else 'checked_frozen_snapshot',
        parity_states=sum(v['rows']+1 for c,v in selected),parity_mismatches=0,
        source=identity,source_checkpoint_sha256=saved,dataset_d_sha256=d_sha,
        source_scan_measurements=checkpoint.get('scan_measurements'),
        source_header_counts=selection['counts'],source_header_rejections=selection['header_rejections'],
        source_eligible_by_day=selection['eligible_occurrences_by_day'],
        implementation_sha256={p:sha256_file(p) for p in [
            'src/tinychess/chess/pgx_env.py','src/tinychess/chess/action_codec.py',
            'src/tinychess/chess/state_features.py','src/tinychess/data/human.py',
            'src/tinychess/data/hf_month.py','src/tinychess/data/hf_shards.py','uv.lock']},
        selected_ids_sha256=selected_sha,coverage=coverage,validation_rejections=rejections,
        schema='human_v3',encoder=ENCODER_VERSION,action_map=ACTION_MAP_VERSION,rules=RULES_POLICY_VERSION)
    # Freeze a small approval before publication. A repeated build with different
    # selection or rules cannot overwrite a previously verified month namespace.
    receipt_path=uploader.root/f'{run_id}-{month}.approval.json'
    approval_existed=receipt_path.exists()
    if approval_existed and json.loads(receipt_path.read_text())!=approval:
        raise ScanError('Month approval differs from previous build')
    partials=list(uploader.root.glob(f'{run_id}-{month}-*.parquet.partial'))
    if partials and not approval_existed:
        raise ScanError('Unowned unsealed shards found without a matching month approval; retained')
    for partial in partials:
        partial.unlink()  # Never sealed/uploaded; exact verified inputs regenerate it.
    atomic_json(receipt_path,approval,uploader.guard)
    prefix=('smoke/' if synthetic else 'data/')+run_id
    frozen_receipt=None
    if d_sha is not None:
        frozen_copy=uploader.root/f'frozen-d-{d_sha}.json'
        uploader.guard.check(Path(frozen_d_path).stat().st_size)
        frozen_copy.write_bytes(Path(frozen_d_path).read_bytes())
        frozen_receipt=stable_receipt(uploader.publish(frozen_copy,f'{prefix}/evaluation/frozen-d-{d_sha}.json'))
    shard_receipts=[]
    def on_shard(descriptor):
        tick=time.perf_counter()
        results=[]
        for file in descriptor['files']:
            name=f"{descriptor['part']:05}-{file['sha256']}.{file['kind']}.parquet"
            path=f"{prefix}/{file['kind']}/{descriptor['split']}/{month}/{name}"
            results.append(dict(kind=file['kind'],**stable_receipt(uploader.publish(file['path'],path))))
        result={k:v for k,v in descriptor.items() if k!='files'}
        result['files']=results
        shard_receipts.append(result)
        timing['upload_verify_seconds']+=time.perf_counter()-tick
        return result
    outbox=ShardOutbox(uploader.root,month,uploader.guard,on_shard,max_file_bytes=max_file_bytes,max_games=max_games,namespace=run_id)
    try:
        for candidate,summary in selected:
            uploader.check_runtime()
            tick=time.perf_counter()
            replayed=replay_game(candidate['pgn'],month,candidate['source_ordinal'],identity['url'],replay_config)
            timing['parse_replay_parity_encode_seconds']+=time.perf_counter()-tick
            if logical_game_sha(replayed)!=summary['logical_sha256'] or feature_sha(replayed)!=summary['feature_sha256']:
                raise ScanError('Second pass differs from fully validated month')
            outbox.append(replayed)
        outbox.close()
    except Exception:
        if outbox.position_writer:
            outbox.position_writer.close()
            outbox.position_sink.close()
        raise
    # Selected PGNs are small reproducibility inputs, not a month archive. Keep
    # only a bounded compressed selection file, then upload/verify/delete it.
    selection_path=uploader.root/f'{run_id}-{month}.selected.jsonl.zst'
    selection_tmp=selection_path.with_suffix(selection_path.suffix+'.partial')
    with selection_tmp.open('wb') as target:
        with zstd.ZstdCompressor(level=3).stream_writer(target,closefd=False) as writer:
            for candidate,summary in selected:
                uploader.guard.check(2**20)
                writer.write((canonical(dict(candidate=candidate,validation=summary))+'\n').encode())
                writer.flush(zstd.FLUSH_BLOCK)
                if target.tell()>max_file_bytes:
                    raise ScanError('Selected PGN cache exceeds bounded artifact cap; stop before next month')
    if selection_tmp.stat().st_size>max_file_bytes:
        raise ScanError('Sealed selected-PGN input exceeds artifact cap; retained')
    if selection_path.exists():
        if sha256_file(selection_path)!=sha256_file(selection_tmp):
            raise ScanError('Retained selection differs from regeneration; neither copy silently discarded')
        selection_tmp.unlink()
    else:
        selection_tmp.replace(selection_path)
    selection_receipt=stable_receipt(uploader.publish(selection_path,f'{prefix}/provenance/{month}/selected.jsonl.zst'))
    approval_receipt=stable_receipt(uploader.publish(receipt_path,f'{prefix}/provenance/{month}/approval.json',delete_after_verify=False))
    complete=dict(**approval,status='COMPLETED_MONTH',shards=shard_receipts,selection=selection_receipt,
                  approval=approval_receipt,
                  frozen_dataset_d=frozen_receipt,
                  sharding=dict(max_file_bytes=max_file_bytes,max_games=max_games),
                  training_positions=sum(d['rows'] for d in shard_receipts),
                  artifact_bytes=sum(f['bytes'] for d in shard_receipts for f in d['files']))
    complete['shard_digest']=hashlib.sha256(canonical([f['sha256'] for d in shard_receipts for f in d['files']]).encode()).hexdigest()
    complete_path=uploader.root/f'{run_id}-{month}.complete.json'
    # Receipt timestamps/revisions may differ between repeated executions; a
    # separate deterministic comparison is based on shard checksums and IDs.
    atomic_json(complete_path,complete,uploader.guard)
    result=uploader.publish(complete_path,f'{prefix}/manifests/{month}.json',delete_after_verify=False,allow_update=False)
    timing['elapsed_seconds']=time.perf_counter()-started
    timing['accepted_games']=quota
    timing['training_positions']=complete['training_positions']
    timing['positions_per_second_including_upload']=complete['training_positions']/timing['elapsed_seconds']
    timing['accepted_games_per_second_including_upload']=quota/timing['elapsed_seconds']
    timing['pure_parser_seconds']=None
    timing['pure_pgx_kernel_seconds']=None
    atomic_json(uploader.root/f'{run_id}-{month}.measurements.json',timing,uploader.guard)
    return dict(month=month,complete=complete,manifest_receipt=result,
                shard_digest=complete['shard_digest'])


def reproduce_month(complete,config,frozen_d_path,uploader,*,run_id,synthetic=False,audit=None):
    """Second independent feature build from verified cloud selection, no archive.

    A selected-game input is <=32 MiB compressed in RAM; decoded JSONL is read
    one game at a time. No second complete dataset copy/download cache exists.
    """
    if not re.fullmatch('[A-Za-z0-9_-]+',run_id):
        raise ScanError('Unsafe run identifier')
    frozen,d_sha=run_evaluation_inputs(config,frozen_d_path,synthetic=synthetic)
    if d_sha!=complete['dataset_d_sha256']:
        raise ScanError('Reproduction Dataset D snapshot differs')
    payload=uploader._transport().read_verified_small(complete['selection'])
    month=complete['month']
    prefix=('smoke/' if synthetic else 'data/')+run_id
    hashes,ids=[],[]
    shard_index=0
    def on_shard(descriptor):
        nonlocal shard_index
        if (shard_index>=len(complete['shards']) or
                [f['sha256'] for f in descriptor['files']] != [f['sha256'] for f in complete['shards'][shard_index]['files']]):
            raise ScanError('Regenerated shard differs before any reproduction upload')
        shard_index+=1
        for file in descriptor['files']:
            remote=f"{prefix}/{file['kind']}/{descriptor['split']}/{month}/{descriptor['part']:05}-{file['sha256']}.{file['kind']}.parquet"
            # Existing immutable objects must compare equal; publish is idempotent.
            uploader.publish(file['path'],remote)
            hashes.append(file['sha256'])
    for partial in uploader.root.glob(f'{run_id}-{month}-*.parquet.partial'):
        partial.unlink()  # Verified cloud input + original digest permit safe rebuild.
    writer=ShardOutbox(uploader.root,month,uploader.guard,on_shard,**complete['sharding'],namespace=run_id)
    if audit:
        audit.begin()
    try:
        with zstd.ZstdDecompressor().stream_reader(io.BytesIO(payload)) as decoded:
            with io.TextIOWrapper(decoded,encoding='utf-8') as text:
                while True:
                    line=text.readline(4*2**20+1)
                    if not line:
                        break
                    if len(line)>4*2**20:
                        raise ScanError('Selected-game JSONL record exceeds bounded line')
                    record=json.loads(line)
                    uploader.check_runtime()
                    candidate,summary=record['candidate'],record['validation']
                    if hashlib.sha256(candidate['pgn'].encode()).hexdigest()!=candidate['raw_pgn_sha256']:
                        raise ScanError('Reproduction PGN hash differs')
                    game=replay_game(candidate['pgn'],month,candidate['source_ordinal'],complete['source']['url'],
                                     dict(config,frozen_evaluation_fens=frozen['position_fens']))
                    if logical_game_sha(game)!=summary['logical_sha256'] or feature_sha(game)!=summary['feature_sha256']:
                        raise ScanError('Reproduction logical game records differ')
                    if audit:
                        audit.add(game,summary['logical_sha256'])
                    writer.append(game)
                    ids.append(candidate['game_id'])
        writer.close()
    except Exception:
        if writer.position_writer:
            writer.position_writer.close()
            writer.position_sink.close()
        if audit:
            audit.rollback()
        raise
    digest=hashlib.sha256(canonical(hashes).encode()).hexdigest()
    ids_sha=hashlib.sha256(canonical(ids).encode()).hexdigest()
    if digest!=complete['shard_digest'] or ids_sha!=complete['selected_ids_sha256'] or len(ids)!=complete['accepted_games']:
        if audit:
            audit.rollback()
        raise ScanError('Second build IDs/shard hashes/count differ')
    if audit:
        audit.commit()
    result=dict(status='REPRODUCTION_PASS',month=month,games=len(ids),shard_digest=digest,
                selected_ids_sha256=ids_sha,source_selection_sha256=complete['selection']['sha256'])
    path=uploader.root/f'{run_id}-{month}.reproduction.json'
    atomic_json(path,result,uploader.guard)
    result['receipt']=stable_receipt(uploader.publish(path,f'{prefix}/manifests/{month}.reproduction.json',delete_after_verify=False))
    return result
