"""Freeze a small derived sample from verified Dataset A; never modify originals."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from tinychess.data.training_loader import (VerifiedStore, inventory, iter_records,
    parquet_batches, validate_game, GAME_SCHEMA, collate, sha256, require, REVISION)

FIELDS = ('planes','context','legal_mask','action','wdl_class','policy_weight',
          'value_weight','training_eligible')


def rank(seed, *parts):
    return hashlib.sha256(json.dumps([seed,*parts],separators=(',',':')).encode()).digest()


def logical_hash(arrays, identities):
    h=hashlib.sha256(json.dumps(identities,sort_keys=True,separators=(',',':')).encode())
    for name in FIELDS:
        x=np.ascontiguousarray(arrays[name])
        h.update(name.encode());h.update(x.dtype.str.encode());h.update(str(x.shape).encode());h.update(x.tobytes())
    return h.hexdigest()


def choose_rows(records, candidates, size, seed):
    """One hash-ranked eligible position per candidate game, then unique inputs.

    This sample-only diversity filter is explicit; Dataset A is not deduplicated.
    The caller consumes complete shards, so existing count checks still execute.
    """
    best={};scanned=0
    for row in records:
        scanned+=1
        if not row['training_eligible'] or row['game_id'] not in candidates:continue
        key=rank(seed,'position',row['game_id'],row['ply'])
        old=best.get(row['game_id'])
        if old is None or key<old[0]:best[row['game_id']]=(key,row)
    selected=[];seen={};duplicates=conflicts=0
    for gid in candidates:
        if gid not in best:continue
        row=best[gid][1]
        # Only model inputs define equivalence, not labels or rule-state hashes.
        key=hashlib.sha256(row['planes']+row['context']).digest()
        label=(row['action'],row['wdl_class'] if row['value_weight'] else None)
        if key in seen:
            duplicates+=1;conflicts+=int(seen[key]!=label);continue
        seen[key]=label;selected.append(row)
        if len(selected)==size:break
    require(len(selected)==size,'Not enough distinct eligible candidate-game inputs')
    return selected,dict(scanned_positions=scanned,duplicate_inputs_excluded=duplicates,
                         conflicting_duplicate_inputs_excluded=conflicts,candidate_games_with_rows=len(best))


def build_sample(store_root, output, *, size=1024, seed=20261031):
    require(type(size) is int and 1<=size<=1024,'Stage 3A sample is bounded at 1024')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    require(not (output/'sample.npz').exists(),'Sample already exists; use a fresh directory')
    store=VerifiedStore(store_root,download=False)
    shards,catalog=inventory(store);train=[s for s in shards if s.split=='train']
    games=[]
    for shard in train:
        with store.file(shard.games) as path:
            for chunk in parquet_batches(path,GAME_SCHEMA,128):
                for row in chunk:
                    game,_=validate_game(row,shard);games.append(game['game_id'])
    require(len(games)==35000 and len(set(games))==35000,'Training game inventory mismatch')
    candidates=sorted(games,key=lambda gid:(rank(seed,'game',gid),gid))[:2*size]
    rows,coverage=choose_rows(iter_records(store,train,'train'),candidates,size,seed)
    original=collate(rows,'train')
    arrays={k:np.asarray(getattr(original,k)) for k in FIELDS}
    identities=[dict(game_id=r['game_id'],ply=r['ply'],month=r['source_month'],
                     position_hash=r['position_hash'].hex()) for r in rows]
    # Host validation only; retain original float32 features and all loss weights.
    from tinychess.model.objectives import validate_batch
    validate_batch(arrays)
    target=output/'sample.npz';np.savez_compressed(target,**arrays)
    sources=[dict(month=s.month,positions_path=s.positions.path,positions_sha256=s.positions.sha256,
                  games_path=s.games.path,games_sha256=s.games.sha256) for s in train]
    manifest=dict(format='gambit-zero-overfit-sample-v1',dataset_revision=REVISION,split='train',
        seed=seed,positions=size,games=len(set(original.game_id)),history_dropout=0.,
        algorithm='SHA256-ranked 2N games from all 35K; minimum position hash rank per game; first N unique encoded inputs',
        sample_sha256=sha256(target),logical_sha256=logical_hash(arrays,identities),identities=identities,
        source_shards=sources,source_counts=catalog['counts']['train'],coverage=coverage,
        months=dict(Counter(original.source_month)),wdl_classes=dict(Counter(map(int,original.wdl_class))),
        value_masked_rows=int((original.value_weight==0).sum()),downloaded_bytes=store.downloaded_bytes,
        sample_bytes=target.stat().st_size)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest


def load_sample(directory):
    directory=Path(directory);m=json.loads((directory/'manifest.json').read_text())
    require(m['format']=='gambit-zero-overfit-sample-v1' and m['dataset_revision']==REVISION
            and m['split']=='train' and m['history_dropout']==0.,'Wrong sample provenance')
    require(sha256(directory/'sample.npz')==m['sample_sha256'],'Sample checksum mismatch')
    with np.load(directory/'sample.npz',allow_pickle=False) as z:
        require(set(z.files)==set(FIELDS),'Wrong sample tensor fields');arrays={k:z[k] for k in FIELDS}
    require(logical_hash(arrays,m['identities'])==m['logical_sha256'],'Logical sample mismatch')
    require(len(arrays['action'])==m['positions']==len(m['identities'])<=1024,'Sample count mismatch')
    require(len({r['game_id'] for r in m['identities']})==m['games']==m['positions'],'Expected one position per game')
    require(all(r['month'] in {f'2026-0{i}' for i in range(1,8)} for r in m['identities']),'Wrong sample month')
    from tinychess.model.objectives import validate_batch
    validate_batch(arrays)
    return arrays,m
