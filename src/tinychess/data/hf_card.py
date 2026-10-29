"""Dataset card and small reproducible contracts; no credentials in metadata."""
import json
from pathlib import Path

import yaml

from tinychess.data.archive_scan import atomic_json
from tinychess.data.hf_month import stable_receipt


def publish_docs(uploader,completed,*,run_id,synthetic=False,fully_reproduced=False,audit_report=None):
    counts={s:dict(games=0,positions=0) for s in ['train','validation','test']}
    groups={}
    months=[entry['complete'] if 'complete' in entry else entry for entry in completed]
    deferred=not synthetic and bool(months) and all(m.get('dataset_d_status')=='not_checked_deferred' for m in months)
    for month in months:
        counts[month['split']]['games']+=month['accepted_games']
        counts[month['split']]['positions']+=month['training_positions']
        for shard in month['shards']:
            for file in shard['files']:
                groups.setdefault(file['kind'],{}).setdefault(month['split'],[]).append(file['path_in_repo'])
    actual={k:v['games'] for k,v in counts.items()}
    final=(not synthetic and fully_reproduced and actual==dict(train=35000,validation=7500,test=7500))
    status='synthetic-smoke-only' if synthetic else 'complete-50k' if final else 'building-not-accepted'
    configs=[]
    for kind,splits in sorted(groups.items()):
        configs.append(dict(config_name=('smoke_' if synthetic else '')+('human_v3' if kind=='positions' else 'games'),
            default=kind=='positions',data_files=[dict(split=split,path=sorted(paths)) for split,paths in sorted(splits.items())]))
    metadata=dict(license='cc0-1.0',language=['en'],tags=['chess','jax','policy-value','human-games'])
    if configs:metadata['configs']=configs
    card='---\n'+yaml.safe_dump(metadata,sort_keys=False)+'---\n\n'
    card+=f'''# Gambit Zero — Human Chess Dataset A

**Status: {status}.** {'All uploaded games in this test are synthetic fixtures. No real 50K dataset exists yet. Do not train or evaluate on the smoke fixtures.' if synthetic else 'Only completed, checksum-verified monthly artifacts listed in the manifest are usable. Building datasets are not the accepted 50K pilot.'}

Repository visibility is {'private' if uploader.private else 'public'}, as requested by the owner.
Run: `{run_id}`. This repository stores dataset artifacts, not model weights.
No M2 model or self-play training is included.

## Size and splits actually published in this catalog

| Split | Games | Training positions |
|---|---:|---:|
'''
    card+='\n'.join(f"| {split} | {v['games']} | {v['positions']} |" for split,v in counts.items())+'\n'
    card+='''
Final real-data targets are 35,000 training games (5,000 each January–July 2026),
7,500 validation games (August), and 7,500 test games (September). A game never
crosses splits. The table above reports this catalog, not promised target counts.

## Sources, selection and verification

Real sources: public CC0 Lichess monthly standard-rated PGN archives,
https://database.lichess.org/. Exact URLs, published hashes and verification
status are in `source_provenance.json` and each completed month manifest.
Published archive hashes alone do not mean the complete sources were verified.
Synthetic smoke artifacts are explicitly marked and are not Lichess samples.

Streaming consumes original compressed bytes in bounded frames without retaining
full archives. Month checkpoints are accepted only after byte-count/EOF and full
SHA256 verification against the official monthly checksum. Sampling uses
`utc-day-rating-bottomk-sha256-v1`, seed 20261025, full UTC calendar-day strata,
deterministic SHA256 priorities and bounded reserves. At least 60% of accepted
games per day have both player ratings >=2000; all remaining games are >=1800.
The lower rating defines the band. Exhausted reserves/coverage gaps stop the run.

## Quality and targets

Standard rated chess from the initial position, base clock >=300 seconds
independent of increment, valid complete results and legal moves. Exclude
variants, abandoned/aborted/rules-infraction games, malformed metadata and games
that cannot be completely replayed under the 512-ply horizon. Time forfeits
retain policy targets and default value weight 0. Generic Normal endings are not
misrepresented as proven resignation/agreement. Labels are noisy human outcomes,
not strategic ground truth. No Stockfish/[%eval] labels are consumed.

Replay every accepted trajectory from the initial corrected Pgx state, preserving
history. Full legal actions, position/history, terminal status and reward parity
are checked against python-chess offline at every ply. Actual final checkmate is
checked against Pgx; non-board PGN results do not require environment termination.
Rules: `tinychess-auto-fide-pgx-v1`, no voluntary claims, fivefold/150-halfmove
automatic draws; truncations never supply WDL labels. Remaining dead-position and
hash-collision limitations are inherited from the accepted rules contract.

## Schema and encoder

`human_v3`; `schema.json` defines every physical field. Policy uses Pgx's native
4,672-action order, codec `pgx-2.6.0-chess-v2-uci-v1`. Encoder
`pgx-v2-history8-ep-context8-v1`: planes float32 (8,8,120), context float32 (8,),
legal mask bool (4672,). Feature bytes are little-endian float32; masks are
584 bytes, little bit order. See `encoder_contract.md` for channel/context order.
WDL class order is win/draw/loss (0/1/2), scalar +1/0/-1 from side to move.
Game metadata/action sequences are provenance; the primary training inputs are
the materialized positions configuration.

## Evaluation integrity and limitations

Before production upload, freeze versioned Dataset D puzzle IDs/source snapshot
and exact opening positions. Match by pieces, side to move, castling rights and
legally capturable en passant; ignore clocks/history conservatively. Exact Dataset
D positions are physically absent from training position shards. Game-level
provenance retains complete actions; do not turn it into an unfiltered training
reader. FEN cannot establish full repetition history. Report overlap of unique
boards between splits overall and at ply >=10 using the compact exact audit index;
repeated board positions are not necessarily duplicate games. Frozen evaluation
snapshots/hashes, exclusion counts and coverage are stored in monthly manifests.

## Upload, reproducibility and recovery

Small game-aligned Parquet shards are grouped by kind, split and source month.
Each immutable filename includes its full SHA256. Only a completed month is
listed in the catalog. Uploads use authenticated huggingface_hub; files are read
back at an immutable revision, streaming without a download cache, and compared
by SHA256 and byte count before the Mac copy is removed. An upload/read-back
failure retains the local file and halts generation of further shards.

A second feature build can consume the verified compact selected-PGN input from
Hugging Face, one bounded selection file/game at a time, and compare regenerated
shard hashes without a second local dataset. Reproduction is accepted only when
actually run; completed reproduction manifests record results. Store the upload
journal and completed-month receipts for recovery. Interrupted archive months
restart from byte zero; verified completed months are reused. No blind compressed
byte-range game access is assumed.

## Usage without downloading the whole dataset

Authenticate using `hf auth login`; never embed a token in a notebook or commit.
Install `datasets` in your consumer environment, then use its streaming mode:

```python
from datasets import load_dataset
# Use the configuration listed in this card; smoke data is synthetic only.
rows = load_dataset("ishagarg1103/gambit-zero", name="CONFIG_NAME",
                    split="train", streaming=True)
row = next(iter(rows))
```

Decode a position row (NumPy):

```python
import numpy as np
planes = np.frombuffer(row["planes"], dtype="<f4").reshape(8, 8, 120)
context = np.frombuffer(row["context"], dtype="<f4")
legal = np.unpackbits(np.frombuffer(row["legal_mask"], dtype=np.uint8),
                      bitorder="little").astype(bool)
assert legal[int(row["action"])]
```

Use policy_weight/value_weight separately, retain perspective, and pin the Hub
revision in any experiment. Outcome value loss must respect the timeout mask.
This dataset makes no claim of optimal moves, model strength or frontier-GPT wins.
'''
    name=('smoke_' if synthetic else '')+'human_v3'
    if deferred:
        start=card.index('Before production upload, freeze versioned Dataset D')
        end=card.index('\n## Upload, reproducibility and recovery',start)
        card=card[:start]+'''Dataset D is deferred and its overlap has NOT been checked. No Dataset D
positions were excluded. Game identities, exact board position hashes, game
actions and source metadata are retained for the later overlap audit. Before
model training, run that audit and publish a revised Dataset A if needed.
Split/game integrity and exact board-position overlap between train, validation
and test are audited independently, overall and at ply >=10. FEN alone cannot
recover complete repetition history; trajectories preserve it.
'''+card[end:]
    card=card.replace('name="CONFIG_NAME"',f'name="{name}"')
    root=uploader.root
    uploader.guard.check(len(card.encode())+256*2**10)
    card_path=root/'README.md';card_path.write_text(card)
    schema_path=root/'schema.json';schema_path.write_bytes(Path('configs/human_v3_schema.json').read_bytes())
    encoder_path=root/'encoder_contract.md';encoder_path.write_bytes(Path('reports/encoder_contract.md').read_bytes())
    sources=json.loads(Path('reports/m1_sources_50k.json').read_text())
    provenance=dict(source_archive_url_template='https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst',
        checksum_index=sources['checksum_index'],sources=[{k:v for k,v in s.items() if k in ['month','url','archive_size_bytes','published_sha256','full_archive_checksum_verified']} for s in sources['sources']],
        note='Preflight provenance only. Consult completed real-month manifests for actual full-file verification. Smoke fixtures are not archive samples.')
    if not synthetic:
        for entry in provenance['sources']:
            matching=next((m for m in months if m['month']==entry['month']),None)
            if matching:
                entry['full_archive_checksum_verified']=True
                entry['evidence_manifest']=f"data/{run_id}/manifests/{entry['month']}.json"
    p=root/'source_provenance.json';atomic_json(p,provenance,uploader.guard)
    manifest=dict(status=status,run_id=run_id,synthetic=synthetic,counts=counts,
        months=[dict(month=m['month'],accepted_games=m['accepted_games'],training_positions=m['training_positions'],shard_digest=m['shard_digest'],
                     source=m['source'],dataset_d_sha256=m['dataset_d_sha256'],
                     manifest_path=('smoke/' if synthetic else 'data/')+run_id+'/manifests/'+m['month']+'.json') for m in months],
        repo_id=uploader.repo_id,private=uploader.private,fully_reproduced=fully_reproduced)
    manifest['dataset_d_status']='not_checked_deferred' if deferred else 'synthetic_fixture_only' if synthetic else 'checked_frozen_snapshot'
    m=root/'dataset_manifest.json';atomic_json(m,manifest,uploader.guard)
    receipts=[]
    if audit_report is not None:
        payload=json.dumps(audit_report,sort_keys=True,indent=2)+'\n'
        import hashlib
        name='audit-'+hashlib.sha256(payload.encode()).hexdigest()+'.json'
        a=root/name
        a.write_text(payload)
        audit_receipt=stable_receipt(uploader.publish(a,'audits/'+name))
        manifest['audit']=audit_receipt
        atomic_json(m,manifest,uploader.guard)
    for path in [schema_path,encoder_path,p,m,card_path]:
        receipts.append(stable_receipt(uploader.publish(path,path.name,delete_after_verify=False,
            allow_update=path.name in ['README.md','dataset_manifest.json','source_provenance.json'])))
    return dict(manifest=manifest,revision=receipts[-1]['revision'],receipts=receipts,
                url=f'https://huggingface.co/datasets/{uploader.repo_id}')
