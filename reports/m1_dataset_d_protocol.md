# Frozen Dataset D protocol — 29 October 2026

This is an evaluation-only fixture freeze, not model evaluation or training.
The user deferred Dataset D while Dataset A was extracted, then authorized
finishing the M1 checklist. The original pre-extraction ordering was therefore
not met. Preserve the original Dataset A revision and publish a lossless filtered
derivative before any model training; never retroactively claim original exclusions.

## Selection fixed before overlap inspection

Configuration: `configs/dataset_d.yaml`, seed **20261029**,
`sha256-canonical-seed-kind-id-bottomk-v1`.

- Puzzles: 200 each in rating bands [0,1000), [1000,1500), [1500,2000),
  [2000,2500), [2500,infinity): **1,000 IDs**. Require popularity >=0,
  at least 100 plays, rating deviation <=100. These are chosen quality criteria,
  not a representative sample of every Lichess puzzle/player.
- Consume the complete official compressed export, retaining bounded hash-priority
  reservoirs. Validate every selected FEN and every solution move. An invalid
  selected solution or quota shortfall stops the freeze; do not tune selection
  after looking at Dataset A overlap or model results.
- Openings: **32** exact roots, 7/7/6/6/6 across ECO A/B/C/D/E source files,
  from pinned commit `e76b8b4ec5c8aeeba41dac6caf4a60b938f652db` of
  [lichess-org/chess-openings](https://github.com/lichess-org/chess-openings/tree/e76b8b4ec5c8aeeba41dac6caf4a60b938f652db).
  Keep 8–24-ply nonterminal legal lines. Select one hash-ranked representative
  per name prefix before `:`, then the quota per file; reject duplicate final
  roots. Name prefixes are an explicit grouping heuristic, not a comprehensive
  chess taxonomy. Paired-color match use is frozen, not run here.

The full puzzle stream contained **6,157,341 rows**; 4,461,898 met the frozen
metadata filters. Full source SHA256 is
`76335bfa7d7c4a7f93c1366d81549e53951ebb79dd43d34904cab8f22d962f8d`.
It is a computed fingerprint of the entire **307,234,795-byte** compressed stream,
not a publisher checksum. HTTP ETag, Last-Modified, retrieval date, counts and
the immutable, verified HF receipt are in `m1_dataset_d_freeze.json`.
The post-run configuration pins that computed source fingerprint for reruns.
The immutable first-freeze artifact records the original configuration before
that fingerprint was available. No source archive was retained locally.

## Root and solution semantics

The [official puzzle format](https://database.lichess.org/#puzzles) places the FEN
**before the opponent setup move**. Apply the first UCI move to obtain the solver
root; the remaining moves form the solution line. The freezer checks all moves
for legality and stores the setup, root, full solution and original source row.
It does not claim to independently prove strategic uniqueness/optimality.
Later puzzle scoring must measure complete-line success and accept any actual
checkmating move in a mate-in-one fixture, as documented by Lichess.

The frozen source rows/IDs, canonical selected-row hashes and complete legal
opening trajectories are stored in the evaluation artifact, **separately** from
Dataset A. Puzzle solutions must never become Dataset A policy/value targets.
No engine labeling, inference, model selection or playing-strength test ran.

## Exact exclusion contract

Use SHA256 of pieces, side to move, castling rights and legally capturable
en passant, matching the accepted Dataset A board-identity contract. Ignore
clocks and repetition history conservatively for overlap. Puzzle FENs do not
recover prior repetition rights; do not fabricate a complete history from them.
Opening lines do preserve their history from the standard initial state.

- Puzzle exclusion set: root plus every solution pre-move board, including
  opponent response states; not every related position from the source game.
- Opening exclusion set: exact final root only; not all related opening families
  or every opening prefix.
- Deduplicate by the stated board identity: **4,178 unique evaluation boards**.
- Count original Dataset A/D matches separately by train/validation/test, overall
  and at ply >=10. Also report source game-ID overlap separately.
- Physically remove exact matching training rows. Keep accepted game identities,
  complete action provenance and chronological held-out rows; report held-out
  overlap. Compact game replay requires the same exclusions before training.
- Compare every retained scalar/string/tensor byte with the verified original.
  Filter changed shards twice, check identical output hashes, upload, read back
  at an immutable revision and verify SHA256/size before deleting local output.
  Unaffected shards reuse the original verified hashes.
- Recompute split/D overlap after exclusions; require zero train/D overlap.
  The original diagnostic subset remains conservatively disjoint from the reduced
  training set, but is not claimed maximal against that derivative.

## Actual status and reproduction

Freeze, exclusions and independent publication verification are **COMPLETE**.
`m1_dataset_d_exclusions.json` and
`m1_dataset_d_publication_verification.json` both have status `PASS`.
Final revision: `b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c`.

Exactly **150 training rows covering 14 unique evaluation boards** were removed.
All 50,000 game identities are preserved; retained positions total **3,793,600**.
Training/D overlap is zero, overall and at ply >=10. Held-out overlaps remain
reported: validation 36 rows / 13 unique boards; test 24 rows / 6 unique boards.
No puzzle-source game IDs are shared with any Dataset A split.
All 796 final primary shard objects have matching LFS SHA256/size metadata;
small manifests/card were fully read and checked. Changed shards had full body
read-back verification during upload and byte-identical repeated filtering.

The first exclusion attempt stopped on an upload error; one idempotent retry
confirmed HTTP 429 with a 3,471-second server delay. See
`m1_dataset_d_upload_diagnostic.json` for the exact no-earlier-than retry time.
The catalog stayed unchanged throughout that failed attempt. Its sealed local
shard was retained until the successful continuation verified it remotely.
No automated retry job is running.
The prepared continuation groups at most 8 files / 64 MiB into one commit,
retains per-file SHA256/size verification, and publishes the card/catalog atomically.
Those paths passed local failure/recovery tests and the live continuation after
the server cooldown. An import-order/no-Xet guard failure discovered in
targeted tests was corrected before final regression; no chess-rule failure occurred.

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/freeze_dataset_d.py
.venv/bin/python scripts/freeze_dataset_d.py --execute --publish
.venv/bin/python scripts/apply_dataset_d_exclusions.py
.venv/bin/python scripts/apply_dataset_d_exclusions.py --execute
.venv/bin/python scripts/verify_m1_dataset_d_publication.py
.venv/bin/python scripts/report_m1_pilot.py
```

Default commands preflight only. Freeze execution streams another 307 MB;
reusing the immutable published suite avoids that transfer. A source snapshot
change or SHA256 mismatch must stop reproduction rather than silently update v1.
Exclusion execution requires the completed original 50K metadata/index audit.
It reads bounded existing cloud shards, never the monthly Lichess archives.
All Git commands and the M2 decision remain the user's responsibility.
