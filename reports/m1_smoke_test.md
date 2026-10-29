# Gambit Zero M1 — Dataset A smoke-test acceptance report

Date: 25 October 2026. **Completed: local implementation and the 1,000-game
correctness/reproducibility gate. No new branch, Git mutation, M2, full transformer,
self-play/RL, 50K extraction or full monthly-archive download was performed.**

**Disposition: PASS for the bounded M1 correctness smoke; CONDITIONAL for scaling.
Do not launch 50K yet.** The current prefix sample is intentionally biased and
is not a representative training/evaluation corpus. Review the month-wide
sampling/network budget and the measured storage choice before authorizing scale.
New Blackwell large-batch/peak-memory profiling remains unexecuted.

## 1. Specification and scope

Read the root README v0.4, the repository's `Untitled` M1 checklist, M0 parity and
commit reports, M0 adapter/codec/adjudication/tests, and the supplied reviewer
prompt. The user's adjustments override the checklist's branch creation,
automatic scaling, unconditional mate-in-two discovery and default storage choice.
M0 acceptance stays passed; `m0_followup.md` documents targeted hardening.

Main policy head is now explicitly **4672 native Pgx actions (8×8×73)**.
Attention-style policy is a later ablation. The one shared encoder is
`pgx-v2-history8-ep-context8-v1` (see `encoder_contract.md`). No rules or action
semantics changed. The underlying corrected adapter SHA256 is pinned in
`pgx_m1_pin.json` and remained unchanged throughout.

## 2. Implemented files

- `configs/data_human.yaml`: frozen months/splits, quotas, filters, download/output
  budgets, sampling-bias declaration and measured primary storage selection.
- `configs/human_v3_schema.json`: exact physical schemas, feature byte layouts,
  versions, mask packing and label semantics.
- `src/tinychess/chess/state_features.py`: canonical pure-JAX scalar/vmap encoder.
- `src/tinychess/validation/parity.py`: independent history-aware fail-fast oracle.
- `scripts/check_pgx_parity.py`: staged random-position and explicit draw-history audit.
- `scripts/probe_mctx_tactics.py`: deterministic PUCT/Gumbel diagnostics and oracle.
- `benchmarks/bench_pgx_safe.py`: synchronized memory-budgeted batch profiler.
- `src/tinychess/data/sources.py`: bounded compressed prefix acquisition, immutable
  cached provenance, streaming decompression and bounded game framing/recovery.
- `src/tinychess/data/human.py`: parser, filters, whole-game split, Pgx trajectory
  replay, full legal/history/outcome checks, action encoding and PGN WDL labels.
- `src/tinychess/data/storage.py`: compact game and materialized position Parquet
  prototypes, lossless feature decoding and incompatible-version rejection.
- `src/tinychess/data/audits.py`: split/game/position overlap, distributions and
  train-disjoint held-out index.
- `src/tinychess/data/build_human.py`: bounded 100/1000-game orchestration,
  schema/quota/source checks, per-rejection ledger, checksummed manifests and metrics.
- `benchmarks/bench_human_storage.py`: sequential/random access, full replay and
  feature-equivalence benchmark on both storage prototypes.
- `scripts/audit_m1_sample.py`: every-row label/mask/metadata audit, real special-move
  coverage, temporal coverage and mutually disjoint held-out index.
- `tests/test_m0_followup.py`, `tests/test_human_data.py`: additional automated checks.
- `pyproject.toml`, root lockfiles and README: data extras and frozen head/encoder
  contract. All M0 pins retained. New pins: PyArrow 20.0.0, zstandard 0.23.0;
  existing PyYAML 6.0.3 / psutil 7.2.2 explicitly declared for data tooling.

The user's original checklist file is untouched. Raw prefixes and generated
Parquet/indices stay under ignored `data/raw/` and `data/processed/`. Only small
configs, source, schemas and measurement/provenance reports are review artifacts.

## 3. Sources, budgets and sampling actually used

Source: [Lichess standard-rated monthly PGN exports](https://database.lichess.org/#standard_games),
CC0. January–July 2026 are train, August validation, September held-out test,
with each game entirely in one split. Split is determined by **source month**;
UTC date/time is retained and audited separately (zero source-month mismatches
in this sample). No engine evaluations or external strategic labels were used.

Each of the nine months was fetched with a bounded **offset-zero** HTTP prefix:
`Range: bytes=0-4194303`. All responses were HTTP 206. Total compressed archive
payload: **37,748,736 bytes (36 MiB)** plus the small published checksum index.
Each full archive is about 28–31 GB: none was downloaded in full. These are
prefixes beginning at a valid stream start, not arbitrary compressed offsets.

`m1_sources.json` records URLs, full sizes, ETags, Last-Modified, retrieval times,
prefix lengths/SHA256s and the publisher's full-archive SHA256 from its
[checksum index](https://database.lichess.org/standard/sha256sums.txt). The index
itself is checksummed. **Prefix hashes were verified; full archives were not.**
A publisher full-file digest is provenance, not proof that our prefix verifies
that full file. Repeated extractions use the same frozen cached bytes/metadata.

Configured download cap is 40 MiB, output cap 1 GiB per smoke run. Actual local
free disk before the final first run was 84,261,920,768 bytes. This is local
filesystem availability, not a Molab quota claim. All decoded reads are streaming;
game text is capped at 1 MiB and line reads at 65,536 characters. Framing assumes
Lichess's Event-first game headers. Corrupted source games are rejected/resynced;
unexplained Pgx/oracle mismatches abort the whole extraction with `FAIL_STOP.json`.

### Severe measured chronology bias

Every selected game in each source month is on UTC day **01**. Final selected
start timestamps range only from 00:00:00–00:00:34 in January through at most
00:01:01 in May; the other months end within 26–55 seconds of midnight. Exact
ranges are in `m1_sample_audit.json`. This is **not month-wide coverage** and
must not be presented as a representative sample or independent strength test.

### Proposed future sampling — not implemented or run

For representative 50K sampling, sequentially decompress each frozen monthly
source from offset zero (or from an independently verified frame/checkpoint
index), bucket eligible games by UTC day, and keep bounded per-day priority
reservoirs ranked by `SHA256(seed || source_month || game_id)`. Allocate quotas
across days before processing; keep deterministic reserves to replace corrupted,
unreplayable, duplicate or frozen-evaluation candidates. Replay every selected
accepted game through Pgx before final acceptance. Record all quotas, rankings,
rejections, source digests and any stratum shortfall; do not silently substitute
more day-01 games for missing days.

Without an authoritative seekable/frame index, a complete sequential scan is
needed to establish full-month coverage. Nine monthly sources imply roughly
260–270 GB of compressed network traffic even if bytes are never all stored.
Parsing/decompression compute must also be budgeted. A reusable day-shard/frame
index could be built during that scan for later resumption, with verified frame
and PGN boundaries and checksums. **Do not issue random HTTP byte ranges and
assume they form independently decompressible `.zst` files.** This larger scan,
reservoir/index implementation and its budget require separate review.

## 4. Filtering and ending semantics

Accept only standard rated games from the standard initial position, both Elo
ratings ≥1800, base time ≥180 seconds, valid complete PGN result, and legal parsed
mainline. Optional Elo≥2000 can be set with `min_rating`. Reject variants,
nonstandard starts, invalid/missing metadata/results, corruption/illegal moves,
Abandoned/Rules infraction/Aborted endings, empty games, >512 plies, any actual
512-ply truncation, and continuations after the declared automatic board end.
Every rejected source game has a reason/hash/month/ordinal in the ignored ledger.
Reasons use the **first failing filter**, not independent overlapping counts.

Human PGN results are not interchangeable with environment termination:

- Actual final-board checkmate: assert Pgx checkmate/termination and result agreement.
- Actual automatic board draws: validate board/Pgx equivalence and result agreement.
- Nonterminal decisive/draw PGNs: retain the human result; do not falsely require
  Pgx terminal just because the game ended by resignation/agreement/claim.
- A generic `Termination="Normal"` does not identify the exact human ending.
  Preserve `decisive_nonboard_unspecified` or `draw_nonboard_unspecified` rather
  than inventing evidence of resignation, agreement or a claim. Explicit reported
  resignation/agreement/claim headers have separate classifier branches.
- Time forfeit: keep **policy_weight=1** and PGN result provenance, but set
  **value_weight=0** for every position. Labels and weights remain separate.
- 512-ply truncation: reject/count the game; never manufacture a draw/WDL label.
  Our M0 horizon is unchanged.

Every accepted pre-move position and final board passes full legal-action, FEN,
board-history, repetition, counter, flag and reward parity. Training states are
always sequential Pgx replay, never independently imported per-position FENs.

## 5. Actual incremental extraction results

First smoke: **100 accepted / 752 encountered / 652 rejected**, **7,270 positions**,
21 actual checkmates and 18 time-forfeit games. Runtime 37.24 s includes initial
source acquisition/cold initialization. Peak host RSS 511,213,568 bytes.
`m1_smoke100_manifest.json` and `m1_smoke100_measurements.json` preserve evidence.

The complete automated suite passed **110 tests** before the 1K gate. After
additional quota, fail-stop, frozen-fixture and reader-version checks, the final
suite passed **117 tests in 51.36 s**. The safe-path rerun also passed **117 tests in 50.99 s**
(`m1_tests_safepath.txt`), after removing a cwd-dependent test import for Molab compatibility.

Final 1K sample:

| Split | Encountered | Accepted | Rejected | Position records |
|---|---:|---:|---:|---:|
| Train, Jan–Jul | 3,820 | 700 | 3,120 | 52,500 |
| Validation, Aug | 1,339 | 150 | 1,189 | 10,879 |
| Test, Sep | 793 | 150 | 643 | 11,877 |
| **Total** | **5,952** | **1,000** | **4,952** | **75,256** |

Train has exactly 100 accepted games per source month. Full per-month counts
are in `m1_manifest_run1.json`. Accepted game length ranged from 7 to 266 plies.

| First-failure rejection reason | Games |
|---|---:|
| Rating below minimum | 3,690 |
| Base time below minimum | 1,170 |
| Not rated | 78 |
| Abandoned termination | 12 |
| Invalid time control | 2 |

No real sampled games exceeded the horizon, were corrupted, required illegal-move
rejection, or failed result consistency. Those branches are tested synthetically;
zero observed rejections is not evidence they are unnecessary on broader data.

Final accepted endings: 192 board checkmates; 15 insufficient-material draws;
7 stalemates; 1 fivefold repetition; 507 decisive nonboard/unspecified endings;
35 nonboard/unspecified draws; 243 time forfeits. No actual claim/agreement is
inferred merely from `Normal` or a claimable final position.

## 6. Correctness, reproducibility and distributions

- **Zero full legal-action/history/status/reward parity mismatches**. Each 1K
  run checked 75,256 pre-move positions plus 1,000 final boards: 76,256 states.
- Played moves all map to legal native Pgx actions. The read-back audit verified
  every row's mask, action sequence, FEN identity, versions, finite features,
  side to move, WDL sign/class, weights and game metadata.
- WDL classes are **0=win, 1=draw, 2=loss**; scalar labels are +1/0/−1 from STM.
  Counts including masked rows: 34,104 win; 7,453 draw; 33,699 loss.
  Value-eligible counts: 23,874 win; 6,140 draw; 23,581 loss.
- **21,661 rows** have zero value weight due to time forfeits. Policy supervision
  remains present. No truncated rows were accepted.
- All **192 actual final checkmates** passed the explicit final-board assertion.
- Observed played special moves: **1,724 castlings, 41 EP captures, 161 queen
  promotions and 1 knight underpromotion**. There were 339 repeated-position rows.
  Bishop/rook underpromotions were absent in the real smoke sample; unit tests
  cover all promotion types for White and Black, plus EP/castling/history.
- Games: 460 White wins, 471 Black wins, 69 draws; 37,875 White-to-move rows and
  37,381 Black-to-move rows. Descriptive ply phases: 19,911 opening (<20), 44,267
  middlegame (20–79), 11,078 late (≥80); these are not engine labels.
- Rating-minimum buckets: 636 games in 1800s, 271 in 2000s, 70 in 2200s, 17 in
  2400s, 4 in 2600s and 2 in 2800s. Time controls/openings are fully counted in
  `m1_audit_run1.json`. Largest ECO buckets: A40=51, B00=49, D00=42. No opening
  downsampling was silently applied. A future training sampler must cap/rebalance
  frequent positions/opening families, without collapsing different human moves.

### Determinism

Two final extractions with the same frozen inputs/config produced **22 identical
hashed files and identical manifests**. Dataset digest:

```text
7ae6d8202b4079171ffd3ccab22611d598a06aec24b66ca49a3a2ed15ed6e428
```

Only wall-time/memory/disk measurement JSON is excluded from deterministic data
hashes. A separately derived strict disjoint index and its audit also match
across the two final runs. `m1_determinism.json` records this explicitly. Two
prior prototype runs had the same shard digest as well. Final run configuration
selects the measured materialized layout; its SHA256 is recorded in the manifests.
`m1_implementation_hashes.json` binds the delivered source/config/lock snapshot.

## 7. Leakage audits and disjoint held-out subsets

Games remain wholly in one split: **1,000 unique IDs, zero duplicate IDs or
repeated complete action trajectories** in this sample. Different positions
from the same game never cross splits.

Position key is SHA256 of pieces, side to move, castling rights and **legally
capturable** EP. It deliberately excludes clocks/history for a conservative
board-level overlap audit. Shared board identity is not necessarily identical
model context. Exact dataset output/feature hashes are separate.

| Unique-position comparison | Validation | Test |
|---|---:|---:|
| Unique positions, all plies | 10,185 | 11,174 |
| Shared with train, all plies | 393 (3.86%) | 371 (3.32%) |
| Unique positions, ply≥10 | 9,297 | 10,327 |
| Shared with train, ply≥10 | 31 (0.333%) | 45 (0.436%) |

Row-level train overlap: validation 992/10,879 overall, 32/9,382 at ply≥10;
test 1,010/11,877 overall, 48/10,377 at ply≥10. Train has 48,388 unique board keys
at all plies and 45,193 at ply≥10. Validation/test share 215 unique keys overall,
8 when both sides exclude ply<10.

The base index excludes train positions from each held-out split. The **stricter**
`strict_disjoint_eval.jsonl` additionally excludes validation identities from
test: **9,887 validation rows and 10,824 test rows** remain. It selects evaluation
rows, not whole new games, and preserves legitimate repeated human policy labels.
Its deterministic checksum and definition are in `m1_sample_audit.json`.

No frozen puzzle/GPT-match fixture manifest was supplied, so zero actual external
fixtures were excluded and **no puzzle/test-suite non-overlap claim is made**.
Configured frozen FEN exclusion is implemented/tested; freeze Dataset D inputs
before making later evaluation-contamination claims.

## 8. Exact record/encoder/storage contract

`human_v3` has two Parquet representations of the same games. File metadata pins
schema, encoder, rules, action map, WDL order, feature dtype and mask bit order.

**Game row:** `game_id:string`, `metadata:string` (canonical JSON),
`actions:list<uint16>`. Metadata includes source month/ordinal/URL/raw PGN hash,
whole-game split, PGN result/termination, observed final-board reason/FEN, ratings,
TC, UTC date/time, opening, number of plies, label source, value weight and version
IDs. All actions replay from the standard initial board and preserve full state.

**Position row (physical columns):**

| Columns | Physical type |
|---|---|
| game_id, source_month, split, human_move_uci, fen, label_source | string |
| ply, action, remaining_plies | uint16 |
| outcome_stm | int8 |
| wdl_class, color, repetition_count | uint8 |
| value_weight, policy_weight | float32 |
| training_eligible | bool |
| position_hash | fixed binary[32] |
| planes | fixed binary[30720], little-endian float32 `(8,8,120)` |
| context | fixed binary[32], little-endian float32 `(8,)` |
| legal_mask | fixed binary[584], all 4672 bits, little bit order |

Per-game metadata is joined on the unique game ID, rather than redundantly
copied into every physical row. Compact actions plus metadata reconstruct the
full rules history; cached position features are supervised model inputs, not
standalone search-state substitutes. Context order and every plane are defined
in `encoder_contract.md`; no second encoder path exists. The compact reader
rejects incompatible schema/encoder/action/rules metadata.

A real inspected row is game `mGJ1eas9`, train/2026-01, ply 12, White to move,
UCI `e2e4`, action **2426**, scalar outcome +1, WDL class 0, value/policy weights
1, repetition count 1. Its human game ended nonterminal with PGN result 1-0;
we do not label that ending as proven resignation. `m1_sample_audit.json`
contains three representative records at different plies, complete legal action
indices/packed-mask hex, context, plane SHA256 and full game metadata. Binary
plane payloads are in ignored Parquet, not copied as large report examples.

## 9. Measured storage/access decision

Benchmark: CPU, 1,000 games / 75,256 positions, same accepted records. Warm filesystem,
preloaded small game/index tables, JAX initialized. It compares complete sequential
feature streams byte-for-byte and a fixed 64-position random batch from 61 games.
Full source/feature parity and **all materialized/replayed features equal**.
Raw result: `m1_storage_benchmark.json`.

| Measurement | Compact games + replay | Materialized positions |
|---|---:|---:|
| Parquet bytes | 356,717 | 41,310,315 |
| Bytes / position | 4.74 | 548.93 |
| Sequential feature read/replay | 13.640 s | 1.669 s |
| Encoded/decoded positions/s | 5,517 | 45,102 |
| Random 64-position minibatch | 0.909 s (2,784 replay steps) | 0.175 s |

Compact metadata/action decode alone took 0.0206 s; that is **not** full feature
availability. Random compact timing includes replay, encode and equality checks;
materialized timing includes row-group read and decoding. This is a single
warm-access microbenchmark, not an optimized production training reader or cold
I/O benchmark. Peak benchmark host RSS: 1,079,164,928 bytes.

**Decision:** use **materialized position Parquet as primary** for the initial
supervised reader, because measured sequential availability is ~8.2× faster and
random access ~5.2× faster. Retain compact game Parquet as canonical replay /
provenance representation. Materialized data costs ~115.8× the compact bytes,
but only 41.3 MB here. Both prototypes remain available, and the choice must be
rechecked for real training batch sizes, shuffling, GPU transfer and larger data.
No transformer/dataloader training performance was measured.

## 10. Extraction throughput and memory

Final runs used cached prefixes, exact same quotas/bytes/config:

| Metric | Final run 1 | Final run 2 |
|---|---:|---:|
| Elapsed | 54.852 s | 63.679 s |
| Encountered games/s | 108.51 | 93.47 |
| Accepted games/s | 18.23 | 15.70 |
| Position records/s | 1,371.98 | 1,181.80 |
| Peak host RSS, bytes | 527,089,664 | 519,389,184 |
| Base output bytes (both prototypes + audit/manifest) | 43,766,401 | 43,766,401 |

Measurements include parse/filter/replay/parity/encode/write/audit/checksums;
exclude the original download for these cached final runs. They include varying
shared-host load; no deterministic timing claim is made. A few additional MB of
ignored audit indices and measurement JSON are separate from the base digest.
Extra prototype/final repeats are reproducibility work on the same 1,000 unique
games, not an expanded corpus. Scratch use remains under the bounded local budget.

PyArrow emitted sandbox `sysctlbyname` CPU-cache/NEON discovery warnings on this
Mac. Runs completed, read-back/checksums/equality passed, and warnings were not
reclassified as chess errors. No GPU data-pipeline/encoder-throughput or device
peak-memory result is claimed for M1.

## 11. Reproduction commands

Run from this repository directory. Output directories must be new; extraction
refuses to overwrite evidence. No Git commands are needed.

```bash
uv sync --locked --extra test --extra data
.venv/bin/python -m pytest -q
PYTHONSAFEPATH=1 .venv/bin/python -m pytest -q

# First gate; fetches at most the configured bounded prefixes.
.venv/bin/python -m tinychess.data.build_human --stage 100 --output data/processed/reproduce_100

# Only after the 100-game gate/tests pass; cached input, no larger download.
.venv/bin/python -m tinychess.data.build_human --stage 1000 --offline --output data/processed/reproduce_1000_a
.venv/bin/python -m tinychess.data.build_human --stage 1000 --offline --output data/processed/reproduce_1000_b

.venv/bin/python scripts/audit_m1_sample.py --dataset data/processed/reproduce_1000_a --output reports/reproduced_m1_audit.json
.venv/bin/python benchmarks/bench_human_storage.py --dataset data/processed/reproduce_1000_a --output reports/reproduced_m1_storage.json
```

Compare manifest `dataset_digest` and `checksums` for the two independent outputs;
ignore `measurements.json` for deterministic hashing. Final observed config digest
is in `m1_determinism.json`. The only authorized CLI stages are 100 and 1000;
quotas must sum exactly to that stage. There is no 50K execution path.

## 12. Remaining limitations / recommendation

Completed evidence: staged 10K/50K random **positions**, explicit long histories,
CPU tactical/sign tests, safe local batch-128 profiling with explicit larger-size
skips, 100→1000 human extraction, final regression suite, double deterministic
extraction, full read-back/feature equality, storage measurements and leakage audit.

Unexecuted/proposed work: several-hundred-thousand-position stress; new Blackwell
1K/4K/8K/peak-memory profiling; cross-device encoder checks; full archive checksum
verification; representative month-wide sampling/index construction; cold/durable
training I/O and real minibatch sizes; actual Molab storage quota; external frozen
puzzle/GPT fixture audit; 50K extraction; M2 transformer and all self-play/RL.

No unexplained legal-action/history/terminal/reward mismatch occurred, and no M0
rules policy was changed. The sample does not exhaust rare chess positions;
insufficient-material logic is not a general dead-position solver and repetition
hashing is probabilistic. Human `Normal` endings remain semantically ambiguous.

**Recommendation:** accept the 1K correctness/storage smoke for review. Do **not**
scale the current prefix sampler to 50K or approve M2 on this sample. After
reviewing representative acquisition, compute/network/storage budgets, fixed
external evaluation exclusions and the storage reader tradeoff, authorize a
separate bounded 50K pilot. Stop here as requested.
