# M1 — Dataset A 50K pilot acceptance report

Assessment date: 29 October 2026. **CONDITIONAL PASS; M2 is not authorized.**
The real 50K generation and independent reproduction are complete. Dataset D
exclusions and the diagnostic index are separate acceptance conditions below;
their completion must be supported by actual receipts. The separate Blackwell
benchmark remains pending when no GPU session is available.

## Repository and immutable cloud evidence

- Source branch: `m1/human-dataset`; HEAD at report generation: `4f956b343bd0925cc596834990fa4237b496e234`.
  This identifies the report's base, not the subsequent commit containing it.
  The user will execute source staging, commit and push manually.
  The final commit identity is recorded by Git history, avoiding a self-referential
  hash embedded inside that same commit. No merge into `main` or M2 is authorized.
- Public data: https://huggingface.co/datasets/ishagarg1103/gambit-zero
- Original 50K data revision: `d409fce8bd444c2d456ef4c343dbc9b5e25981eb`.
- Original inventory: 846 verified artifacts,
  2,137,251,914 bytes (1.990 GiB).
  Small source/recovery backups and later evaluation/diagnostic objects are additional.
- Recovery receipt: `recovery/human-v3-pilot-20261025/recovery.json`,
  SHA256 `4c168272b622932a13ccbcf546a784e1dea405c6e28ffce5ad59e693a8341cb3`.
- Source-code backup: `recovery/human-v3-pilot-20261025/generation-source.zip`,
  SHA256 `9a9fe70f5120d00008299e45bd5d02aabe26e5eb36489a7cab13bc8c8b23d442`.

## Final original split counts and coverage

| Month | Split | Accepted / reproduced | Rows | Both >=2000 | Days | Scan hours | Primary build hours |
|---|---|---:|---:|---:|---:|---:|---:|
| 2026-01 | train | 5,000 / 5,000 | 380,274 | 3,016 (60.32%) | 31 | 1.554 | 0.313 |
| 2026-02 | train | 5,000 / 5,000 | 381,007 | 3,012 (60.24%) | 28 | 1.331 | 0.293 |
| 2026-03 | train | 5,000 / 5,000 | 380,484 | 3,016 (60.32%) | 31 | 1.212 | 0.270 |
| 2026-04 | train | 5,000 / 5,000 | 380,334 | 3,020 (60.40%) | 30 | 1.293 | 0.307 |
| 2026-05 | train | 5,000 / 5,000 | 375,511 | 3,016 (60.32%) | 31 | 1.206 | 0.255 |
| 2026-06 | train | 5,000 / 5,000 | 376,760 | 3,020 (60.40%) | 30 | 1.830 | 0.310 |
| 2026-07 | train | 5,000 / 5,000 | 380,630 | 3,016 (60.32%) | 31 | 1.530 | 0.279 |
| 2026-08 | validation | 7,500 / 7,500 | 569,124 | 4,524 (60.32%) | 31 | 1.822 | 0.408 |
| 2026-09 | test | 7,500 / 7,500 | 569,626 | 4,500 (60.00%) | 30 | 3.542 | 0.428 |

Total: **35,000 train / 7,500 validation / 7,500 test games**;
**2,655,000 / 569,124 / 569,626 position rows**, respectively.
All 273 calendar days are represented.
Exact accepted/strong counts per day: [coverage](m1_evidence_archive.md#reports-m1-50k-coverage-json).
Header and full-validation rejection counts per month:
[rejections](m1_evidence_archive.md#reports-m1-50k-rejections-json). Header rejection is the first failed filter,
not all possible reasons for rejecting the same game. Eligible but unselected
candidates are not mislabeled as correctness failures.

## Frozen sources, selection and reproducibility

URL template: `https://database.lichess.org/standard/lichess_db_standard_rated_YYYY-MM.pgn.zst`.
Each successful scan consumed the complete compressed stream and matched its
official published SHA256 at exact EOF. No compressed/decompressed archive was
retained. Interrupted attempts do not establish a verified source hash.

| Month | Compressed bytes | Verified source SHA256 | Reproduced shard digest |
|---|---:|---|---|
| 2026-01 | 30,661,381,578 | `53388f5c5be7eb2c3ce4d5071d7453a35ada6cea4f965bc6118cf1edb05e40e5` | `4928abcb463b75e88f146cbd678c155af3a4e08dca709d264e268b48bd04d19c` |
| 2026-02 | 27,755,967,701 | `feeffa53b5056b2776f8e2e6c638a83eaaa1d1ffbaa88f711299db6c2b156c01` | `a9f42f05787a9ddf33f4fcfdd827e5d0abb9a94c2523e5f18e3d957691c782b8` |
| 2026-03 | 29,351,713,061 | `d3adc2bcc58e85f4398ece2f7f8ea422c5ec5269a7d2921257bc31c5b914180f` | `38c352f0173a8b5076331e4beba9a68a0fc3d26a7930b71509f160e152ae9a7e` |
| 2026-04 | 29,325,351,334 | `a6dc976bda3e6f543929e9448420482f2fe827b09e16c1005ca9a6ef66e7336d` | `967a44304c3d139c317f15a074997f7575c6f1f1e6d6787adab42fb54cfee457` |
| 2026-05 | 29,689,176,290 | `249b4c13e92442ba6694d7f0bdb5b4639047bf74caba1611722a132a012ab294` | `246436d0fcfe3717d6954abd4b43c354b4c9a7a12da0ddcffc699045b6d8db9a` |
| 2026-06 | 28,241,946,492 | `8fd81071f56511e7546cb77e38db5cf32f7e8a437fb906e26959cc064d8b1f79` | `c681621e09f46c7f73ac209a5138d83e46dbc589c88a22ea8143ea3396738bfe` |
| 2026-07 | 29,051,046,504 | `68738b1c448f051dc8d42db645d5b01749988a3bc1c24981adfe44ea92060dc7` | `2540a4ae8dc929df9438bb239121ffeeab491f33314acee7923c3dce13891f5b` |
| 2026-08 | 30,145,862,359 | `6bf6fa8a5dee7bb81d1874ac312160060daf12f18a29dc2740a3bf6f5e5e6248` | `ac9b5d2b2cb0c259a35da5f182e4d233e5aa7a910d1886fcf3824cb42c153629` |
| 2026-09 | 29,224,520,887 | `1de8eb64fed443f4be310c779e8d60a13bcfc6e3b104bcb39a018eb056edec89` | `d08368c27aa3ba1902561efa8f29eed95aa95711d70cc0c081ef81c24784f14f` |

Selection: `utc-day-rating-bottomk-sha256-v1`, seed **20261025**.
Rank complete games deterministically by seed/month/stable source identity
within UTC-day/rating strata. Daily quotas cover the full month; bounded reserves
allow later full-game rejection without quietly changing quotas. Each selected
PGN retains its raw SHA256, source ordinal/day/ID and archive identity. These are
reusable selected-game manifests, not a random-access index into arbitrary zstd
byte ranges. Generation checked source frame structure; live suffix-range
network recovery preserves the same decoder/hash state and does not invent
independent decompression of arbitrary offsets. A process restart rescans an
unfinished source from zero; verified completed months are reused.

All nine months were built independently a second time from verified immutable
cloud selection artifacts. Selected IDs, logical record hashes, encoded-feature
hashes and actual regenerated Parquet shard hashes matched. Monthly
`data/human-v3-pilot-20261025/manifests/YYYY-MM.reproduction.json` records the
comparison; this is not a second full archive download or an independently
reimplemented chess engine/encoder. Byte-identical output is supported by those
actual second-build receipts.

## Schema, history and target contracts

- `human_v3`, physical schema in [human_v3_schema.json](../configs/human_v3_schema.json).
- Encoder `pgx-v2-history8-ep-context8-v1`: float32 planes `(8,8,120)`,
  context `(8,)`, legal mask `(4672,)`. Stored feature bytes are little-endian
  float32 and masks are 584 bytes, little bit order. Exact channel ordering:
  [encoder contract](encoder_contract.md).
- Codec `pgx-2.6.0-chess-v2-uci-v1`, native 4,672-action Pgx order.
- Rules `tinychess-auto-fide-pgx-v1`: no voluntary draw claims;
  fivefold/150-halfmove automatic draws, checkmate precedence;
  512 plies is truncation, never a draw or a final WDL source.
- Replay complete action trajectories from the initial corrected Pgx state.
  No FEN-only feature reconstruction; actual repetition/history is preserved.
  Full legal masks, board/history/status/rewards are compared offline to
  python-chess, including the final board.
- Policy is the human action, not an engine best move. WDL order win/draw/loss
  (0/1/2); scalar +1/0/-1 from the recorded side to move. Policy/value weights
  are separate. Time forfeits keep policy supervision with value weight zero.
  Generic Normal PGNs do not prove resignation, agreement or a draw claim;
  actual board checkmate alone requires equivalent Pgx checkmate/winner.
- All games: standard rated chess, both ratings >=1800, base clock >=300 seconds
  independent of increment. At least 60% strong games in every source month.
  Rating bands use the lower player rating. No Stockfish or `[%eval]` labels are
  consumed by Dataset A. Future training must respect timeout masks and caps on
  common positions/openings; distribution measurements alone are not rebalancing.

Compact action games are retained as provenance; materialized position Parquet
is the primary training representation, selected using measured 1K storage,
decoding/replay and random-minibatch costs in [the smoke report](m1_smoke_test.md).
No transformer, training loop or self-play data was started.

## Correctness and regression evidence

- Required generation checks: **3,843,750 states, 0 mismatches**
  (3,793,750 pre-move rows plus 50,000 final boards).
- Independent cloud feature reproduction: all 50,000 games, all nine months.
- Additional fresh downloaded-row replay audit: January–June, 30,000 games,
  2,274,370 rows; every field/feature byte compared, zero mismatches.
  See `m1_hf_rows_audit*.json` and companion reports. July–September already
  passed required generation/reproduction; another downloaded-row replay is
  optional assurance, not an unperformed required parity check.
- Prior full regression after transport recovery: 204 passed in 48.81 seconds.
- Closeout regression: [actual output](m1_closeout_tests.txt). This file is the
  authority for the current count; do not infer a successful run from code alone.

## Exact board-position overlap in the original 50K

Identity: SHA256(pieces, side to move, castling rights, legally capturable EP).
Clocks/history are excluded deliberately for conservative board overlap; it is
not a claim that two model inputs or repetition rights are identical. Whole-game
identity isolation is checked separately. Repeated action trajectories can
occur across distinct source games and are reported, not hidden.

| Scope | Train unique | Validation unique | Test unique | Train/val shared (% val) | Train/test shared (% test) | All three |
|---|---:|---:|---:|---:|---:|---:|
| all_plies | 2,286,117 | 503,218 | 504,064 | 23,589 (4.6876%) | 23,178 (4.5982%) | 10,405 |
| ply_ge_10 | 2,234,036 | 485,263 | 485,844 | 12,665 (2.6099%) | 12,143 (2.4994%) | 4,054 |

The full metadata closeout independently reproduced these aggregates and checked
all 50K game identities, source months, rating/time filters, row order and WDL/
timeout masks. It did not regenerate tensor features or replace prior chess parity.
Position-disjoint diagnostic index: **{'test': 484240, 'validation': 482926}**.
Selection excludes any held-out board found in training at *any* ply. Validation
and test may overlap each other; repeated unseen boards remain observations.
Index rows identify source path, zero-based row offset, game, ply and board hash;
they contain no feature tensors or invented labels. The original chronological
dataset is unchanged. Receipts and distributions:
[diagnostics](m1_evidence_archive.md#reports-m1-50k-diagnostics-json).
Published revision: `73960d7712d6fba5360a643bf09c8a46a60944e8`.
The first-run raw summary's legacy `peak_working_disk_bytes` field denotes
index size at completion, not a continuously sampled high-water measurement;
the current implementation names this `index_disk_bytes_at_end`.

Repeated complete action-trajectory groups across distinct games:
**16**. These are distinct source game
identities, not necessarily identical player/clock/outcome records.
Original WDL row counts (win/draw/loss class 0/1/2):
`{'0': 1719638, '1': 375674, '2': 1698438}`. Value-masked rows:
**769,665**; nonzero-value-weight rows:
**3,024,085**.
Phase-by-ply and side-to-move distributions are in the diagnostic JSON; phase
means opening <20, middlegame <80, late >=80 plies, not engine/material judgments.

| Month | Lower-player rating bands (200-point bins) | Ending semantics | ECO labels present |
|---|---|---|---:|
| 2026-01 | `{'1800': 1984, '2000': 2541, '2200': 428, '2400': 26, '2600': 6, '2800': 10, '3000': 5}` | `{'board_checkmate': 843, 'board_fivefold_repetition': 1, 'board_insufficient_material': 65, 'board_stalemate': 20, 'decisive_nonboard_unspecified': 2953, 'draw_nonboard_unspecified': 250, 'time_forfeit': 868}` | 334 |
| 2026-02 | `{'1800': 1988, '2000': 2559, '2200': 385, '2400': 47, '2600': 4, '2800': 6, '3000': 11}` | `{'board_checkmate': 882, 'board_fivefold_repetition': 1, 'board_insufficient_material': 59, 'board_stalemate': 25, 'decisive_nonboard_unspecified': 2952, 'draw_nonboard_unspecified': 224, 'time_forfeit': 857}` | 325 |
| 2026-03 | `{'1800': 1984, '2000': 2554, '2200': 417, '2400': 27, '2600': 4, '2800': 11, '3000': 3}` | `{'board_checkmate': 813, 'board_insufficient_material': 71, 'board_stalemate': 27, 'decisive_nonboard_unspecified': 2988, 'draw_nonboard_unspecified': 219, 'time_forfeit': 882}` | 349 |
| 2026-04 | `{'1800': 1980, '2000': 2592, '2200': 376, '2400': 26, '2600': 13, '2800': 5, '3000': 8}` | `{'board_checkmate': 842, 'board_fivefold_repetition': 1, 'board_insufficient_material': 61, 'board_stalemate': 20, 'decisive_nonboard_unspecified': 3047, 'draw_nonboard_unspecified': 210, 'time_forfeit': 819}` | 342 |
| 2026-05 | `{'1800': 1984, '2000': 2579, '2200': 401, '2400': 20, '2600': 8, '2800': 6, '3000': 2}` | `{'board_checkmate': 853, 'board_fivefold_repetition': 1, 'board_insufficient_material': 77, 'board_stalemate': 17, 'decisive_nonboard_unspecified': 2989, 'draw_nonboard_unspecified': 226, 'time_forfeit': 837}` | 331 |
| 2026-06 | `{'1800': 1980, '2000': 2501, '2200': 452, '2400': 33, '2600': 13, '2800': 17, '3000': 4}` | `{'board_checkmate': 823, 'board_insufficient_material': 64, 'board_stalemate': 19, 'decisive_nonboard_unspecified': 3014, 'draw_nonboard_unspecified': 233, 'time_forfeit': 847}` | 331 |
| 2026-07 | `{'1800': 1984, '2000': 2557, '2200': 398, '2400': 28, '2600': 13, '2800': 16, '3000': 4}` | `{'board_checkmate': 874, 'board_fivefold_repetition': 1, 'board_insufficient_material': 58, 'board_stalemate': 26, 'decisive_nonboard_unspecified': 2980, 'draw_nonboard_unspecified': 231, 'time_forfeit': 830}` | 329 |
| 2026-08 | `{'1800': 2976, '2000': 3828, '2200': 573, '2400': 72, '2600': 22, '2800': 22, '3000': 7}` | `{'board_checkmate': 1238, 'board_fivefold_repetition': 3, 'board_insufficient_material': 86, 'board_stalemate': 39, 'decisive_nonboard_unspecified': 4536, 'draw_nonboard_unspecified': 344, 'time_forfeit': 1254}` | 373 |
| 2026-09 | `{'1800': 3000, '2000': 3772, '2200': 593, '2400': 60, '2600': 33, '2800': 37, '3000': 4, '3400': 1}` | `{'board_checkmate': 1298, 'board_insufficient_material': 87, 'board_stalemate': 32, 'decisive_nonboard_unspecified': 4461, 'draw_nonboard_unspecified': 343, 'time_forfeit': 1279}` | 365 |

`unknown` ECO is a missing source annotation, not an inferred opening family.

## Dataset D disposition

Frozen independently of Dataset A overlap/model performance:
1,000 puzzles and 32 opening lines,
4,178 exact board identities.
Source snapshot, IDs, source checksums and verified immutable receipt:
[freeze evidence](m1_dataset_d_freeze.json).
Puzzle checksum is a **computed full compressed-source fingerprint**, not a
publisher checksum. Puzzle FEN precedes the opponent setup move; the actual root
is after that move. Solver/opponent continuation boards are included in the
exact overlap set; openings contribute only their exact final roots, never
entire related opening families. Puzzle repetition history is unknown.
The suite is evaluation only and provides no Dataset A strategic targets.

Physical exclusions and revised catalog: [verified results](m1_evidence_archive.md#reports-m1-dataset-d-exclusions-json).
Original matching rows: {'test': 24, 'train': 150, 'validation': 36}.
Original matching unique boards: {'test': 6, 'train': 14, 'validation': 13}.
Every exact matching **training** row was physically removed; held-out matches
are reported and retained. Train/D shared unique boards after filtering: **zero**
both across all plies and at ply >=10. Accepted game counts remain exactly 50K;
compact actions remain provenance, not an unfiltered replacement training reader.
Revised position counts: {'test': {'games': 7500, 'positions': 569626}, 'train': {'games': 35000, 'positions': 2654850}, 'validation': {'games': 7500, 'positions': 569124}}.
Revised final HF revision: `b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c`.
Exclusion pass: 887.61 seconds,
2,070,323,156 source Hub payload bytes read,
peak process RSS 942,800,896 bytes.
Its conservative temporary-disk plan is 1.25 GiB,
including the index/rollback limit, bounded shard spool and metadata margin;
20 GiB remains reserved. This is a planning bound, not a continuously measured
disk high-water mark. No source monthly archive was rescanned.
Referenced primary game/position artifacts: 2,043,334,039
compressed bytes; new derivative artifacts published: 571,546,097
bytes (logical artifact sizes, not metered HTTP traffic).
Every changed shard was independently filtered twice with identical bytes;
every retained field/tensor byte was compared to the verified original. This
is repeated lossless filtering on top of the original full Pgx reproduction,
not a claim of a second new PGN replay after exclusions.
Original data remains accessible at its pinned revision. The original
position-disjoint index remains a conservative valid subset for the reduced
training set; it is not claimed maximal against that derivative.

| Postfilter scope | Train/validation shared | Train/test shared | All three | Train/D | Validation/D | Test/D |
|---|---:|---:|---:|---:|---:|---:|
| all_plies | 23,577 | 23,172 | 10,399 | 0 | 13 | 6 |
| ply_ge_10 | 12,658 | 12,140 | 4,051 | 0 | 8 | 3 |

Puzzle-source game IDs shared with A (separate from board overlap): {'test': 0, 'train': 0, 'validation': 0}; IDs are in the exclusion report.

## Final immutable publication verification

Independent post-publication check: **PASS**, revision
`b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c`. All nine month manifests, frozen evaluation receipt,
split counts, human-readable card counts and exact `human_v3`/`games` file lists
agree. 796 unique primary objects were checked
at that final revision: 796 by LFS SHA256/size
and 0 regular objects by full body
read-back. Small-object payload read: 2,314,636 bytes.
Large shards had already been read back in full during upload/original audits;
this extra verification compares their final LFS content IDs, not a second
download of every tensor file. See [publication verification](m1_dataset_d_publication_verification.json).

## Measured resource accounting and limitations

- Successful complete source streams: **263,446,966,206 bytes**, **15.321 hours**
  combined source/network/header-scan wall time; aggregate 4.776 MB/s.
- Primary feature construction/validation/upload: **2.863 hours**
  combined across nine months; this excludes independent reproduction, extra
  row audits, failed attempts and human idle time.
- Final September invocation: **4.236 hours**;
  it is not the total project duration. Its peak process RSS:
  1,240,334,336 bytes; do not claim this is a monitored
  lifetime maximum across all previous processes.
- Historical archive allowance 504,427,436,757 bytes
  is conservative reservation, **not measured actual transfer**. Actual total
  traffic including failed prefixes/retransmissions/TLS overhead was not metered.
- Parsing, Pgx replay, parity and encoding are measured as a combined phase;
  pure parser/kernel throughput is unavailable. Source checksum work overlaps
  scanning rather than an independently timed phase.
- Fine-grained rating bands for rejected source games were not retained by the
  original fast prefilter. Accepted-game rating bands and first-rejection
  reasons are exact; a complete rejected-game rating histogram is unavailable.
  Recovering that missing telemetry would require another source scan and is
  not silently authorized. No new 263 GB scan was launched to fill a report.
- Original published inventory averages
  563.4 compressed bytes/position,
  42745.0 bytes/game, including companion
  metadata; these are not raw tensor sizes.
- Original disk plan: 2.5 GiB additional storage, 20 GiB reserve. Temporary size
  caps were explicitly waived by the user; free-space and host-memory checks
  remained. An exact historical peak working-disk measurement was not retained.
- Generated local data/caches were cleaned after verified upload. Closeout uses
  one bounded compressed shard in memory plus a temporary SQLite hash index,
  planned <=1 GiB disk and the same 20 GiB reserve. Failed uploads retain their
  local shard until verification. Completed cleanup is reported explicitly.

No exhaustive chess-correctness claim: coverage is finite, dead-position/hash
limitations remain documented by M0, human outcomes/moves are noisy, FEN board
identity is not full history identity, repeated openings can bias imitation,
and month/day/rating balancing changes the natural Lichess distribution.
The source population is filtered Lichess play, not all chess players or Elo scales.

## Separate Blackwell benchmark: PENDING

No connected GPU session is available in this closeout. Run in the validated
Molab CUDA environment, with the repository and its pinned CUDA dependencies:

```bash
JAX_PLATFORMS=cuda .venv/bin/python benchmarks/bench_pgx_safe.py \
  --batches 128 1024 4096 8192 --repeats 30 \
  --output reports/m0_large_batch_blackwell.json
```

This progressive benchmark preflights memory and skips unsafe allocations with
an explicit reason. It separates trace/compile, synchronized warm execution and
peak device memory. Peak memory is a process cumulative high-water mark, not a
per-batch delta. Environment steps/sec is not end-to-end self-play throughput.
The specification permits this documented pending disposition; no GPU results
are fabricated and it does not invalidate independent CPU data parity.

## Reproduction and changed closeout files

No archive rescans are needed to check the already published pilot:

```bash
uv sync --locked --extra test --extra data
.venv/bin/python -m pytest -q
.venv/bin/python scripts/finalize_m1_diagnostics.py
.venv/bin/python scripts/finalize_m1_diagnostics.py --execute --publish
.venv/bin/python scripts/report_m1_pilot.py
```

The diagnostic preflight makes no network calls; execution pins the original
immutable data revision. Repeated uploads use content-addressed paths and
verified remote read-back. To extend the optional full row replay audit:

```bash
.venv/bin/python scripts/audit_hf_rows.py \
  --revision d409fce8bd444c2d456ef4c343dbc9b5e25981eb \
  --months 2026-07 2026-08 2026-09 \
  --config configs/audit_human_50k.yaml \
  --output reports/m1_hf_rows_audit_july_september.json
```

This optional audit transfers about 0.85 GB of existing shards/metadata and
replays 20,000 games; it has not been run merely by listing the command.
Generation reproduction uses the backed-up runtime source and frozen per-month
configurations from `generation-source.zip`, selected PGNs and manifests in
the recovery receipt; do not relaunch full scans using stale local runtime paths.

Closeout source/config/test files: `src/tinychess/data/closeout.py`,
`src/tinychess/data/frozen_eval.py`, `src/tinychess/data/eval_filter.py`,
`scripts/apply_dataset_d_exclusions.py`, `tests/test_eval_filter.py`,
`scripts/finalize_m1_diagnostics.py`,
`scripts/freeze_dataset_d.py`, `scripts/report_m1_pilot.py`,
`scripts/audit_hf_rows.py`, `configs/audit_human_50k.yaml`,
`configs/dataset_d.yaml`, `tests/test_m1_closeout.py`,
`tests/test_frozen_eval.py`, `tests/test_hf_row_audit.py`,
`src/tinychess/data/hf_batch.py`, `src/tinychess/data/hf_root.py`,
`src/tinychess/data/hf_upload.py`, `scripts/watch_m1_closeout.py`,
`tests/test_hf_batch.py`, `tests/test_hf_root.py`, `tests/test_hf_upload.py`.
Publication verification: `src/tinychess/data/publication.py`,
`scripts/verify_m1_dataset_d_publication.py`, `tests/test_m1_publication.py`.
Relevant generated small reports are listed/linked above; no raw PGNs,
Parquet datasets, caches, environments or credentials belong in Git.

## Recommendation

**CONDITIONAL PASS for M1; do not start M2.** The required real Dataset A
generation, parity, checksum verification and reproduction passed.
CPU dataset correctness/reproducibility/exclusion checks are PASS only once
their verified receipts above are complete. Review the separate GPU pending
disposition and the documented historical telemetry limitations before declaring
every M1 checklist item closed. No unavailable measurement is represented as done.
No M2 training, Dataset B/self-play, larger extraction or automatic merge is authorized.

<!-- m1-blackwell-followup:start -->
## Blackwell follow-up — verified 29 October 2026

The later Molab environment benchmark gate is **PASS**. See the
[measured benchmark](m1_blackwell_benchmark.md) and
[supplemental acceptance](m1_blackwell_acceptance.json).
Measured batches: [128, 1024, 4096, 8192]; skipped: [].
CUDA regression: 264 passed in 383.84 seconds.
This supersedes the historical GPU-pending statements above.
The original CPU dataset findings and telemetry limitations remain unchanged.
These are independent initial-position environment-step rates, not model,
training, MCTS or self-play throughput. Dataset revisions and protected chess
implementation are unchanged; M2 remains unauthorized.
<!-- m1-blackwell-followup:end -->
