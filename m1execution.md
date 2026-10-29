# Gambit Zero — M1 Final Phase: 50K Human Chess Dataset

We are continuing **M1 — Dataset A: Human Chess Game Dataset**.

The initial M1 implementation and 1,000-game validation have already been reviewed and accepted for correctness. M0 is also accepted.

Your objective is to finish M1 by producing a reproducible, validated dataset of **50,000 human chess games** suitable for supervised training of our future 100M-parameter chess policy–value transformer.

**Do not start M2 or train a model.**

## 1. Read the existing implementation first

Before changing anything, read:

- `README.md` — research plan v0.4, especially Sections 3, 4, 7, 14 and 15.
- The complete M1 checklist in the repository.
- The existing M1 implementation, tests, and accepted 1K-game review report.
- `reports/m0_rules_parity.md`
- `reports/m0_commit_verification.md`
- The M0 Pgx adapter, action codec, adjudication logic and search tests.

Preserve all accepted M0 and M1 functionality. Do not redesign the existing data pipeline without a documented technical reason.

## 3. Dataset sizes and splits

Build exactly the following target dataset, subject to passing correctness and resource checks:

| Split | Source | Accepted games |
|---|---|---:|
| Train | January–July 2026 | 35,000 |
| Validation | August 2026 | 7,500 |
| Test | September 2026 | 7,500 |
| Total | | 50,000 |

For the training split, select **5,000 games from each of the seven months**.

The final accepted game counts, generated positions and filtering results must be reported separately for every source month and split.

Never mix positions from the same game across splits.

## 4. Resource preflight — before scanning archives

The proposed sampling strategy requires streaming through all nine monthly Lichess archives, potentially transferring hundreds of gigabytes of compressed data.

Before starting:

1. Measure available local disk space, disk quota, memory and network throughput.
2. Obtain and record the actual compressed sizes of the nine source archives.
3. Estimate total network transfer, processing cost and temporary storage requirements.
4. Check whether interrupted scans can be resumed safely without corrupting sampling or checksum validation.
5. Confirm that the proposed workload fits the available resource budget.

Stream compressed archives without unnecessarily retaining complete downloaded copies.

If the scan is not feasible, stop and report the measured blocker and a technically sound alternative. Do not silently switch to start-of-archive sampling.

## 5. Implement representative month-wide sampling

The earlier 1K smoke test was intended to establish correctness. The 50K pilot must also establish sampling quality.

For every monthly archive:

1. Stream the complete `.pgn.zst` source.
2. Use a fast header-only pre-filter before invoking the full `python-chess` parser.
3. Inspect relevant headers including `WhiteElo`, `BlackElo`, `TimeControl`, `UTCDate`, `Termination`, `Variant`, `Rated`, and game identity where available.
4. Reject obviously ineligible games before expensive full-game parsing.
5. Use deterministic, per-UTC-day reservoir or equivalent bounded top-K sampling.
6. Rank candidates using SHA256 of an unambiguously encoded combination of seed, month and stable game ID.
7. Ensure the resulting sample covers the full calendar month rather than disproportionately selecting early-archive games.
8. Account for candidates rejected by subsequent full parsing and validation, without silently violating sampling quotas.

Freeze the selection seed and algorithm in the dataset configuration.

Report the number of eligible candidates and accepted games per day, including any unavoidable gaps or redistribution.

### Source verification

Retrieve official Lichess SHA256 checksums where available.

Compute each monthly archive's checksum over the original compressed bytes while consuming the complete stream.

Compare it against the official checksum.

An incomplete scan must not be reported as checksum-verified.

Record source URLs, dates, bytes processed, hashes and scan status in the manifests.

### Reusable indexing

Create a reusable sampling/index artifact for potential future 1M–5M-game data scaling.

However, **do not assume that an arbitrary `.zst` archive supports random access to individual chess games**.

First verify the source's actual compression-frame structure.

If independent seekable frames are unavailable, use an accurate alternative such as:

- Deterministic daily sampling manifests
- Indexed selected-game caches
- Source checksums and logical game identifiers
- Verified offsets only where they are actually reusable

Clearly distinguish a true random-access index from a manifest that requires rescanning the source.

Do not invent seekability.

## 6. Game quality filters

Apply the following requirements:

- Standard rated chess only; exclude Chess960 and variants.
- Both players rated at least 1800 Lichess.
- Starting clock/base time at least 300 seconds (5 minutes), independent of increment.
- Valid game result: `1-0`, `0-1`, or `1/2-1/2`.
- Exclude abandoned games, rules infractions, corrupted games and illegal moves.
- Record original time control, result and termination reason.

### Rating distribution

At least **60% of accepted games must have both players rated 2000 or higher**.

The remainder must still satisfy the 1800 minimum.

For reporting, define rating bands using the lower of the two player ratings, so mixed 1800/2200 games are not accidentally classified as 2000+ games.

Aim to preserve the quota separately in train, validation and test. Document any shortages instead of silently changing the criteria.

Report actual rating distributions and game counts by rating band.

### Time-forfeit games

Keep eligible time-forfeit games for human-move policy supervision.

Make their value-loss weight configurable, with:

`value_weight = 0` by default.

Never treat a timeout as proof of poor positional value.

Report the time-forfeit rate under the new filters, including its impact on usable value-supervision examples.

## 7. Freeze Dataset D before extraction

Dataset D is for evaluation, not training.

Before the 50K extraction:

1. Freeze a versioned Lichess puzzle snapshot.
2. Record its source, snapshot date, puzzle IDs and checksum.
3. Freeze a fixed chess-opening evaluation suite, including exact positions and metadata.
4. Create a reproducible set of evaluation position identifiers for overlap checks.
5. Exclude exact frozen evaluation positions from Dataset A training rows.
6. Report exclusion counts and any unavoidable overlap with held-out diagnostic data.

Avoid accidentally invalidating the training distribution by excluding overly broad opening families or all positions merely related to an evaluation opening.

Define exact position matching explicitly, including side to move, castling and en passant semantics. Document when history-dependent information cannot be inferred from FEN alone.

Dataset D must remain separate from Dataset A.

## 8. Generate the human training data

Reuse the accepted M1 implementation.

For each accepted game:

1. Parse the complete PGN using `python-chess` offline.
2. Replay every move from the initial position through our corrected Pgx chess adapter.
3. Preserve game history, including repetition-relevant state.
4. At each pre-move position, generate the canonical model-input encoding.
5. Encode the human move using the existing 4,672-action Pgx codec.
6. Verify that the human move is legal.
7. Compare the complete decoded Pgx legal-move set with `python-chess` at every ply.
8. Assign the final game result from the side-to-move perspective.
9. Preserve source, game, player, split, termination and label-quality metadata.

**Per-ply legality mismatch count must be zero.**

Do not construct independent training positions by repeatedly importing FENs without their actual history.

### Outcome handling

Distinguish source PGN results from environment-generated terminal outcomes:

- Checkmate: if the final board is checkmate, verify corresponding Pgx termination and winner.
- Resignation: use the valid PGN result; Pgx need not be terminal.
- Agreed or claimed draw: use the valid PGN result; the M0 no-voluntary-claim convention may differ.
- Time forfeits: keep valid policy targets and default to zero value weight.

Retain the accepted M0 rule policy `tinychess-auto-fide-pgx-v1`.

Respect the 512-ply environment horizon. For this pilot, reject and count games that cannot be represented safely through complete replay under the existing horizon.

Never turn truncation into an ordinary draw or a fabricated WDL target.

Any new chess-rule inconsistency is a correctness blocker.

## 9. Keep one canonical model-input encoder

Use the encoder selected and documented during the M1 correctness phase.

It must be derived from the validated Pgx state and consistently represent the relevant board, side-to-move, castling, en passant, clock and available history information.

The same encoding must be usable for:

- Human-game positions in Dataset A
- Future GPU self-play states in Dataset B
- The 100M transformer in M2

Document exact shapes, dtypes and version identifiers.

Retain Pgx's native 4,672-action policy representation as the main policy head. The alternative attention-style policy head remains a future ablation.

Do not implement the transformer itself.

## 10. Dataset storage and manifests

Preserve the accepted M1 storage strategy unless new benchmarks demonstrate that it is unsuitable.

If choosing between game-level action sequences and materialized position features, compare storage consumption, decoding/replay cost, and random minibatch access. Record the decision.

Generate versioned, checksummed data artifacts and manifests containing:

- Source files and source hashes
- Sampling seed and selection algorithm
- Filtering rules and rejection counts
- Dataset/schema versions
- Pgx rules and action-codec versions
- Encoder version
- Accepted games and generated positions
- Outcome label provenance and weights
- Shard names, sizes and checksums
- Split assignment and duplicate diagnostics

Keep large generated data files outside Git.

## 11. Full correctness and leakage audit

Before accepting the final dataset, verify:

- Every selected human action is legal in Pgx.
- All Pgx legal moves match `python-chess` at every accepted ply.
- Correct action-index encoding for both colors.
- Castling, en passant, promotions and underpromotions.
- Correct WDL labels for White wins, Black wins and draws.
- Appropriate handling of resignations, time forfeits and claimed draws.
- Correct game termination and history handling.
- No whole-game overlap across train, validation and test.
- Dataset D evaluation positions excluded as specified.
- Correct rating quotas, time-control filters and source-month assignments.
- No unintended Stockfish or `[%eval]` training supervision.

### Position leakage report

Report unique-position overlap for:

1. Train versus validation.
2. Train versus test.
3. All three splits.
4. Train versus frozen Dataset D.

Report overlap both:

- Across all plies.
- Excluding opening plies before ply 10.

Use a clearly documented position-equivalence definition. Do not equate all repeated positions with duplicate games.

Provide total positions, unique positions and overlap percentages.

## 12. Reproducibility

Using the same frozen source data, schema, seed and configuration:

1. Run the 50K dataset construction twice, reusing verified immutable source/selection artifacts where appropriate.
2. Confirm identical selected game IDs.
3. Confirm identical dataset records and canonical output hashes.
4. Ensure shard ordering and compression metadata are deterministic, or separately hash canonical logical records when byte-for-byte file determinism is not guaranteed.
5. Record any differences and their causes.

Never claim identical results without actually performing the comparison.

## 13. Performance measurements

Report:

- Total network bytes transferred
- Archive scan throughput
- Eligible and accepted games per second
- PGN parse and Pgx replay throughput
- Generated training positions per second
- Total wall-clock time
- Peak CPU memory
- Disk space and output size
- Bytes per position/game
- Per-day and per-month sampling coverage
- Time spent in checksum verification, parsing, validation and output writing where measurable

Separate extraction time from source scanning and network transfer time.

### Separate Blackwell benchmark

Run the existing `benchmarks/bench_pgx_safe.py` on the NVIDIA RTX PRO 6000 Blackwell at batch sizes:

- 1,024
- 4,096
- 8,192

If the existing benchmark has different exact batch parameters, document the chosen settings.

Before large allocations, estimate memory requirements and run progressively.

Measure synchronized warm execution throughput, compilation time and peak GPU memory separately.

Do not cause avoidable OOMs or interpret environment steps/sec as complete self-play throughput.

If remote GPU access is unavailable, prepare exact reproduction commands and explicitly report the benchmark as pending rather than fabricating results.

This benchmark must not modify the M0 rules implementation or block independent CPU-based dataset validation without a concrete correctness reason.

## 14. Tests and regression requirements

Preserve all previously accepted M0 and M1 tests.

Add or extend tests for:

- Month-wide deterministic sampling and UTC-day quotas
- Correct handling of malformed/missing PGN headers
- Rating and time-control filter boundaries
- Deterministic tie-breaking and stable game identity
- Archive checksum failures and incomplete scans
- Full-game split isolation
- Side-to-move WDL and value weights
- Per-ply Pgx/python-chess legal parity
- History-dependent state replay
- Special moves and terminal positions
- 512-ply horizon handling
- Dataset D exclusion and position overlap
- Manifest integrity and reproducibility

Run the full available regression suite.

If any actual chess correctness test fails, stop the extraction and investigate. Do not work around mismatches by silently removing problematic positions.

## 15. Final deliverables

After processing the 50K pilot, create a comprehensive report:

`reports/m1_50k_pilot.md`

Include:

1. Current commit and branch.
2. Files created and modified.
3. Final schema and encoder contracts.
4. Exact source archives, checksums and sampling algorithm.
5. Accepted and rejected games by month, rating band and reason.
6. Training, validation and test game/position counts.
7. Per-day and per-month coverage.
8. Per-ply legal-action parity mismatch counts.
9. Outcome/WDL correctness and time-forfeit statistics.
10. Dataset D snapshot and exclusion results.
11. Position overlap and leakage statistics.
12. Reproducibility results with actual hashes.
13. Storage, memory, throughput, network and runtime measurements.
14. Separate Blackwell benchmark results, or explicitly documented pending status.
15. Known limitations, remaining risks and potential data-quality concerns.
16. A clear **PASS / CONDITIONAL PASS / FAIL** recommendation for M1 and whether the dataset is ready for M2.

Include the exact commands needed to reproduce the implementation and reports.

Do not invent measurements. Separate measured results, estimates and unresolved items.

## 16. Execution rules and stopping point

Work sequentially:

**Stage A — Preflight and source preparation**
Verify resources, freeze Dataset D and finalize sampling/index design.

**Stage B — Sampling implementation**
Implement and test month-wide deterministic sampling, header filtering, source hashing and manifests using bounded fixtures.

**Stage C — Full 50K pilot**
Only if the resource preflight and correctness gates pass, stream the required archives and produce the specified dataset.

**Stage D — Final verification**
Run the legality/leakage/reproducibility audits and generate the final report. Run the separate Blackwell benchmark where available.

**STOP after the completed M1 50K pilot report.**

Do not begin M2, implement the 100M transformer, start self-play RL, generate Dataset B, or launch larger 1M–5M-game extractions.

Do not automatically merge the M1 branch into `main`.

If an essential correctness test fails or the workload exceeds the available resource budget, stop safely, preserve the completed work and report the blocker.

Prioritize correctness, reproducibility and representative sampling over speed. Follow our README v0.4 and existing accepted architecture rather than introducing unnecessary new frameworks.