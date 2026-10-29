# M1 initial acceptance-gap assessment — 29 October 2026

**Historical gap assessment.** Follow-up work has since published the full
position-disjoint diagnostic index, frozen Dataset D, excluded 150 matching
training rows and independently verified the revised publication. The first
upload attempt stopped safely on HTTP 429; its continuation passed after the
cooldown. Final revision: `b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c`.
Current evidence, exact retry window, regression results and remaining conditions
are in [the live checklist](m1_checklist_status.md),
[the acceptance report](m1_50k_pilot.md), and
[machine-readable acceptance](m1_final_acceptance.json).
Do not treat the initial remaining-work list below as the current live status.

Assessment against `m1execution.md`, README v0.4, final pinned HF catalog and
checksummed audit; later user instructions override the original Dataset D timing.

## Completed

- Exact 50K: 35,000 train / 7,500 validation / 7,500 test; 3,793,750 positions.
- All nine full compressed source checksums verified; 263,446,966,206 successful
  source bytes processed. This is not total historical traffic including failed
  attempts, retransmissions or transport overhead.
- Month-wide deterministic daily/rating sampling, filters, history-preserving
  replay, target weights and horizon checks. All 3,843,750 recorded chess states
  passed parity with zero mismatches (positions plus final game boards).
- All nine months independently reproduced; matching IDs, records and shard digests.
- Final inventory: 846 artifacts / 2,137,251,914 bytes verified; small recovery
  and source backups additionally uploaded. Generated local caches removed.
- Full regression: 204 passed in 48.81 seconds following the transport fix.
- Final split-overlap audit published and checksum-verified; locally saved as
  `reports/m1_final_split_audit.json`. Whole-game isolation is enforced by the
  audited game identity/content index. Board-position overlap is measured, not
  incorrectly asserted zero.
- Extra post-upload full row audit completed for January–June: 30,000 games,
  2,274,370 rows. This is additional assurance beyond the required full-generation
  parity and independent reproduction that already cover all 50K.

## Remaining acceptance work

1. **Dataset D is explicitly deferred, not checked.** Freeze an independently
   selected versioned puzzle/opening suite, perform exact overlap matching,
   exclude matching training positions and publish a revised A if needed before
   model training. Original extraction-before-D sequencing was changed by the
   user; do not claim that original requirement was fulfilled. No original
   Lichess archive rescan should be needed for overlap filtering.
2. **Final position-disjoint diagnostic subset.** The bounded 1K smoke produced
   one, but the streaming 50K destination publishes overlap aggregates without
   producing the corresponding full-pilot validation/test row-selection index.
   Produce/publish that index for reporting alongside the ordinary chronological
   metrics, with an explicit board-identity definition. This does not mean
   discarding every repeated opening position from the whole dataset.
3. **Finish the comprehensive report and acceptance disposition.** The current
   `m1_50k_pilot.md` is an automated completion summary. Consolidate source hashes,
   per-month rejection/rating/outcome/timeout/coverage statistics, actual
   reproducibility hashes, overlap tables/percentages, schema/encoder contracts,
   storage/runtime evidence, limitations, exact commands and current branch/commit.
   Its 4.236-hour elapsed time refers to the final September invocation, not
   the complete multi-day project. Reservation allowances are not measured
   historical network bytes. End with PASS / CONDITIONAL PASS / FAIL and a
   separate explicit M2-readiness statement.
4. **Separate large-batch Blackwell benchmark.** 1,024 / 4,096 / 8,192 synchronized
   measurements with separate compile time and peak memory remain pending. The
   spec allows an explicit pending disposition and exact commands if remote
   access is unavailable; this is not an unreported CPU-data correctness failure.
   Existing command: `.venv/bin/python benchmarks/bench_pgx_safe.py --output reports/m0_large_batch_blackwell.json`
   on the validated Molab GPU environment (`JAX_PLATFORMS=cuda`).

After remaining code changes, run their regression checks. A full 204-test run
already exists; it must not be mislabeled as never executed.

## Optional extra assurance, not missing generation validation

Repeat the independent downloaded-row audit for July–September to extend the
extra audit from 30K to all 50K games. Those months already passed all required
per-ply generation checks and independent cloud reproduction. Before this extra
audit, adapt the audit helper's 5,000-game assumption for 7,500-game held-out
months and its configuration lookup for the now-cleaned runtime.

## Final published overlap evidence

| Scope | Train/validation shared unique positions | % of validation unique | Train/test shared unique positions | % of test unique | Shared in all three |
|---|---:|---:|---:|---:|---:|
| All plies | 23,589 | 4.6876% | 23,178 | 4.5982% | 10,405 |
| Ply >=10 | 12,665 | 2.6099% | 12,143 | 2.4994% | 4,054 |

Identity: SHA256(pieces, side to move, castling, legally capturable EP); clocks
and repetition history are excluded for this conservative board-overlap metric.

Pinned data revision: `d409fce8bd444c2d456ef4c343dbc9b5e25981eb`.

Assessment: Dataset A generation/reproduction is COMPLETE; final M1 acceptance
remains conditional. Do not start M2 or larger extraction. User controls Git
review, staging, commit, push and merge. No Git commands were executed here.
