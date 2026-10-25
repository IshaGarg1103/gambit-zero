# M1 checklist disposition at the 1,000-game stop gate

The user's latest scope takes precedence over `Untitled` and the reviewer draft.

| Area | Disposition |
|---|---|
| Branch/merge/commit/push | Intentionally not performed; user will manage Git manually |
| M0 acceptance | Preserved; targeted hardening passed, no rules/action policy change |
| Random parity | 10K then 50K random positions plus explicit draw histories completed; larger stress proposed only |
| MCTS tactics | Both-color mate-in-one/sign tests passed; finite-budget mate-in-two diagnostics reported separately |
| Large batches | CPU 128 measured; unsafe 1K/4K/8K skipped with reasons; new GPU/device-memory results pending |
| Head and encoder | Native 4672-action head adopted; one versioned Pgx-derived encoder frozen; transformer not implemented |
| Data config and schema | Implemented, versioned, bounded to 100/1000 quotas |
| Streaming source and provenance | Nine offset-zero 4 MiB prefixes acquired, streamed and cached with hashes; full archives not downloaded/verified |
| Filters and outcomes | Implemented; nonboard Normal endings remain ambiguous; time-forfeit value weights masked |
| History-preserving replay | Full accepted-game action replay and legal/history/status/reward parity completed |
| Horizon | Reject/count beyond horizon or actual truncation; synthetic rejection tests; no such real cases observed |
| Two storage prototypes | Both Parquet representations implemented and benchmarked; materialized primary selected from measurements |
| Tests | Full regression/data and safe-path suites run; see exact test outputs |
| 100-game gate | Completed before 1K, real human PGNs |
| 1000-game gate | Completed twice with identical base data/manifests and strict disjoint derived index |
| Leakage and distributions | Whole-game IDs, board identities overall/ply≥10, disjoint subsets and distributions audited |
| Frozen external fixtures | Configurable exclusion tested; no actual frozen puzzle/GPT fixture manifest supplied |
| Representative month-wide sampling | Proposed reproducible UTC-day hash reservoirs; not implemented or executed; current prefix bias explicitly measured |
| Opening/position rebalancing | Distribution measured; no silent smoke downsampling; future training sampler still required |
| Reports/reproduction | M0 follow-up, M1 smoke, schema, encoder contract, source/provenance, measurements and exact commands delivered |
| 50K extraction / M2 / self-play | Not started; separate review required |

Recommendation: PASS for the bounded correctness/storage smoke. Do not scale
prefix sampling or begin M2. Review representative acquisition, budgets, frozen
evaluation exclusions and reader performance before authorizing a 50K pilot.
