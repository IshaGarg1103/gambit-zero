# Current M1 status — 29 October 2026

Current evidence: [50K report](m1_50k_pilot.md). Historical 1K notes below
describe an earlier gate and are not the present extraction status.

| Area | Actual disposition |
|---|---|
| Real Dataset A generation | COMPLETE: 50K games, exact chronological quotas |
| Source SHA256 and full per-ply/final parity | COMPLETE: all 9 streams; 3,843,750 states; zero mismatches |
| Independent feature/shard reproduction | COMPLETE: all 9 months |
| Extra downloaded-row replay | Jan–June, 30K games; July–September optional |
| Full-50K metadata and position-disjoint index | COMPLETE and published |
| Independent Dataset D freeze | COMPLETE: 1K puzzles + 32 opening lines |
| Dataset D overlap and physical training exclusions | COMPLETE; zero retained train/D overlap |
| Revised immutable card/catalog and complete file inventory | VERIFIED |
| Comprehensive acceptance report | Updated with measured results and explicit gaps |
| New large-batch Blackwell benchmark | PENDING; exact commands supplied; permitted deferral |
| Historical resource/rejection telemetry | Explicit gaps documented; no invented measurements |
| Regression | See actual `m1_closeout_tests.txt` |
| M2, larger extraction, merge into main | Not performed or authorized |
| GitHub source handoff | Prepared for user-run commit/push to `m1/human-dataset` |

Recommendation: CONDITIONAL PASS for M1 review. CPU Dataset A checks are
accepted only with the complete receipts above. GPU deferral and unavailable
historical telemetry must be acknowledged; no 100% checklist claim is made.
