# Historical initial M1 checklist

This is the original pre-implementation checklist, preserved for provenance.
Its unchecked boxes and branch instructions are historical, superseded by
[m1execution.md](../m1execution.md), subsequent user instructions and
[the current acceptance report](../reports/m1_50k_pilot.md).

1. Set up the M1 branch and data configuration

1/5
[ ] Merge the accepted M0 PR into main, then create m1/human-dataset.
[ ] Preserve all M0 dependency locks and the corrected Pgx environment.
[ ] Create configs/data_human.yaml to hold source months, filtering, sampling, split and output settings.
[ ] Define a versioned Dataset A schema and manifest format before writing large datasets.
[ ] Measure actual usable storage/quota and establish a download and output budget.


2. Build the Lichess PGN streaming pipeline

0/8
[ ] Use the Lichess standard-rated monthly PGN archives.
[ ] Use January–July 2026 for training, August for validation, and September for held-out testing.
[ ] Read compressed .pgn.zst streams incrementally using zstandard.
[ ] Use python-chess to parse complete games and their mainline moves offline.
[ ] Do not require entire archives to be downloaded and decompressed locally.
[ ] Record source month, archive URL, source size, retrieval provenance, and available checksums.
[ ] Create deterministic selection/resumption behavior so runs can be reproduced.
[ ] Handle malformed games gracefully without crashing the entire ingestion pipeline.


3. Implement game filtering

0/9
[ ] Accept standard rated chess only; reject Chess960 and all variants.
[ ] Require both players to have Lichess ratings of at least 1800.
[ ] Support an optional stronger-player subset using ratings of at least 2000.
[ ] Prefer rapid/classical or longer blitz, and retain original time-control metadata.
[ ] Accept only complete, successfully parsed games with a valid result: 1-0, 0-1, or 1/2-1/2.
[ ] Reject aborted, corrupted and illegal-move games.
[ ] Record game termination and handle time-forfeit games separately; exclude or downweight unreliable outcome targets.
[ ] Capture acceptance/rejection reason counts and examples.
[ ] Avoid an excessive concentration of repeated openings and monitor game-phase distribution.


4. Reconstruct game states using our M0 chess adapter

0/9
[ ] Replay each accepted game's moves in order, preserving the actual chess history.
[ ] Store the position immediately before each human move.
[ ] Use the versioned M0 UCI↔Pgx action codec to encode the played move into the 4,672-action policy space.
[ ] Generate the complete legal-action representation using the corrected Pgx adapter.
[ ] Assert that the human move is present in the legal-action mask.
[ ] Preserve castling rights, en passant, halfmove clock, side to move, and repetition/history-dependent information.
[ ] Verify state and action consistency against python-chess as the offline oracle.
[ ] Do not reconstruct every position from an independently imported FEN; M0 established that this discards prior repetition history.
[ ] Define explicit handling/rejection of games that cannot be represented safely under the M0 512-ply horizon.


5. Build one supervised training record per eligible position

0/10
[ ] Store a stable game ID, source month, split and ply number.
[ ] Store board representation, full FEN, and sufficient move/history features.
[ ] Store the human's UCI move and versioned Pgx policy-action index.
[ ] Store every legal action (or a lossless, versioned packed action mask).
[ ] Store side to move, ratings, time control, game termination and provenance.
[ ] Assign the final WDL result from the perspective of the player to move at that position.
[ ] Store WDL as both a class and/or numeric target: win +1, draw 0, loss -1.
[ ] Keep labels and weights separate when a game's ending makes the value target unreliable.
[ ] Optionally retain remaining plies for the later moves-left experiment.
[ ] Record label_source=human_game; never incorporate Stockfish or external evaluation labels.


6. Write efficient, versioned dataset files

0/6
[ ] Use sharded Parquet as the primary training format.
[ ] Optionally use compressed JSONL for small debugging samples.
[ ] Decide and document the storage encoding for 4,672-dimensional legal masks.
[ ] Generate checksums for produced shards.
[ ] Generate dataset manifests recording schema, split, filters, version identifiers, source and row counts.
[ ] Keep raw PGNs and derived training datasets out of Git; commit configurations, schemas, reports and small tests.


7. Run leakage and distribution audits

0/7
[ ] Keep complete games in exactly one split; never split their positions across train and validation/test.
[ ] Check for duplicate game IDs and repeated board positions across games and splits.
[ ] Report train/validation/test position overlap and create a position-disjoint evaluation subset.
[ ] Do not collapse all identical board positions into a single move target; different human moves can be valid examples.
[ ] Cap or rebalance overrepresented positions/openings when sampling for training.
[ ] Report WDL, phase, opening, side-to-move, rating, time-control and termination distributions.
[ ] Reserve frozen evaluation fixtures from training where known, and document any overlap.


8. Create automated data correctness tests

0/9
[ ] Test UCI/action-index round trips and complete legal-action masks.
[ ] Test castling, en passant, queen promotions and all underpromotions.
[ ] Test WDL signs independently for White wins, Black wins and draws.
[ ] Test parser corruption, aborted games, invalid ratings/results and variants.
[ ] Test repeated positions, EP state, castling rights and history-sensitive repetition.
[ ] Test whole-game split separation and deterministic processing.
[ ] Test time-forfeit exclusions/weights and handling of unusually long games.
[ ] Check that all accepted records have valid targets and version metadata.
[ ] Rerun M0 regression tests to ensure Dataset A code has not broken the environment.


9. Run the incremental extraction pilots

0/7
[ ] First generate 100 complete accepted games and inspect their records.
[ ] Expand to 1,000 accepted games; verify legality, labels, metadata and outputs.
[ ] Optionally run a 10K-game intermediate scaling check to measure throughput and disk usage.
[ ] Only after smoke-test acceptance, build the 50K accepted-game M1 pilot.
[ ] Include validation/test data from their separately frozen months, with explicitly documented sample sizes.
[ ] Measure processed games/sec, accepted games/sec, rows/sec, bytes/position, peak memory and rejection rates.
[ ] Do not scale to the proposed 1M–5M-game main-training dataset automatically.


10. Produce the M1 acceptance report

0/7
[ ] Document sources, schema and exact extraction configuration.
[ ] Report accepted/rejected game counts and detailed filtering reasons.
[ ] Report total positions, unique-position counts and position overlap.
[ ] Report label distributions, action legality and history verification results.
[ ] Report output shard sizes, checksums, runtime and storage requirements.
[ ] Document known limitations and a reproducible regeneration command.
[ ] Give an explicit PASS / CONDITIONAL PASS / FAIL recommendation for M2 readiness.