"""Render the acceptance report from saved measured evidence, never promises."""
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from tinychess.data.evidence import EvidenceStore, rewrite_archived_links
EVIDENCE = EvidenceStore(ROOT)


def read(name):
    return EVIDENCE.read_json('reports/'+name, required=False)


def git_identity():
    """Capture the source HEAD used to generate the report without self-reference."""
    head = (ROOT / '.git/HEAD').read_text().strip()
    if not head.startswith('ref: '):
        return 'detached', head
    ref = head[5:]
    path = ROOT / '.git' / ref
    sha = path.read_text().strip() if path.exists() else None
    if sha is None and (ROOT / '.git/packed-refs').exists():
        sha = next((line.split()[0] for line in (ROOT / '.git/packed-refs').read_text().splitlines()
                    if line.endswith(' '+ref)), 'unavailable')
    return ref.removeprefix('refs/heads/'), sha


def render():
    status = read('m1_50k_run_status.json')
    audit = read('m1_final_split_audit.json')['audit']
    diagnostics = read('m1_50k_diagnostics.json')
    frozen = read('m1_dataset_d_freeze.json')
    exclusions = read('m1_dataset_d_exclusions.json')
    upload_block = read('m1_dataset_d_upload_diagnostic.json')
    publication = read('m1_dataset_d_publication_verification.json')
    captured_acceptance = read('m1_final_acceptance.json')
    if EVIDENCE.manifest and captured_acceptance:
        branch = captured_acceptance['local_branch']
        commit = captured_acceptance['source_head_at_report_generation']
    else:
        branch, commit = git_identity()
    months = status['months']
    scan_seconds = sum(m['source_scan_measurements']['elapsed_seconds'] for m in months.values())
    construction_seconds = sum(m['construction_measurements']['elapsed_seconds'] for m in months.values())
    source_bytes = sum(m['source']['archive_size_bytes'] for m in months.values())
    parity = sum(m['parity_states'] for m in months.values())
    mismatch = sum(m['parity_mismatches'] for m in months.values())
    complete_diag = diagnostics and diagnostics['status'] == 'PASS' and diagnostics.get('temporary_cleanup_completed')
    complete_d = exclusions and exclusions['status'] == 'PASS' and exclusions.get('temporary_cleanup_completed')
    complete_publication = (complete_d and publication and publication['status'] == 'PASS'
                            and publication['revision'] == exclusions['final_revision'])
    rate_blocked = bool(not complete_d and upload_block and exclusions
                        and exclusions.get('status') == 'FAIL'
                        and (exclusions.get('http_status') == 429 or 'http_status' not in exclusions))
    # CPU data correctness can pass while the broader M1 review retains the
    # explicitly permitted GPU deferral and honest historical telemetry gaps.
    disposition = 'CONDITIONAL PASS'
    coverage = {month: data['coverage'] for month, data in months.items()}
    rejection = {month: dict(header=data['header_rejections'], full_validation=data['validation_rejections'])
                 for month, data in months.items()}
    # Frozen coverage/rejection tables live in the verified archive. Do not
    # recreate removed generated reports in the tracked source directory.
    if EVIDENCE.manifest is None:
        for filename, data in [('m1_50k_coverage.json', coverage), ('m1_50k_rejections.json', rejection)]:
            (ROOT / 'reports' / filename).write_text(json.dumps(data, sort_keys=True, indent=2)+'\n')
    text = f'''# M1 — Dataset A 50K pilot acceptance report

Assessment date: 29 October 2026. **{disposition}; M2 is not authorized.**
The real 50K generation and independent reproduction are complete. Dataset D
exclusions and the diagnostic index are separate acceptance conditions below;
their completion must be supported by actual receipts. The separate Blackwell
benchmark remains pending when no GPU session is available.

## Repository and immutable cloud evidence

- Source branch: `{branch}`; HEAD at report generation: `{commit}`.
  This identifies the report's base, not the subsequent commit containing it.
  The user will execute source staging, commit and push manually.
  The final commit identity is recorded by Git history, avoiding a self-referential
  hash embedded inside that same commit. No merge into `main` or M2 is authorized.
- Public data: https://huggingface.co/datasets/ishagarg1103/gambit-zero
- Original 50K data revision: `{status['final_revision']}`.
- Original inventory: {status['verified_artifacts']:,} verified artifacts,
  {status['verified_artifact_bytes']:,} bytes ({status['verified_artifact_bytes']/2**30:.3f} GiB).
  Small source/recovery backups and later evaluation/diagnostic objects are additional.
- Recovery receipt: `{status['recovery_receipt']['path_in_repo']}`,
  SHA256 `{status['recovery_receipt']['sha256']}`.
- Source-code backup: `{status['source_snapshot_receipt']['path_in_repo']}`,
  SHA256 `{status['source_snapshot_receipt']['sha256']}`.

## Final original split counts and coverage

| Month | Split | Accepted / reproduced | Rows | Both >=2000 | Days | Scan hours | Primary build hours |
|---|---|---:|---:|---:|---:|---:|---:|
'''
    for month, m in sorted(months.items()):
        text += f'| {month} | {m["split"]} | {m["validated_games"]:,} / {m["cloud_reproduced_games"]:,} | {m["positions"]:,} | {m["strong_games"]:,} ({100*m["strong_games"]/m["validated_games"]:.2f}%) | {len(m["coverage"])} | {m["source_scan_measurements"]["elapsed_seconds"]/3600:.3f} | {m["construction_measurements"]["elapsed_seconds"]/3600:.3f} |\n'
    text += f'''
Total: **35,000 train / 7,500 validation / 7,500 test games**;
**2,655,000 / 569,124 / 569,626 position rows**, respectively.
All {sum(len(m['coverage']) for m in months.values())} calendar days are represented.
Exact accepted/strong counts per day: [coverage](m1_50k_coverage.json).
Header and full-validation rejection counts per month:
[rejections](m1_50k_rejections.json). Header rejection is the first failed filter,
not all possible reasons for rejecting the same game. Eligible but unselected
candidates are not mislabeled as correctness failures.

## Frozen sources, selection and reproducibility

URL template: `https://database.lichess.org/standard/lichess_db_standard_rated_YYYY-MM.pgn.zst`.
Each successful scan consumed the complete compressed stream and matched its
official published SHA256 at exact EOF. No compressed/decompressed archive was
retained. Interrupted attempts do not establish a verified source hash.

| Month | Compressed bytes | Verified source SHA256 | Reproduced shard digest |
|---|---:|---|---|
'''
    for month, m in sorted(months.items()):
        text += f'| {month} | {m["source"]["archive_size_bytes"]:,} | `{m["source"]["published_sha256"]}` | `{m["shard_digest"]}` |\n'
    text += '''
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
'''
    text += f'''
## Correctness and regression evidence

- Required generation checks: **{parity:,} states, {mismatch} mismatches**
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
'''
    for label in ['all_plies', 'ply_ge_10']:
        a = audit[label]
        text += f'| {label} | {a["train_unique"]:,} | {a["validation_unique"]:,} | {a["test_unique"]:,} | {a["train_validation_shared"]:,} ({a["train_validation_shared_percent_of_validation"]:.4f}%) | {a["train_test_shared"]:,} ({a["train_test_shared_percent_of_test"]:.4f}%) | {a["all_three_shared"]:,} |\n'
    if complete_diag:
        text += f'''
The full metadata closeout independently reproduced these aggregates and checked
all 50K game identities, source months, rating/time filters, row order and WDL/
timeout masks. It did not regenerate tensor features or replace prior chess parity.
Position-disjoint diagnostic index: **{diagnostics['diagnostic_index_counts']}**.
Selection excludes any held-out board found in training at *any* ply. Validation
and test may overlap each other; repeated unseen boards remain observations.
Index rows identify source path, zero-based row offset, game, ply and board hash;
they contain no feature tensors or invented labels. The original chronological
dataset is unchanged. Receipts and distributions:
[diagnostics](m1_50k_diagnostics.json).
Published revision: `{diagnostics.get('published_revision', 'not published')}`.
The first-run raw summary's legacy `peak_working_disk_bytes` field denotes
index size at completion, not a continuously sampled high-water measurement;
the current implementation names this `index_disk_bytes_at_end`.
'''
        counts = diagnostics['counts']
        distributions = diagnostics['distributions']
        text += f'''
Repeated complete action-trajectory groups across distinct games:
**{diagnostics['repeated_trajectory_groups']:,}**. These are distinct source game
identities, not necessarily identical player/clock/outcome records.
Original WDL row counts (win/draw/loss class 0/1/2):
`{distributions['wdl_class']}`. Value-masked rows:
**{counts['value_masked_rows']:,}**; nonzero-value-weight rows:
**{counts['positions']-counts['value_masked_rows']:,}**.
Phase-by-ply and side-to-move distributions are in the diagnostic JSON; phase
means opening <20, middlegame <80, late >=80 plies, not engine/material judgments.

| Month | Lower-player rating bands (200-point bins) | Ending semantics | ECO labels present |
|---|---|---|---:|
'''
        for month in sorted(months):
            rating = distributions[month+':rating_band_lower_player']
            ending = distributions[month+':ending']
            eco = distributions[month+':eco']
            text += f'| {month} | `{rating}` | `{ending}` | {len(eco)} |\n'
        text += '\n`unknown` ECO is a missing source annotation, not an inferred opening family.\n'
    else:
        text += '\nFull-50K position-disjoint index: **not yet complete**; see running diagnostic report.\n'
    text += '\n## Dataset D disposition\n\n'
    if frozen:
        text += f'''Frozen independently of Dataset A overlap/model performance:
{frozen['puzzles']:,} puzzles and {frozen['openings']} opening lines,
{frozen['unique_evaluation_boards']:,} exact board identities.
Source snapshot, IDs, source checksums and verified immutable receipt:
[freeze evidence](m1_dataset_d_freeze.json).
Puzzle checksum is a **computed full compressed-source fingerprint**, not a
publisher checksum. Puzzle FEN precedes the opponent setup move; the actual root
is after that move. Solver/opponent continuation boards are included in the
exact overlap set; openings contribute only their exact final roots, never
entire related opening families. Puzzle repetition history is unknown.
The suite is evaluation only and provides no Dataset A strategic targets.
'''
    else:
        text += 'Dataset D freeze is pending; no puzzle overlap or exclusions are claimed.\n'
    if complete_d:
        text += f'''
Physical exclusions and revised catalog: [verified results](m1_dataset_d_exclusions.json).
Original matching rows: {exclusions['original_dataset_d_matching_rows']}.
Original matching unique boards: {exclusions['original_dataset_d_matching_unique']}.
Every exact matching **training** row was physically removed; held-out matches
are reported and retained. Train/D shared unique boards after filtering: **zero**
both across all plies and at ply >=10. Accepted game counts remain exactly 50K;
compact actions remain provenance, not an unfiltered replacement training reader.
Revised position counts: {exclusions['counts']}.
Revised final HF revision: `{exclusions['final_revision']}`.
Exclusion pass: {exclusions['elapsed_seconds']:.2f} seconds,
{exclusions['source_downloaded_payload_bytes']:,} source Hub payload bytes read,
peak process RSS {exclusions['peak_process_rss_bytes']:,} bytes.
Its conservative temporary-disk plan is {exclusions['plan']['planned_peak_local_bytes']/2**30:.2f} GiB,
including the index/rollback limit, bounded shard spool and metadata margin;
20 GiB remains reserved. This is a planning bound, not a continuously measured
disk high-water mark. No source monthly archive was rescanned.
Referenced primary game/position artifacts: {exclusions['referenced_primary_artifact_bytes']:,}
compressed bytes; new derivative artifacts published: {exclusions['newly_published_artifact_bytes']:,}
bytes (logical artifact sizes, not metered HTTP traffic).
Every changed shard was independently filtered twice with identical bytes;
every retained field/tensor byte was compared to the verified original. This
is repeated lossless filtering on top of the original full Pgx reproduction,
not a claim of a second new PGN replay after exclusions.
Original data remains accessible at its pinned revision. The original
position-disjoint index remains a conservative valid subset for the reduced
training set; it is not claimed maximal against that derivative.
'''
        text += '\n| Postfilter scope | Train/validation shared | Train/test shared | All three | Train/D | Validation/D | Test/D |\n|---|---:|---:|---:|---:|---:|---:|\n'
        for scope in ['all_plies', 'ply_ge_10']:
            a = exclusions['postfilter_overlap'][scope]
            text += f'| {scope} | {a["train_validation_shared"]:,} | {a["train_test_shared"]:,} | {a["all_three_shared"]:,} | {a["train_dataset_d_shared_unique"]:,} | {a["validation_dataset_d_shared_unique"]:,} | {a["test_dataset_d_shared_unique"]:,} |\n'
        text += '\nPuzzle-source game IDs shared with A (separate from board overlap): '
        text += str({split: len(ids) for split, ids in exclusions['dataset_d_source_game_ids_shared'].items()})+'; IDs are in the exclusion report.\n'
    else:
        text += '''
**Dataset A overlap/exclusion remains unresolved.** Original generation explicitly
deferred Dataset D at the user's request and applied no D exclusions. Freeze
alone is not an overlap audit or a training-ready revised Dataset A. The original
pre-extraction Dataset D ordering cannot be retroactively claimed.
'''
        if rate_blocked:
            text += f'''
**Current blocker: Hugging Face HTTP 429.** A single idempotent retry received
`Retry-After: {upload_block['retry_after_seconds']}`. Uploads stopped safely.
Do not retry before **{upload_block['resume_not_before_ist']} (IST)**.
The original active card/catalog remain unchanged. Verified staging shards are
unreferenced by the active catalog; the one unverified local shard is retained
for recovery. First attempt: [saved failure](m1_dataset_d_exclusions_attempt1.json).
Safe HTTP diagnostic: [retry window](m1_dataset_d_upload_diagnostic.json).

The resumed uploader batches up to 8 immutable files / 64 MiB per commit,
while retaining 32 MiB per-shard bounds and per-file streamed verification.
This reduces commit pressure within service limits; it does not bypass the
retry window. Atomic root publication and upload batches passed local failure/
recovery tests; their new live production execution remains pending.

After the retry window, from the project root:

```bash
.venv/bin/python scripts/apply_dataset_d_exclusions.py --execute
.venv/bin/python scripts/verify_m1_dataset_d_publication.py
.venv/bin/python scripts/report_m1_pilot.py
```

Execution refuses early retries before any network request. It reuses verified
immutable uploads, preserves failed local files and rebuilds only the bounded
metadata index. It does not rescan Lichess monthly archives.
'''
    text += '\n## Final immutable publication verification\n\n'
    if complete_publication:
        text += f'''Independent post-publication check: **PASS**, revision
`{publication['revision']}`. All nine month manifests, frozen evaluation receipt,
split counts, human-readable card counts and exact `human_v3`/`games` file lists
agree. {publication['unique_primary_objects']:,} unique primary objects were checked
at that final revision: {publication['lfs_objects_checked']:,} by LFS SHA256/size
and {publication['regular_objects_read_back']:,} regular objects by full body
read-back. Small-object payload read: {publication['full_read_payload_bytes']:,} bytes.
Large shards had already been read back in full during upload/original audits;
this extra verification compares their final LFS content IDs, not a second
download of every tensor file. See [publication verification](m1_dataset_d_publication_verification.json).
'''
    else:
        text += 'Independent final-revision verification has not yet passed.\n'
    text += f'''
## Measured resource accounting and limitations

- Successful complete source streams: **{source_bytes:,} bytes**, **{scan_seconds/3600:.3f} hours**
  combined source/network/header-scan wall time; aggregate {source_bytes/scan_seconds/10**6:.3f} MB/s.
- Primary feature construction/validation/upload: **{construction_seconds/3600:.3f} hours**
  combined across nine months; this excludes independent reproduction, extra
  row audits, failed attempts and human idle time.
- Final September invocation: **{status['elapsed_seconds']/3600:.3f} hours**;
  it is not the total project duration. Its peak process RSS:
  {status['peak_process_rss_bytes']:,} bytes; do not claim this is a monitored
  lifetime maximum across all previous processes.
- Historical archive allowance {status['archive_payload_allowance_bytes']:,} bytes
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
  {status['verified_artifact_bytes']/status['training_positions']:.1f} compressed bytes/position,
  {status['verified_artifact_bytes']/50000:.1f} bytes/game, including companion
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
JAX_PLATFORMS=cuda .venv/bin/python benchmarks/bench_pgx_safe.py \\
  --batches 128 1024 4096 8192 --repeats 30 \\
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
.venv/bin/python scripts/audit_hf_rows.py \\
  --revision {status['final_revision']} \\
  --months 2026-07 2026-08 2026-09 \\
  --config configs/audit_human_50k.yaml \\
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

**{disposition} for M1; do not start M2.** The required real Dataset A
generation, parity, checksum verification and reproduction passed.
CPU dataset correctness/reproducibility/exclusion checks are PASS only once
their verified receipts above are complete. Review the separate GPU pending
disposition and the documented historical telemetry limitations before declaring
every M1 checklist item closed. No unavailable measurement is represented as done.
No M2 training, Dataset B/self-play, larger extraction or automatic merge is authorized.
'''
    text = rewrite_archived_links(text, 'reports/m1_50k_pilot.md', root=ROOT)
    (ROOT / 'reports/m1_50k_pilot.md').write_text(text)
    checklist = ROOT/'reports/m1_checklist_status.md'
    existing = checklist.read_text()
    historical = existing.split('\n---\n', 1)[1] if '\n---\n' in existing else existing
    current = f'''# Current M1 status — 29 October 2026

Current evidence: [50K report](m1_50k_pilot.md). Historical 1K notes below
describe an earlier gate and are not the present extraction status.

| Area | Actual disposition |
|---|---|
| Real Dataset A generation | COMPLETE: 50K games, exact chronological quotas |
| Source SHA256 and full per-ply/final parity | COMPLETE: all 9 streams; 3,843,750 states; zero mismatches |
| Independent feature/shard reproduction | COMPLETE: all 9 months |
| Extra downloaded-row replay | Jan–June, 30K games; July–September optional |
| Full-50K metadata and position-disjoint index | {'COMPLETE and published' if complete_diag else 'IN PROGRESS / not accepted yet'} |
| Independent Dataset D freeze | {'COMPLETE: 1K puzzles + 32 opening lines' if frozen else 'PENDING'} |
| Dataset D overlap and physical training exclusions | {'COMPLETE; zero retained train/D overlap' if complete_d else 'PENDING / not claimed'} |
| Revised immutable card/catalog and complete file inventory | {'VERIFIED' if complete_publication else 'PENDING'} |
| Comprehensive acceptance report | Updated with measured results and explicit gaps |
| New large-batch Blackwell benchmark | PENDING; exact commands supplied; permitted deferral |
| Historical resource/rejection telemetry | Explicit gaps documented; no invented measurements |
| Regression | See actual `m1_closeout_tests.txt` |
| M2, larger extraction, merge into main | Not performed or authorized |
| GitHub source handoff | Prepared for user-run commit/push to `m1/human-dataset` |

Recommendation: CONDITIONAL PASS for M1 review. CPU Dataset A checks are
accepted only with the complete receipts above. GPU deferral and unavailable
historical telemetry must be acknowledged; no 100% checklist claim is made.
'''
    checklist.write_text(current if EVIDENCE.manifest else current+'\n---\n'+historical)
    tests = (ROOT/'reports/m1_closeout_tests.txt').read_text()
    passed = re.search(r'(\d+) passed in ([\d.]+)s', tests)
    tests_pass = bool(passed and not re.search(r'\d+ (?:failed|errors?)', tests))
    acceptance = dict(recommendation='CONDITIONAL_PASS',
        cpu_dataset_checks='PASS' if complete_diag and complete_d and complete_publication and tests_pass else 'INCOMPLETE',
        original_revision=status['final_revision'],
        revised_revision=exclusions.get('final_revision') if complete_d else None,
        dataset_d_exclusions_complete=bool(complete_d), diagnostic_index_complete=bool(complete_diag),
        publication_verification_complete=bool(complete_publication),
        regression=dict(status='PASS' if tests_pass else 'NOT_VERIFIED',
                        passed=int(passed[1]) if passed else None, seconds=float(passed[2]) if passed else None),
        gpu_large_batch_benchmark='PENDING; explicitly permitted deferral',
        historical_unavailable_metrics=['total failed-attempt/retransmission traffic',
            'fine rejected-game rating histogram', 'historical exact peak working disk',
            'separate parser and Pgx kernel timings'],
        m2_started=False, m2_authorized=False,
        local_branch=branch, source_head_at_report_generation=commit,
        github_repository='IshaGarg1103/gambit-zero', github_target_branch='m1/human-dataset',
        github_commit_push_executor='user', agent_git_commands_allowed=False,
        source_commit_identity='Git history after the user commits this prepared work',
        merge_to_main_authorized=False)
    if EVIDENCE.manifest and captured_acceptance:
        result_keys = ['cpu_dataset_checks','original_revision','revised_revision',
                       'dataset_d_exclusions_complete','diagnostic_index_complete',
                       'publication_verification_complete','regression']
        if any(acceptance[key] != captured_acceptance[key] for key in result_keys):
            raise ValueError('Frozen validation results changed; publish a new evidence version explicitly')
        acceptance = captured_acceptance
    if rate_blocked:
        acceptance['upload_blocker'] = upload_block
    (ROOT/'reports/m1_final_acceptance.json').write_text(json.dumps(acceptance, sort_keys=True, indent=2)+'\n')
    readme = ROOT/'README.md'
    begin, end = '<!-- m1-current-status:start -->', '<!-- m1-current-status:end -->'
    note = f'''{begin}
## Completed M1 Dataset A milestone — 29 October 2026

The dated M0/1K/preparation notes above are historical. The real 50K Dataset A
is generated, cloud-verified and independently reproduced: 35K/7.5K/7.5K games,
3,793,600 retained positions after 150 exact evaluation-overlap exclusions,
and zero recorded parity mismatches in the original full replay.
The train/validation/test position counts are 2,654,850 / 569,124 / 569,626.

Published dataset: [ishagarg1103/gambit-zero](https://huggingface.co/datasets/ishagarg1103/gambit-zero).
Immutable revision: `b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c`.
Use that revision for reproduction; data artifacts remain on Hugging Face.
The full diagnostic index is {'published' if complete_diag else 'pending'};
Dataset D is {'frozen and its exclusions are verified' if complete_d else 'frozen; physical exclusions are not yet complete'}.
{'The exclusion upload is stopped on HTTP 429; obey the recorded server retry window.' if rate_blocked else ''}
See [the current M1 acceptance report](reports/m1_50k_pilot.md),
[machine-readable acceptance](reports/m1_final_acceptance.json), and
[Dataset D protocol](reports/m1_dataset_d_protocol.md).
The separate large-batch Blackwell benchmark is pending. No M2, model training,
self-play or larger extraction has started. The source handoff is prepared for
the user to commit/push to `m1/human-dataset`; `main` remains unchanged. CPU dataset checks PASS; overall M1
acceptance retains the documented GPU deferral and historical telemetry limits.

From the repository root, reproduce the offline regression and final publication
checks without rescanning the monthly Lichess archives:

```bash
uv sync --locked --extra test --extra data
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_m1_dataset_d_publication.py
```

The last command authenticates through the local Hugging Face login and reads
the pinned publication; it does not generate training data. Archive-run templates
are portable and disabled by default. New extraction needs a separate resource
review. Raw PGNs, Parquet shards, caches, environments, credentials and checkpoints
are excluded from the GitHub source handoff.
{end}
'''
    body = readme.read_text()
    if begin in body:
        before, rest = body.split(begin, 1)
        _, after = rest.split(end, 1)
        body = before+note.rstrip()+after
    else:
        body += '\n'+note
    readme.write_text(body)
    if (ROOT/'reports/m0_large_batch_blackwell.json').is_file():
        # Preserve the frozen CPU acceptance, and reattach later measured GPU
        # evidence so regenerating historical results cannot erase its update.
        import runpy
        runpy.run_path(str(ROOT/'scripts/report_blackwell_followup.py'))['render'](ROOT)


if __name__ == '__main__':
    render()
