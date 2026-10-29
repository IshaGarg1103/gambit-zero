# M1 incremental Hugging Face destination

25 October 2026. **Local implementation/tests and authenticated Hugging Face smoke PASS.
No full archive scan or M2 run started.**

Configured dataset repository:
https://huggingface.co/datasets/ishagarg1103/gambit-zero.
Visibility: **public**, following the user's explicit update to the original
private-repository requirement. Existing repositories must match the requested
visibility; the code does not automatically expose an existing private repository.

The authenticated live smoke passed in the personal account (not WootzappLab).
Actual Hub revision: `0f1fbee6e8c820d60bc77c322bf0efb9fd95baa6`. All 12 artifacts
(262,679 bytes) passed streamed remote SHA256/size verification.
The payload contains synthetic smoke data, not the real 1K/50K corpus.
Zero local Parquet files remain; about 87,111 bytes of recovery metadata remain.
`m1_hf_live_smoke.json` preserves the actual result; offline evidence stays separate.

## Storage answer for the Mac

**The complete 50K feature dataset is not required on the Mac.** Stream the
archives; retain a bounded candidate cache for the current month; validate;
write small Parquet pairs; upload/read-back/verify/remove each pair; proceed.
Do not retain a complete compressed or decompressed monthly archive.

| Component | Conservative local allowance |
|---|---:|
| Current verified candidate cache / scanner state | 1 GiB combined cap |
| Shard outbox, retries, journals and small controls | 256 MiB combined cap |
| Exact compact audit index + SQLite rollback space | 1 GiB combined allowance (DB max 512 MiB) |
| Extra metadata allowance | 128 MiB |
| Other bounded temporary allowance | 128 MiB |
| **Additional local planning budget** | **2.5 GiB** |
| Free-space reserve | **20 GiB** |
| **Minimum starting free space** | **22.5 GiB** |

Fresh measured free space: 82,008,518,656 bytes (~76.38 GiB); available RAM:
4,062,576,640 bytes (~3.78 GiB). Storage and memory planning gates pass; available
space remains dynamic. Raw evidence: `m1_hf_storage_plan.json`. The space reserve
is checked, not preallocated against other applications.

Each sealed Parquet file is capped at **32 MiB**, up to 128 whole games per pair.
Typical files can be much smaller. The outbox allows retry duplicates/atomic
replacements, never an unbounded upload queue. Candidates/metadata are bounded
independently. Decoder and game/feature working buffers are in RAM, not archive
files on disk. Final audit metadata and manifests remain locally for recovery;
feature shards are removed only after verification.

The earlier 1K storage measurement extrapolates to roughly **2–4 GB for the final
50K dataset on Hugging Face**, including conservative room for changed game
lengths and metadata. This is an estimate, not a measured 50K size. The input
archives still require **263.45 GB of network transfer**, which is different from
final dataset storage. Upload plus full read-back/reproduction adds several GB
of Hub traffic; its speed/cost/quota have not been measured or allocated.

Existing historical ignored inputs/results were left intact: about 37 MiB under
`data/raw` and 182 MiB under `data/processed`, before this cloud integration.
They are not a newly generated 50K dataset and are already included in free-space
measurements. The offline HF smoke's temporary directory was removed entirely.
No prior research evidence or user files were deleted to free space.

## Pipeline and acceptance gates

1. The existing bounded PZstandard scanner consumes original compressed bytes,
   preserves month/day quotas and verifies complete monthly SHA256/size/EOF.
2. Frozen sampler artifacts must match their own hash, official frozen source
   URL/size/SHA256, algorithm, seed, source month, reserves and target quota.
   Production targets remain 5K each Jan–Jul, 7.5K Aug and 7.5K Sep.
3. Freeze real Dataset D before production. Require exact versioned FENs, puzzle
   IDs, opening suite, source URLs/snapshot dates/checksums. A synthetic Dataset D
   fixture cannot satisfy the production gate. Publish the frozen manifest too.
4. Validate every selected game through the accepted `replay_game` path, including
   complete per-ply legal/history/FEN/terminal/reward parity, finite canonical
   features, legal action index, correct side-to-move WDL and timeout weights.
   Complete the whole month's checks before the first upload of that month.
   This pass holds one game's tensors at a time and discards them afterward.
5. Enforce both ratings >=1800, base >=300 seconds independent of increment,
   daily/monthly >=60% both-player-2000+ quotas, valid results, normal initial
   board and complete safe replay under the accepted 512-ply horizon. Use bounded
   ranked reserves for expected source rejections. Unexplained `ParityError` or
   target/history inconsistency aborts; no silent position deletion workaround.
6. Build an exact compact SQLite overlap index without feature tensors. A game
   ID cannot cross splits or change contents. Store board hashes and overall/
   ply>=10 counts for train/validation/test/all-three overlap. Physically omit
   exact frozen Dataset D matches from training position shards, not merely a
   flag that consumers could ignore. Report held-out Dataset D overlap separately.
7. Re-encode one game at a time, compare logical and feature hashes to the complete
   validation pass, and write bounded, game-aligned position/game Parquet shards.
   Source month and split are separate path components. Stop generation when
   upload/verification fails; do not accumulate the rest of the month on disk.
8. Upload using authenticated `huggingface_hub`. Stream each object back at the
   returned **immutable commit revision**, compare actual payload SHA256 and byte
   count to local values, durably journal success, then unlink the local shard.
9. Upload verified compact selected-PGN inputs and completed-month manifests.
   Perform another independent Parquet feature build from those cloud selection
   inputs. Compare regenerated game IDs, logical/feature digests and exact shard
   hashes; differing shards stop **before** reproduction uploads. Matching remote
   objects are reused without duplicate commits. No second complete local dataset
   is required. There is a validation pass plus two independently written shard
   passes; none retains a whole month's feature tensors.
10. Publish completed-month catalogs, source provenance, schema, encoder contract,
    frozen evaluation manifest, overlap audit and dataset card. The final card
    cannot claim complete-50k until counts equal 35K/7.5K/7.5K and all months have
    actually completed both builds. Larger candidate caches are retired only
    after these month receipts and cloud-backed reproduction pass.

Frozen policy remains `tinychess-auto-fide-pgx-v1`; canonical encoder remains
`pgx-v2-history8-ep-context8-v1`; policy actions remain native 4,672. Original M0
and M1 replay/search/encoder implementations are unchanged. No transformer,
training loop, Dataset B, self-play or teacher labels were introduced.

## Remote layout and usage

Production objects use paths like:

```text
data/<run-id>/positions/train/2026-01/<part>-<sha256>.positions.parquet
data/<run-id>/games/train/2026-01/<part>-<sha256>.games.parquet
data/<run-id>/provenance/2026-01/selected.jsonl.zst
data/<run-id>/provenance/2026-01/approval.json
data/<run-id>/evaluation/frozen-d-<sha256>.json
data/<run-id>/manifests/2026-01.json
data/<run-id>/manifests/2026-01.reproduction.json
schema.json
encoder_contract.md
source_provenance.json
dataset_manifest.json
README.md
```

Smoke objects are under `smoke/hf-smoke-v1/`, separate from production data.
The card's explicit Parquet configurations select only completed artifacts and
keep game-level provenance separate from training positions. Game actions retain
full trajectories, including excluded evaluation positions; consumers must not
create an unfiltered training reader from that provenance configuration.

Schema metadata pins features/masks/rules/codec. Binary planes are little-endian
float32 (8,8,120), context (8,), legal mask 584 bytes with little bit order; WDL
classes are win/draw/loss (0/1/2), scalar +1/0/-1 from side to move. Use the dataset
card's streaming `load_dataset(..., streaming=True)` example to consume remotely
without downloading a complete dataset. Pin the actual Hub revision in training.

The complete card preview is `m1_hf_dataset_card.md`. It includes source/license,
filters, source verification caveats, WDL weights, dimensions, leakage semantics,
recovery, reproducibility and usage instructions. It currently identifies the
small synthetic smoke, not an accepted real 50K dataset.

## Resume and deletion behavior

- Stable run/month/part/content-hash paths and immutable receipts make uploads
  idempotent. Lost upload responses are recovered by checking existing remote
  content, not blindly making a new commit. Conflicting content is an error.
- Upload or read-back failure retains the sealed local file and stops the outbox.
  Rerunning regenerates/compares deterministic parts and reuses verified objects.
  Never delete sealed unverified files manually to make the job continue.
- A crash while writing an unsealed `.partial` file can be rebuilt from identical
  verified inputs under the matching run/month approval. Such files have never
  been uploaded. Unknown ownership/mismatched approval stops instead of deleting.
- Sleep/connection loss during archive scanning restarts the current month from
  byte zero for a fresh full SHA256; it can cost another ~28–31 GB. Already completed
  HF months are verified/reused without rescanning their original archives.
- Sleep/connection loss during Hub upload retains the current shard and verified
  source checkpoint. Resume does not require retransferring its original archive.
- Keep the small upload journal/completed-month manifests and compact audit DB.
  Cloud selection inputs can also rebuild audit metadata while reproducing data.
  Automated recovery assumes those local completed-month receipts remain; recovery
  after deleting all local state is not claimed as a tested disaster scenario.
- Runtime checks occur between games/uploads and during read-back. Individual SDK
  network calls may overshoot a deadline by a blocking timeout; no indefinite
  local lock acquisition is configured. No automatic unbudgeted archive retries.

## SDK, authentication and cache behavior

Pinned and inspected `huggingface_hub==2.3.0` and `filelock==4.2.4`; hashes are in
`uv.lock` and both exported lockfiles. Installed API signatures/CommitInfo.oid/
HTTP streaming interface are captured in `m1_hf_api_contract.json`. JAX/jaxlib
0.7.2, Pgx 2.6.0, mctx 0.0.6, Flax and Optax core pins remain unchanged.

Use cached CLI login or HF_TOKEN environment authentication; no token argument is
accepted by our scripts, no token is embedded in files/notebooks/URLs, and no
credential value is printed/journaled. Error details are reduced to safe classes;
tests inject credential-like exception text to ensure it does not escape.
The official HF endpoint is required. Repository visibility is checked explicitly.

`HF_HUB_DISABLE_XET=1` is set before SDK import and required by the transport,
so upload uses HTTP/LFS rather than creating a disk Xet cache. Read-back uses the
SDK session's streaming HTTP interface, not `hf_hub_download`/snapshot caching.
The installed HTTP client strips bearer Authorization on cross-origin redirects.
The dependency wheel may be installed; its runtime data cache is not used.

## Actual tests and small-shard results

Final regression: **165 passed in 48.76 seconds**, no skips/xfails. Syntax
compilation passed. Locked offline sync dry-run makes no changes; `uv pip check`
reports all 54 installed distributions compatible. The default plan passes with
full execution disabled. No large archive/large-batch benchmark was launched.

Offline end-to-end smoke (`m1_hf_offline_smoke.json`):

- 31 synthetic games spanning every January UTC day; two actual fixture mates,
  non-board results/draws/time forfeits, full corrected-Pgx trajectory replay.
- 69 pre-move positions +31 final boards = **100 parity states**, zero mismatch.
- 31 exact synthetic evaluation-position matches physically excluded; **38
  training position rows** remain, with 31 game provenance rows.
- Two bounded Parquet files total **225,516 bytes**; **12 simulated artifacts**
  total **262,679 bytes** including metadata/card/schema. These are not actual
  bytes uploaded to Hugging Face.
- Cloud-interface second build PASS; shard digest
  `5bd014bc99e53f5246f0d9de6d87f29fccd3d1582cdaed07c873041c51b840cc`.
- 0 local Parquet files remain after verified simulated publication; temporary
  offline fixture/input/audit directories were removed after the run.

Automated tests include complete-month-before-upload gating, source SHA failure,
exact Dataset D exclusion, deterministic second build, existing-object conflict,
lost response, upload failure, remote mismatch, unchanged audit counts on retry,
whole-game split conflict, exact all-three/late overlap, interruption recovery,
owned unsealed-file rebuild, sealed failed-shard retention/backpressure, runtime
exhaustion, explicit visibility checks and suppression of sensitive SDK text.

**Live status:** `PASS_LIVE_HF_SMOKE`. Revision `0f1fbee6e8c820d60bc77c322bf0efb9fd95baa6`;
262,679 verified artifact bytes, 12/12 successful remote read-backs.
The live test also reproduced identical shard hashes from cloud selection inputs.
Real 50K game counts remain **not built**, not a completed empty dataset.

## Commands

From the repository directory, reproduce local no-network checks:

```bash
uv sync --locked --extra test --extra data
.venv/bin/python -m pytest -q
.venv/bin/python scripts/check_hf_upload.py --output reports/m1_hf_offline_smoke.json
.venv/bin/python scripts/scan_m1_archives.py --output reports/m1_hf_storage_plan.json
```

Authenticate locally, then perform ONLY the small synthetic live smoke:

```bash
.venv/bin/hf auth login
.venv/bin/hf auth whoami
HF_HUB_DISABLE_XET=1 .venv/bin/python scripts/check_hf_upload.py --upload --output reports/m1_hf_live_smoke.json
```

Follow the CLI browser/token flow; if asked for a token, use a write-capable one
only at the credential prompt, never in a command argument or this chat. Git credential storage is not needed. The last command creates/reuses
the public **dataset** repository, uploads only clearly marked smoke artifacts,
verifies them and records the actual URL/revision/size/counts. It cannot enable
the full scan or M2. On a failure, keep its local spool for retry.

## Files and remaining gate

New modules: `hf_upload.py`, `hf_shards.py`, `hf_audit.py`, `hf_month.py`,
`hf_destination.py`, `hf_card.py`; new smoke script and automated HF tests.
Updated scan destination/config/storage plan, data dependencies/locks, README
and ignored cache paths. `m1_hf_implementation_hashes.json` binds the source/test/
config snapshot. No Git commands, branch creation, commits, pushes or merges ran.

The authenticated small live smoke passed. Full extraction still needs real
frozen Dataset D, transfer/runtime/awake-host allocation and review of
these component bounds. All full-scan enable flags remain false. Stop before
large extraction; no M2.
