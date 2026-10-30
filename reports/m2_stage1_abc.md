# M2 Stage 1 Parts A–C — 30 October 2026

**PASS for sample byte fidelity and full training-split accounting on CPU.**
Part C's optional cache remains deferred. The default 200K-buffer profile is
implemented and memory-guarded; full-scale execution of that profile needs a host
with more available RAM and is not claimed here.
Stage 1 Parts D/E, architecture, training and self-play remain pending.
Machine-readable measured results: [m2_stage1_abc.json](m2_stage1_abc.json).

## Source and scope

- Branch: `m2/stage-1`. The owner merged PR #4 and rebased this work. At initial implementation, local Stage 1,
  main and origin/main pointed to base `7dc8e1540f81ae62e9c1fcab42cb64b3758a84bd`.
  Before that operation, Stage 1 branched from Stage 0 at
  `360c94be8798c7be16cedaec4241fd9e99e332b0` with a clean tracked worktree.
  Git index/blob contents and refs were read directly; no Git command was run by
  the agent. No commit, push, PR creation or merge was performed by the agent.
- Dataset: [ishagarg1103/gambit-zero](https://huggingface.co/datasets/ishagarg1103/gambit-zero/tree/b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c),
  immutable revision `b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c`.
- Catalog SHA256: `ab791031be6b4aa2f7962ddc527fee9fbb91840223759339686d55ab1ab9daa1`;
  independently matches the original accepted catalog receipt in archived M1
  exclusion evidence. Monthly manifest names contain their verified SHA256.
- Read complete README v0.4, M1/M2 execution specifications, Stage 0 report,
  canonical encoder/schema, accepted M1 results and existing data code first.
  Existing storage, version constants and feature decoder are reused.

Changed files: `src/tinychess/data/training_loader.py`,
`scripts/check_training_loader.py`, `tests/test_training_loader.py`, this report,
its JSON summary, `.gitignore` report allowlist, and an appended status section
in `m2execution.md`. Earlier roadmap text and research results are preserved.

## Part A — API, memory and accounting

```python
from tinychess.data.training_loader import VerifiedStore, inventory, iter_batches

store = VerifiedStore("data/processed/training-originals", download=True)
shards, catalog = inventory(store)  # tiny pinned catalog + hashed manifests
train = [s for s in shards if s.split == "train"]
print(sum(s.positions.size + s.games.size for s in train))  # transfer estimate
for batch in iter_batches(store, train, "train", batch_size=256,
                          seed=20261030, epoch=0):  # default buffer 200K; host RAM permitting
    # CPU NumPy inputs; no model or training is implemented.
    print(batch.planes.shape, batch.max_positions_per_game)
    # Consume the iterator to exhaustion to verify epoch/shard accounting.
```

`download=False` is the default and fails if a verified original is missing.
Existing processed originals can be placed/symlinked under their manifest-relative
paths in the explicit store root; size and SHA256 are checked before every use.
Wrong local checksums fail instead of silently overwriting/re-downloading.
All network reads resolve the immutable revision, including shards whose receipts
were created at earlier commits. Never select old unfiltered paths via globbing.
The complete inventory is 398 position/game pairs across nine months.

| Batch field | Shape at B=256 | dtype / contract |
|---|---|---|
| planes | `(256,8,8,120)` | little-endian float32, exact stored bytes |
| context | `(256,8)` | little-endian float32, exact stored bytes |
| legal_mask | `(256,4672)` | bool; unpacked little-bit-order 584-byte rows |
| action / wdl_class | `(256,)` | int32; native action / win=0, draw=1, loss=2 |
| outcome_stm | `(256,)` | int8; +1/0/-1 from side to move |
| policy_weight / value_weight | `(256,)` | float32; unchanged stored weights |
| training_eligible | `(256,)` | bool; every emitted row is eligible |
| ply | `(256,)` | uint16 |
| game_id, position_hash, source_month | length 256 | original identity tuples |
| split | scalar string | exactly one chronological split |

Every row is validated **before** filtering `training_eligible=False`; malformed
excluded rows also stop. Time-forfeit policy rows remain; their value weights
remain zero. No final WDL is re-inferred, no loss is computed here, and no history
is reset/replayed. A future trainer must multiply value loss by `value_weight`.
The last batch is short: no padding, `drop_last`, duplication, deduplication or
per-game cap. Source count checks run when the iterator is exhausted; an early
consumer stop is only partial coverage. Dataset D's physically removed train
plies are allowed as gaps; duplicates/backward plies are rejected.

Arrow decodes at most 128 rows by default (configurable 1–512), and accepted row
groups are checked to contain at most 512 rows. The default shuffle buffer holds
200,000 **original packed records** (configurable), with a conservative host
allocation estimate of 7,227,648,000 bytes at B=256/chunks=128 plus a required
1 GiB available-memory reserve. This is host RAM, not GPU VRAM. An insufficient
available-memory check fails before loading; use an explicit smaller buffer on
the Mac. The initial sample below used 16,384 records, not this new default. Shards are permuted with NumPy PCG64 and
`SeedSequence([seed, epoch])`; a replacement buffer spans Arrow chunks, games and
shards, followed by a random drain. Every eligible record appears exactly once.
This is a bounded shuffle, not a uniform full-dataset permutation: mixing depends
on buffer size and sampled game lengths. No natural repeated boards are removed.
Same seed/epoch and immutable inputs reproduce order independent of Arrow chunk
size; another seed/epoch changes it. Evaluation retains deterministic source order.

Training now retains newly downloaded originals by default
(`VerifiedStore(..., keep_downloads=True)`). Repeated epochs and process restarts
reuse the same manifest-relative paths, validating size/SHA256 on every access;
local corruption is an error, never an overwrite. Per-file locks serialize
concurrent persistent downloads and atomic publication. No second copy per epoch.
Use `VerifiedStore(..., keep_downloads=False)` in a separate audit root for
one-off temporary reads: only newly acquired originals are deleted after use,
with another checksum check before deletion; pre-existing files are retained.
The sample-audit command also retains its explicit temporary-cleanup behavior.
At most one <=32 MiB shard per reader is acquired at a time, with an
additional 20 GiB free-space reserve enforced. No global Hub dataset cache is
used. Interrupted small-shard downloads fail, remove their own incomplete file,
and can restart that shard; completed verified files are reused. This is not a
mid-epoch checkpoint/resume implementation. HF credentials come from the SDK's
normal local authentication, never source/config/report strings.

## Part B — actual validation and diagnostics

No local Parquet shards existed at preflight. Free disk was 76.85 GiB; host RAM
16 GiB, 10 logical CPUs, macOS 26.6.1 arm64, Python 3.11.15. Expected full original
position download: 2,025,652,303 bytes (1.89 GiB), **not acquired**. No source PGNs
were acquired. Downloaded only five shard pairs, 21,526,068 bytes (20.53 MiB):
January part 0, April part 20, July part 39, August part 0, September part 0.
The April sample is a revised Dataset D-filtered shard, exercising missing-ply
gaps. Original metadata/game lengths remain intact after physical exclusions.

The pinned catalog and all nine hashed manifests were verified; their current
counts agree with retained M1 validation evidence. These are **existing accepted
full-data totals / expected eligible row counts**, not a new full tensor traversal:

| Split | Games | Retained/expected eligible positions |
|---|---:|---:|
| train, Jan–Jul | 35,000 | 2,654,850 |
| validation, Aug | 7,500 | 569,124 |
| test, Sep | 7,500 | 569,626 |

The following are **new measurements on the five-shard sample only**. It is a
fixed engineering sample, not a representative estimate of full-data frequencies.
All sampled rows were eligible. Each of 39,766 emitted feature/target records
matched an independent original-byte fingerprint, keyed by game/ply; no missing,
duplicate or changed emitted record. Planes/context are float32 byte-identical,
legal masks repack identically and labels/weights/hashes/identities match.
Synthetic test fixtures are separate and never uploaded/mixed with real data.

| Sample | Games | Rows | W / D / L positions | Zero value weight | ply <10 |
|---|---:|---:|---|---:|---:|
| train | 264 | 19,942 | 9,040 / 1,963 / 8,939 | 2,885 | 2,631 |
| validation | 128 | 10,000 | 4,403 / 1,257 / 4,340 | 1,885 | 1,278 |
| test | 128 | 9,824 | 4,349 / 1,180 / 4,295 | 2,721 | 1,280 |

| Sample | Unique boards | Identities occurring >1x | Occurrences beyond first | Max frequency |
|---|---:|---:|---:|---:|
| train | 18,531 | 338 | 1,411 | 264 |
| validation | 9,399 | 181 | 601 | 128 |
| test | 9,262 | 160 | 562 | 128 |

Board identity is the accepted SHA256 of pieces, side to move, castling and
legally capturable EP; excludes clocks/history. Equal identities need not have
identical model features. These are within-sample repetition statistics; existing
full-data overlap remains in [M1 validation](m1_validation_summary.json).

Sample game lengths (min/mean/max plies): train 5/75.54/187; validation
8/78.125/236; test 10/76.75/190. Sample time-controls and ending distributions are
in the JSON summary. The original full-game diagnostic evidence reports 8,473
forfeit games among 50K, unchanged by position filtering. Its original row WDL/
769,665 masked-row totals precede 150 exclusions and are **not substituted for
new counts at the final revision**. No new full-data WDL/zero-weight recount is
claimed. This task verifies loader preservation, not a new per-ply rules replay.

Channel inventory over all sample positions found exact 0/1 values in 118 of
120 planes; channels 113 (episode ply/512) and 118 (halfmove/100) were nonbinary.
Observed ranges: channel 113, 0–0.458984375; channel 118,
0–0.9899999499320984. No negative zero observed in the sample; synthetic tests
verify negative-zero and nonbinary float bytes survive decoding. This sample
cannot prove binary channels over every possible state or authorize bitpacking.
All context bytes are preserved, including their nonbinary normalized counters.

Training shuffle at B=256 mixed **at least 144 distinct games per batch**, with
**at most 10 positions from one game**, including the final short batch. Final
sizes: train 230, validation 16, test 96. Evaluation is intentionally unshuffled:
maximum single-game contributions were 207 (validation), 152 (test). The mixing
requirement applies to training; evaluation computes fixed-order observations.
Training order SHA256:
`c7a8a81a92db71679952864f138f4f9fafcca42601d6c77a1c1077aef95fae6d`.
Three real-data repeated iterations matched it; synthetic tests additionally
cover different seeds/epochs and mixing across three shards and Arrow boundaries.

## Part C — measured decision

Three warm local original-Parquet iterations, 19,942 training rows each,
B=256, chunks=128, shuffle buffer=16,384:

| Repeat | Seconds | Positions/s |
|---|---:|---:|
| 1 | 0.580599 | 34,347 |
| 2 | 0.574769 | 34,696 |
| 3 | 0.604758 | 32,975 |

Includes local checksum verification, Parquet decoding, metadata/record checks,
shuffle fill/drain, batch construction and identity-order diagnostics. Excludes
network acquisition and the separate all-byte comparison/channel inventory pass.
OS page-cache effects apply; the CPU regression ran concurrently. These short
sample iterations do not establish sustained remote/full-epoch, GPU transfer,
model, training, search or self-play throughput. Initial sample acquisition took
21.55s; subsequent measured runs reused those verified local files.

Peak process RSS during the final complete audit: 897,302,528 bytes (855.73 MiB).
This includes reference fingerprints and diagnostics as well as the loader;
not a GPU-memory measurement. Peak retained sample Parquet size: 20.53 MiB.
**No cache built:** >6x the requested ~5K/s floor leaves no demonstrated need for
conversion/compression complexity or additional disk use. Original Parquet stays
the canonical loading path. No binary-channel compression proof or cache lossless
audit is claimed; those checks are unnecessary until a cache is proposed.

## Tests, reproduction and preservation

```bash
uv sync --locked --extra test --extra data
JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_training_loader.py -x -q
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -x -q
JAX_PLATFORMS=cpu .venv/bin/python scripts/check_training_loader.py --download
```

Actual: **36 focused tests passed in 2.43s; 305 full CPU tests passed in 56.39s**.
Tests cover canonical stored bytes/dtypes, sign/WDL, all three results, forfeit
weights, eligibility, split isolation, same/different RNG seeds/epochs, cross-game/
chunk/shard mixing, exact-once accounting, partial and empty batches, nonbinary/
negative-zero bytes, malformed records (also ineligible), schemas, counts, missing/
duplicate rows, checksums, disk reserve, interrupted/short/oversized/corrupt
transfers, atomic cleanup and verified offline reuse. Failures stop with source
path/game/ply context where available; nothing malformed is silently skipped.

The sample command reports download cost first, emits ignored runtime JSON under
`data/processed/m2_stage1/`, and normally removes newly acquired sample originals
at completion. `--keep-sample` explicitly keeps them for offline repeats. Small
manifest/recovery evidence remains; actual dataset shards never enter Git.
The temporary sample originals used here were removed after verification.

A final read-only HF metadata check verified both dataset main and the pinned
revision remain `b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c`. All earlier production
source, configs, schemas, rules, codec, encoder, locks and accepted M1 reports
match their initial SHA256 fingerprints. No dataset/annotation changes, archive
scan, augmentation, evaluation-suite changes, model, GPU experiment or training.

Initial sample limitations (before the PR #5 follow-up): a complete new
3,793,600-row loader epoch was not executed;
new byte checks and throughput are sample-only. Full statistical acceptance
continues to rely on the existing M1 evidence. Mid-epoch restart and future
trainer loss masking still need integration when separately authorized. These
are not claims that Stage 1 D/E or model/training stages are complete.
**Stop for review of Parts A–C.**

The PR #5 follow-up below supersedes the earlier full-training-epoch limitation;
the initial sample measurements and original scientific findings above are historical.

## PR #5 final loader-readiness follow-up — 30 October 2026

**Full training accounting: PASS. Larger-host 200K-buffer execution: unverified.**
PR #5 was verified open/unmerged, and the clean local `m2/stage-1` matched its
head `19d4cabe15c89daf0b5bf258c2e78bd61b11b9fe` before these changes. The PR base
is `7dc8e1540f81ae62e9c1fcab42cb64b3758a84bd`. No Git command, commit, push,
PR comment or merge was performed. The user must push these local changes.

Final machine-readable evidence: [full training audit](m2_stage1_full_train.json).
The original sample JSON/results remain historical and unchanged.

### Persistent training originals and temporary audit mode

`VerifiedStore` now defaults to **`keep_downloads=True`**. The first acquisition
stores each original at its exact manifest-relative path; later epochs/processes
verify its size and SHA256 and reuse it. Per-file locks serialize concurrent
check/download/publication; atomic replacement publishes only a fully verified
new file. Corrupt existing originals fail without replacement. No per-epoch
copies, repacking, global HF cache, derived tensors or schema changes.

For one-off audits retain `keep_downloads=False` with a **separate audit root**.
It removes only originals newly acquired by that temporary store, after a final
checksum check; it never deletes pre-existing user files. Do not mix persistent
and temporary clients in the same root. The existing sample audit still offers
its explicit `--keep-sample`/temporary-cleanup modes.

Persistent originals used here are retained under
`data/processed/m2_training_originals/`: **560 files, 1,417,011,959 bytes
(1.32 GiB)**. These are only the current 280 training shard pairs; no validation/
test payloads or Lichess PGNs were fetched. All files were rehashed after the
run; no partials, extra Parquet copies, deletion or content changes. The data
remain ignored by Git. They can be reused offline on the next run.

### Larger configurable shuffle default and resource preflight

Training defaults to **200,000 positions**, still configurable via
`shuffle_buffer` / `--shuffle-buffer`. It is deliberately a host-memory buffer,
not a VRAM allocation. At B=256 and chunk=128 the conservative estimate is
7,227,648,000 bytes plus **1 GiB available-memory reserve**; 250K would estimate
9,027,648,000 bytes plus the reserve. Capacity is capped by actual selected rows
for small fixtures/samples. Invalid settings fail before acquisition, and
insufficient RAM fails explicitly rather than silently shrinking the buffer.

This Mac has 16 GiB total RAM, 10 logical CPUs; approximately 3.9 GiB was available
at preflight. Its 200K preflight was **correctly rejected**. The executed full
split used an explicit **32,768-position buffer**, estimated 1,207,296,000 bytes
plus reserve. This is only 1.23% of the full training population, so the mixing
measurements below exercise a streaming shuffle rather than a near-full shuffle.
The 200K profile still needs a full audit on a larger-RAM host; the Blackwell's
96 GB VRAM alone does not establish its host RAM. No GPU run was attempted.

Disk preflight: 82,590,810,112 free bytes, 20 GiB reserve, 1 GiB extra work allowance
for the identity index/journal/metrics and incomplete shards, plus 1.42 GB missing
Parquet. Total Parquet payload transferred by the audit and a bounded three-worker
prefetch helper was **exactly 1,417,011,959 bytes**, matching the unique manifest
inventory. Per-file locks prevented duplicate fetches between those processes.
HTTP headers/protocol overhead were not measured. No archive scan was run.

### Exact-once full training audit

The new `scripts/audit_training_split.py` reads the immutable catalog/manifests,
preflights resources, and validates every original training row. A bounded SQLite
index stores the exact eligible `(game_id, ply, source_month)` identities. Each
emitted batch must mark previously unseen identities, so duplicate, unknown,
ineligible or wrong-month records fail immediately. At epoch end no expected
identity may remain unseen, and emitted counts must equal the eligible reference.
This is exact set accounting, not only a sum or probabilistic digest.

Raw row/game/month totals are checked against the manifests. Malformed records
still fail before eligibility filtering. Training cannot include August/September
shards; batch split and eligibility are checked again. Tests inject missing,
duplicate, ineligible and wrong-month rows and verify failure. No record is
silently skipped and no repeated board is deduplicated.

Reference result: **35,000 games, 2,654,850 rows, all eligible, zero ineligible**:

| Month | Verified eligible positions |
|---|---:|
| January | 380,256 |
| February | 380,988 |
| March | 380,458 |
| April | 380,313 |
| May | 375,493 |
| June | 376,737 |
| July | 380,605 |

Two complete shuffled epochs (seed 20261030, epoch 0/1) each emitted
**2,654,850 positions in 10,371 batches**, with a final batch of 130.
Both had **zero missing, duplicate or split violations** and **zero downloaded
bytes during the epoch**. Checksums and record validation were repeated on access.
Both used the same persistent originals and produced different epoch order hashes:

- Epoch 0: `b6a1c922a06749476467389e2668625e2da3445be93adf41aec2af898e6323ec`.
- Epoch 1: `12fb64d5de2bc077bca3b2fc13ed8fe9f71d0fe7388c7461966795dced3dd70e`.

These full passes add accounting/split evidence to the initial byte-fidelity
sample; they do not claim a fresh all-ply python-chess/Pgx replay or a new
full-feature bytewise comparison. Existing M1 rules-parity evidence is preserved.
Validation/test were not fully traversed again; their earlier sample checks and
M1 acceptance remain the evidence for those splits.

### Mixing measured over the full population

The metrics below are for **full B=256 batches**; no per-game cap was imposed:

| Epoch | Max positions from one game | Distinct games min / mean / max | Distinct source months min / mean / max |
|---|---:|---|---|
| 0 | 8 | 183 / 217.75 / 236 | 3 / 6.18 / 7 |
| 1 | 9 | 178 / 217.83 / 239 | 3 / 6.46 / 7 |

Final 130-row batches: epoch 0 had 120 games / 5 months / max 3 from one game;
epoch 1 had 115 games / 7 months / max 3. Full histograms are in the JSON report.
This measures the explicit 32K Mac profile; it does **not** certify 200K host
mixing or claim a uniform permutation of the whole training split.

Every batch's epoch/index, size, maximum single-game contribution, distinct-game
count and distinct-source-month count is recorded in:
`data/processed/m2_stage1_followup/full_train.batches.jsonl`.
**20,742 records, 2,300,892 bytes**, SHA256:
`e662eab45ae2e4a38aa52f5b71762ee4ccd14a1e711ac644482a84fb7b584d66`.
The small local per-batch evidence remains ignored; it is reproducible from pinned
originals and the commands below. The final compact summary retains its digest.

### Resources, tests and reproduction

Full reference pass (including acquisition and index creation): **591.48s**.
Epoch 0: **103.63s, 25,617 positions/s**; epoch 1: **101.88s, 26,060 positions/s**.
Total audit wall time **797.58s (13m18s)**. The epoch rates include checksum reads,
validation, shuffle, SQLite exact accounting, order hashing and per-batch metrics;
network acquisition is excluded from epoch timing. They are CPU audit rates,
not model, GPU-transfer, training, MCTS or self-play rates. No new Part C tensor
cache is justified or implemented.

Peak process RSS: **1,463,877,632 bytes (1.36 GiB)**. The accounting DB's final main
file was 76,333,056 bytes; it and its disposable work directory were removed after
PASS. Persistent originals and small recovery/evidence files were retained.
Exact aggregate peak disk/journal use was not instrumented; the conservative
1 GiB work allowance and 20 GiB reserve passed. Final free space was about
74.42 GiB. Failed audits retain their identity index for diagnosis; rerunning
starts fresh accounting while reusing verified originals, not a mid-epoch resume.

```bash
# Full-source audit on a host with enough RAM: default 200K buffer.
JAX_PLATFORMS=cpu .venv/bin/python scripts/audit_training_split.py --download --preflight-only
JAX_PLATFORMS=cpu .venv/bin/python scripts/audit_training_split.py --download

# Executed Mac profile; after initial acquisition this needs no network.
JAX_PLATFORMS=cpu .venv/bin/python scripts/audit_training_split.py --shuffle-buffer 32768

# Focused and full CPU regression.
JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_training_loader.py -x -q
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -x -q
```

The initial executed Mac command added `--download` to the Mac command above.
No download flag is needed to repeat from the now-complete persistent originals.
Optional `--root` selects persistent storage on the training host; `--output`
selects audit evidence storage. The full audit requires persistent mode and has
no original-data cleanup option. Subsequent runs reuse/recheck files; a failed
or interrupted acquisition leaves completed verified files intact.

**Final tests: 47 focused passed in 2.61s; 316 full CPU tests passed in 57.75s.**
New tests cover repeated persistent reuse/corruption rejection, explicit temporary
mode, memory guards, invalid preflight parameters, complete two-epoch accounting,
per-batch metrics and injected accounting failures. Existing rules/encoder/data
regressions remain unchanged. HF metadata confirms dataset main still equals
`b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c`.

Changed in this follow-up: `training_loader.py`, new `loader_audit.py`, new
`audit_training_split.py`, loader tests, this report, the compact full-audit JSON
and its `.gitignore` allowlist. No source datasets, chess rules, encoder, codec,
schemas, dependency locks, Parts D/E, architecture, MCTS, training or self-play.
**Ready for review of the local fixes; 200K-host execution remains a documented
follow-up. Do not merge automatically.**
