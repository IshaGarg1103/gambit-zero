# M2 Stage 1 Parts A–C — 30 October 2026

**PASS for the loader and validation in the tested sample scope. Part C's optional
cache is deferred because the original Parquet loader exceeds 5K positions/s.**
Stage 1 Parts D/E, architecture, training and self-play remain pending.
Machine-readable measured results: [m2_stage1_abc.json](m2_stage1_abc.json).

## Source and scope

- Branch: `m2/stage-1`. The owner merged PR #4 and rebased this work. Local Stage 1,
  main and origin/main now point to base `7dc8e1540f81ae62e9c1fcab42cb64b3758a84bd`.
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
                          shuffle_buffer=16384, seed=20261030, epoch=0):
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
16,384 **original packed records**, approximately 490 MiB of feature bytes plus
Python metadata, not an entire dataset. Shards are permuted with NumPy PCG64 and
`SeedSequence([seed, epoch])`; a replacement buffer spans Arrow chunks, games and
shards, followed by a random drain. Every eligible record appears exactly once.
This is a bounded shuffle, not a uniform full-dataset permutation: mixing depends
on buffer size and sampled game lengths. No natural repeated boards are removed.
Same seed/epoch and immutable inputs reproduce order independent of Arrow chunk
size; another seed/epoch changes it. Evaluation retains deterministic source order.

Newly downloaded originals are normally deleted after use; pre-existing user
files are retained. At most one <=32 MiB shard is acquired at a time, with an
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

Remaining limits: a complete new 3,793,600-row loader epoch was not executed;
new byte checks and throughput are sample-only. Full statistical acceptance
continues to rely on the existing M1 evidence. Mid-epoch restart and future
trainer loss masking still need integration when separately authorized. These
are not claims that Stage 1 D/E or model/training stages are complete.
**Stop for review of Parts A–C.**
