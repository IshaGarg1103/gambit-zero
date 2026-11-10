# M2 Stage 3B — 25M supervised pilot preparation

Date: 9 November 2026. Branch: `m2/stage-3b-supervised`.
Base: `9c53aabad8ef8eeaa4bc7373fa307d1f94683284`, matching local main/origin/main
and containing accepted Stage 3A `f191ebb0830fca8385a73d054cf2d3b610a1b00d`.

**Scope: implementation and CPU verification only. No LR sweep, real-data model
training, new GPU experiment or dataset download was run.** The only optimizer
updates in preparation were tiny synthetic unit-test fixtures, not 25M experiments.

## Implementation

- Reuse accepted 25M model, masked policy/WDL losses, global-normalized gradient
  accumulation and once-per-update clipping. No model/codec/encoder/rules changes.
- Add optional schedule/weight-decay arguments to existing create_state, preserving
  Stage 3A defaults and fp32 parameters/moments. Stage 3B uses bf16 compute.
- Extend existing Tracker with optional group/name/job/artifact settings, preserving
  Stage 3A defaults. New groups: m2-lr-sweep and m2-supervised, project gambit-zero.
  Credentials remain environment-only; offline tracking is supported.
- Reuse persistent VerifiedStore and canonical loader. Bound train to January–July,
  validation to August; never open test Parquet. Verify source manifests/checksums,
  record validity, labels and full emitted split counts. Keep original repeat-board
  distribution; history dropout remains 0%.
- Final partial batches are padded with legal feature copies carrying zero policy
  AND value weights. No rows dropped. Losses use population weight denominators;
  forfeits contribute no WDL loss. Validation aggregates numerators/denominators
  over the full split, not a mean of batch means.
- Fresh seeded initialization for each experiment; pre-update loss sanity check.
  Checkpoint recovery is explicitly Stage 3B-only and bound to source/config hashes.
- Latest checkpoint every 500 updates, separate best checkpoint after full validation.
  Deterministic mid-epoch replay restores the exact next batch/optimizer state,
  with bounded shuffle memory. Refuse accidental output replacement; changed-recipe
  restore failures preserve the existing checkpoint.
- Default CLI performs metadata/resource preflight only. Explicit execution flag,
  one selected LR, Blackwell/pin checks, host/disk checks, conservative compiled
  memory analysis and time budget are required before real updates.
- Future executions record compile/init time separately, synchronized update rate,
  loader wait, end-to-end throughput, physical/JAX/host memory, norms and clipping
  counts, W&B run URL, full validation metrics and zero-based early-ply strata.

## Proposed experiments — require review and approval

See [the run guide](../docs/m2_stage3b_run.md) and
[reviewable configuration](../configs/stage3b_25m.json).

- Microbatch 32 × 32 accumulation = 1,024; validated 32,768 host shuffle buffer.
- Three independent one-epoch candidates: LR 1e-4 / 3e-4 / 6e-4, same seed 20261109
  and data order. Each is a separate invocation; no automatic sweep-to-pilot launch.
- AdamW beta=(0.9,0.95), epsilon 1e-8, matrix-kernel weight decay 0.01,
  clip 1.0 after accumulation. 1,000-update warmup and cosine to 10% of peak.
- Select LR using entire validation policy NLL, then seek pilot approval. Pilot
  starts fresh, up to 20 epochs, patience two complete validation epochs (min_delta
  1e-4). Sweep and pilot use their respective schedule horizons; no claim of
  identical cross-mode trajectories.
- 2,593 updates per epoch; final train batch 642 real rows. Full validation:
  569,124 rows / 17,786 microbatches, final batch four real rows.
- Proposed caps: one active wall-hour per candidate (three hours total), up to
  12 hours for the selected pilot. Not granted/consumed budgets or runtime forecasts.
  Inspect actual first-candidate throughput, validation cost, memory and stability
  before approving a longer pilot. Budget checks cannot preempt an in-flight call.

## Actual local preflight — no downloads

CPU metadata/resource preflight intentionally returned `PREFLIGHT_BLOCKED` (exit 2)
because available host RAM was below the conservative requirement. No reserves
were reduced to bypass the stop. This does not establish GPU-host availability.

| Quantity | Measured/planned bytes |
|---|---:|
| Available Mac host RAM at preflight | 3,505,225,728 |
| Existing loader RAM estimate | 1,262,592,000 |
| Additional model/work reserve | 3,221,225,472 |
| Required available host RAM | 4,483,817,472 |
| Free disk at store/output filesystem | 71,199,330,304 |
| Train+validation original Parquet inventory | 1,730,442,889 |
| Missing original Parquet (validation; NOT downloaded) | 313,430,930 |
| Free-disk reserve | 21,474,836,480 |
| Checkpoint/atomic-copy/W&B staging allowance | 3,221,225,472 |

Disk checks passed; host RAM check failed. Existing train originals remain intact.
A blank GPU-host store needs roughly 1.61 GiB of originals plus small manifests;
allow about 24.7 GiB free after environment setup. A fresh CUDA environment/cache
may additionally need up to about 15 GiB. These are conservative plans, not
measured Stage 3B GPU peaks. No full data was created or cached twice on the Mac.
Raw preflight/test logs are ignored under data/processed/stage3b_preparation/.

## Tests and reproducible commands

New focused suite: **28 passed in 15.17s**. Covers fixed configuration boundaries,
seeded initialization, schedule endpoints, correct tail accumulation/AdamW update,
zero-weight forfeits, population-weighted validation, early-ply partition, decay
mask, exact optimizer/checkpoint continuation, replay suffix, early stopping,
no-test-split access, launch guard and W&B grouping/credential boundaries.

Complete CPU regression: **423 passed in 89.72s** on the final source implementation.
Stage 3A and all prior environment/encoder/codec/loader/model regressions remain passing.

Tiny synthetic orchestration additionally verifies interrupted/full-epoch training
against an uninterrupted reference: final serialized model+optimizer checkpoint
SHA256 is identical; metadata mismatch never overwrites existing weights. A fresh
run refuses existing output. These tests stub hardware/telemetry and parameter
count only for the tiny fixture; they are not Blackwell/25M measurements. Existing
model tests independently verify real 25M/100M parameter shapes and precision.

```bash
# Executed: metadata/resource preflight; expected exit 2 on this Mac snapshot.
JAX_PLATFORMS=cpu .venv/bin/python scripts/run_supervised_pilot.py \
  --mode sweep --lr-index 0 --output data/processed/stage3b_preparation/preflight
# Executed: focused preparation tests.
JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_supervised_pilot.py -q
# Executed: complete regression.
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q
```

## Limitations and stopping point

Stage 3B GPU numerical/memory behavior, live logging from this new runner, real
full-split training and full validation traversal remain unverified. Prior Stage
3A learning/GPU evidence is preserved and is not presented as a new Stage 3B run.
Exact cloud-log recovery and latest mid-epoch checkpoint cloud durability are not
claimed. A forced kill can lose elapsed time since the last checkpoint; no automatic
restart or time-budget extension is provided. Recovery requires persistent originals
and latest checkpoint; the remote best artifact is a separate recovery point.

Rating-stratified metrics, training-board-disjoint validation diagnostics, WDL ECE/
class diagnostics and puzzle/match evaluation remain additional evaluation work
before a final M2 verdict. No chess-strength, generalization or final-B0 claim is
made. History dropout, openings_v2, 100M training, MCTS and self-play remain deferred.

**Recommendation: CONDITIONAL PASS for preparation.** CPU correctness and small
orchestration evidence support review; approve a GPU-host preflight and one bounded
candidate only after reviewing configuration, resources and the above limitations.
No sweep/pilot launch, commit, push or merge was performed.
