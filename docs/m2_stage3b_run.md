# Stage 3B — run guide and executed recipe

The proposal below records the reviewed preparation recipe. The owner subsequently
authorized the smoke, three-point sweep and fresh 25M pilot; these completed on
9–10 November 2026. See [the actual execution report](../reports/m2_stage3b.md)
for early stopping, measured results, immutable checkpoints and the unresolved
value-quality concern. The historical [preparation report](../reports/m2_stage3b_preparation.md)
and [configuration](../configs/stage3b_25m.json) remain available. This guide does
not authorize any additional execution.
The original B0 → independent B1/B2 research sequence remains unchanged.

## Data and initialization

Use original Dataset A at revision
`b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c`: 2,654,850 eligible train positions
(January–July) and 569,124 validation positions (August). The trainer never opens
September test shards. Inventory checks still read pinned metadata for all months.
No PGN scan, repacking, new data, feature changes, board deduplication or history
dropout. Preserve natural repeated positions. Forfeits keep policy weight 1 and
value weight 0. Malformed records fail immediately through the accepted loader.

Each candidate starts with freshly seeded 25,622,092-parameter model weights,
seed 20261109. Reuse Stage 3A's zero output-head initialization; check initial
policy CE against weighted mean ln(legal moves) and WDL CE against ln(3).
Never initialize from the overfit checkpoint. A Stage 3B resume is separately
checked against source/config hashes, dataset revision, model and optimizer step.

## Proposed optimization and budgets — not yet executed

| Setting | Proposal |
|---|---|
| Model/precision | 25M; bf16 compute, fp32 master parameters and moments |
| Microbatch / accumulation | 32 × 32 = 1,024 positions per update |
| Shuffle | Existing deterministic shard order + 32,768-row replacement buffer |
| LR comparison | Three independent one-epoch runs: 1e-4, 3e-4, 6e-4 |
| Optimizer | Existing AdamW beta=(0.9,0.95), epsilon 1e-8, clip 1.0 once after accumulation |
| Weight decay | 0.01 on matrix kernels; exclude bias, norms and positional embeddings |
| Schedule | 1,000-update linear warmup, cosine to 10% of peak |
| Pilot duration | Up to 20 epochs; stop after 2 full validation epochs without NLL improvement ≥1e-4 |
| Validation | Entire August split at each completed epoch; no validation sampling for selection |
| Checkpoints | Latest every 500 updates and epoch end; separate best by validation policy NLL |
| Logging | Batch metrics/mixing/LR/norms every 25 updates; full validation per epoch |
| Proposed time caps | 1 active wall-hour per LR candidate; up to 12 active wall-hours for selected pilot |
| W&B | Project gambit-zero; groups m2-lr-sweep and m2-supervised |

One epoch is 2,593 updates, including a final 642-position batch. All positions
contribute exactly once per completed epoch; padding repeats legal input rows
with BOTH weights zero. No row is discarded. Policy and WDL gradients use
separate global weight denominators, including the partial batch; clipping and
AdamW run once after summation. Validation uses microbatches of 32 (17,786 batches;
last contains 4 real positions), sums loss numerators/denominators over the entire
split and reports policy NLL/top-1/top-3, weighted WDL NLL/accuracy/Brier and
separate zero-based ply <10 versus ply ≥10 metrics.

A full sweep is at most 7,779 updates / three epochs total. Each candidate uses
the same initial seed, first-epoch ordering and one-epoch schedule. Pilot cosine
decay spans its configured 20-epoch horizon (51,860 updates); it therefore differs
from the one-epoch sweep schedule. Do not compare their curves as identical
optimization trajectories. Select LR using full validation policy NLL, then
review the sweep before authorizing a NEW pilot initialized from scratch.
No automatic sweep loop, LR selection, fallback LR or pilot launch is provided.

The caps include initialization, compilation, loader/replay work, validation and
checkpoint work until the next check. Operations are not preempted mid-JIT,
HTTP call or artifact upload; an operation can cross the limit. Resume carries
elapsed time from its checkpoint; a forced kill can lose elapsed time since that
checkpoint. BUDGET_STOP is not a completed/passing experiment and must be reviewed;
there is no automatic restart or budget extension.

## Reuse, recovery and storage

The existing VerifiedStore keeps checksum-verified Parquet locally on the GPU
host and reuses it across epochs/candidates. No file is deleted from that store.
Downloads default OFF; --download explicitly permits missing pinned originals
only. A blank store needs 1,730,442,889 bytes of train+validation Parquet, plus
small manifests. Existing complete train originals on the Mac leave 313,430,930
bytes missing for validation; those bytes were NOT downloaded in preparation.

Preflight reserves 20 GiB free disk and 3 GiB checkpoint/output staging, in
addition to missing originals. On a fresh GPU host, allow roughly 24.7 GiB free
after environment setup. The isolated CUDA environment/caches can additionally
require up to about 15 GiB if not already installed. Latest/best model+Adam state
are kept separately; each 25M state is about 307 MB. Atomic replacement, local
serialization and W&B staging explain the larger 3 GiB work allowance. No claim
of measured aggregate peak disk use is made yet.

Host memory planning uses the accepted 36,000-byte/row estimate, plus decoded
batch/chunk allowances: 1,262,592,000 bytes for this loader profile, plus a 3 GiB
model/work reserve (about 4.18 GiB available RAM required). Check cgroup limits as
well as host RAM. Blackwell VRAM does not establish available host RAM. Compiled
buffer safety and physical/JAX memory are checked/recorded on the GPU before any
update. The 200K shuffle buffer is not part of this preparation.

Recovery restores exact model/optimizer state and `(epoch, next_batch)`, recreates
the seed+epoch loader and replays/discards the previously consumed prefix. This
is bounded-memory recovery, not serialization of the Python shuffle buffer.
It costs repeated reads/checksums/CPU decoding, but no data redownload. Final
batch and epoch boundaries are checked. Keep the latest checkpoint, original
shards and config/source snapshot persistently; Molab scratch can disappear when
the session ends. Best checkpoints are uploaded to W&B; latest mid-epoch recovery
is local. Do not promise cloud recovery of the latest 500-update boundary.
Cloud metric logging across interruptions is not claimed to be exactly once.

## Commands

Use the existing pinned CPU environment; GPU setup remains the accepted Stage 2B
CUDA setup. W&B is separately pinned in requirements-wandb.txt, with credentials
from WANDB_API_KEY/Molab Secrets only. Offline mode stores tracking data locally;
sync it later using the W&B CLI. No token goes into source, config or reports.

Safe local metadata/resource preflight (no shard downloads, model init or training):

```bash
JAX_PLATFORMS=cpu .venv/bin/python scripts/run_supervised_pilot.py \
  --mode sweep --lr-index 0 --output data/processed/stage3b_preflight
JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_supervised_pilot.py -q
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q
```

Future GPU commands below are examples for review, NOT authorization. Each needs
separate approval. Set a persistent GPU-host --store path and fresh --output.
Do not put --execute-approved in a notebook auto-run cell before approval.

```bash
# One approved candidate only; indices 0,1,2 correspond to the three reviewed LRs.
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
JAX_DEFAULT_MATMUL_PRECISION=highest \
.venv/bin/python scripts/run_supervised_pilot.py --mode sweep --lr-index 0 \
  --store /persistent/gambit-zero-originals --output /persistent/stage3b/lr-1e-4 \
  --download --execute-approved --wandb-mode online

# Example pilot AFTER sweep review/approval; 3e-4 is not a selected result.
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
JAX_DEFAULT_MATMUL_PRECISION=highest \
.venv/bin/python scripts/run_supervised_pilot.py --mode pilot --lr 0.0003 \
  --store /persistent/gambit-zero-originals --output /persistent/stage3b/pilot \
  --download --execute-approved --wandb-mode online
```

For approved interrupted-run recovery, use the exact same command and source/
config with --resume; keep the same output and LR. A fresh run refuses to replace
existing output. Recovery refuses Stage 3A checkpoints or changed recipes/source.

Before a long pilot, inspect the first candidate's full validation result,
initial-loss sanity, finite-update checks, clip fraction, loader wait, synchronized
update throughput, end-to-end throughput and physical/JAX/host peaks. Compilation
is reported separately. Extrapolate time from actual candidate measurements,
including full validation and I/O; Stage 3A tiny-set throughput is insufficient.

## Remaining scope

The bounded GPU gate, LR comparison and selected 25M pilot are complete within
their recorded scope; the execution report supersedes preparation-only status.
Rating-stratified metrics, validation disjoint from all
training board hashes, WDL ECE/class diagnostics and external puzzle/match evaluation
remain additional evaluation work before a final M2 verdict. They are not claimed
by this trainer. History dropout, openings_v2, 100M training, MCTS and self-play
remain unstarted/deferred. These are pilot checkpoints, never an automatic B0 freeze.
