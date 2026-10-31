# Stage 3A reproduction and manual Git handoff

Stage 2A/2B is already merged into main at `68e8381bd3457cd2dcbbb0a76588fc07df8192e7`.
Do not merge it again. Stage 3A work starts on `m2/stage-3a-overfit` from that commit.
All Git commands are for the owner to run manually.

## Data and environment

Use the existing persistent Stage 1 original-shard store; do not rescan PGNs.
`freeze_overfit_sample.py` disables downloads and verifies original train shards.
It scans all 2,654,850 train positions and freezes 1,024 positions from 1,024 games.
Selection is hash-seeded across all games, with one eligible position per game;
identical encoded inputs are explicitly excluded for this memorization diagnostic.
This does not change Dataset A or prescribe deduplication for future pilots.

```bash
uv sync --locked --extra test --extra data
JAX_PLATFORMS=cpu .venv/bin/python scripts/freeze_overfit_sample.py \
  --store data/processed/m2_training_originals \
  --output data/processed/m2_stage3a/fresh_sample

.venv/bin/python scripts/prepare_stage3a_bundle.py \
  --sample data/processed/m2_stage3a/fresh_sample \
  --output artifacts/gambit-zero-stage3a-input.zip
```

The derived sample and large checkpoints remain ignored. The real sample's
compressed tensors occupy about 0.4 MB; decoded tensors occupy about 36 MB.
No new original Dataset A downloads or weights are needed on the Mac. On Molab,
plan up to 15 GiB for the isolated CUDA environment and caches, plus approximately
2 GiB for checkpoint serialization, atomic replacement and W&B artifact staging.
Actual host/disk observations and virtualized-filesystem limits are in the report.

Run on an RTX PRO 6000 Blackwell using Python 3.11.15 and unchanged core locks:

```bash
uv sync --locked --python 3.11.15 --extra test --extra data --extra cuda
uv pip install --python .venv/bin/python -r requirements-wandb.txt --constraint requirements-lock.txt
uv pip check --python .venv/bin/python
```

W&B is optional and separately pinned at 0.28.1; core pins are unchanged. Record
installed transitive versions from the run environment. The SDK reads
`WANDB_API_KEY` from the environment only. Molab Secrets supplies it; never put
its value in code, command arguments, reports or Git. No `.env` is bundled.
The tracking destination is the approved sole-member `ishagarg-research/gambit-zero`.

The live-notebook source transfer was verified against its SHA256 before extraction.
From the extracted bundle root, the isolated job is:

```bash
python scripts/molab_stage3a.py
```

Alternatively after setup, run the trainer directly (the extracted bundle contains
`input/` and `STAGE3A_BUNDLE.json`):

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
JAX_DEFAULT_MATMUL_PRECISION=highest \
.venv/bin/python scripts/run_stage3a.py \
  --sample input --output data/processed/stage3a_manual \
  --steps 3000 --batch 32 --lr 0.001 --wandb-mode online
```

For no network, use `--wandb-mode offline`. Keep its local W&B directory and sync
it later with `wandb sync <offline-run-directory>` after reconnecting. Offline mode
does not require a key. Online mode fails if credentials/network are unavailable;
it never silently changes logging mode. `disabled` is for unit/local diagnostics,
not an assertion that a complete online experiment was logged.

Checkpoint receipts bind source manifest, sample hash, seed, LR, microbatch and
model settings. Save/restore verifies file size, SHA256, exact state arrays, step,
optimizer moments and evaluation metrics. Training continues from restored state
at periodic checkpoints. `--resume <old-checkpoint-directory>` can restore into a
fresh output with the same immutable inputs/config. Remote W&B interruption,
service restart and exact deduplicated cloud-log recovery are not acceptance claims.
The sample batch order derives solely from seed and optimizer step. No random
augmentation or history dropout is used.

The final checkpoint is explicitly logged as a W&B checkpoint artifact; result
ZIPs contain only metadata, logs and receipts, not the ~300 MB model state.
Molab scratch storage is not a durable checkpoint backup. Verify the artifact
before relying on it after sandbox shutdown.

## Tests

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_stage3a.py -q
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q
```


## Accepted full-batch confirmation

The successful Stage 3A recipe is frozen in
[`configs/stage3a_fullbatch_accepted.json`](../configs/stage3a_fullbatch_accepted.json).
It is a post-run configuration receipt, not a new runtime-config parser. The
unchanged `scripts/confirm_stage3a_fullbatch.py` reproduces the settings in the
pinned Molab bundle with the original frozen sample. No new run is authorized by
saving this configuration.

Accepted settings: 25M, bf16 compute/fp32 masters, effective batch 1,024 through
32x32 globally normalized accumulation, LR 3e-4, warmup 100 updates, AdamW
beta=(0.9,0.95), epsilon 1e-8, no weight decay/history dropout, global clip 1.0
once per full gradient. PASS at 281 updates: policy accuracy 99.0234%, CE 0.041589.
This is an overfit engineering recipe, not a validated full-data pilot recipe.
The original failed minibatch run and diagnosis remain part of the evidence.

See the [accepted report](../reports/m2_stage3a_fullbatch_confirmation.md),
[measured results](../reports/m2_stage3a_fullbatch_confirmation_results.json), and
[W&B run](https://wandb.ai/ishagarg-research/gambit-zero/runs/f9hphmku).
Large checkpoints remain in W&B; sample tensors, archives, caches and credentials
are excluded from Git. Stage 3B remains unstarted.

## Manual commit and push after review

```bash
git switch m2/stage-3a-overfit
git status --short
git add -- .gitignore requirements-wandb.txt m2execution.md \
  src/tinychess/train/ \
  scripts/freeze_overfit_sample.py scripts/prepare_stage3a_bundle.py \
  scripts/molab_stage3a.py scripts/run_stage3a.py scripts/report_stage3a.py \
  scripts/diagnose_stage3a.py scripts/audit_overfit_checkpoint.py \
  scripts/report_stage3a_diagnosis.py scripts/confirm_stage3a_fullbatch.py \
  scripts/report_fullbatch_confirmation.py \
  tests/test_stage3a.py tests/test_stage3a_diagnosis.py \
  tests/test_fullbatch_confirmation.py \
  configs/stage3a_diagnosis.json configs/stage3a_fullbatch_accepted.json \
  docs/m2_stage3a_run.md \
  reports/m2_stage3a.md reports/m2_stage3a_results.json \
  reports/m2_stage3a_learning_curve.svg \
  reports/m2_stage3a_diagnosis.md reports/m2_stage3a_diagnosis_results.json \
  reports/m2_stage3a_diagnosis_curve.svg \
  reports/m2_stage3a_fullbatch_confirmation.md \
  reports/m2_stage3a_fullbatch_confirmation_results.json \
  reports/m2_stage3a_fullbatch_confirmation_curve.svg
git diff --cached --check
git diff --cached --stat
git diff --cached --name-only
git diff --cached
git commit -m "feat(m2): complete stage 3a overfit validation"
git push -u origin m2/stage-3a-overfit
git fetch origin
git rev-parse HEAD
git rev-parse origin/m2/stage-3a-overfit
git status --short
```

Verify the two SHAs match and review all staged files, including anything staged
before this command. No automatic Git actions or later-stage experiment is authorized.
