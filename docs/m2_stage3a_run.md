# Accepted Stage 3A reproduction

Use `m2/stage-3a-overfit`. Stage 3A passed its bounded engineering gate; Stage 3B
is not authorized. This cleanup is a normal follow-up to commit
`3670cf67ac19602b3a0c6bdc41fc974b713aed1b`, not a history rewrite.

## Verified recipe

[Accepted configuration](../configs/stage3a_fullbatch_accepted.json): 25,622,092
parameters, bf16 compute, fp32 masters/moments, 32 microbatches × 32 positions per
optimizer update, LR 3e-4, 100-update warmup, AdamW beta=(0.9,0.95), epsilon 1e-8,
no weight decay/history dropout and global norm clip 1.0 **after accumulation**.
The runner now reads and validates this configuration. Alternate configurations
are rejected; this entry point is not a pilot or sweep driver.

The recorded run reached 99.0234% policy accuracy / CE 0.041589 at update 281.
See the [final report](../reports/m2_stage3a_fullbatch_confirmation.md),
[compact verified measurements](../reports/m2_stage3a_fullbatch_confirmation_results.json)
and [W&B run](https://wandb.ai/ishagarg-research/gambit-zero/runs/f9hphmku).
Recorded source hashes identify the original GPU execution, before cleanup;
cleanup does not relabel its original source bytes as newly GPU-tested code.

## Recover the exact sample

The sample SHA256 is
`cc00bda4d5feb8a980be6e32208712f50eb94bb197ae60d01aeaff6e47ced590`.
The source/sample ZIP and manifest remain in W&B artifact
`ishagarg-research/gambit-zero/stage3a-25m-provenance:v0`, attached to the original
run. Recover only `input/sample.npz` and `input/manifest.json` from its input ZIP;
use the cleaned checkout for code, not the historical failed runner in that ZIP.
The sample is derived from Dataset A's immutable revision
`b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c`; validation/test are never used.

Alternatively regenerate it from the already verified original training-shard
store. This reads the original Parquet, verifies checksums/counts and uses the
fixed-seed selection algorithm. It downloads nothing and never scans Lichess PGNs:

```bash
uv sync --locked --extra test --extra data
JAX_PLATFORMS=cpu .venv/bin/python scripts/freeze_overfit_sample.py \
  --store data/processed/m2_training_originals \
  --output data/processed/stage3a_reproduced_sample
.venv/bin/python scripts/prepare_stage3a_bundle.py \
  --sample data/processed/stage3a_reproduced_sample \
  --output artifacts/gambit-zero-stage3a-input.zip
```

The packager includes accepted source/tests/configuration, locks and the small
verified sample. It excludes raw reports, logs, environments, checkpoints and
credentials. The generated ZIP and sample stay outside Git. The sample is about
394 KB compressed /36 MB decoded. Allow remote CUDA setup/cache space of up to
15 GiB, plus about 2 GiB temporary checkpoint/artifact headroom. Molab scratch
storage is not a durable checkpoint backup.

## Pinned GPU environment and explicit reproduction

Only perform a new training run when separately authorized. In a fresh Blackwell
Molab bundle extraction, provide `WANDB_API_KEY` through Molab Secrets/environment.
Never write its value into code, commands, logs or reports.

```bash
uv sync --locked --python 3.11.15 --extra test --extra data --extra cuda
uv pip install --python .venv/bin/python -r requirements-wandb.txt --constraint requirements-lock.txt
uv pip check --python .venv/bin/python
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
JAX_DEFAULT_MATMUL_PRECISION=highest \
.venv/bin/python scripts/confirm_stage3a_fullbatch.py \
  --config configs/stage3a_fullbatch_accepted.json --sample input \
  --output data/processed/stage3a_fullbatch_confirmation --wandb-mode online
```

`--wandb-mode offline` stores local tracking data without a key or network;
`wandb sync <offline-run-directory>` can synchronize it later. Online mode fails
if credentials/network are unavailable; no silent logging-mode switch occurs.
The destination is `ishagarg-research/gambit-zero`, your approved sole-member team.
W&B is separately pinned to 0.28.1; the core JAX/Flax/CUDA locks are unchanged.

The runner applies exactly one clipped AdamW update per complete population,
checks finite values, evaluates the gate every update, and stops at its first
crossing or 2,000 updates. Use a fresh output directory; an atomic guard prevents
accidental repeat execution. Checkpoint SHA256/metadata and restored parameters/
metrics are verified. The checkpoint artifact is explicitly uploaded to W&B.
The result JSON records the curve, predictions, all missed rows and strata.
This is a tiny-set engineering result, not held-out strength or pilot performance.

## Tests and report reproduction

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
  tests/test_stage3a.py tests/test_fullbatch_confirmation.py \
  tests/test_policy_head_mapping.py tests/test_stage3a_reproduction.py -q
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q
.venv/bin/python scripts/report_fullbatch_confirmation.py \
  --results reports/m2_stage3a_fullbatch_confirmation_results.json \
  --output reports/m2_stage3a_fullbatch_confirmation.md
```

Historical runs remain on W&B and historical code remains recoverable from the
parent Git commit. Only the accepted implementation and final results are kept
in the current tree. No W&B run or dataset was deleted by cleanup.
