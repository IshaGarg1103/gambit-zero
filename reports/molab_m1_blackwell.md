# M1 Blackwell benchmark follow-up: Molab protocol

Status: **Molab run completed and independently verified: PASS**.
All four batches measured; 264 CUDA tests passed in 383.84 seconds.
See the [measured report](m1_blackwell_benchmark.md) and
[verification receipt](m1_blackwell_verification.json).
The local branch `m1/blackwell-benchmark` starts at merged main
`69fd040d5a9d979038bd222eb5e8b7e7ba9935d0`. The accepted M1 tag is
`m1-dataset-50k-v1` (accepted source `9893c2edcdb99dd57a58da8cd688b4157753bf93`).
No Git commands are executed by the runner or bundle builder. The user handles
commits, pushes and the eventual small follow-up PR after GPU verification.

## Inspection and scope

The existing benchmark uses JAX 0.7.2 compiled memory analysis and synchronized
`block_until_ready` stepping. The CUDA extra is locked in `uv.lock` and
`requirements-cuda-lock.txt`: JAX/jaxlib/CUDA plugin/PJRT 0.7.2, Pgx 2.6.0,
mctx 0.0.6, Flax 0.12.0, Optax 0.2.6; CUDA runtime 12.9.79 and cuDNN 9.27.0.42.
No dependency versions or locks were changed. The runner installs Python 3.11.15
and the locked test/data/CUDA extras in its own `.venv`, using uv 0.10.9.
Notebook JAX packages are not used or upgraded. Setup requires network access
to the dependency registries and temporary Molab storage for CUDA wheels/cache;
no Mac dataset storage or Lichess/Hugging Face data access is needed.

The follow-up adds atomic per-batch reports, explicit GPU enforcement, refreshed
free-memory checks and failure capture. The Molab controller executes batches
128, 1024, 4096, 8192 in separate fresh interpreters with 30 repeats each. Each
uses three warmups and synchronization on every dispatch. Trace, compile and
warm execution times are separate. Initial state construction is excluded.

Both pre-allocation (1 MiB/state estimate) and post-compilation (twice compiled
buffers plus 256 MiB reserve) gates apply, with a budget capped at 35% of current
free memory and 16 GiB. Missing memory analysis/telemetry leads to explicit skips.
Unexpected worker failure stops larger batches and packages diagnostics. This
reduces OOM risk; it cannot guarantee protection against concurrent GPU users or
compiler/driver allocations absent from memory estimates.

Peak GPU memory is the JAX allocator high-water mark, cumulative within each
isolated worker and including compilation/warmup. It excludes some driver and
other-process allocations. Physical free/total memory, driver, system nvcc,
backend CUDA version and all installed CUDA/JAX packages are recorded separately.
No invented device-memory figure or driver-CUDA/runtime equivalence is reported.

The full regression suite runs in a fresh CUDA process after the progressive
benchmark. It covers rules/termination/history, codec, encoder and mctx value
signs/mates. Protected source/configuration hashes are checked before and after.
No source under `src/`, dataset schema, Dataset A/D objects, dataset revision,
training code or self-play code is changed by this follow-up.

## Build and execute

On the Mac, regenerate the small source-only input ZIP with:

```bash
.venv/bin/python scripts/prepare_molab_blackwell.py
```

This writes `artifacts/gambit-zero-m1-blackwell-input.zip` and prints its SHA256.
It includes source, tests, locked configuration, required small report inputs
and a per-file checksum manifest. It excludes Git, environments, caches,
credentials, dataset shards, raw archives and checkpoints.

Create a fresh Molab notebook manually, select the RTX PRO 6000 Blackwell runtime,
and upload that ZIP. Run the notebook cell supplied in chat, using the actual
uploaded path and printed ZIP SHA256. It extracts to a fresh directory, checks
all source members, runs the standalone controller and returns a results ZIP.
Keep the remote session alive while it runs; no archive-scan resume logic is
needed here. An interruption may require rerunning this bounded benchmark.

## Returned evidence and acceptance

The results ZIP includes:

- `reports/m0_large_batch_blackwell.json`: aggregate actual metadata/rows/failures.
- `reports/m1_blackwell_benchmark.md`: readable measured results and limitations.
- `reports/m1_blackwell_acceptance.json`: supplemental GPU acceptance gate.
- `reports/m1_blackwell_tests.txt`: actual CUDA regression output, when reached.
- Worker logs/raw batch results and host/stack fingerprints.
- `BLACKWELL_BUNDLE.json` and `RESULTS_MANIFEST.json` for source/result checksums.

Gate PASS requires the expected GPU, at least one measured batch, all four batch
sizes accounted for (unsafe ones explicitly skipped), 30 repeats for measured
rows, positive measured GPU peak telemetry, correct transitions, a successful
full regression and unchanged protected source hashes. Failed/incomplete runs
remain marked accordingly and their ZIP should still be shared for diagnosis.

The historical final M1 acceptance JSON/report stays frozen. The measured
supplement updates the GPU disposition only after a successful Molab run; it
preserves all original scientific results and historical telemetry limitations.
Once the returned evidence is verified, add a link to the measured supplement
in the M1 acceptance documentation and prepare the small PR into main.
The returned ZIP and all sealed member checksums have now passed independent verification.

Rates describe repeated independent first-ply environment transitions on resident
initial states. They are **not** midgame trajectories, model inference, MCTS,
end-to-end training or self-play throughput.

Dataset revision remains `b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c`.
No M2 or self-play is authorized or started.

## Local preparation validation (CPU, not Blackwell evidence)

Full regression: **264 passed in 49.87 seconds**, including 11 new benchmark
safety/provenance tests. See [actual local output](m1_blackwell_local_tests.txt).
The source-only ZIP was extracted into a fresh temporary directory and all 121
listed payload files verified against its manifest (122 ZIP members including
the manifest). Notebook cell syntax passed. All protected source/schema hashes
and frozen M1 acceptance results remain unchanged. The later returned Molab run supplied the verified GPU measurements linked above.

## Final import and documentation verification

Returned ZIP SHA256:
`028cdf39821683bdb67791cb2fc4e0af8506c2f77c90439b459df170c11bae02`.
All 21 sealed result members, prepared input bundle, runtime source fingerprints,
locks, per-worker rows and throughput arithmetic verified. All four requested
batches measured; no failures or skips occurred.

The dated GPU supplement is linked from the M1 report/checklist/README and
survives historical report regeneration. Original CPU findings and acceptance
JSON remain unchanged. Final local regression after adding documentation checks:
**267 passed in 51.74 seconds**;
[actual output](m1_blackwell_verification_tests.txt). The Molab run's separate
CUDA regression remains **264 passed in 383.84 seconds**;
[actual GPU output](m1_blackwell_tests.txt). No new GPU runs are implied by the
extra local documentation tests.

```bash
.venv/bin/python scripts/verify_blackwell_results.py
.venv/bin/python scripts/report_blackwell_followup.py
```
