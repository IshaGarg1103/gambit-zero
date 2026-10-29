# M1 Blackwell environment benchmark follow-up

Benchmark gate: **PASS**. Recorded UTC: 2026-10-29T10:56:55.224653+00:00.
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition.
Python: 3.11.15; JAX backend: PJRT C API
cuda 12090.
Actual physical VRAM, driver, CUDA package versions, host RAM/CPU and source/lock
hashes are recorded in `m0_large_batch_blackwell.json`, under `host`.

Host logical CPUs / RAM GiB: 20 / 160.00.
JAX / jaxlib / Pgx / mctx: 0.7.2 / 0.7.2 / 2.6.0 / 0.0.6.
GPU inventory (including total/free VRAM bytes and driver):
`[{"driver_version": "595.71.05", "free_bytes": 101361647616, "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "pci_bus_id": "[Unknown Error]", "total_bytes": 102641958912, "uuid": "GPU-39663416-2c50-404b-b302-bdfeaa446975"}]`.

| Batch | Status | Compile s | Steps/s | Peak JAX GPU MiB | Repeats |
|---|---|---:|---:|---:|---:|
| 128 | MEASURED | 2.767 | 203,791.758 | 55.795 | 30 |
| 1024 | MEASURED | 2.857 | 693,665.300 | 385.316 | 30 |
| 4096 | MEASURED | 2.968 | 690,539.659 | 1,538.000 | 30 |
| 8192 | MEASURED | 2.802 | 748,291.938 | 3,073.562 | 30 |



Regression: {'status': 'PASS', 'passed': 264, 'seconds': 383.84}.
Protected source/schema hashes unchanged: True.

Each batch uses a fresh interpreter, three warmup calls and 30 measured repeats.
Each dispatch synchronizes the complete output tree. Compilation and tracing
are excluded from warm execution throughput. Initialization/setup is not timed.
Every call steps the same resident initial-position batch once; this is not a
multi-ply rollout or representative midgame workload. The JAX allocator peak
includes compilation/warmup, is cumulative within each isolated batch process,
and does not measure all device allocations by the driver or other processes.
Compiled buffer estimates and host peak RSS are recorded separately.

Safety: conservative pre-allocation estimates plus compiled-memory analysis and
fresh physical/logical free-memory checks gate execution. Unsafe sizes are
explicitly skipped. An unexpected failure stops further batch execution and is
retained in the output; subprocess isolation cannot guarantee avoidance of every
external OOM condition.

These are **environment-only** rates. No network inference, MCTS, data loading,
optimizer update or self-play is included. They must not be used as end-to-end
training or self-play throughput estimates. Dataset A/D, chess rules, action
codec, encoder, pinned dataset revision and frozen M1 acceptance remain unchanged.
The supplement supersedes the historical GPU deferral only when its gate is PASS;
it does not erase historical telemetry limitations or authorize M2.

## Independent result import verification

Returned ZIP: 29,404 bytes;
SHA256 `028cdf39821683bdb67791cb2fc4e0af8506c2f77c90439b459df170c11bae02`.
All 21 sealed result members passed SHA256
verification, and the source manifest matched the prepared input ZIP. Runtime
source hashes and dependency locks matched; per-worker rows and synchronized
throughput arithmetic were independently checked. See
[verification receipt](m1_blackwell_verification.json),
[source manifest](m1_blackwell_source_manifest.json) and
[result manifest](m1_blackwell_results_manifest.json).

All four requested batches were measured; no failures or safety skips occurred.
The short timing windows are single runs with 30 repeats, not confidence intervals
or sustained-load/power benchmarks. PCI bus telemetry returned `[Unknown Error]`
inside the Molab sandbox; GPU name, UUID, driver and physical memory were available.
This metadata limitation does not change the measured stepping results.

Recheck retained results without GPU or dataset access:

```bash
.venv/bin/python scripts/verify_blackwell_results.py
.venv/bin/python scripts/report_blackwell_followup.py
```
