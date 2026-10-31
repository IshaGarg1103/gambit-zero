# M2 Stage 2B — returned Blackwell model validation

**Recommendation: PASS.**
Stage 3 was not executed. Readiness below is a technical result, not authorization.

Source base: `44a7d28bb23830111863c65ddc25a3cb90dddb48`. Input ZIP SHA256: `fbc97689f96de84cb8a7722dec62dbf40bc1e1648fee095b1259b84813303696`.
Exact source manifest SHA256: `a27afa4062b2c06b7f876f0846ca04d0cd46227791e8c7e0b87c48a46e951ae6`.
Dataset A remains pinned; no dataset, loader benchmark, model redesign, MCTS or self-play run.

Started UTC: 2026-10-31T06:43:13.507453+00:00; completed UTC: 2026-10-31T06:54:41.456587+00:00.

## Hardware and pinned environment

GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition. Packages/runtime/driver/VRAM/host metadata: the retained [`m2_stage2b_results.json`](m2_stage2b_results.json) (`host` field).
Host available RAM: 171234029568; logical CPUs: 20; disk preflight: {'free_bytes': 9223372036852248576, 'minimum_bytes': 16106127360}.
Physical GPUs/VRAM/driver: `[{"driver_version": "595.71.05", "free_bytes": 101361647616, "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "pci_bus_id": "[Unknown Error]", "total_bytes": 102641958912, "uuid": "GPU-6c61f193-70da-1d6f-30b5-e512ee4e6f5a"}]`.
Backend: `PJRT C API
cuda 12090`; Python: `3.11.15`.
Installed versions: `{"chess": "1.11.2", "flax": "0.12.0", "jax": "0.7.2", "jax-cuda12-pjrt": "0.7.2", "jax-cuda12-plugin": "0.7.2", "jaxlib": "0.7.2", "mctx": "0.0.6", "nvidia-cublas-cu12": "12.9.2.10", "nvidia-cuda-cccl-cu12": "12.9.27", "nvidia-cuda-cupti-cu12": "12.9.79", "nvidia-cuda-nvcc-cu12": "12.9.86", "nvidia-cuda-nvrtc-cu12": "12.9.86", "nvidia-cuda-runtime-cu12": "12.9.79", "nvidia-cudnn-cu12": "9.27.0.42", "nvidia-cufft-cu12": "11.4.1.4", "nvidia-cusolver-cu12": "11.7.5.82", "nvidia-cusparse-cu12": "12.5.10.65", "nvidia-nccl-cu12": "2.32.3", "nvidia-nvjitlink-cu12": "12.9.86", "nvidia-nvshmem-cu12": "3.8.0", "optax": "0.2.6", "pgx": "2.6.0", "psutil": "7.2.2", "pytest": "8.4.1", "python-chess": "1.999"}`.
nvidia-smi driver capability, system nvcc and pip CUDA runtime are recorded separately, not conflated.

## Actual isolated cases

| Model | Compute | B | Status | Parameters | Updates | JAX peak bytes | Sampled physical used bytes |
|---|---|---:|---|---:|---:|---:|---:|
| 25m | bfloat16 | 4 | PASS | 25622092 | 1 | 733935104 | 1724907520 |
| 100m | bfloat16 | 4 | PASS | 100031308 | 1 | 2846116352 | 4969201664 |
| 25m | float32 | 4 | PASS | 25622092 | 1 | 735474688 | 1731198976 |
| 100m | float32 | 4 | PASS | 100031308 | 1 | 2850035712 | 4956618752 |
| 25m | bfloat16 | 32 | PASS | 25622092 | 1 | 957575680 | 1731198976 |
| 100m | bfloat16 | 32 | PASS | 100031308 | 1 | 2901006080 | 4962910208 |

## Phase timings and numerical checks

| Model / compute / B | Phase | Trace s | Compile s | Synchronized execution s |
|---|---|---:|---:|---:|
| 25m / bfloat16 / 4 | adamw_once | 0.549591 | 0.959252 | 0.016890 |
| 25m / bfloat16 / 4 | backward | 0.777485 | 3.890301 | 0.013812 |
| 25m / bfloat16 / 4 | forward | 0.165883 | 2.319318 | 0.004939 |
| 25m / bfloat16 / 4 | initialize | 0.976170 | 8.441876 | 0.005205 |
| 25m / bfloat16 / 4 | post_update_forward | 0.001687 | 0.000179 | 0.003989 |
| 100m / bfloat16 / 4 | adamw_once | 1.023692 | 1.528630 | 0.043681 |
| 100m / bfloat16 / 4 | backward | 1.204736 | 4.827733 | 0.020132 |
| 100m / bfloat16 / 4 | forward | 0.284025 | 2.929573 | 0.009043 |
| 100m / bfloat16 / 4 | initialize | 1.048032 | 13.683542 | 0.009400 |
| 100m / bfloat16 / 4 | post_update_forward | 0.002675 | 0.000108 | 0.006493 |
| 25m / float32 / 4 | adamw_once | 0.558739 | 1.026890 | 0.017998 |
| 25m / float32 / 4 | backward | 0.726546 | 2.772857 | 0.015088 |
| 25m / float32 / 4 | forward | 0.149808 | 1.325378 | 0.004650 |
| 25m / float32 / 4 | initialize | 0.968231 | 8.407533 | 0.005007 |
| 25m / float32 / 4 | post_update_forward | 0.001645 | 0.000030 | 0.003976 |
| 100m / float32 / 4 | adamw_once | 1.086986 | 1.500545 | 0.043825 |
| 100m / float32 / 4 | backward | 1.102248 | 3.639408 | 0.023683 |
| 100m / float32 / 4 | forward | 0.327560 | 1.754816 | 0.007740 |
| 100m / float32 / 4 | initialize | 1.064618 | 13.544680 | 0.008225 |
| 100m / float32 / 4 | post_update_forward | 0.002320 | 0.000048 | 0.006858 |
| 25m / bfloat16 / 32 | adamw_once | 0.527772 | 1.034752 | 0.010182 |
| 25m / bfloat16 / 32 | backward | 0.745006 | 4.807199 | 0.023129 |
| 25m / bfloat16 / 32 | forward | 0.176622 | 3.643582 | 0.005421 |
| 25m / bfloat16 / 32 | initialize | 0.980223 | 8.385538 | 0.005344 |
| 25m / bfloat16 / 32 | post_update_forward | 0.001695 | 0.000032 | 0.004593 |
| 100m / bfloat16 / 32 | adamw_once | 1.014396 | 1.445906 | 0.013847 |
| 100m / bfloat16 / 32 | backward | 1.167311 | 5.729748 | 0.068381 |
| 100m / bfloat16 / 32 | forward | 0.291604 | 4.430453 | 0.013504 |
| 100m / bfloat16 / 32 | initialize | 1.039980 | 13.556086 | 0.009313 |
| 100m / bfloat16 / 32 | post_update_forward | 0.002395 | 0.000049 | 0.012288 |

| Model / compute / B | Loss before / after | Gradient norm squared | Parameter delta squared | Host RSS bytes |
|---|---|---:|---:|---:|
| 25m / bfloat16 / 4 | 5.231551647186279 / 3.201472759246826 | 6721.2958984375 | 0.2552567422389984 | 2302111744 |
| 100m / bfloat16 / 4 | 4.2124223709106445 / 3.318087577819824 | 2832.67626953125 | 0.995091438293457 | 2744258560 |
| 25m / float32 / 4 | 5.219195365905762 / 3.2052807807922363 | 6687.72412109375 | 0.2552228569984436 | 2138456064 |
| 100m / float32 / 4 | 4.2215576171875 / 3.3186943531036377 | 2853.025634765625 | 0.9950222969055176 | 2572873728 |
| 25m / bfloat16 / 32 | 5.258713722229004 / 3.3496646881103516 | 6767.80078125 | 0.2552064061164856 | 2320404480 |
| 100m / bfloat16 / 32 | 4.106322765350342 / 3.585386276245117 | 2700.09619140625 | 0.9943730235099792 | 2701881344 |

Each successful case uses a fresh initialization and exactly one AdamW update;
no worker trains a trajectory beyond one step. Up to six independent updates total.
Synthetic labels are applied to four scripted canonical Pgx positions; no human-data acquisition.
AdamW smoke constants: LR=1e-4, beta=(0.9,0.95), weight_decay=0.01; these are not tuned pilot settings.
Every case JSON records before/after losses, parameter delta, fp32 parameter/gradient/optimizer checks,
separate tracing/compilation and synchronized execution times for init, forward, backward, update and post-forward.
JAX peak covers its allocator in that fresh worker. nvidia-smi is sampled every ~0.2s and may miss short peaks;
sampled physical use includes other processes (maximum across physical devices), not just this model.
Host RSS is recorded per case. Estimates are preflight safeguards, not measured-memory replacements.

## Tests and recommendation

Relevant CUDA regression: return code 0, passed 86.
Safe measured starting bf16 microbatches: `{'25m': 32, '100m': 32}`. No effective-batch or accumulation claim.
Stage 3A technical readiness established: **True**; owner review/authorization still required.
If any essential smoke/test/source-integrity check failed, do not start overfit training.

## Reproduction

The exact commands, host metadata and all six case results are retained in [`m2_stage2b_results.json`](m2_stage2b_results.json). Raw logs and per-case JSON remain in the results ZIP.
Reproduce by uploading the matching input ZIP and running its generated Molab cell in a fresh GPU notebook.
The runner uses uv 0.10.9, Python 3.11.15, unchanged locked test/data/CUDA extras.
Missing dependencies/interpreter may require network; no Lichess/HF dataset transfer is allowed.
The results ZIP includes checksummed evidence and BLACKWELL_BUNDLE.json, including failure diagnostics.


## Returned-evidence verification — 31 October 2026

All 25 indexed evidence files match their SHA256 hashes. Results ZIP SHA256:
`bd2caae129706a751836639ff73dbc5d6372f658f782c1dcfea32296ad351e1a`.
The source manifest is byte-identical to the corrected input bundle; every protected
source file also matches the local checkout. Individual worker JSON agrees with
the aggregate results. Independently reassessing the evidence returns PASS;
regenerating the original remote report reproduces it byte-for-byte.
See [`m2_stage2b_verification.json`](m2_stage2b_verification.json) for the evidence index.
Raw evidence is retained locally under `data/processed/m2_stage2b/returned/` and
`artifacts/gambit-zero-stage2b-results.zip`; neither is automatically committed.

The complete CPU regression passed **373 tests in 67.99s** after the launcher
import fix; the focused Stage 2B suite passed **25 tests in 4.51s**. The CUDA
correctness suite passed **86 tests in 343.55s**. No tests failed or cases skipped.

At bf16 batch 32, peak JAX allocator use was **0.892 GiB (25M)** and
**2.702 GiB (100M)**; sampled device-wide peaks were **1.612 / 4.622 GiB**.
Across all cases the maximum sampled device-wide use was **4.628 GiB**, and
maximum host worker RSS was **2.556 GiB**. These are smoke measurements, not
end-to-end loader/training throughput or integrated training peak estimates.
Start Stage 3A conservatively at microbatch **32** for either model and remeasure
memory in its actual training loop; no larger batch was tested.

Molab's filesystem preflight reported an implausible near-2^63 free-byte value.
Treat this as virtualized filesystem telemetry, not established disk capacity.
Host RAM is host-visible rather than independently guaranteed container capacity;
workers also checked available cgroup limits. The small smoke completed normally.
No correctness or numerical blocker was found. Full real-data training, overfit
convergence, accumulation and checkpoint/resume remain untested here. Stage 3A
is technically ready but has not been started or authorized by this result.
