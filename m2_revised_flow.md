# Gambit Zero — Revised M2 Execution Flow

**Date:** 30 October 2026  
**Purpose:** Practical stage-by-stage handoff reflecting the agreed order of work. This supplements `m2execution.md`; it does not replace the canonical research specification or authorize all stages at once.

## 1. Research objective (unchanged)

Build a chess-specialized transformer **from scratch** in JAX/Flax, first using human-game supervised policy + WDL learning, then AlphaZero-style self-play RL. The ~100M model is the primary arm; ~25M is the reference comparison. **No pretrained-LLM fine-tuning or GRPO.**

The long-term experiment remains:

```text
Verified chess environment (M0)
        ↓
Human Dataset A (M1)
        ↓
M2: human-data supervised pilot (25M versus 100M)
        ↓
Decision: is more human data needed to train final B0?
        ↓
Freeze selected final human-only B0 checkpoint
        ├── B1: PUCT-32 self-play from frozen B0
        └── B2: Gumbel-32 self-play from the SAME frozen B0
                   ↓
         Compare gains per GPU-hour at matched compute
```

B1 and B2 must have separate buffers, training runs, and checkpoints. The 50K-game pilot is **not automatically** the final frozen B0. Any 1M+ dataset expansion requires an explicit evidence-based decision.

## 2. Current status and evidence

| Work | Status | Evidence / qualification |
|---|---|---|
| M0: Pgx chess environment, codec, feature encoder | Complete | Chess-rule parity and Stage 0 successor mapping verified |
| M1: 50K accepted human games | Complete | 3,793,600 retained positions across train/val/test; immutable Dataset A |
| M1 Blackwell *environment* benchmark | Complete | Pgx transition benchmark only—not network training performance |
| M2 Stage 0: final correctness gate | Complete | 3,384 successor transitions, 269 tests reported |
| Stage 1A: Parquet training loader | Complete | Original-data loader, batched decoding |
| Stage 1B: data integrity and shuffle | Complete | 2 full train-split epochs, each 2,654,850 positions; zero missing/duplicate/split errors |
| Stage 1C: optional lossless cache | Assessed; deferred | Original Parquet loader adequate; no extra cache built |
| Stage 1D: history-dropout option | **Pending; deliberately deferred** | Baseline 0%; later controlled 0% versus 10% at 25M |
| Stage 1E: `openings_v2`, Stockfish setup | **Pending; deliberately deferred** | Required before comparative chess matches, not before basic training |
| Stage 2A: 25M / 100M architecture | **Implemented locally, reported; review/preservation needed** | 25,622,092 and 100,031,308 parameters; 348 reported CPU tests, fp32/bf16 forward/backward |
| Stage 2B: limited Blackwell *model* smoke | **Next** | Not a large standalone GPU benchmark |
| Stage 3: supervised learning | Not started | Overfit → 25M LR sweep/pilot → 100M pilot → evaluations |

**Important:** The Stage 2A report said the implementation was not committed or pushed when reported. Before remote Blackwell testing, preserve the code in a reviewable commit/PR or otherwise make the exact working tree available on the server. The table records reported results, not independently rerun tests. Check current repository/PR state before Git operations.

**Immutable Dataset A:** `ishagarg1103/gambit-zero` at HF revision `b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c`. Train Jan–July 2026 (35,000 games), validation August (7,500), test September (7,500). Keep Dataset D evaluation-only and preserve source splits/labels.

## 3. Next execution sequence — authorize ONE checkpoint at a time

### Step 1 — Review and preserve Stage 2A (immediate housekeeping)

- Inspect code diff and `reports/m2_stage2a.md`; verify actual model parameter counts, feature/token ordering, 73-plane policy mapping, legal masking, WDL signs and precision contract.
- Explicit mapping: board token `planes[row, col]` corresponds to `source = col*8 + (7-row)`; action index = `source*73 + action_plane`. Never use naive row-major flattening.
- Keep parameters fp32, bf16 matmul/attention compute where supported, fp32 LayerNorm/softmax/losses; mask illegal logits with a large finite fp32 value and reject all-illegal policy rows.
- Preserve the working implementation; prepare a small reviewable PR. Do not start training merely because CPU smoke tests passed.

**Gate:** architecture reviewed, code available on training host, reproducible tests/report.

### Step 2 — Stage 2B: LIMITED Blackwell model-readiness smoke (next actual experiment)

**Goal:** Verify the newly built networks can take gradient updates on the RTX PRO 6000 Blackwell and establish a *safe* initial minibatch size.

- Inspect GPU, JAX/CUDA, host RAM and available device VRAM.
- Initialize both exact models (~25M and ~100M); run a few small forward and backward passes in fp32 and bf16 compute modes.
- On synthetic and/or small verified real loader batches, check masked policy CE and value-weighted WDL CE are finite, gradients finite, and one AdamW update succeeds.
- Try **only a few safe batch sizes**, increasing gradually. Record compile time, synchronized forward/backward timing and actual peak GPU memory. Do not assume a 1K–4K batch is feasible without measurements.
- For real data use the **already validated 32,768 host-RAM shuffle buffer**. No separate 32K/65K/200K loader bakeoff. The buffer is **CPU RAM**, not Blackwell VRAM.
- Write `reports/m2_stage2b.md` with exact environment, commands, outcomes, memory, recommended initial batch, and PASS / CONDITIONAL PASS / FAIL.

**Boundary:** Not full supervised training, no 1K-position overfit yet, no mctx search sweep, self-play, history-dropout implementation or dataset expansion.

**Gate:** both model configurations reliably execute and update on GPU with finite values and an identified safe operating configuration. Fix actual blockers before continuing.

### Step 3 — Stage 3A: 1,000-position overfit test

**Goal:** Confirm the network can genuinely **learn**, rather than merely produce gradients.

- Use a frozen, reproducible sample of approximately 1,000 eligible training positions; never use held-out validation/test to optimize.
- Run short repeated updates (begin with the 25M model, test 100M if practical). Track policy NLL, top-1 move match, WDL loss and finite gradients; ensure legal action probability normalization.
- Expect meaningful training-loss reduction. Diagnose shape/mapping/masking/optimizer errors if losses do not decrease. Distinguish genuine failure from intrinsically ambiguous chess labels.
- Keep the experiment bounded. No full-dataset pilot until the overfit gate is understood.

**Gate:** model demonstrates a reproducible learning signal with the intended losses.

### Step 4 — Stage 3B: supervised baseline pilot

**Goal:** Compare baseline human-game learning in the ~25M and ~100M models; **history dropout stays 0%** for the initial runs.

- AdamW β=(0.9,0.95), weight decay 0.01–0.05, global grad clip 1.0.
- Initial LR guidance: ~6e-4 (25M) and ~3e-4 (100M); 1–2K warmup steps, cosine decay.
- Short **three-value LR sweep on 25M** before the 100M run; document deviations.
- Batch size target 1K–4K **only if Stage 2B establishes feasibility**; use smaller safe batches or accumulation as appropriate.
- Loss: **masked policy CE + 1.0 × (`value_weight`-scaled WDL CE)**. Report both component magnitudes. Change value coefficient only in documented ablation.
- Up to 20 epochs per pilot, with early stopping after two epochs without validation policy-NLL improvement; track checkpoints, seeds, GPU-hours and reproducibility. Approximate Stage 3 planning budget: one week, not a guaranteed wall-clock runtime.
- Preserve natural repeated-position distribution; report metrics for **ply < 10** separately. Any per-board cap is a distinct ablation.
- Keep test split frozen; do not tune hyperparameters or select checkpoints using test results.

**Gate:** documented learning curves, validation behavior and 25M/100M comparison; no unapproved final-B0 freeze.

### Step 5 — Stage 1D: history-dropout ablation (after baseline is established)

**Goal:** Test whether exposing the model to missing game history helps FEN-only puzzle generalization.

- Implement deterministic, training-only transformation matching the actual FEN-only encoder semantics; original Parquet bytes untouched.
- Start with a **0% baseline versus 10%** experiment **at 25M**, with matched compute and seeds where feasible.
- Report **full-history validation** and **FEN-only puzzle** results separately; do not alter canonical validation/test histories.
- Try 100M dropout only if 25M shows a meaningful improvement on FEN-only evaluation without harming full-history validation.
- A missing-history encoding must not accidentally corrupt current board, legal masks, labels, value weights, clocks or repetition semantics. Validate representation equivalence before training an ablation.

**Gate:** clear ablation result or explicit deferral; negative/neutral findings are acceptable.

### Step 6 — Stage 1E: independent chess-match evaluation preparation

**Goal:** Measure **playing strength**, not only human-move prediction.

- Build `openings_v2` with reproducible source/provenance and balanced openings, using a bounded 2025 Lichess source read (e.g. December 2025) for frequency selection. Preserve the original frozen opening suite.
- Pin Stockfish binary version, configuration, seeds/protocol where applicable, and fixed node budgets **1 / 10 / 100 / 1000** for evaluation-only opponents; use paired colors from identical starting positions.
- Report match score and confidence intervals; clearly distinguish network-only and any search-assisted match settings.
- Prepare evaluation infrastructure alongside the supervised pilot if convenient, but **complete it before making final chess-strength/scaling conclusions**. It must not block Stage 2B or 1K overfit.
- No Stockfish labels in the primary engine-free training path. Dataset D remains evaluation-only.

**Gate:** pinned, repeatable evaluation harness/fixtures and honest uncertainty reporting.

### Step 7 — M2 decision gate

Compare train versus validation losses, 25M versus 100M gains, puzzle and match strength, calibrated WDL, legal-move behavior, GPU-hours, and any history-dropout effects.

- **Potentially data-limited:** early large train/validation gap and no 100M advantage over 25M supports *considering* 1M-game extraction; verify optimization/regularization explanations too.
- **Still improving with scale:** 100M significantly ahead and improving means the 50K pilot may not be saturated; report that rather than reflexively requesting more games.
- **Diagnostics, not automatic verdicts:** unusually low top-1 validation move match (particularly ply ≥10) warrants investigation; any illegal move from legal-masked policy is a correctness bug. Search-assisted play weaker than the same network-only baseline warrants investigation with appropriate uncertainty and compute budgets.
- Decide **separately** whether to expand Dataset A toward 1–5M accepted human games and produce/freeze the real B0 checkpoint. Do not initiate acquisition without authorization.
- PUCT/Gumbel mctx (32/64 simulation) verification must be ready before M3/B1/B2 self-play. Extended search benchmarks may be deferred from the earliest supervised pilots but not omitted from self-play readiness.

## 4. What was intentionally deferred, and why

| Item | Current policy | Revisit when |
|---|---|---|
| Standalone Blackwell loader benchmark | **Dropped as a separate milestone**; existing 32K buffer is validated | An observed data-input bottleneck or memory failure |
| 200K shuffle buffer | Optional high-host-RAM setting, **not required** | Training host has capacity and comparative measurement adds value |
| Lossless compressed cache | Deferred; original Parquet loader adequate | Storage/I/O measurements justify complexity |
| History dropout | Retained as research ablation, not prerequisite for baseline | After 25M baseline and before final ablation conclusions |
| `openings_v2` / Stockfish | Retained as evaluation requirement | Before comparative match/scaling decisions |
| Extensive mctx searches | Deferred from minimal GPU smoke/initial pilot | Before search-assisted evaluation and especially M3 self-play |
| Exact mid-epoch training resume | Stage 3 integration gap, **not Stage 1 PR blocker** | Before costly long-running training if recovery cost is material |

## 5. Execution discipline

1. **Only authorize the next gate in Codex prompts.** This document is context, not permission to execute all steps.
2. Keep bounded PRs and reports. No automatic commit/push/merge without the user's explicit approval.
3. Results must identify actual run, machine, dataset revision, config/seed, code SHA, pass/fail and any untested scope. Do not call a CPU test a Blackwell GPU benchmark.
4. Preserve accepted rules, encoder, policy codec, `human_v3`, Dataset A and Dataset D. Any semantic change requires its own reviewed correctness gate.
5. **Immediate next authorization:** Stage 2B *only*, after Stage 2A code review/preservation. Do **not** restart Stage 1D/E or 200K buffer work first.

## 6. Reference documents

- Canonical research specification: `README.md` (v0.4) and original TinyChess Notion export.
- Detailed implementation plan and hyperparameter addendum: `m2execution.md`.
- Stage 0: `reports/m2_stage0.md`.
- Stage 1: `reports/m2_stage1_abc.md`, `reports/m2_stage1_full_train.json`.
- Stage 2A: `reports/m2_stage2a.md` (reported locally; check whether committed).
- GitHub: https://github.com/IshaGarg1103/gambit-zero

**Bottom line:** Finish the model GPU smoke, prove learning on 1K positions, train the baseline pilots, then run the retained optional ablations and strength evaluations before deciding the final B0 data scale. The original B0 → independent B1/B2 research design remains unchanged.
