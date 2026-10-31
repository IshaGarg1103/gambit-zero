# Gambit Zero — M2 Execution Plan (pilot, not final B0)

**Status:** Proposed execution specification; no M2 training is claimed.  
**Date:** 2026-10-29  
**Canonical repo:** https://github.com/IshaGarg1103/gambit-zero  
**Source hierarchy:** Original TinyChess implementation plan (user-provided DOCX exported from Notion), `README.md` (research plan v0.4), `m1execution.md`, frozen M1 reports, Claude's independent review, and this M2 execution specification.  
**Original Notion source:** https://app.notion.com/p/TinyChess-implementation-403b3fee384880ffa4f3fdb500b5cc60 — reviewed via the user-provided DOCX export on 29 October 2026. Its diagrams and research design are preserved in Section 0.1; decisions that differ are labeled as M2-stage refinements.

## 0. Scope and objective

M2 is a **pilot** to validate a reproducible JAX/Flax policy–value training pipeline on the existing 50K human-game Dataset A, compare measured ~25M and ~100M models, and make an evidence-based decision about scaling human data for the real B0 checkpoint. It is **not** final B0, M3 self-play, a 1M-game extraction, or a claim of frontier-LLM chess strength.

The original research objective remains a 100M main model trained through the B0 → B1 → B2 sequence; report network-only and search-assisted strength separately. No engine-derived training labels in the primary track. Pgx + mctx remain the intended on-device environment/search stack; python-chess is an offline correctness oracle.

### 0.1 Original research plan — scope and explicit M2-stage refinement

The original Notion implementation plan defines **TinyChess-100M**, a **bidirectional** transformer trained **from scratch** on board features, not a fine-tuned pretrained LLM. Learning is **human-game supervised policy+WDL** followed by **AlphaZero-style self-play RL**; **GRPO is not used**. It proposes a 25–50M scale reference, 100M primary model, and only conditionally a 600M scale experiment. The intended chess stack is JAX/Flax/Optax + Pgx + mctx. The network emits a 73-plane move policy and 3-class WDL; its search scalar is `P(win) - P(loss)` from the appropriate perspective.

Original experiment order: **B0** human-only supervised checkpoint, then independent **B1 PUCT-32** and **B2 Gumbel-32** self-play branches. **B1 and B2 each initialize from exactly the same frozen B0**, never from each other. Their replay buffers and checkpoints remain separate for a fair comparison. Later B3/B4 address value-target and human/self-play mixing; optional B5–B7 and scaling arms follow only on evidence. Dataset C with Stockfish teacher labels is an explicitly separate control, not part of the engine-free primary path. Dataset D is evaluation-only. Ultimately compare network-only vs search-assisted strength and use paired, pinned GPT-match protocol—not puzzle accuracy—as evidence for beating a frontier model.

**Important reconciliation:** The original plan envisioned a frozen human-only **B0 after the 50K pilot** if the pipeline was satisfactory. Claude's later independent review recommends treating 50K as an **engineering/training pilot** and *considering* a 1M+ game extraction before the final B0. This M2 execution plan follows the owner's later requested **pilot-first decision**, **not** an experimentally proven 1M requirement. Stage 3 may produce well-evaluated pilot checkpoints, but it must **not call them the final frozen B0**. The final B0 dataset scale and freeze require a separate explicit decision after the pilot. This is a change in milestone timing, not the research hypothesis or model family.

Candidate original architecture mentioned roughly **14 layers, width 768, 12 heads and FFN 3072**, but the plan expressly says this is *not guaranteed* to equal 100M parameters; count and tune actual models. M2 pilot must use a measured ~100M configuration, with measured ~25M comparator; do not lock the 14×768 proposal in without a parameter count and feasibility check.

## 1. Current accepted foundation and immutable references

- M0: corrected Pgx chess adapter, JAX state representation, 4,672-action codec, 8×8×120 planes + 8 context features, MCTS integration and chess-rule checks.
- M1 source PR (merged): https://github.com/IshaGarg1103/gambit-zero/pull/2
- M1 source tag: https://github.com/IshaGarg1103/gambit-zero/tree/m1-dataset-50k-v1 ; peeled commit `9893c2edcdb99dd57a58da8cd688b4157753bf93`.
- M1 data: 50,000 rated Lichess games (35K train Jan–Jul 2026, 7.5K validation Aug, 7.5K test Sep); 3,793,600 retained position rows; 150 overlapping training rows excluded from Dataset D; approximately 2.65M *training* rows (verify exact eligible count from frozen artifacts).
- Dataset A pinned revision: `b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c` at https://huggingface.co/datasets/ishagarg1103/gambit-zero . Never train from superseded `d409fce8bd444c2d456ef4c343dbc9b5e25981eb`.
- M1 acceptance: https://github.com/IshaGarg1103/gambit-zero/blob/main/reports/m1_50k_pilot.md ; historical CPU acceptance frozen.
- Archived original PR evidence commit: `5edc8bec33e5a8b398f5a4af5b501e846998a93e`; see `reports/m1_evidence_archive.json` and `reports/pr2_cleanup.md`.
- Blackwell follow-up PR: https://github.com/IshaGarg1103/gambit-zero/pull/3 ; PR open when this document was drafted; **verify status before acting**. Reported 264 CUDA tests + 267 final local tests, benchmark PASS, four independent initial-position batch sizes measured (128, 1024, 4096, 8192), ~203,792 / 693,665 / 690,540 / 748,292 environment steps/s. This is not full-game, inference, training, MCTS, or self-play throughput.
- Blackwell report: https://github.com/IshaGarg1103/gambit-zero/blob/m1/blackwell-benchmark/reports/m1_blackwell_benchmark.md
- The independent reviewer reports no found chess-logic defect in its targeted action/FEN, encoder/history/EP, WDL/sign, and mctx checks, but independent runtime reproduction and hardware scope must be distinguished from assertions.

## 2. Invariants: non-negotiable

1. Do **not** modify M0 chess rules, termination/adjudication, action codec, board orientation, encoder contract, `human_v3` semantics, original Dataset A/D shards, or published dataset revision. Any necessary semantic change requires a new version ID, separate review, regeneration and parity validation; stop and ask for approval rather than silently changing existing semantics.
2. Derived training caches must be stored outside the canonical dataset and must have explicit source revision, format version, build config, checksums and decode tests.
3. `float32` input feature values and context are the **semantic truth**. A compact representation is allowed only if each original record can be reconstructed **bit-for-bit** in the required endian order. No reduced-precision storage for nonbinary values; do not assume all 120 planes are binary.
4. Treat history dropout as an **optional, separately tracked ablation**, not a default. Baseline uses **0%** dropout. Do not destructively edit original features. Keep legal mask/targets and history-dependent cases consistent; verify reconstructed FEN-only feature behavior. Never generate self-play game states from FEN-only resets.
5. Use existing locked dependencies, exact dataset revision, deterministic seeds, and train/validation/test separation. Prevent evaluation leakage. Do not introduce Stockfish training targets into the primary track; Stockfish may only evaluate opening balance under a pinned version/configuration.
6. No new 263GB Lichess source scan; no 1M-game extraction; no M3 self-play; no final B0 freeze; no automatic merge/push/data publication. One short per-stage section in a single `reports/m2_pilot.md`; keep execution lean and report only real results.

## 3. Stage 0 — close M1 follow-up and strengthen correctness

**Order:**

1. Confirm PR #3 status, review relevant changed files/test results, merge it into `main` with a regular merge commit **only with owner authorization**; otherwise provide the exact human-run `gh pr merge 3 --merge --match-head-commit <verified_sha>` command. Ensure `main` is synchronized. Optionally create/push a milestone tag for the accepted Blackwell follow-up after merge.
2. Document that `uv sync --locked` may download Python 3.11.15 and therefore requires network; document supported system Python 3.11 fallback with `requirements-lock.txt` and exact supported package/runtime constraints. Don't weaken pins.
3. Add a deterministic independent property test: sample random valid game states, enumerate every legal Pgx action, decode to UCI, push on python-chess, compare complete resulting FEN/necessary state fields. Cover promotions, both colors, castling, en passant and repetition-relevant positions. Report number of positions/actions actually tested.
4. Add a policy-index ↔ encoder mapping test: for each legal action `a`, `s = a // 73`; verify `planes[7 - (s % 8), s // 8, 0:6]` has the side-to-move's own piece according to the accepted board-plane convention. If the convention differs, do not change the encoder blindly; escalate with a concrete counterexample.
5. Run pinned full CPU regression and CUDA smoke/regression where available. Stage 0 must PASS before Stage 1.

**Exit:** M1 follow-up merged or explicitly blocked; tests passing; no semantic/source dataset drift; document exact HEAD and pending deviations.

## 4. Stage 1 — lossless loader, history handling, evaluation quality

### 4.1 Data loading and optional lossless training cache

- Inspect real `human_v3` field values and shapes first: planes are 30,720 bytes per position (`8×8×120` little-endian float32), context 32 bytes, legal mask already packed into 584 bytes. Inventory channel-by-channel unique values/ranges, including unusual states. Confirm any binary channels are *exactly* 0.0/1.0 in float32; handle +0/-0 and unusual bit patterns by byte-exact validation.
- **Baseline path:** implement a working, correct streaming/shuffled minibatch loader directly from the immutable published Parquet revision **without repacking**. This is the fallback and correctness oracle.
- **Optional optimization:** build a separate, versioned, training-only cache from pinned Dataset A, containing only eligible train rows, larger shuffled row groups, compact binary channels **only after proof**, separately lossless scalars/nonbinary features and original weight/label/mask metadata. Preserve full training semantics and per-row identities. Do not claim that compression itself changes float32 model input or bf16 compute.
- Validate a representative deterministic sample and structured edge cases, then perform a **complete rowwise checksum/equality audit** if feasible before enabling the cache as a training default. Decoding must reconstruct original `planes` and `context` **byte-identically** and legal mask/targets/weights/IDs identically. If any mismatch: fail-stop and use the original Parquet loader. Do not merely compare approximate tensors.
- Benchmark both paths (data batches/s, CPU RAM, disk/network reads and decoder overhead) before selecting an optimization. Retain canonical dataset provenance and training-visible row count.

### 4.2 FEN-only puzzle mismatch

- Characterize the actual `from_fen` versus replay encoder discrepancy (reported 84 history channels plus ply plane; verify from current code). Freeze original puzzle set and do not silently change its results.
- Implement a *configurable* no-history augmentation matching actual FEN initialization exactly, restricted to training. **Pilot baseline = 0%**. Compare **0% vs 10%** on identical seeds/data/order where possible; optionally 5% only if results justify it. Report full-history validation and FEN-only puzzle metrics separately. Ensure truly draw-relevant history-dependent state remains meaningful; do not claim augmentation fixes the underlying information gap.
- Self-play and match games must be initialized by replay from initial state, not arbitrary FEN resets.

### 4.3 Opening evaluation suite and data diagnostics

- Preserve original 32 opening lines as frozen historical suite. Create a separate `openings_v2` suite of about 100–250 common, diverse, balanced positions from a month **outside Dataset A**, with exact provenance and no training-label leakage. Pinned Stockfish evaluation for selection only (e.g., abs eval <= ~0.5 pawns at documented depth/time); no engine labels in training.
- Pair both colors and fix match protocol/seeds. Report sample uncertainty; neither 64 games nor the new suite alone establishes reliable Elo.
- **Report only (do not edit Dataset A):** time-control distribution, number of decisive games under 10 plies, and repetition-count-3 incidence among `draw_nonboard_unspecified` games.

**Exit:** canonical loader correct, optional cache accepted only after lossless checks, opening-suite provenance frozen, diagnostics measured, augmentation off by default.

## 5. Stage 2 — model implementation and GPU throughput, no training

**Dependency order matters:** implement architecture and correctness **before** benchmarking a model that does not yet exist.

1. Implement shared 64-square-token transformer with a context token, 120-channel linear input embedding, learned positional embeddings, pre-LN blocks, 73 action logits per source square, and three WDL logits. Match accepted source-square index ordering; legal-mask outside-network logits appropriately. Parameter counts for ~25M and ~100M must be **actually measured**, not inferred from labels. Define init/dtype/RNG/key handling and checkpoint metadata.
2. Test shape/index permutation, mask behavior, no illegal logits after masking, WDL order/perspective, finite forward/backward and save/restore with synthetic samples. Frozen schema is not changed.
3. Bench Flax bf16 compute with fp32 master params on the Blackwell at batch 256, 1024 and 4096 as memory allows. Include throughput, compilation, synchronized latency, measured memory, skips and OOM handling; inference microbenchmark ≠ full-training throughput.
4. Integrate random-initialized policy/value model with **one mctx search call**, batch 1024 and 4096, 32 and 64 simulations, PUCT and Gumbel, only as memory permits. Report execution and peak memory, correctness (legal actions and value perspective), and constraints of ~46KB/node Pgx state / cached observation (verify actual measured numbers). Do not redesign state in this stage. Confirm mctx interface semantics and parameter-sign correctness.
5. Stop if memory or integration is unsafe; record blocker rather than inventing a pass.

**Exit:** validated model shape and policy ordering, reproducible benchmark with GPU fingerprint, decision about feasible pilot batch sizes.

## 6. Stage 3 — supervised pilot, not final B0

1. **1K-position overfit smoke:** verify label and value-loss behavior, finite gradients, optimizer updates, checkpoint save/resume, and reproducibility. Make failing tiny-set fit a blocking diagnostic.
2. **Pilot train** measured ~25M baseline and measured ~100M candidate using only `training_eligible` train positions (~2.65M; read exact count), from pinned source/verified derived cache. Avoid uncontrolled repeated identical positions per epoch. Compare both at documented token/position/step and wall-time budgets. Training duration and GPUs based on real Stage 2 data. No blind full-run expense before smoke. Mixed-precision bf16 math, fp32 params/optimizer state as supported.
3. **Policy loss:** cross-entropy of human action over legal-masked 4,672 actions. **Value loss:** three-class WDL cross-entropy in accepted perspective, multiplied by `value_weight` (0 for forfeits). Explain policy/value balancing and masking before training. Keep data splits frozen.
4. **Metrics:** train/val policy NLL; top-1/top-3 on full validation plus training-position-disjoint diagnostics; breakdown by game phase/rating; value-eligible Brier/ECE and WDL class quality; finite-weights/illegal-policy checks (legal move rate 100% under legal masking); checkpoint trends and train-val gaps rather than predicting overfit; puzzle accuracy with explicit FEN-history caveat; paired-color match games vs random baseline and earlier checkpoints on `openings_v2` with uncertainty. Track inference vs search-assisted results separately.
5. **History ablation:** 0% baseline first, 10% history-dropout variant as a separate experiment if compute permits; same architecture/seed controls. Report if not run; do not quietly enable 10%.
6. **Pilot final decision:** assess whether data loader, model, optimization and evaluation are reliable; determine what evidence justifies 1M-game expansion and real B0; list uncertainties and specific next experiments. No 1M extraction or M3.

**Stop after Stage 3.**

## 7. Execution and Git/PR discipline

- Start by checking GitHub `main` and PR #3, not assuming previously reported status is still current.
- Merge PR #3 only with human approval; capture resulting merge SHA and optionally tag Blackwell closeout.
- Create `m2/model-training` **from updated `main`**, with clean working tree. Prefer one branch with separately reviewable PRs scoped to preparation, model/benchmark, then pilot. Do not rewrite/rebase accepted M0/M1 history or force-push.
- At each stage run tests and update **one** report: `reports/m2_pilot.md` with dated actual evidence, commands, pass/fail counts, source commit, data revision, measured checksums, environment versions, quantitative results, caveats, and blockers. No placeholder PASS or fabricated tables.
- Codex should not commit/push/merge or modify Hugging Face source dataset without explicit owner permission. Provide commands for the owner when the stage is reviewable.
- Stage 3 is explicitly permitted to train small pilot models after prior gates pass; stop there. Final B0 checkpoints, M3 self-play, 1M games and >100M scaling remain separate approvals.

## 8. Deliverables / acceptance gates

- `m2execution.md` (this plan, committed into repo after review).
- `reports/m2_pilot.md` with Stage 0–3 sections, evidence and actual outcomes (start with status NOT STARTED and update incrementally).
- Added Stage 0 correctness tests.
- A validated canonical-source loader and optional *proven lossless* training cache with format version and verification; may defer optimization if it adds risk or yields no benefit.
- Frozen `openings_v2` and measured data diagnostics.
- Tested model definitions/checkpoint metadata for measured 25M and 100M variants.
- GPU forward/mctx benchmarks with actual memory numbers and scope cautions.
- Actual 1K overfit and pilot training/evaluation results, subject to successful previous gates.
- **Final recommendation: GO / CONDITIONAL GO / NO-GO** for the larger Dataset A extraction and real B0 run, with costs, evidence and unresolved risks.

## 9. Critical references

- Research plan v0.4: https://github.com/IshaGarg1103/gambit-zero/blob/main/README.md
- M1 execution: https://github.com/IshaGarg1103/gambit-zero/blob/main/m1execution.md
- M1 PR #2: https://github.com/IshaGarg1103/gambit-zero/pull/2
- M1 dataset: https://huggingface.co/datasets/ishagarg1103/gambit-zero/tree/b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c
- Blackwell PR #3: https://github.com/IshaGarg1103/gambit-zero/pull/3
- Blackwell results (PR branch until merged): https://github.com/IshaGarg1103/gambit-zero/blob/m1/blackwell-benchmark/reports/m0_large_batch_blackwell.json
- Original Notion plan (reviewed via user-provided DOCX export): https://app.notion.com/p/TinyChess-implementation-403b3fee384880ffa4f3fdb500b5cc60


## Verified Stage 0 completion — 29 October 2026

**Stage 0: PASS (tested CPU scope).** See [reports/m2_stage0.md](reports/m2_stage0.md).
PR #2 and PR #3 are already merged; no merge was attempted. Reused branch
`m2/stage-0` from verified main `60e32a93926a2ea9698b8c636e0e5cc718035282`.
The two independent checks passed: 122 roots / 3,384 legal successor transitions
and policy-source mappings, zero mismatches. Full CPU regression: 269 passed;
existing-Python CPU fallback verified. Rules, codec, encoder, schema, locks and
Dataset A/D remain unchanged. Dataset revision stays
`b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c`. No new GPU experiment was run.
**Stages 1–3 remain NOT STARTED; await the owner's Stage 0 review.**

## Verified Stage 1 Parts A–C — 30 October 2026

**Parts A/B: PASS in the tested sample scope; Part C: explicitly deferred.**
See [reports/m2_stage1_abc.md](reports/m2_stage1_abc.md) for source/base identity,
exact loader API, bounded cross-game shuffle, fail-stop validation and measurements.
After the owner merged PR #4 and rebased, `m2/stage-1` starts from main
`7dc8e1540f81ae62e9c1fcab42cb64b3758a84bd`.
All 39,766 sampled real positions decoded identically; 36 focused tests and
305 full CPU tests passed. Training batches mixed at least 144 games; maximum
10 positions from one game per 256-position batch. Warm original-Parquet loading
measured 32,975–34,696 positions/s; no derived cache was needed or built.
Full-data manifest counts are distinguished from the newly measured sample;
a new complete 3.79M-row loader epoch was not run. Dataset A revision, chess rules,
encoder, codec, schemas and dependency pins remain unchanged.
**Stage 1 Parts D/E and Stages 2/3 remain pending and were not implemented.**

## Verified Stage 2A — chess transformer implementation — 30 October 2026

**Stage 2A: PASS for CPU architecture/correctness smoke scope.** See
[reports/m2_stage2a.md](reports/m2_stage2a.md). PR #5 is merged; the owner created
`m2/model-architecture` from verified main `e36829f216baaa6de4f943aaad06e7913e9c25d1`.
Implemented a bidirectional pre-LN Flax transformer with the exact spatial-to-Pgx
policy permutation, checked legal masking and policy/WDL loss primitives.
Initialized parameter counts: **25,622,092** (8×512) and **100,031,308** (14×768).
Both profiles passed real-data B=2 inference and one full backward in fp32 and
bf16-compute modes; every parameter/gradient stayed fp32. Zero optimizer updates.
New focused suite: 32 passed; full CPU suite: 348 passed. Rules, datasets, encoder,
codec, schema and dependency locks remain unchanged. No training or self-play.

**Deferred, not cancelled:** Stage 1D history dropout, Stage 1E `openings_v2`,
separate Blackwell loader profiling, and remaining Stage 2 GPU inference/search
integration. No Blackwell model result is claimed. Stage 3 overfit/pilot training
has not started. The original B0 → independent B1/B2 research plan is unchanged;
this completes Stage 2A only, not all of Stage 2 or M2. Stop for review.


## Verified Stage 2B — Blackwell model validation — 31 October 2026

**Stage 2B: PASS within the bounded smoke scope.** See
[`reports/m2_stage2b.md`](reports/m2_stage2b.md). Both accepted models
(25,622,092 / 100,031,308 parameters) initialized and passed finite forward,
backward and exactly one AdamW update per fresh case: fp32/bf16 at batch 4,
and bf16 at batch 32. All six cases passed; 86 CUDA correctness tests passed.
The CPU regression passed 373 tests. Measured bf16 batch-32 JAX peaks were
0.892 / 2.702 GiB. Batch 32 is a conservative starting microbatch for Stage 3A,
subject to memory checks in the actual loop. Dataset A revision, rules, codec,
encoder and model are unchanged. Stage 3A was not executed. History dropout and
openings_v2 remain deferred under the revised flow; original research objectives
and B0 → B1/B2 sequencing remain unchanged. Stop for owner review.

## Stage 3A — bounded 1K-position diagnostic — 31 October 2026

**Stage 3A: FAIL; Stage 3B remains blocked.** See
[reports/m2_stage3a.md](reports/m2_stage3a.md). The owner merged Stage 2A/2B;
`m2/stage-3a-overfit` starts at `68e8381bd3457cd2dcbbb0a76588fc07df8192e7`.
Implemented a minimal JIT 25M trainer, fixed-seed real training sample, environment-only
W&B integration, and atomic checksummed checkpoint save/restore. The sample has
1,024 positions from 1,024 games, with original histories and 172 zero-value-weight
rows. Two sample builds reproduced identical bytes. Full CPU regression: 385 passed;
12 focused tests also passed in the isolated Molab environment.

The single bf16/microbatch-32 run used 100-step warmup to LR 1e-3, no weight decay,
clip 1.0 and 0% history dropout. Initial policy/WDL losses matched uniform baselines.
At exactly 3,000 updates, policy accuracy was 25.3906% / CE 2.68855; weighted WDL
accuracy 59.2723% / CE 0.80280. These do not satisfy the overfit gate. Finite checks
and six checkpoint roundtrips passed, including continued training after restore.
Stop for diagnosis/review; no LR fallback, sweep, pilot, history dropout, openings_v2,
MCTS or self-play was started. Original model/rules/encoder/codec/loader/data/core
locks are unchanged. The B0 → independent B1/B2 research objectives are preserved.

## Stage 3A — single full-batch confirmation — 31 October 2026

**Stage 3A: PASS under the owner's policy confirmation gate.** See
[reports/m2_stage3a_fullbatch_confirmation.md](reports/m2_stage3a_fullbatch_confirmation.md).
One run, unchanged frozen 1,024-position sample and 25M model, LR 3e-4, warmup
100 updates, clip 1.0, no weight decay or history dropout. Correct globally
normalized accumulation of 32 microbatches produced one full-batch update.
The policy gate was first met at 281 updates: 99.0234% accuracy / CE 0.041589;
weighted WDL accuracy 97.4178% / CE 0.050313. Clip-active fraction 270/281 (96.09%).
Nine of ten residual policy misses occur at ply 0–10; both shared-board-hash
rows are correct. All missed identities and denominator-aware breakdowns are
retained in the report. D had no saved checkpoint, so its per-row analysis is
unavailable and was not reconstructed by another training run.
This clears the bounded engineering memorization gate; previous failed evidence
remains intact. Batch noise and presentation budget changed jointly, not as
individually isolated causes. Full CPU regression: 394 passed. Stop for review;
Stage 3B, model changes, sweeps and later experiments remain unstarted.
