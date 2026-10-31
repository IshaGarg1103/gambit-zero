# M2 Stage 2A — chess transformer — 30 October 2026

**PASS for Stage 2A's CPU architecture/correctness scope.** Both parameter-counted
profiles passed real-data inference and a full backward in both precision modes.
No optimizer step, overfit experiment, training, search integration or GPU
benchmark ran. This is not acceptance of all Stage 2 or the complete M2 milestone.
Blackwell execution/performance remains unverified and requires its own gate.

## Accepted base and scope

- Branch: `m2/model-architecture`, created by the owner from
  `e36829f216baaa6de4f943aaad06e7913e9c25d1`.
- GitHub verified PR #5 merged at that SHA. Local main/origin/main matched it;
  a live GitHub comparison to main was identical (0 ahead/behind). The initial
  tracked worktree was clean. No Git command, commit, push, PR or merge was
  performed by the agent; main was not edited.
- Read README v0.4, complete M1/M2 plans, Stage 0 correctness evidence, Stage 1
  loader/storage code and sample/full-training reports. The owner's current
  Stage 2A scope supersedes the earlier roadmap's broader execution permissions.
- Model inputs remain encoder `pgx-v2-history8-ep-context8-v1`; actions remain
  `pgx-2.6.0-chess-v2-uci-v1`; rules remain `tinychess-auto-fide-pgx-v1`.
- Dataset A remains `b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c`. No data downloads,
  source archive reads, dataset regeneration/repacking or publication occurred.

New source: `src/tinychess/model/{__init__,transformer,objectives}.py`.
New checks: `tests/test_transformer.py`, `scripts/smoke_chess_transformer.py`.
Also changed this report, its `.gitignore` allowlist and appended verified status
in `m2execution.md`. Existing source, tests, research text and lock versions are
preserved; no parallel chess rules/codec/encoder were introduced.

## Architecture and actual parameter counts

Flax **Linen**, initialized from scratch, with no dropout/stochastic layers:

1. Input float32 planes `(B,8,8,120)` are reshaped into 64 spatial row-major board
   tokens and linearly embedded. Context `(B,8)` is projected and added to a learned
   context token, prepended to the board sequence. Learned positions cover all
   **65 tokens**. The input state/history features are never rewritten.
2. Each block is pre-LayerNorm attention + residual, followed by pre-LayerNorm
   GELU feed-forward + residual. Multihead attention is fully bidirectional;
   there is no causal/triangular mask. Every square and the context can interact.
3. Final LayerNorm. A shared per-square linear head emits 73 board logits, then
   applies the explicit Pgx permutation below. Context-token features feed a
   width→width GELU MLP and width→3 WDL head.
4. Raw model outputs are fp32 `(B,4672)` policy logits and `(B,3)` WDL logits.
   Policy logits are **unmasked** until passed to the checked masking/loss helper.
   No position import, chess rules call, python-chess, mctx or search is used here.

Immutable `ModelConfig` profiles (defined together in `transformer.PROFILES`):

| Profile | Layers | Width | Heads / dimension | FFN width | Initialized parameters | FP32 parameter bytes |
|---|---:|---:|---|---:|---:|---:|
| ~25M | 8 | 512 | 8 / 64 | 2,048 | **25,622,092** | 102,488,368 |
| ~100M | 14 | 768 | 12 / 64 | 3,072 | **100,031,308** | 400,125,232 |

Counts were summed from **actual initialized parameter arrays** for each mode;
independent `jax.eval_shape(model.init)` trees agreed. Every embedding, position,
LayerNorm scale/bias, projection bias and both heads is included. Group counts:

| Group | ~25M | ~100M |
|---|---:|---:|
| One transformer block | 3,152,384 | 7,087,872 |
| All transformer blocks | 25,219,072 | 99,230,208 |
| Board embedding | 61,952 | 92,928 |
| Context embedding + learned context token | 5,120 | 7,680 |
| Learned positions | 33,280 | 49,920 |
| Final LayerNorm | 1,024 | 1,536 |
| Policy head | 37,449 | 56,137 |
| Value hidden + WDL projection | 264,195 | 592,899 |

`ModelConfig.metadata()` records model version `tinychess-transformer-v1`, dimensions,
compute/master dtypes, input shapes, 65-token convention, policy ordering and WDL
ordering. Initialization takes an explicit JAX key. Same key/config/input produced
identical parameters and outputs in tests; smoke seed was 20261030. Within-host
reproducibility is tested; cross-device/compiler bytewise equality is not claimed.
In-memory Flax serialization round-trips preserve params exactly. No persistent
training checkpoint format, optimizer state or mid-epoch resume is implemented.

## Exact source-square policy contract

Board tokens retain ordinary spatial order *inside* the trunk; the head explicitly
transforms its `(B,row,col,73)` grid:

```python
source_index = col * 8 + (7 - row)
action_index = source_index * 73 + action_plane
# Implementation: reverse rows, transpose col before row, flatten source×plane.
```

All **4,672 indices** were checked against independent row/column/plane loops,
including bijectivity. Further composition tests use canonical encoded Pgx states
and UCI decoding for both colors, EP and promotions. The unchanged Stage 0 suite
also passed in the full regression: 122 roots / 3,384 successor and policy-source
checks. Its own-piece contract is exactly
`planes[7 - source % 8, source // 8, 0:6]`; no encoder/codec permutation changed.

## Precision and numeric safeguards

`compute='float32'` is the reference; `'bfloat16'` selects bf16 projection and
attention operands. **Every master parameter remains float32**, explicitly set
on Dense, LayerNorm, positions and the context token. All tested gradients remain
float32 too. No parameter tree is cast to bf16.

- Backbone projections use the selected operand dtype. Attention QK and weighted-V
  einsums explicitly request fp32 accumulation. Attention softmax is fp32; its
  weights are converted to the selected dtype for the weighted-V multiplication.
- LayerNorm uses fp32 inputs/parameters/reductions, `use_fast_variance=False` and
  `force_float32_reductions=True`. Residual streams and GELU are fp32.
- Both heads project in fp32. Policy masking, softmax/log-softmax, WDL softmax,
  cross-entropy and weighted reductions are fp32.
- Tests inspect the actual JAXPR: bf16 dot operands exist in bf16 mode; softmax
  exponentials and norm/softmax reductions are fp32. The same fp32 parameters
  also gave close fp32/bf16 outputs on the reduced model (rtol/atol 0.05).

Installed APIs were inspected before implementation: Flax 0.12.0 Dense/LayerNorm
signatures and LayerNorm source, JAX 0.7.2 einsum preferred accumulation type.
CPU/XLA bf16 computations passed; native Blackwell kernel behavior, speed and
memory have not been tested. CPU support is not a GPU-performance claim.

## Legal masking, losses and the loader boundary

`masked_policy(logits, mask)` accepts `(B,4672)`, casts logits to fp32, centers by
**the legal maximum**, then substitutes **finite `-1e9`** for illegal entries.
Centering prevents illegal sentinels dominating legitimate logits below -1e9.
It explicitly zeros illegal probabilities and normalizes the legal mass.
Off-board actions and every other false mask entry receive **exact zero**.

All-illegal inference rows return an **all-zero policy** and
`has_legal_actions=False`, with finite masked logits. Consumers must honor that
flag/use the returned probabilities; independently softmaxing the terminal
sentinel vector would be incorrect. WDL predictions remain finite for actual
encoded checkmate/stalemate states; scalar search value is always
`P(win) - P(loss)` in side-to-move W/D/L order `[win,draw,loss]`.

`validate_batch` is a host-only boundary for a Stage 1 `TrainingBatch` or mapping.
It checks exact shapes/dtypes, finite features/weights, eligibility, nonempty
legal sets and legal action/WDL targets, rejecting invalid inputs before JIT.
It does not belong in future MCTS/model hot paths and imports no chess parser.

`policy_cross_entropy` computes **only from masked fp32 logits**; illegal actions
have zero logit gradients. Empty legal sets and invalid targets return an invalid
flag + NaN loss in compiled code, providing a fail-stop diagnostic if host checks
are bypassed. `supervised_loss` adds policy and WDL weighted means, with equal
unit coefficients for this smoke primitive. It preserves stored weights: timeout
rows retain policy targets but zero value weight. A zero total value weight yields
zero value loss and zero WDL-logit gradients. Fractional weights use their actual
positive denominator. Shape guards prevent unintended weight broadcasting.
Invalid targets/weights/eligibility or nonfinite losses yield invalid + NaN total.

These are loss/gradient primitives, not a trainer: no LR, optimizer, scheduler,
step/update loop, fitting, checkpoint selection or model-quality evaluation.

## Actual CPU smoke evidence

Host: macOS 26.6.1 arm64, 10 logical CPUs, 16 GiB total RAM. Python 3.11.15;
JAX/jaxlib 0.7.2, Flax 0.12.0, existing pinned environment; CPU device only.
Each profile/precision ran in a **fresh process**, sequentially. RAM preflight
uses parameter bytes plus conservative compiler/temporary allowances and a
1 GiB reserve, with a further check before optional full backward. Unsafe full
backward is explicitly skipped with a reason; **no skip occurred here**.

Inputs were the first two pre-move positions of real pinned `human_v3` game
`vlLZPjB0`, plies 0/1, from the existing verified local training shard. Store SHA256,
metadata, action legality and loader checks ran; **zero bytes downloaded** in all
four smokes. No synthetic examples were mixed with real records or published.
Separate reduced synthetic/encoded-state tests cover shape and boundary cases.

| Profile / compute | Init incl. compile (s) | First forward incl. compile (s) | Full backward incl. compile (s) | Peak process RSS (bytes) |
|---|---:|---:|---:|---:|
| ~25M / fp32 | 0.989 | 0.226 | 1.417 | 964,132,864 |
| ~25M / bf16 | 0.964 | 0.267 | 1.635 | 944,046,080 |
| ~100M / fp32 | 1.613 | 0.381 | 3.541 | 2,047,623,168 |
| ~100M / bf16 | 1.511 | 0.474 | 3.748 | 1,819,688,960 |

All four materialized full parameter trees had the counted sizes and fp32 dtype.
All four B=2 outputs had correct shapes, finite fp32 values, normalized legal
policies and exact zero illegal probabilities. Full and reduced (26,988 params,
2×32, FFN 64, 4 heads) forward/backward checks were finite in both modes. Full
smoke losses were 4.243156 / 4.237737 (~25M fp32/bf16) and 5.185051 / 5.205776
(~100M). These random-initialization loss values are diagnostics, not strength.
**Zero parameter updates**. First-call timings include compilation; no warmed
throughput, separately isolated compiler timing or GPU memory was measured.

## Reproduction and test outcomes

```bash
uv sync --locked --extra test --extra data
JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_transformer.py -x -q
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -x -q

for profile in 25m 100m; do
  for compute in float32 bfloat16; do
    JAX_PLATFORMS=cpu .venv/bin/python scripts/smoke_chess_transformer.py \
      --profile "$profile" --compute "$compute" --full-gradient \
      --data-root data/processed/m2_training_originals \
      --output "data/processed/m2_stage2a/${profile}_${compute}.json"
  done
done
```

If verified local shards are unavailable, omit `--data-root` for explicitly
synthetic smoke inputs; that does not reproduce the real-data claim. The runner
never downloads data. With `--full-gradient`, it attempts one full backward only
when both resource checks pass, otherwise records a skip and performs the reduced
check. No optimizer update happens in either case. Existing Stage 0 setup caveats
for network/Python 3.11.15 installation remain applicable; locks were not changed.

Actual focused result: **32 passed in 12.26s**.
Actual full CPU regression: **348 passed in 65.13s**, including unchanged M0/M1,
loader and Stage 0 checks. Tests cover counts/dtypes, deterministic init/forward,
all policy indices, encoder/codec composition, masking/off-board actions,
normalization and CE reference, invalid masks/targets/weights/shapes, WDL signs,
zero-value-weight gradients, fp32/bf16 JAXPR semantics, finite forward/full reduced
backward, non-causal/context influence and in-memory save/restore.
Four additional full-profile real-data smokes passed as recorded above.
Raw local logs/JSON are ignored under `data/processed/m2_stage2a/`; no model weights
were written to disk. This report is the retained final Stage 2A evidence.

## Remaining limits and next gate

No found correctness blocker in the tested CPU scope. Full-profile backward was
limited to one B=2 batch per configuration/precision; no overfit, optimizer-state,
training checkpoint/resume or extended numerical stability experiment occurred.
Blackwell mixed-precision execution, host/device transfer and feasible large-batch
inference/backward memory/performance remain **unverified**, not inferred from M1's
environment-only benchmark. Search integration is also untested for this model.

Stage 1D history dropout, Stage 1E `openings_v2` and separate Blackwell loader
profiling are **deferred, not cancelled**. Baseline input history stays intact.
B0 → independent B1/B2 research order and engine-free labels are preserved.
Remaining Stage 2 GPU/search gates and all Stage 3 training require further owner
authorization. **Stop after Stage 2A and review before any next experiment.**
