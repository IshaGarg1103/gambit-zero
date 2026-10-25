# Official Pgx AlphaZero example audit

The official [Pgx AlphaZero example](https://github.com/sotetsuk/pgx/tree/87278d2d6e677fd87248c457207b59cfa42e578d/examples/alphazero)
was inspected and executed locally before implementing any TinyChess self-play
trainer. Source is pinned to Pgx v2.6.0 commit
`87278d2d6e677fd87248c457207b59cfa42e578d`. Exact original files, README,
requirements and SHA256 manifest are in `benchmarks/official_pgx/upstream/`.
Pgx's Apache license is retained at `licenses/PGX-LICENSE`.

## Actual results

Core: Python 3.11.15, Pgx 2.6.0 chess-v2, mctx 0.0.6, JAX/jaxlib 0.7.2,
Flax 0.12.0, Optax 0.2.6. Additional example dependencies: Haiku 0.0.15,
W&B 0.22.2, OmegaConf 2.3.0, Pydantic 2.12.3. An independent lockfile and
venv under `benchmarks/official_pgx/` preserve the main M0 environment.

1. **Unmodified official chess CLI: FAILS**, exit 1 at
   `pgx.make_baseline_model("chess_v0")`. That baseline does not exist in the
   installed Pgx registry. The JSON outputs capture the actual traceback.
2. **Bounded official routines, native Pgx: PASS.**
3. **Bounded official routines, corrected TinyChess Pgx adapter: PASS.**
4. **Synthetic terminal-return/mask check: PASS**, returns [-1,+1,0,0]
   with value-target mask [true,true,false,false] for the known sequence.
5. **Upstream truncation semantics with wrapped environment: FAIL.** At the
   actual adapter cutoff, `truncated=True` and `terminated=False`, upstream
   recurrence returns discount -1 instead of 0. The harness asserts and records
   this observed limitation; it does not hide it or certify the trainer for use.
   Our separate TinyChess recurrence already cuts bootstrap at either flag.

The harness removes exactly one AST assignment, the unused baseline loader;
never executes upstream `__main__` or evaluation; uses the original network,
recurrence, selfplay, compute_loss_input and train functions. For the wrapped
probe it explicitly substitutes the environment's init/step functions. No
installed package or upstream source file is edited. Source SHA256s are checked
before execution. This is an instrumented reference run, not a claim that the
unmodified end-to-end chess example works.

Each smoke network has **613,042 parameters**, 8 channels, 1 residual block;
batch 2 × 4 plies × 4 Gumbel simulations. Selfplay runs twice (compile/cold,
then warm). There is exactly one tiny optimizer update per backend. No dataset,
checkpoint, training replay, W&B run, or baseline weights are saved/downloaded.
There are **zero completed games** in these short chess traces, so value loss is
0 and there are no valid outcome value targets. The synthetic test verifies
return signs/masking separately; this is not evidence of chess learning.

| CPU reference run | Cold selfplay including compilation | Warm selfplay | Compile + one update | Policy loss |
|---|---:|---:|---:|---:|
| Native Pgx | 2.384 s | 0.00779 s | 0.479 s | 8.23082 |
| Corrected adapter | 3.035 s | 0.00925 s | 0.412 s | 8.23082 |

Actual evidence: `official_pgx_native_cpu.json`,
`official_pgx_wrapped_cpu.json` and corresponding `.log` files. CPU results do
not establish Blackwell compatibility. Pydantic emitted field-metadata warnings;
these did not prevent the bounded routines from running. An initial isolated
environment without Flax failed during Haiku/Pydantic introspection; including
the project's pinned Flax resolved it. Dependencies were not silently upgraded.

## What to reuse, and what must change later

The example uses **Haiku ResNet**, scalar value regression, Gumbel selfplay,
pmap batching, scanned rollout windows, auto-reset, and return masking. TinyChess
still uses the README's **Flax transformer and WDL head**, starting with B0 then
PUCT B1 before Gumbel B2. Running this reference does not reorder experiments.

Keep the demonstrated parent-player reward indexing, negative nonterminal
discount, root invalid-actions mask and recurrent legal logits. Preserve the
full environment/history embedding. Do not copy the upstream native chess draw
rules or its terminated-only truncation checks. Its window-end value mask and
auto-reset logic need validation for explicit environment truncations before
reuse; raw Pgx's 512-step cutoff is incorrectly classed as termination for our
purposes. The simple baseline evaluation is debugging-only and is neither the
frozen evaluation protocol nor a playing-strength result.

## Reproduce locally

```bash
uv sync --project benchmarks/official_pgx --locked --python 3.11.15
benchmarks/official_pgx/.venv/bin/python scripts/run_official_pgx_example.py --backend native --output reports/official_pgx_native_cpu.json
benchmarks/official_pgx/.venv/bin/python scripts/run_official_pgx_example.py --backend wrapped --output reports/official_pgx_wrapped_cpu.json
```

For Molab GPU execution, see `molab_notebook.md` and `scripts/molab_m0.py`.
The prepared runner requires actual CUDA/Blackwell discovery, re-runs the M0
suite and environment benchmark, then performs these reference probes in the
separate locked CUDA environment. Failures stop the run. Returned results have now been reviewed; see the Blackwell evidence section below.

No TinyChess self-play trainer, Dataset A generation or large-model training
has been implemented or launched. Overall M0 still requires GPU evidence and
review of the declared rules policy.

## Molab sibling-import fix

The first user-supplied Molab screenshot showed the official CLI failing at
`from network import AZNet`, before the expected unsupported-baseline assertion.
This exact failure was reproduced locally with `PYTHONSAFEPATH=1`. The Molab
runtime's actual environment flags have not been inspected, so safe-path mode is
a reproduced cause rather than a confirmed diagnosis of its configuration.
The child process now explicitly receives `PYTHONPATH` set to the verified
upstream directory. Upstream source, checksums, baseline assertion and error gate
remain intact. The parent already explicitly inserts that directory for the
instrumented run. Both native and wrapped reference probes are rechecked locally
with safe-path mode enabled; separate `*_safepath_cpu` evidence records the result.
Existing Molab runs can resume using `scripts/resume_molab_m0.py`, preserving
the prior failure log and completed M0 evidence. Returned GPU evidence now confirms success of the bounded probes (see below).


## Returned Blackwell evidence reviewed — 25 October 2026

Both bounded probes passed on the NVIDIA RTX PRO 6000 Blackwell Server Edition.
Original reports/logs are preserved under `reports/blackwell/`. Source manifests
match this checkout. Native cold/warm selfplay: 12.438 s / 0.00783 s; corrected
adapter: 14.283 s / 0.00820 s. Policy losses were 8.2309809 and 8.2309818;
value losses were zero because no games completed. Synthetic return/mask checks
passed on GPU. The expected missing-baseline assertion and confirmed upstream
truncation-discount defect remain documented. Full M0 GPU suite: 69 tests passed
in 324.12 s. This evidence resolves earlier GPU-pending statements in this report;
see `m0_rules_parity.md` for the current disposition and microbenchmark table.
