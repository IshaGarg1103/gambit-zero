# TinyChess v0.4 — M0 reviewer handoff

Prepared: 25 October 2026. **Implementation and CPU/GPU validation are complete
for the user's immediate M0 scope. The user subsequently supplied a CONDITIONAL
PASS; all four conditions are addressed in `m0_commit_verification.md`. M1 has not started.**

## What M0 establishes

The corrected Pgx chess adapter, action mapping and batched JAX/mctx interfaces
work for the tested positions and games on a Mac CPU and the intended Molab
NVIDIA RTX PRO 6000 Blackwell Server Edition. This is a rules/infrastructure
result, not a trained-model or playing-strength result. Native Pgx and the
unmodified upstream AlphaZero chess trainer do not independently pass our gate.

## Implemented deliverables

| Deliverable | Implementation / evidence |
|---|---|
| Minimal project and isolated Python environment | `pyproject.toml`, `.python-version`, `src/tinychess/`, `.gitignore`; Python 3.11.15 |
| Exact dependency locks | Root `uv.lock`, `requirements-lock.txt`, `requirements-cuda-lock.txt`; separate upstream-example `benchmarks/official_pgx/uv.lock` |
| Versioned action codec | `src/tinychess/chess/action_codec.py`: 4672 Pgx indices ↔ standard UCI, Black orientation, castling, queen/underpromotion; host boundary only |
| JAX Pgx adapter | `src/tinychess/chess/pgx_env.py`: initialization, step, explicit history-discarding FEN import, draw diagnostics, canonical repetition history, termination/truncation and rewards |
| Move-mask correction | `src/tinychess/chess/_pgx_legal.py`: audited Pgx function with candidate buffer 200→434; AST regression ensures only that executable change plus function rename |
| Differential tests | `tests/test_rules.py`: constructed positions and every successor, four seeded games to completion/truncation, periodic all-successor checks, full FEN/clock/rights, legality, checks, outcomes, claims and perspective |
| Native Pgx regression audit | `tests/test_upstream_audit.py`: reproduces native defects; passing these audit tests does not certify native Pgx parity |
| Batched environment/mctx smoke | `src/tinychess/search/mctx_recurrent.py`, `tests/test_search.py`: dummy policy/value, JIT/vmap, PUCT/Gumbel, masks, values, rewards, absorbing nodes and bootstrap cutoff |
| Flax/Optax compatibility smoke | `tests/test_stack.py`: tiny Dense inference and optimizer-state initialization |
| Batch throughput benchmark | `benchmarks/bench_pgx_batching.py`: batches 1,8,32,128; trace/lower, compile and warm-up separate from steady execution |
| Installed-source/API inspection | `scripts/inspect_stack.py`, CPU/GPU stack JSONs with versions, exact signatures, upstream hashes, hardware/compiler discovery |
| Official AlphaZero example audit | Original pinned upstream files, checksums, separate environment, bounded runner and `reports/official_pgx_example.md` |
| Manual Molab reproduction | `scripts/molab_m0.py`, `scripts/resume_molab_m0.py`, `reports/molab_notebook.md` |
| Results and provenance | `reports/m0_rules_parity.md`; original remote files and checksummed review under `reports/blackwell/` |

Core pins: JAX/jaxlib 0.7.2, Flax 0.12.0, Optax 0.2.6, Pgx 2.6.0 chess-v2,
mctx 0.0.6; offline oracle python-chess 1.999 / chess 1.11.2.

## Native Pgx defects and explicit corrections

- The 200-candidate move buffer omits 18 moves from a valid 218-move stress
  position. Keep all 434 pre-filter candidate slots. Test all 218 successors.
- Native automatic threefold and 100-halfmove stopping differ from the selected
  automatic-only game convention. Use fivefold and 150 halfmoves; checkmate wins
  precedence. Expose current/intended-move claimability separately.
- Native 512-ply cutoff is termination. Reclassify it as truncation with no
  adjudicated WDL outcome; stop mctx bootstrapping at the boundary.
- EP hashing distinguishes a non-capturable EP target. Include EP only when an
  actual legal EP capture exists, including pin checks.
- Native initial hash is stale. Compute it from the actual state.
- Native experimental FEN conversion inherits default initial history. Clear
  history explicitly; require `history="discard"`. Replay full trajectories when
  prior repetition/history is required.
- Native core terminal masks use all-True sentinels. Keep the real chess mask
  and expose no playable UCI moves after episode end; search terminal embeddings
  are absorbing and use finite dummy priors.

Private Pgx APIs are version-pinned. Upstream source has not been modified in
site-packages. The vendored function preserves the Pgx copyright and license.
No Rust/PyTorch fallback was activated because these corrections passed the
selected differential gate. A future unresolvable essential defect must reopen
M0 and trigger the README's separately validated fallback path.

## Actual validation

- CPU suite: **69 passed**, no skips/expected failures. Original run 41.18 s;
  pre-Molab repeat 39.27 s, retained separately.
- Blackwell suite: **69 passed in 324.12 s (5m24s)**.
- GPU identity, core versions, action/rules policy IDs and recorded upstream
  source SHA256s were checked against the local evidence and match.
- Raw returned GPU logs/JSON are preserved byte-for-byte; `blackwell/review.json`
  records archive and per-file hashes. The agent inspected user-returned evidence,
  not a live remote session.

Corrected GPU environment microbenchmark:

| Batch | Trace/lower s | XLA compile s | Steps/s |
|---:|---:|---:|---:|
| 1 | 1.380 | 2.589 | 1,952 |
| 8 | 1.264 | 2.765 | 15,359 |
| 32 | 1.291 | 2.810 | 62,631 |
| 128 | 1.280 | 2.693 | 229,335 |

These are synchronized independent one-ply steps from resident initial states,
not full trajectory/search/game throughput. They include rules, observations,
canonical history and host dispatch. They exclude initialization, action
selection, network inference and resets. No 100M training feasibility is claimed.

## Official Pgx AlphaZero example

Pinned upstream commit: `87278d2d6e677fd87248c457207b59cfa42e578d` (v2.6.0).

- The unmodified chess CLI fails at nonexistent `chess_v0` baseline loading.
- With exactly that unused module-level assignment bypassed, the actual official
  Haiku network, Gumbel rollout, loss-input conversion and one update ran on CPU
  and GPU, with both native Pgx and the corrected adapter. This is instrumented
  reference execution, not success of the unmodified complete CLI.
- Tiny network: **613,042 parameters**, eight channels, one block; two boards,
  four plies, four simulations. Compile and warm execution are separate.
- Zero completed chess games and zero valid outcome targets in these tiny traces;
  value loss 0 is expected. A synthetic terminal sequence separately verifies
  backward-return signs and value masking.
- The original recurrence ignores `truncated` and retains discount -1 at our
  episode cap. The audit records this defect; do not reuse that trainer unchanged.
  TinyChess's separate dummy recurrence correctly uses discount 0 there.
- Molab's sibling-import failure was fixed by explicitly supplying the verified
  upstream directory to the child import path. Safe-path-mode regressions pass
  locally. Upstream sources/checksums were not changed.

No TinyChess trainer, human dataset, replay dataset, large checkpoint or
production self-play loop was created. Only the bounded reference smoke runs
performed tiny optimizer updates; no checkpoints or dataset rows were saved.

## Decisions and limitations for reviewer attention

1. **Rules convention:** no voluntary draw claims. The adapter detects threefold
   and 50-move claims (including intended moves), but the move policy has no claim
   action. Automatic draws use fivefold/75 moves. Is this acceptable for training
   and disclosed future match protocols, or should claims become an explicit action?
2. **Episode horizon:** 512 plies means truncation, not a draw. Such episodes must
   never receive final-outcome WDL labels. History storage and metadata must be
   revised together if the horizon changes.
3. **History correctness:** FEN alone cannot preserve repetition. Review the replay
   path, canonical EP identity, initial hash and randomized player-order mapping.
4. **Scope of parity:** four random games and selected edge cases provide evidence,
   not exhaustive correctness. General FIDE dead-position solving and collision-free
   repetition are not established. Repetition uses a two-word Zobrist hash.
5. **Input boundary:** the FEN loader expects valid standard-chess FEN, not arbitrary
   untrusted/Chess960 input. The M1 offline parser must validate source games.
6. **Reproducibility:** review the vendored mask/private APIs and source pins. Actual
   GPU model memory, BF16 throughput and completed games/hour remain later work.
   Molab's enormous reported free-disk value is virtual accounting; measure quota
   before allocating data storage.

No currently known essential corrected-adapter defect failed the tested gate.
Accepting conventions is a research decision; it should not be confused with an
upstream implementation fix or proof of all chess positions.

## Reproduce

```bash
uv sync --locked --extra test
.venv/bin/python -m pytest -q
.venv/bin/python benchmarks/bench_pgx_batching.py
uv sync --project benchmarks/official_pgx --locked --python 3.11.15
benchmarks/official_pgx/.venv/bin/python scripts/run_official_pgx_example.py --backend native --output /tmp/official_native.json
benchmarks/official_pgx/.venv/bin/python scripts/run_official_pgx_example.py --backend wrapped --output /tmp/official_wrapped.json
```

GPU commands and manual notebook steps are in `reports/molab_notebook.md` and
`scripts/molab_m0.py`. Do not rerun benchmark scripts blindly over archived GPU
results; preserve evidence under `reports/blackwell/`.

## Suggested independent-review request

Review TinyChess M0 against the root README and the user's immediate M0 scope.
Inspect implementation and tests, not just the passing counts. Prioritize legal
move completeness, action orientation/promotions, repetition and FEN history,
draw claims versus automatic rules, reward/value perspective, terminal/truncated
search dynamics, dependency reproducibility and provenance. Review the declared
no-claim policy and 512-ply horizon explicitly. Distinguish raw Pgx/official-trainer
defects from corrected-adapter failures. Give concrete file/line findings and a
PASS, CONDITIONAL PASS or FAIL recommendation; do not begin M1 or model training.

## After review: M1 only

1. Freeze a versioned Dataset A schema and manifest: source, whole-game split,
   filtering reasons, rules/action versions, history and outcome provenance.
2. Implement the offline streaming PGN parser and validate **100–1,000 complete
   games** first. No full monthly-archive downloads.
3. Replay accepted moves through the corrected Pgx adapter; retain all legal
   actions, clocks, castling/EP/repetition history and side-to-move labels.
4. Test outcomes/signs, parse rejection, variants, timeout handling, checksums,
   whole-game split separation and duplicate-position/leakage diagnostics.
5. After that smoke gate, run a bounded **50K accepted-game pilot**, with an
   explicit source/storage budget. Model training remains M2.

Git preparation updated 25 October 2026: the implementation is prepared for its
initial commit on the user-requested `pgx-mctx-infrastructure` branch. `origin`
points to `https://github.com/IshaGarg1103/gambit-zero.git`. No push or merge is
performed by this preparation. The earlier review ZIP predates final commit
preparation; use the committed tree and `reports/m0_commit_verification.md` for
final checks. The user subsequently supplied the independent review's CONDITIONAL PASS and
four conditions. Their disposition and final verification are recorded in
`reports/m0_commit_verification.md`; the original remote evidence is preserved.
