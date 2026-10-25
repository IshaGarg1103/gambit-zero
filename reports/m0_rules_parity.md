# M0 rules parity and infrastructure gate

Initial run: 24 October 2026; GPU evidence reviewed: 25 October 2026.
Scope: M0 only. No dataset archives, TinyChess model training,
self-play RL, teacher labels, or large checkpoints were produced.

**Disposition: native Pgx FAILS parity. The explicitly corrected Pgx adapter
passes the CPU and Molab Blackwell test suites. GPU validation is verified
from the returned evidence on 25 October 2026. The supplied independent review conditions have been addressed in final commit
preparation; see `m0_commit_verification.md` for the updated local suite and policy adoption.** No backend switch
has occurred. Do not advance to M1/training on the basis of native Pgx alone.

## Environment and reproducibility

The supplied plan was named `README (4).md`; its entire text was read and copied
to `README.md`. The 100M primary model, B0→B1→B2 order, separate supervision
sources, evaluation protocol and gated scaling arms are preserved. Only M0
modules are implemented, without placeholder training or data-generation code.
The folder initially inherited an unrelated parent Git checkout. An independent
local repository was initialized on `codex/m0`, with `origin` pointing to the
supplied TinyChess GitHub URL. The parent history, remote and existing changes
have not been altered. GitHub was verified read-only; nothing has been pushed.

The actual test host is macOS arm64, 10 logical CPUs and 16 GiB RAM, **JAX CPU
only**. It is not the remote GPU. Python 3.11.15; JAX/jaxlib 0.7.2; Flax 0.12.0;
Optax 0.2.6; Pgx 2.6.0 with chess environment **v2**; mctx 0.0.6;
python-chess 1.999 / chess 1.11.2; pytest 8.4.1. Initial resolution with JAX 0.6.2
was rejected because mctx 0.0.6 requires JAX/jaxlib ≥0.7.0; the incompatible
combination was never used for tests.

`uv.lock` pins the cross-platform dependency graph with hashes, including an
optional Linux CUDA 12 dependency set. `requirements-lock.txt` records all 40
third-party distributions actually tested locally. `requirements-cuda-lock.txt`
is the hashed CUDA/test export of `uv.lock`; the stack has now executed
on Blackwell in Molab (returned evidence below). `stack_cpu.json` records exact signatures,
upstream source SHA256s, CPU/RAM/disk, compiler and device discovery. NVIDIA
hardware, driver and initial memory metadata are in `blackwell/stack_blackwell.json`.
BF16/model feasibility is not established by these float32 smoke tests.

From this project directory:

```bash
UV_CACHE_DIR=/tmp/tinychess-uv-cache uv sync --locked --extra test
.venv/bin/python scripts/inspect_stack.py --output reports/stack_cpu.json
.venv/bin/python -m pytest -q
.venv/bin/python benchmarks/bench_pgx_batching.py --output reports/pgx_batching_cpu.json
```

Individual checks:

```bash
.venv/bin/python -m pytest tests/test_rules.py -q
.venv/bin/python -m pytest tests/test_upstream_audit.py -q
.venv/bin/python -m pytest tests/test_search.py tests/test_stack.py -q
TINYCHESS_RANDOM_GAMES=32 .venv/bin/python -m pytest tests/test_rules.py -k seeded_random -q
```

The last command expands random-game coverage on demand; the default four games
are the reproducible baseline. `pytest-output.txt` is the actual final local
suite output: **69 passed in 41.18 seconds**, no skips or expected failures.
`uv pip check` also passed for all 41 installed distributions including TinyChess.
Upstream audit tests deliberately **assert the existence of native
Pgx defects**; their success is not evidence of native Pgx parity.

## Verified source contracts and upstream defects

We inspected the installed `pgx/chess.py`, `pgx/core.py`,
`pgx/_src/games/chess.py`, `pgx/experimental/chess.py`, and
`mctx/_src/policies.py`, then captured signatures via `inspect.signature`.
The source hashes are in `stack_cpu.json`.

| Native Pgx 2.6.0 observation | Adapter correction / regression |
|---|---|
| Move generation retains only 200 pseudo-legal candidates before check filtering. A valid 218-move position loses 18 legal moves. | Vendor that single move-mask function; retain all **434 = 16×27+2** candidate slots. AST test ensures executable code differs only in function name and this buffer size. No installed package modification or global monkeypatch. |
| `Game.is_terminal` stops at 100 halfmoves without a claim and automatically at threefold. | Explicit policy: continue until fivefold or 150 halfmoves; expose claim diagnostics separately. |
| 512 plies is `terminated=True, truncated=False`. | Mark `truncated=True, terminated=False`, unless a genuine terminal rule occurs on the same move. Never treat truncation as a drawn outcome. |
| EP target is always part of the repetition hash, even if no legal EP capture exists. | Canonicalize EP using the actual legal-action mask, including pinned-pawn cases. |
| Hard-coded initial hash differs from the computed hash of the initial position. | Compute the initial hash, then maintain canonical hashes after every transition. |
| Experimental FEN import inherits default initial history and cannot preserve repetition. | Clear history explicitly; require `history="discard"`. Replay trajectories instead of re-importing each FEN when history is required. |
| Core API replaces terminal legal masks with all-True sentinels. | Keep the actual chess mask; codec exposes no playable moves after termination/truncation. Search uses finite dummy logits solely for absorbing terminal embeddings. |

218-move regression FEN:

```text
R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1
```

Native Pgx: 200 moves; python-chess and adapter: 218. Native Pgx misses:

```text
h4g3 h4g4 h4g5 h4h1 h4h2 h4h3 h4h5 h4h6 h4h7
h8b8 h8c8 h8d8 h8e8 h8f8 h8g8 h8h5 h8h6 h8h7
```

Fixture provenance: [Lichess analysis position](https://lichess.org/analysis/fromPosition/R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1_w_-_-),
linked from the author's [branching-factor investigation](https://blog.lukesalamone.com/posts/most-ramified-chess-position-2023/).
Move counts and missing moves were independently computed locally.
The retained Pgx Apache 2.0 copyright/license appears in `_pgx_legal.py` and
`licenses/PGX-LICENSE`.

## Declared rules and state contract

Rules policy ID: `tinychess-auto-fide-pgx-v1`. Standard chess only, no Chess960.
Automatic terminal conditions: checkmate, stalemate, insufficient material
according to the tested Pgx/python-chess detector, fivefold repetition and
150 halfmoves without capture/pawn movement. Checkmate takes precedence over
the 75-move rule. Legal moves, complete six-field FENs including raw EP target,
clocks, check state, player-order mapping and rewards are compared to the oracle.

Players **never voluntarily claim draws** under this policy. Threefold and
50-move claimability are separate from automatic draws. `ENV.claims` returns
threefold/fifty-move booleans, including claims available by an intended legal
move, and is checked against `can_claim_threefold_repetition` /
`can_claim_fifty_moves`. This expensive JAX audit helper is excluded from MCTS.
There is no claim action in the 4672-move policy space. Any future GPT match must
explicitly disclose this convention or add a separately tested claims protocol;
changing it requires a new policy version.

Episode horizon is 512 plies from initialization/import. Truncated games have
zero transition reward but **no adjudicated outcome** and must not supply WDL
training labels. This horizon also bounds the stored 513×2 uint32 hash history.
Increasing the horizon requires enlarging history and revalidating the adapter.
Repetition is based on a two-word Zobrist hash, with probabilistic collision
risk; this is not a collision-free proof of position equality.

FEN import is a host/offline operation, supports valid orthodox six-field FENs,
and resets episode ply and prior history. Validate untrusted input offline.
It retains halfmove/fullmove clocks and explicitly acknowledges loss of prior
repetition/history. Observation history is seeded only with the supplied board.
`step` accepts scalar action indices in [0,4672); illegal chess actions forfeit;
already finished states absorb actions and return zero rewards.

This is tested equivalence to python-chess's implemented outcome rules,
**not a proof of every FIDE dead position**. Both libraries' insufficient-material
detection is more limited than a solver for arbitrary dead/blocked positions.

## Action codec

Codec ID: `pgx-2.6.0-chess-v2-uci-v1`. Pgx squares are **file-major** (a1=0,
a2=1), and Black's rank orientation is reversed. Index = source×73+plane.
Planes 0–8 encode rook/bishop/knight underpromotions; ordinary pawn moves to the
last rank imply a queen. Standard UCI castling is king-to-destination (e1g1,
e1c1, e8g8, e8c8). All promotions and capture promotions in both orientations
are round-tripped. Invalid indices, illegal UCI and spurious promotion suffixes
are rejected. Legal UCI moves must map one-to-one to indices.

This string codec is a host I/O boundary. Search carries integer action indices
and full Pgx state directly, without UCI conversion or python-chess calls.

## Test coverage and search integration

The suite includes:

- All legal moves and every resulting state from 21 constructed starting FENs,
  including the high-mobility composition and both colors.
- Four seeded legal games, run to terminal or 512-ply truncation. At every ply:
  full FEN, legal set, check, termination, rewards and codec equivalence. Every
  64 plies: all successor positions are checked too.
- Castling both sides/colors, blocked/attacked transit, check, rights lost by
  rook/king moves, rook captures, and rights not restored by returning a rook.
- EP captures, vertical and horizontal pins, canonical repetition identity for
  legal versus non-capturable EP, and histories after a pawn double push.
- Queen/rook/bishop/knight promotions and capture promotions for both colors.
- Mate/stalemate, mate at halfmove 150, bare kings, single bishop/knight,
  same/opposite-color bishops, and two knights as a non-draw control.
- Clocks 49,50,98,99,100,149,150; present and intended-move claims; threefold
  claim availability before fivefold; real trajectory replay; FEN history reset.
- 512-ply truncation, genuine termination precedence, terminal absorption,
  illegal-action forfeits and both player-order/perspective conventions.
- JIT/vmap transition batch size 8; mctx PUCT and Gumbel batch size 2 with four
  simulations each; tiny Flax Dense inference and Optax state initialization.

mctx 0.0.6 contract: root fields `(prior_logits,value,embedding)`; recurrence
`(params,rng_key,action,embedding) -> (RecurrentFnOutput,new_embedding)` where
output fields are `(reward,discount,prior_logits,value)`. Root invalid-actions
mask uses **True for illegal**. Interior masking is supplied through recurrent
logits. Reward is indexed for the **parent player**, child value is side-to-move;
discount is **-1** during play and **0** at terminal/truncated nodes. Dummy value
0.25 makes reversal observable. Tests include mate reward +1, draw reward 0,
zero terminal value, absorption and truncation bootstrap cutoff for multiple
randomized player assignments.

PUCT uses `mctx.muzero_policy` with Dirichlet noise disabled for this smoke test.
Gumbel uses `mctx.gumbel_muzero_policy` with four considered actions and noise
scale 0. Both must select legal root actions, produce finite nonnegative weights
summing to one with zero illegal mass, and preserve legality on expanded edges.
PUCT action weights are checked against visit probabilities; Gumbel weights are
its improved policy output, not relabeled visit counts. Terminal embeddings use
finite dummy logits; their actions are absorbing indices, not played chess moves.

No python-chess imports exist in the environment or search path (AST guard).
No transformer, training loop, replay generation or RL is implemented in M0.

## CPU benchmark

`pgx_batching_cpu.json` contains raw measurements for the **corrected adapter**.
Each timed call applies one legal move to resident initial-position states;
200 repetitions, three warm-ups, synchronization after each call. Includes full
rules, legal mask, canonical history, observation and host dispatch; excludes
initialization, action selection, network, search and resets. Repeating an active
input avoids measuring terminal no-ops. These are microbenchmark results, not
trajectory, search, completed-game or GPU throughput estimates.

| Batch | Trace/lower s | XLA compile s | Warm-up s | Environment steps/s |
|---:|---:|---:|---:|---:|
| 1 | 0.489 | 0.322 | 0.0013 | 5,971 |
| 8 | 0.537 | 0.363 | 0.0029 | 11,625 |
| 32 | 0.493 | 0.362 | 0.0070 | 15,254 |
| 128 | 0.407 | 0.437 | 0.0230 | 19,234 |

Timing varies with host load. Compile time excludes tracing/lowering, which is
reported separately. CPU runs expose no GPU-memory statistics. No inference
throughput for a 100M network, games/hour, or strength/GPU-hour is claimed.

## Molab GPU reproduction and remaining gate

In a Molab Linux environment with this project mounted/copied, use a terminal
(or notebook subprocess with this project as its working directory):

```bash
UV_CACHE_DIR=/tmp/tinychess-uv-cache uv sync --locked --extra test --extra cuda
JAX_PLATFORMS=cuda .venv/bin/python scripts/inspect_stack.py --require-gpu --output reports/stack_blackwell.json
JAX_PLATFORMS=cuda .venv/bin/python -m pytest -q
JAX_PLATFORMS=cuda .venv/bin/python benchmarks/bench_pgx_batching.py --output reports/pgx_batching_blackwell.json
```

Use the isolated project interpreter, not the Molab notebook's preinstalled
Python packages. Fail GPU discovery rather than silently falling back to CPU.
Record `nvidia-smi`, available HBM and actual JAX device kind, driver/runtime
compatibility, compile behavior and memory. Do not upgrade locked dependencies
to work around an error without rerunning M0 and updating the versioned evidence.

**Unresolved / review items:** Blackwell/Molab runtime and benchmark gate passed
from returned evidence (see below). The auto-only/no-voluntary-claim policy and 512-ply truncation need review before later milestones. Valid-input
FEN import is not an untrusted-FEN validator. Tests sample legal play and selected
edge cases rather than proving all reachable positions. Full FIDE dead-position
recognition and collision-free repetition are not established. The vendored move
mask and remaining private Pgx calls require re-audit on any Pgx change.

If a further essential move/rule defect cannot be transparently corrected and
validated, stop the Pgx path and implement the README's **PyTorch + Rust
shakmaty/PyO3 fallback** as a separately identified backend. Repeat differential,
history, action-codec and throughput gates; use a separately validated batched
search driver. mctx must not be claimed to work natively with that fallback.
The defects found here have local tested corrections, so a backend replacement
has not been activated.

Sources: installed source hashes/signatures in `stack_cpu.json`,
[Pgx source](https://github.com/sotetsuk/pgx),
[mctx source](https://github.com/google-deepmind/mctx), and
[JAX installation guidance](https://docs.jax.dev/en/latest/installation.html).
Installed source and runtime tests, rather than unpinned web documentation,
define the contracts used here.

## Follow-up: official AlphaZero reference and manual Molab runner

The official Pgx v2.6.0 AlphaZero example has now been inspected and its bounded
routines executed locally on native and corrected environments, in a separate
locked environment. The unmodified chess CLI cannot load `chess_v0`, and its
recurrence mishandles explicit truncation. See `official_pgx_example.md` for
actual evidence and precise instrumentation; this is not a production trainer.
Manual GPU notebook instructions are in `molab_notebook.md`.


## Verified Molab Blackwell evidence — 25 October 2026

**GPU correctness/stack gate: PASS. M0 ready for policy review.** Returned archive
`9842-83-zaeWcXbs.zip` passed ZIP CRC validation; original nine files are preserved
in `reports/blackwell/`. `review.json` records archive/per-file SHA256s and checks.
This is user-run remote evidence, not an agent-controlled live Molab session.

- Device: NVIDIA RTX PRO 6000 Blackwell Server Edition, GPU backend, one visible
  JAX device. NVIDIA-SMI reports 97,887 MiB total memory and driver 595.71.05.
  Its CUDA 13.2 field describes driver capability; host nvcc reports 13.3. These
  are not the version identifiers of the locked JAX CUDA wheels.
- Core package versions, Pgx chess-v2 identifier, action/rules policy IDs and all
  captured Pgx/mctx upstream source SHA256s match the local tested environment.
- **69 tests passed in 324.12 seconds (5m24s)**. No skipped/expected failures.
- Bounded official-example native and wrapped runs both passed; source manifests
  match the pinned upstream files. Expected unmodified chess baseline assertion
  remains visible in each JSON. Synthetic terminal return/mask checks passed.
- The wrapped probe again confirms the upstream recurrence's wrong discount at
  explicit truncation (-1). This is a limitation of reusing the official trainer,
  not a failure of TinyChess's separately tested corrected recurrence.
- No dataset/checkpoint saved; the short reference runs completed zero games and
  contain zero valid outcome value targets. Their zero value loss is not learning
  evidence. Pydantic metadata warnings are retained in the original logs.

Corrected environment microbenchmark (200 synchronized independent initial-state
steps per batch, same definition as the CPU benchmark):

| Batch | Trace/lower s | XLA compile s | Warm-up s | Environment steps/s |
|---:|---:|---:|---:|---:|
| 1 | 1.380 | 2.589 | 0.0047 | 1,952 |
| 8 | 1.264 | 2.765 | 0.0053 | 15,359 |
| 32 | 1.291 | 2.810 | 0.0060 | 62,631 |
| 128 | 1.280 | 2.693 | 0.0063 | 229,335 |

Single-position GPU throughput is below the CPU measurement; batching amortizes
launch/dispatch overhead. These numbers are not complete-game/search throughput
or a speed prediction for the future transformer. The native example's cold
selfplay (compilation plus first execution) took 12.438 s and warm selfplay
0.00783 s; wrapped: 14.283 s cold, 0.00820 s warm. Each is only eight positions
with four simulations; one tiny optimizer update was exercised per backend.

The sandbox reports 20 logical CPUs and 160 GiB RAM. Its reported free disk
(~9.22e18 bytes) is implausible virtual-filesystem accounting, not a usable
storage budget. JAX's initial allocator limit (~71.23 GiB) is not total GPU VRAM
or a measurement of future model/search peak memory. Measure actual quota,
BF16/model memory, full trajectories and search throughput in the later relevant
milestones; do not infer large-model feasibility from this smoke run.

Remaining review convention: players do not voluntarily claim draws; fivefold
and 150-halfmove automatic rules apply. A 512-ply cap is truncation with no WDL
label. The review does not establish full FIDE dead-position recognition or
collision-free repetition. No essential unresolved defect was found in the
**corrected adapter** within the tested scope, so the Rust/PyTorch fallback has
not been activated. Native Pgx alone and the unmodified official chess trainer
remain unsuitable for direct use. No M1 or TinyChess training work was started.


## Final conditional-review closure — 25 October 2026

The user supplied an independent CONDITIONAL PASS with four conditions.
The no-voluntary-claim training convention is now adopted explicitly in the
README. The checked host boundary `adjudication.final_wdl` rejects truncations
and ongoing episodes before assigning final labels; tests also reject cleared
rewards after absorbed decisive results. The existing JAX recurrence still cuts
bootstrap at truncation. A 32-seed random differential run passed, alongside the
suite's repetition/history fixtures. The final full local suite is **75 passed
in 38.85 s**; the 32-game run is **32 passed in 76.73 s**. No essential tested
correctness failure remains. The prior 69-test GPU run is archived unchanged;
new host label-guard tests have not been run on GPU and do not alter the tested
JAX environment/search kernels. The supplied review conditions are addressed;
M0 is finalized for a branch commit, with no push, main merge or M1 work.
