# M2 Stage 0 — correctness and environment setup

**Recommendation: PASS for Stage 0's tested CPU scope.** Verified 29 October 2026.
Stage 1–3, model implementation, data repacking, training and self-play remain
**NOT STARTED** and are outside this authorization.

## Repository and changes

- Branch reused: `m2/stage-0`.
- Base: `60e32a93926a2ea9698b8c636e0e5cc718035282`.
  Local main, origin/main and the Stage 0 branch initially pointed to this same
  commit; a read-only GitHub comparison against live main was identical.
- PR #3 was verified **merged**, with that exact merge commit. PR #2 was already
  merged. No merge, Git command, commit, push or PR creation was performed.
- M1 tag: `m1-dataset-50k-v1`; annotated tag object
  `b53ffe9f9cb738ba3e0e64b09a370ef216a5ce83`, peeled source commit
  `9893c2edcdb99dd57a58da8cd688b4157753bf93`. Local tag contents and GitHub's
  tag/commit comparison agreed.
- Read README v0.4, complete M1/M2 specifications, encoder contract and M1
  acceptance/validation/Blackwell results before implementing tests. The latest
  owner instruction limits the roadmap to Stage 0 and supersedes its old PR
  merge instructions and later-stage execution permissions.
- Changed only `tests/test_m2_stage0.py`, this report, a report allowlist entry in
  `.gitignore`, and an appended verified-completion note in `m2execution.md`.

Existing M0 tests exhaust constructed successors and periodically branch during
random games; M1's Oracle already checks full FEN/history/legal/status/rewards.
The two new tests reuse that history-aware oracle and strengthen coverage to
**every** legal action at every newly sampled root. No parallel production rules
or encoder implementation was introduced. python-chess is used only offline.

## Test A — independent legal-action transition parity

Eight seeds: `0, 1, 7, 19, 37, 73, 101, 2026`. Play 128 legal plies per seed,
choosing moves from sorted python-chess legal moves with `random.Random(seed)`.
Check all 1,024 played-path positions; exhaust successors at plies
`0,1,8,9,32,33,64,65,96,97,126,127`: **96 sampled random roots**.
Add **26 named special roots**: 14 replayed prefixes preserving history and
12 sparse FEN fixtures explicitly initialized with history discarded.

At each root enumerate the actual Pgx mask, decode every action through the
accepted UCI codec, and compare the complete move set to python-chess. For each
action independently copy the complete python-chess branch/history and push the
move, step Pgx, and compare six-field FEN including raw EP, resulting legal set,
side to move, castling/EP semantics, clocks, fullmove/episode counters, board
history, repetition counts/hash-history shift, terminal/truncation flags and
rewards. The public python-chess `outcome(claim_draw=False)` supplies an
additional independent outcome reference for the no-voluntary-claims policy.
Mismatch diagnostics contain case name, seed, root FEN, episode ply, action, UCI
and complete replay prefix. Assertions stop at the first mismatch; commands use
pytest `-x` to stop the run too. No failed position is filtered out.

| Actual coverage | Result |
|---|---:|
| Exhausted roots / unique full root FENs | 122 / 115 |
| Legal successor transitions | 3,384 |
| White / Black mover transitions | 1,584 / 1,800 |
| Check evasions / delivered checks | 17 / 211 |
| Legal EP captures, White / Black | 1 / 1 |
| Checkmate / stalemate successors | 17 / 18 |
| Insufficient-material / fivefold draw successors | 4 / 2 |
| Automatic 75-move draw successors | 60 |
| Ongoing successors at >=100 halfmoves | 30 |
| Current-threefold successors | 4 |
| Mate taking precedence at 150 halfmoves | 2 |
| Unexplained transition mismatches | **0** |

All four castling moves (`e1g1`, `e1c1`, `e8g8`, `e8c8`) and queen/rook/bishop/
knight promotions for **both** colors, including captures, were exercised.
Event counts can overlap. Replayed repetition fixtures include an irrelevant
raw EP target so canonical repetition identity is checked with actual history.
The seeded sample stops at ply 127; the unchanged existing M0 suite separately
covers 512-ply truncation/absorption. This is finite coverage, not exhaustive
orthodox chess or general dead-position proof.

Ordered root/action corpus SHA256:
`1bcab19a934b8f78e09b4a8c362d8d4416ad1b6f4fb7b972147a08c61c9b22db`.

## Test B — policy-index/encoder mapping

**3,384 legal actions across the same 122 roots; zero mismatches.** For each
legal action, independently obtain the absolute source square and piece from
UCI/python-chess. Check file-major indexing with Black's rank reversal, then:

```python
source = action // 73
own_piece = planes[7 - source % 8, source // 8, 0:6]
```

The feature must equal the exact six-channel one-hot for the side-to-move piece,
not merely contain any piece. All six piece types were covered. Float32 shape
`(8,8,120)`, legal-mask equality and encoder version
`pgx-v2-history8-ep-context8-v1` are asserted. The expected piece/square is not
read from Pgx's board or the encoder. This verifies the accepted contract;
future source-square logits must respect this permutation rather than blindly
flattening spatial row-major planes. No policy head/model was implemented.

## Commands and actual results

From the repository root:

```bash
TINYCHESS_STAGE0_COVERAGE="$PWD/data/processed/m2_stage0/coverage.json" \
  JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_m2_stage0.py -x -q
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -x -q
```

- New tests: **2 passed in 9.46s**.
- Full existing + new regression: **269 passed in 57.31s**.
- Independent fallback environment: **2 passed in 9.29s**; package compatibility
  check passed (54 installed packages including the project).
- Targeted, full-suite and fallback coverage JSONs were logically identical,
  including the corpus SHA256, event counts and both zero mismatch counts.
- Actual host: macOS 26.6.1 arm64, 10 logical CPUs, `TFRT_CPU_0`.
  Python 3.11.15; JAX/jaxlib 0.7.2; Pgx 2.6.0; mctx 0.0.6; Flax 0.12.0;
  Optax 0.2.6; pytest 8.4.1; python-chess 1.999 / chess 1.11.2; uv 0.10.9.

Small logs, coverage and initial-hash snapshots are in ignored
`data/processed/m2_stage0/`; coverage can be regenerated with the command above.
The fallback environment/cache created for verification was removed afterward.
No new GPU experiment was run. The accepted historical Blackwell suite (264
CUDA tests) is separate evidence; these two new checks were executed on CPU only.

## Pinned development setup and CPU fallback

Canonical setup:

```bash
uv sync --locked --extra test --extra data
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -x -q
```

`.python-version` requests **3.11.15**. `uv sync --locked` may download that Python
when unavailable and therefore needs network access; missing dependency wheels
or the pinned setuptools 80.9.0 build backend can require network too. A lockfile
fixes resolution, not package availability. `--offline` works only with all
required runtime/build artifacts already cached. See [uv Python management](https://docs.astral.sh/uv/concepts/python-versions/).

Primary setup was actually verified with
`UV_CACHE_DIR="$PWD/.uv-cache" uv sync --locked --extra test --extra data`;
the cache override keeps writes inside the workspace in restricted environments.
An initial offline attempt explicitly failed on an uncached build backend;
the unchanged online locked setup then succeeded. Python was already installed,
so a fresh interpreter download was not exercised or claimed.

CPU-only alternative if an existing compatible Python **3.11** is available:
set `STAGE0_PY311` to its absolute executable path, then run:

```bash
"$STAGE0_PY311" -m venv --without-pip data/processed/m2_stage0/cpu-fallback
UV_PYTHON_DOWNLOADS=never uv pip sync \
  --python data/processed/m2_stage0/cpu-fallback/bin/python \
  --require-hashes requirements-lock.txt
UV_PYTHON_DOWNLOADS=never uv pip install \
  --python data/processed/m2_stage0/cpu-fallback/bin/python --no-deps --editable .
uv pip check --python data/processed/m2_stage0/cpu-fallback/bin/python
JAX_PLATFORMS=cpu data/processed/m2_stage0/cpu-fallback/bin/python \
  -m pytest tests/test_m2_stage0.py -x -q
```

This uses uv's [explicit environment selection](https://docs.astral.sh/uv/pip/environments/)
and the CPU requirements export, without the CUDA extra or an automatic Python
download. It may still fetch locked packages/build dependencies. A pip module
inside the environment is unnecessary. Tested with the already available
3.11.15 executable underlying `.venv/bin/python`, not an independently installed
system Python or every other 3.11 patch release. The fallback contained no
`nvidia-*` or `jax-cuda*` packages and exposed only CPU. Fresh offline fallback
installation correctly failed on an uncached jaxlib wheel; online installation
with hash checking succeeded. No versions/lockfiles were loosened or changed.
The project permits Python >=3.11,<3.14; the canonical runtime remains 3.11.15.

## Preservation, dataset check and remaining limits

Production source, rules, codec, encoder, configuration/schema, dependency locks
and original M1 acceptance/results match their pre-task SHA256 fingerprints.
Only ignored generated package metadata changed when reinstalling the editable
project. M0/M1 regressions were retained unchanged.

A read-only authenticated Hugging Face metadata check confirmed both dataset
main and the pinned revision resolve to
`b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c`. No dataset payload was downloaded or
modified. No archive scan, repacking, transformer, training or self-play ran.

No blocking correctness mismatch was found. CPU-only/new-test coverage and
finite sampling are the stated limits, alongside unchanged historical M1
telemetry/dead-position limitations. **Stop here for Stage 0 review.**
