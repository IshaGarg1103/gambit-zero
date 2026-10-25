# M0 commit verification — 25 October 2026

**Status: verified implementation prepared for commit on
`pgx-mctx-infrastructure`; no push or merge performed.** Repository:
https://github.com/IshaGarg1103/gambit-zero. The TinyChess research name and
Python package name are retained; only the GitHub destination changed.

## Final checks actually run

| Check | Outcome / evidence |
|---|---|
| Locked main environment | `uv sync --locked --extra test --offline`: 60 resolved, 41 installed distributions audited; no changes |
| Complete default test suite | **75 passed in 38.85 s**, no skips/expected failures; `pytest_final_m0.txt` |
| Expanded random-game differential test | **32 passed, 48 deselected in 76.73 s**; `pytest_final_random32.txt` (the default four seeds overlap this expanded run) |
| Main installed dependency compatibility | `uv pip check`: all 41 distributions compatible |
| Reference environment lock | Locked offline sync dry-run: 75 resolved, 56 installed distributions audited; no changes |
| Official Pgx example, native | Bounded actual routines passed with `PYTHONSAFEPATH=1`; `official_pgx_native_final_cpu.json` and `.log` |
| Official Pgx example, wrapped | Bounded actual routines passed with `PYTHONSAFEPATH=1`; `official_pgx_wrapped_final_cpu.json` and `.log` |
| Python syntax compilation | `compileall` passed for project source, tests, scripts and vendored example |
| Source/test identity | All **17 source/test files** match the SHA256 manifest in the review snapshot associated with the returned GPU validation |
| Remote evidence integrity | All nine archived Blackwell evidence files match `blackwell/review.json` hashes |

The prior Blackwell run remains **69 passed in 324.12 s**, with both bounded
reference probes passed. No fresh GPU run was claimed during commit preparation;
the source and tests have not changed since that run. Benchmark reports preserve
CPU/GPU compilation and warm-up separately. Peak environment microbenchmark
throughput was 229,335 steps/s at batch 128; this is not complete-game throughput.

## Scope and review disposition — supplied conditions addressed

Known issues discovered during M0 have explicit tested corrections in our adapter:
the Pgx candidate buffer, initial/repetition/EP hashes, FEN history initialization,
and draw/termination conventions. The official chess example's unsupported
baseline loader is bypassed only in the bounded audit harness, and its truncated
bootstrap limitation is recorded rather than silently reused in TinyChess.

The user supplied the independent ChatGPT review disposition: **CONDITIONAL
PASS**, with four closure conditions. This report addresses the supplied feedback;
the full original reviewer response was not independently retrieved.

1. **Adopt no voluntary draw claims:** adopted for training, pinned as
   `tinychess-auto-fide-pgx-v1` and documented in the README. Claims remain
   diagnostics; automatic draws use fivefold repetition and 150 halfmoves.
2. **No truncation draw/WDL labels:** `chess/adjudication.py::final_wdl` now
   rejects ongoing or truncated episodes before producing labels, including
   inconsistent states with both end flags set. Genuine draws and player-indexed
   mate labels are tested; absorbed decisive states with cleared rewards are
   rejected. This host-side boundary is outside MCTS and uses no python-chess.
3. **At least 32 seeded games plus history:** the expanded 32-game differential
   run passed. The complete suite separately covers replayed repetition,
   intended-move claims, legal/non-capturable EP identity and FEN history reset.
4. **Rerun and report:** the complete suite is rerun with the added label-boundary
   tests; its authoritative result is in `pytest_final_m0.txt`. Original remote
   evidence remains unchanged and distinguishable from this final local run.

The new host-side adjudication helper and its six tests were added after the
69-test GPU run. The previously GPU-validated 17 source/test files are unchanged.
No claim is made that the expanded suite was rerun remotely; the new guard has
local tests and does not change the compiled environment/search kernels.

The documented policy remains: no voluntary draw claim action; threefold/50-move
claimability is audited separately; automatic fivefold/75-move draws; 512-ply
truncations carry no adjudicated WDL outcome. Tests are sampled differential
coverage, not exhaustive chess correctness, full dead-position solving or
collision-free repetition. Private Pgx APIs require revalidation on upgrades.

No M1 dataset generation, TinyChess model training, production self-play trainer
or checkpoint generation was performed. The only optimizer updates were tiny
bounded official-example compatibility probes, with no stored training output.

## Reproduce final local checks

```bash
UV_CACHE_DIR=/tmp/tinychess-uv-cache uv sync --locked --extra test --offline
.venv/bin/python -m pytest -q
TINYCHESS_RANDOM_GAMES=32 .venv/bin/python -m pytest tests/test_rules.py -k seeded_random -q
UV_CACHE_DIR=/tmp/tinychess-uv-cache uv pip check --python .venv/bin/python
PYTHONSAFEPATH=1 benchmarks/official_pgx/.venv/bin/python scripts/run_official_pgx_example.py --backend native --output /tmp/official_native_final.json
PYTHONSAFEPATH=1 benchmarks/official_pgx/.venv/bin/python scripts/run_official_pgx_example.py --backend wrapped --output /tmp/official_wrapped_final.json
```

Initial installs need network; use the locked online setup commands in the README
and reference audit report before the offline checks on a clean machine.

## Commit contents and exclusions

Include source, tests, Python configuration, dependency lockfiles, README,
reproduction scripts, relevant CPU/GPU benchmark and test evidence, and upstream
reference code plus its license/source manifest. Preserve original benchmark
reports, including the resolved import-failure log for provenance.

Exclude all virtual environments, caches, downloaded datasets, generated
checkpoints, credentials, packaged ZIP artifacts and the duplicate `README (4).md`.
An explicit staged-file audit checks forbidden paths, file sizes, credential-like
content and Git whitespace errors before committing. The source tree and evidence
are small text files. Nothing is force-pushed, and `main` is not modified.


Final disposition: **M0 correctness checks PASS within the documented scope;
all four supplied CONDITIONAL PASS conditions addressed.** The final complete
local suite includes the six new host-side label tests. Existing GPU evidence is
still 69 tests on unchanged environment/search code. No M1 work, push or merge
is performed. The user will push the committed branch manually.
