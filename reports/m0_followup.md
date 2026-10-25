# M0 follow-up: targeted hardening before M1

M0's acceptance gate remains passed. This follow-up expands evidence without
claiming exhaustive correctness or changing the existing rules/action semantics.
No Git branch, commit or push is performed.

## Upstream pin and local corrections

Pgx remains **2.6.0 chess-v2**, upstream commit
`87278d2d6e677fd87248c457207b59cfa42e578d`; JAX/jaxlib 0.7.2 and mctx 0.0.6.
`pgx_m1_pin.json` captures upstream and adapter SHA256s. The adapter is unchanged.
`pgx_legal_buffer.patch` is the function-level diff against the installed
upstream function; the executable correction is the 200→434 candidate capacity.
Other corrections are explicit wrapper behavior: fresh initial/FEN history,
legal-EP repetition identity, automatic fivefold/150-halfmove draws, and a
512-ply truncation with no final WDL outcome. They are not a silent library
upgrade or site-packages modification.

Upstream reporting distinctions: move-buffer omission, stale initial hash,
non-capturable EP identity and invented FEN history have focused reproductions
in `tests/test_upstream_audit.py`. Automatic draw thresholds, the no-voluntary-
claim convention and classifying the episode cap as truncation are our declared
policy. The official example's terminated-only recurrence is an integration
limitation for that convention, not a demonstrated mctx algorithm defect.

## Staged randomized parity: completed on CPU

| Stage | Seed base | Random positions | Supplementary history positions | Mismatches | Time |
|---|---:|---:|---:|---:|---:|
| Initial | 1000 | 10,000 | 168 | 0 | 7.29 s |
| Expanded | 20000 | 50,000 | 168 | 0 | 23.42 s |

Raw results: `m0_parity_10k.json` and `m0_parity_50k.json`. Every checked position
compares full legal actions, six-field FEN (including raw EP/clocks), check,
current canonical hash, repetition occurrence count, the entire eight-board
history stack, episode ply, color/player mapping, terminal/truncated flags and
reward vector. The independent oracle uses python-chess offline. Any unexplained
mismatch raises `ParityError` and stops, with a fail-stop report and trajectory.

The 50K stage completed 131 terminal random games and 10 truncated random games;
the remaining sampled prefix stopped at its position budget. Counts are position
checks, not 50K distinct games or unique boards. Unique full FEN counts were
10,119 and 49,874 including constructed histories; clocks are part of those
uniqueness counts. The seeded stages are separate, not a claim of 60K distinct
positions. Each supplements random coverage with a true four-cycle knight
repetition trace and a 150-ply quiet sparse-board trace that passes through
threefold claimability / 100-halfmove conditions and automatic draw boundaries.
The 50K stage recorded 12 >=threefold positions, 1,071 >=100-halfmove positions
and 13 >=150-halfmove positions including supplements.

No several-hundred-thousand-position stress run was launched. It remains
separate proposed work. No proof of all legal chess positions, collision-free
repetition or general FIDE dead-position recognition is claimed.

## Tactical search/sign checks: completed on CPU

`scripts/probe_mctx_tactics.py` uses uniform legal priors, zero leaf value,
fixed key 0, depth cap 8, no PUCT Dirichlet noise and Gumbel scale 0 (up to 64
considered actions). Actual mctx API/types are the already pinned installed ones.

- **Mate in one:** PUCT and Gumbel selected the verified mating move for White
  and Black at both 64 and 128 simulations: **8/8 probes passed**.
- **Mate in two:** exact offline three-ply oracle verified a forced mating move
  for each color and verified absence of mate in one. Both algorithms missed the
  selected fixture's solution at 64 and 128 simulations: **0/8 found**. These are
  supplementary finite-budget diagnostics, not correctness failures, and are
  reported separately in `m0_tactics.json`. No simulation budget is claimed to
  guarantee every forced mate under an uninformative evaluator.
- Automated tests additionally verify selected mate-in-one root Q = +1 and
  actual three-edge reward/discount propagation [+1,-1,+1] backward from a
  forced mate, for both colors. Eight new follow-up tests passed in 8.20 s.

A draft mate-in-two fixture initially also had mate in one; the fixture-label
assertion caught it. It was replaced with an independently verified fixture,
not treated as a rules failure. Results above refer only to the corrected probe.

## Large-batch benchmark: completed local portion; GPU pending

`benchmarks/bench_pgx_safe.py` considers 128, 1024, 4096 and 8192. It first checks
a conservative memory estimate against the current available-memory budget,
then compiles from abstract shapes and checks compiled buffers with a 2× factor
and 256 MiB reserve before allocating/executing. Measurements synchronize JAX,
separate trace/lower, compilation, warm-up and execution, and report host peak
RSS and device peak when the backend exposes it. Skips include explicit reasons.

Final local result (`m0_large_batch_cpu_final.json`): batch 128 measured about
19,732 steps/s, 0.460 s trace, 0.455 s compile, 0.0232 s warm-up; host peak
394,051,584 bytes. Earlier local measurements are retained separately. Batches 1K/4K/8K were **skipped before allocation** because the
conservative preflight exceeded 80% of the 853,239,398-byte current budget.
The final profiler also disables default GPU preallocation and caps its budget
against current nvidia-smi free capacity; if capacity is unknown, GPU execution
is skipped. CPU provides no device peak statistic: it is null, not an invented
GPU number. On GPU, peak is a process-cumulative high-water mark; use one batch
per fresh process when per-batch attribution is required.

A new Blackwell large-batch run has not been performed: no live Molab connection
is available here. The existing M0 GPU batch-128 result remains historical,
not a measurement of the new larger sizes. To run safely on Molab after syncing
these files and dependencies:

```bash
JAX_PLATFORMS=cuda .venv/bin/python benchmarks/bench_pgx_safe.py --output reports/m0_large_batch_blackwell.json
```

## Frozen head/encoder decisions

Main policy: **native 4672-action (8×8×73) output**. Attention-style policy is a
later ablation only. README Section 8.2 is updated; no transformer is implemented.
Canonical encoder: `pgx-v2-history8-ep-context8-v1`, defined in
`chess/state_features.py` and documented in `encoder_contract.md`: float32
planes (8,8,120), context (8,), bool mask (4672,). Shared pure-JAX function for
human replay and future self-play; full rules state remains the search embedding.
The M0 rules policy and codec versions are unchanged.

## Reproduce

```bash
uv sync --locked --extra test --extra data
PYTHONPATH=src .venv/bin/python scripts/check_pgx_parity.py --positions 10000 --seed 1000 --output reports/m0_parity_10k.json
PYTHONPATH=src .venv/bin/python scripts/check_pgx_parity.py --positions 50000 --seed 20000 --output reports/m0_parity_50k.json
PYTHONPATH=src .venv/bin/python scripts/probe_mctx_tactics.py
.venv/bin/python -m pytest tests/test_m0_followup.py -q
```

Disposition: targeted legality/history/reward/sign gates passed; finite-budget
mate-in-two discovery is limited; unsafe local sizes skipped; new GPU memory /
large-batch benchmarks and broader stress remain unexecuted. M1 proceeds only
within the authorized 100→1000-game smoke gate, never 50K extraction or M2/RL.

Final full regression evidence: 117 tests passed in 51.36 s, and the safe-path
rerun passed 117 in 50.99 s. See the M1 smoke report for the completed human-data
gate and limits; no new GPU throughput/memory result is inferred from those CPU tests.
