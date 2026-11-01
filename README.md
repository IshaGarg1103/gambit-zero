# Gambit Zero

A research workspace for compact neural models and reproducible learning systems.

The project brings together validated data pipelines, transformer components,
numerical correctness checks and measured CPU/GPU experiments. Work progresses
through reviewed milestones, with configurations and evidence recorded alongside
implementation changes.

## Focus

- Deterministic data processing with versioned manifests and checksum verification.
- Model implementation using JAX, Flax and Optax.
- Independent correctness tests and regression coverage.
- Resource-aware benchmarking and experiment tracking.

## Repository layout

| Directory | Contents |
|---|---|
| `src/` | Core implementation |
| `tests/` | Automated correctness and regression tests |
| `configs/` | Versioned experiment settings |
| `scripts/` | Validation, packaging and reporting tools |
| `benchmarks/` | Performance measurement tools |
| `reports/` | Accepted results and verification summaries |
| `docs/` | Setup and execution guides |

## Development setup

Python and dependency versions are pinned in `.python-version`, `pyproject.toml`
and the committed lockfiles. Using `uv`:

```bash
uv sync --locked --extra test --extra data
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q
```

The first setup may require network access to obtain the pinned interpreter and
packages. GPU execution uses the separately documented CUDA environment and
hardware preflight checks.

## Experiment workflow

Keep seeds, dependency versions and input revisions fixed for reproducible runs.
Record compilation separately from execution, validate numerical behavior and
retain measured results with their configuration. Credentials, local caches,
datasets and checkpoints stay outside source control.

Detailed project specifications and historical milestone notes are preserved in
[PROJECT_PLAN.md](PROJECT_PLAN.md).
