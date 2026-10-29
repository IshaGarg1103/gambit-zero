# M1 source handoff — 29 October 2026

Repository: https://github.com/IshaGarg1103/gambit-zero
Target branch: `m1/human-dataset`.

The user will execute all further Git commands. Before that correction, the
agent fetched origin and renamed the local `human-dataset` branch to
`m1/human-dataset`, preserving its upstream. No staging, commit, push or merge
was performed by the agent. A new commit hash and remote verification are
therefore pending the user's commands below. The implementation and evidence
are prepared for review; no source commit is falsely embedded as its own hash.

## Verified dataset and regression

- Published data: https://huggingface.co/datasets/ishagarg1103/gambit-zero
- Immutable HF revision: `b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c`.
- 50K games: 35K train / 7.5K validation / 7.5K test.
- 3,793,600 retained positions; 150 exact evaluation-overlap training rows removed.
- Zero retained train/Dataset D board overlap, overall and at ply >=10.
- Dataset D: 1K puzzles / 32 opening lines / 4,178 unique evaluation boards.
- Final card, manifest, nine month manifests and 796 primary shard content IDs
  independently verified; changed shards had full streamed read-back during upload.
- Full regression: **241 passed in 51.11 seconds**, recorded in
  `m1_closeout_tests.txt` using `.venv/bin/python -m pytest -q`.
- Dependency lock check: `UV_CACHE_DIR=.uv-cache uv lock --check --offline` passed;
  76 packages resolved without installing dependencies.
- No generated dataset files remain in the managed `data/` tree.

CPU dataset checks PASS. Overall M1 remains CONDITIONAL PASS with the separate
Blackwell benchmark pending and historical telemetry gaps documented. No M2,
model training, self-play, larger extraction or merge into main is authorized.

## Prepared source and evidence

Include root dependency locks/configuration and README, `m1execution.md`,
`src/`, `tests/`, `configs/`, `scripts/`, `benchmarks/`, `reports/` and `docs/`.
The handoff includes streaming/day-rating sampling, HTTP recovery, HF upload
batching/read-back/atomic publication, evaluation freezing/exclusions, independent
audits, regression tests, source manifests and acceptance reports.
`m1_source_handoff_review.json` inventories the reviewed visible files and their
hashes; it excludes itself to avoid a circular checksum. It is a source-tree
review inventory, not a claim that every listed unchanged file is newly committed.
The user's staged Git diff is the authoritative new-commit file list.

Original run templates now use relative project paths and are disabled by
default. Frozen original runtime/config bytes remain in the checksummed HF
recovery backup. These portability/safety changes do not rerun extraction or
alter the accepted dataset, encoder, rules or action codec.

## Intentionally excluded, preserved locally

- `.venv/`, caches, generated `data/` contents and checkpoints.
- Raw compressed PGNs/CSV exports, Parquet shards, compressed JSONL payloads
  and model weight/checkpoint files.
- Credentials, `.env` files, private keys and local HF caches.
- Worker `.log`, launchd `.plist`, lock files and `m1_january_job.json`.
- The loose `Untitled` draft; its full historical text is preserved in
  `docs/m1_initial_checklist.md`, with the current specification clearly linked.

These exclusions are recorded in `.gitignore`. Small historical JSON measurement,
retry and recovery reports are retained as provenance, not active jobs. No
actual dataset shard or complete source archive belongs in GitHub.

## User-run commands, from the project root

Review the staged summary before running the commit command. No force push or
merge is needed.

```bash
git switch m1/human-dataset
git status --short --branch
git add -- .gitignore README.md pyproject.toml uv.lock requirements-lock.txt requirements-cuda-lock.txt m1execution.md src/ tests/ configs/ scripts/ benchmarks/ reports/ docs/
git diff --cached --check
git diff --cached --stat
git diff --cached --name-only
git commit -m "feat(m1): finalize 50k human chess dataset and validation pipeline"
git push -u origin m1/human-dataset
git fetch origin
git rev-parse HEAD
git rev-parse origin/m1/human-dataset
git show --stat --oneline HEAD
git status --short --branch
```

The two revision hashes must match after the fetch. The final status should have
no staged/unstaged/untracked source changes; ignored local operational files may
remain. The commit URL is the repository URL plus `/commit/<printed HEAD hash>`.
The branch URL is https://github.com/IshaGarg1103/gambit-zero/tree/m1/human-dataset.
If Git reports incoming remote commits or a non-fast-forward push, stop and
inspect that difference; do not force-push or merge into main.
