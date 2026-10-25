# Manual Molab M0 run

Upload `tinychess-m0-molab.zip` into the notebook's working directory and select
the RTX PRO 6000 Blackwell GPU runtime. The bundle includes local project files
that have not been pushed to GitHub; cloning the empty repository is insufficient.
Run this single Python cell:

```python
from pathlib import Path
import zipfile

_archive = Path("tinychess-m0-molab.zip").resolve()
tinychess_project = (Path.cwd() / "tinychess-m0").resolve()
assert _archive.is_file(), f"Upload {_archive.name} here first"
tinychess_project.mkdir(exist_ok=True)
with zipfile.ZipFile(_archive) as _zip:
    for _entry in _zip.infolist():
        assert (tinychess_project / _entry.filename).resolve().is_relative_to(tinychess_project)
    _zip.extractall(tinychess_project)
exec(compile((tinychess_project / "scripts/molab_m0.py").read_text(), "molab_m0.py", "exec"))
```

This installs two isolated Python 3.11.15 environments from lockfiles, without
changing the notebook kernel's JAX. It requires CUDA and checks the actual device
name. It runs the TinyChess tests and batch benchmark, then bounded native and
wrapped official-example routines. It stops on errors. The first compilation
and dependency installation can take several minutes. CUDA wheels use several GB
of disk; no chess archives or model checkpoints are downloaded.

Download `tinychess-m0-blackwell-results.zip` after it prints DONE. On an error,
share the traceback and the named log from `tinychess-m0/reports/`.

The official unmodified chess CLI is expected to fail at unsupported
`chess_v0` baseline loading. The harness captures that exact failure, then removes
only that unused assignment and calls upstream routines with a tiny config.
It never calls evaluation, W&B initialization or checkpoint code. One tiny
optimizer update is a compatibility probe, not a TinyChess training run.
