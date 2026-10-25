"""Notebook resume after the official example's sibling-import failure."""
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

_resume_root = Path(tinychess_project).resolve()
_resume_script = _resume_root / "scripts/run_official_pgx_example.py"
_resume_old = 'envvars = {**os.environ, "WANDB_MODE": "disabled"}'
_resume_new = 'envvars = {**os.environ, "WANDB_MODE": "disabled", "PYTHONPATH": str(UPSTREAM)}'
_resume_source = _resume_script.read_text()
if _resume_old in _resume_source:
    _resume_script.write_text(_resume_source.replace(_resume_old, _resume_new, 1))
elif _resume_new not in _resume_source:
    raise RuntimeError("Runner differs from the expected version; inspect before patching")

for _evidence in ["stack_blackwell.json", "pytest_blackwell.txt", "pgx_batching_blackwell.json"]:
    if not (_resume_root / "reports" / _evidence).is_file():
        raise RuntimeError(f"Missing {_evidence}; use the full M0 runner first")

_resume_env = {**os.environ, "JAX_PLATFORMS": "cuda", "WANDB_MODE": "disabled",
               "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
               "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]}
_resume_env.pop("PYTHONPATH", None)
_resume_env.pop("VIRTUAL_ENV", None)
_resume_py = _resume_root / "benchmarks/official_pgx/.venv/bin/python"
for _backend in ["native", "wrapped"]:
    print(f"Running official example: {_backend}", flush=True)
    _log = _resume_root / f"reports/official_pgx_{_backend}_blackwell.log"
    _backup = _log.with_suffix(".before-import-fix.log")
    if _log.exists() and not _backup.exists():
        shutil.copy2(_log, _backup)
    _command = [_resume_py, "scripts/run_official_pgx_example.py", "--require-gpu",
                "--backend", _backend, "--output", f"reports/official_pgx_{_backend}_blackwell.json"]
    with _log.open("w") as _stream:
        _process = subprocess.Popen(list(map(str, _command)), cwd=_resume_root, env=_resume_env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for _line in _process.stdout:
            print(_line, end="", flush=True)
            _stream.write(_line)
            _stream.flush()
        if _process.wait():
            raise RuntimeError(f"Command failed; inspect {_log}")

_resume_results = _resume_root.parent / "tinychess-m0-blackwell-results.zip"
with zipfile.ZipFile(_resume_results, "w", zipfile.ZIP_DEFLATED) as _archive:
    for _path in sorted((_resume_root / "reports").glob("*blackwell*")):
        if _path.is_file():
            _archive.write(_path, _path.name)
print(f"DONE. Download and share: {_resume_results}")
