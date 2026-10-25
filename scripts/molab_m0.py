"""Run from a Molab notebook with exec(Path(...).read_text()).

Assumes this file is already extracted inside the project. Uses subprocesses
so the notebook kernel's JAX installation remains untouched. Stop on any error.
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

# The notebook cell sets this to the extracted project directory.
_project = Path(tinychess_project).resolve()
assert (_project / "uv.lock").is_file(), f"Not a TinyChess bundle: {_project}"
if importlib.util.find_spec("uv") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "uv==0.10.9"], check=True)
_uv = [sys.executable, "-m", "uv"]
_env = {**os.environ, "JAX_PLATFORMS": "cuda", "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "WANDB_MODE": "disabled",
        "UV_CACHE_DIR": str(_project / ".uv-cache")}
# Do not inherit notebook-specific overrides into an isolated interpreter.
_env.pop("PYTHONPATH", None)
_env.pop("VIRTUAL_ENV", None)


def _run(command, log_name=None):
    print("\nRUN:", " ".join(map(str, command)), flush=True)
    if log_name is None:
        subprocess.run(list(map(str, command)), cwd=_project, env=_env, check=True)
        return
    with (_project / "reports" / log_name).open("w") as log:
        process = subprocess.Popen(list(map(str, command)), cwd=_project, env=_env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise RuntimeError(f"Command failed; inspect reports/{log_name}")


if shutil.which("nvidia-smi") is None:
    raise RuntimeError("No nvidia-smi: select a Molab GPU runtime before proceeding")
_run(["nvidia-smi"])
_run(_uv + ["sync", "--project", str(_project), "--locked", "--python", "3.11.15", "--extra", "test", "--extra", "cuda"])
_py = _project / ".venv/bin/python"
_run([_py, "scripts/inspect_stack.py", "--require-gpu", "--output", "reports/stack_blackwell.json"])
_stack = json.loads((_project / "reports/stack_blackwell.json").read_text())
_kinds = " ".join(d["kind"] for d in _stack["devices"])
if "6000" not in _kinds or "Blackwell" not in _kinds:
    raise RuntimeError(f"Expected RTX PRO 6000 Blackwell; actual JAX device(s): {_kinds}")
_run([_py, "-m", "pytest", "-q"], "pytest_blackwell.txt")
_run([_py, "benchmarks/bench_pgx_batching.py", "--output", "reports/pgx_batching_blackwell.json"], "pgx_batching_blackwell.log")

_audit = _project / "benchmarks/official_pgx"
_run(_uv + ["sync", "--project", str(_audit), "--locked", "--python", "3.11.15", "--extra", "cuda"])
_audit_py = _audit / ".venv/bin/python"
for _backend in ["native", "wrapped"]:
    _run([_audit_py, "scripts/run_official_pgx_example.py", "--require-gpu", "--backend", _backend,
          "--output", f"reports/official_pgx_{_backend}_blackwell.json"], f"official_pgx_{_backend}_blackwell.log")

_results = _project.parent / "tinychess-m0-blackwell-results.zip"
with zipfile.ZipFile(_results, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted((_project / "reports").glob("*blackwell*")):
        if path.is_file():
            archive.write(path, path.name)
print(f"\nDONE. Download and share: {_results}")
print("No dataset or checkpoint was saved. M0 review remains required before later milestones.")
