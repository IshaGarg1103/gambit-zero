"""Prepare the exact source overlay and manual Molab cell; never package data."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PREFIX='gambit-zero-stage2b'


def build(output,root=ROOT):
    root=Path(root);paths=set()
    for folder in ['src','tests','scripts','benchmarks','configs','licenses']:
        for p in (root/folder).rglob('*'):
            if any(x in ['.git','.venv','__pycache__','.uv-cache','.uv-python','build','dist']
                   or x.endswith('.egg-info') for x in p.parts):continue
            if p.is_file() and p.suffix in ['.py','.json','.yaml','.md','.txt','.toml','.lock','.patch']:paths.add(p)
    for name in ['README.md','m1execution.md','m2execution.md','m2_revised_flow.md','pyproject.toml','uv.lock',
                 '.python-version','requirements-lock.txt','requirements-cuda-lock.txt','reports/m2_stage2a.md',
                 'reports/m1_validation_summary.json','reports/m1_final_acceptance.json','reports/m1_dataset_d_freeze.json',
                 'reports/pgx_m1_pin.json','reports/encoder_contract.md']:
        p=root/name
        if not p.exists():raise FileNotFoundError(f'Required source/specification missing: {name}')
        paths.add(p)
    for p in paths:
        if not p.resolve().is_relative_to(root.resolve()) or p.stat().st_size>2**20:raise ValueError('Unsafe or oversized source input')
    ref=(root/'.git/HEAD').read_text().strip();commit=(root/'.git'/ref[5:]).read_text().strip() if ref.startswith('ref: ') else ref
    manifest=dict(format='gambit-zero-stage2b-input-v1',source_base_commit=commit,
        source_note='Exact working tree overlay; uncommitted Stage 2B files are pinned by these hashes.',
        dataset_revision='b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c',
        files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)})
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        payloads={PREFIX+'/'+p.relative_to(root).as_posix():p.read_bytes() for p in sorted(paths)}
        payloads[PREFIX+'/BLACKWELL_BUNDLE.json']=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
        for name,payload in sorted(payloads.items()):
            info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,payload)
    digest=hashlib.sha256(output.read_bytes()).hexdigest()
    cell='''# Upload the input ZIP to /marimo, then run this entire cell.
from pathlib import Path
import hashlib
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import zipfile

_s2b_zip = Path("/marimo/@@NAME@@")  # Set the actual upload path if different.
_s2b_hash = "@@HASH@@"
if not _s2b_zip.is_file():
    raise FileNotFoundError(f"Upload the source ZIP first, then set its actual path: {_s2b_zip}")
if hashlib.sha256(_s2b_zip.read_bytes()).hexdigest() != _s2b_hash:
    raise RuntimeError("Input ZIP checksum mismatch: use its matching generated notebook cell")
_s2b_parent = Path("/marimo") if Path("/marimo").is_dir() else Path.cwd()
_s2b_extract = Path(tempfile.mkdtemp(prefix="gambit-zero-stage2b-", dir=_s2b_parent))
with zipfile.ZipFile(_s2b_zip) as _s2b_bundle:
    if sum(_m.file_size for _m in _s2b_bundle.infolist()) > 20 * 2**20:
        raise RuntimeError("Source ZIP exceeds expected extraction bound")
    for _m in _s2b_bundle.infolist():
        _dest = (_s2b_extract/_m.filename).resolve()
        if not _dest.is_relative_to(_s2b_extract.resolve()) or stat.S_ISLNK(_m.external_attr >> 16):
            raise RuntimeError("Unsafe source ZIP member")
    _s2b_bundle.extractall(_s2b_extract)
_s2b_project = _s2b_extract/"gambit-zero-stage2b"
_s2b_process = subprocess.Popen([sys.executable, "-u", str(_s2b_project/"scripts/molab_stage2b.py")],
    cwd=_s2b_project, env=dict(os.environ, STAGE2B_INPUT_ZIP_SHA256=_s2b_hash),
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
try:
    for _s2b_line in _s2b_process.stdout:
        print(_s2b_line, end="", flush=True)
    _s2b_returncode = _s2b_process.wait()
except BaseException:
    os.killpg(_s2b_process.pid, signal.SIGINT)
    try:
        _s2b_process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(_s2b_process.pid, signal.SIGKILL)
        _s2b_process.wait()
    raise
_s2b_results_zip = _s2b_extract/"gambit-zero-stage2b-results.zip"
if _s2b_results_zip.is_file():
    _s2b_download = _s2b_parent/"gambit-zero-stage2b-results.zip"
    shutil.copy2(_s2b_results_zip, _s2b_download)
    print(f"DOWNLOAD AND SHARE: {_s2b_download}")
    print("RESULT SHA256:", hashlib.sha256(_s2b_download.read_bytes()).hexdigest())
    try:
        import marimo as _s2b_mo
        _s2b_mo.output.append(_s2b_mo.download(data=_s2b_download.read_bytes(),
            filename=_s2b_download.name, mimetype="application/zip", label="Download Stage 2B results"))
    except (ImportError, AttributeError):
        print("Download the results from the path above.")
else:
    print(f"No results ZIP produced; share the error. Work directory: {_s2b_project}")
print(f"Runner exit code: {_s2b_returncode}. Nonzero means failed/incomplete; still return the results ZIP.")
'''.replace('@@NAME@@',output.name).replace('@@HASH@@',digest)
    cell_path=output.parent/'molab_stage2b_cell.py';cell_path.write_text(cell)
    return dict(zip=str(output.resolve()),bytes=output.stat().st_size,sha256=digest,notebook_cell=str(cell_path.resolve()),files=len(paths))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=ROOT/'artifacts/gambit-zero-stage2b-input.zip')
    print(json.dumps(build(p.parse_args().output),indent=2))
