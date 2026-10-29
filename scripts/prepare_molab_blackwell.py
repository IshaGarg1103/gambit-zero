"""Build a small source-only Molab input ZIP; never include dataset/cache files."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build(output):
    paths = set()
    for folder in ['src', 'tests', 'scripts', 'benchmarks', 'configs', 'licenses']:
        for path in (ROOT/folder).rglob('*'):
            if any(part in ['.venv', '.git', '__pycache__', '.uv-cache'] for part in path.parts):
                continue
            if path.is_file() and path.suffix in ['.py', '.md', '.txt', '.json', '.yaml', '.toml', '.lock', '.patch']:
                paths.add(path)
    for name in ['README.md', 'm1execution.md', 'pyproject.toml', 'uv.lock', '.python-version',
                 'requirements-lock.txt', 'requirements-cuda-lock.txt',
                 'reports/m1_validation_summary.json', 'reports/m1_final_acceptance.json',
                 'reports/m1_dataset_d_freeze.json', 'reports/pgx_m1_pin.json',
                 'reports/m1_sources_50k.json', 'reports/encoder_contract.md']:
        path = ROOT/name
        if path.is_file():
            paths.add(path)
    ref = (ROOT/'.git/HEAD').read_text().strip()
    source = None
    if ref.startswith('ref: '):
        ref_path = ROOT/'.git'/ref[5:]
        if ref_path.exists():
            source = ref_path.read_text().strip()
    else:
        source = ref
    manifest = dict(format='gambit-zero-blackwell-input-v1',
                    source_base_commit=source, source_tag='m1-dataset-50k-v1',
                    source_note='Uncommitted benchmark follow-up overlay; exact included bytes pinned below.',
                    dataset_revision='b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c',
                    files={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(paths)})
    for path in paths:
        if not path.resolve().is_relative_to(ROOT) or path.stat().st_size > 2**20:
            raise RuntimeError('Unsafe or unexpectedly large source input: '+str(path))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(paths):
            info = zipfile.ZipInfo('tinychess-m1-blackwell/'+path.relative_to(ROOT).as_posix(), (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, path.read_bytes())
        info = zipfile.ZipInfo('tinychess-m1-blackwell/BLACKWELL_BUNDLE.json', (1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        bundle.writestr(info, json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    cell = '''# Paste this entire cell into the fresh Molab GPU notebook.
from pathlib import Path
import hashlib
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile

_bw_zip = Path("/marimo/@@ZIP_NAME@@")  # Edit only if your upload is elsewhere.
_bw_expected = "@@SHA256@@"
if not _bw_zip.is_file():
    raise FileNotFoundError(f"Upload the input ZIP first, then set its actual path: {_bw_zip}")
if hashlib.sha256(_bw_zip.read_bytes()).hexdigest() != _bw_expected:
    raise RuntimeError("Input ZIP checksum mismatch; use the matching newly prepared ZIP and cell")
_bw_parent = Path("/marimo") if Path("/marimo").is_dir() else Path.cwd()
_bw_extract = Path(tempfile.mkdtemp(prefix="gambit-zero-blackwell-", dir=_bw_parent))
with zipfile.ZipFile(_bw_zip) as _bw_bundle:
    for _bw_member in _bw_bundle.infolist():
        _bw_destination = (_bw_extract/_bw_member.filename).resolve()
        if not _bw_destination.is_relative_to(_bw_extract.resolve()) or stat.S_ISLNK(_bw_member.external_attr >> 16):
            raise RuntimeError("Unsafe input ZIP member")
    _bw_bundle.extractall(_bw_extract)
_bw_project = _bw_extract/"tinychess-m1-blackwell"
_bw_env = dict(os.environ, BLACKWELL_INPUT_ZIP_SHA256=_bw_expected)
_bw_run = subprocess.run([sys.executable, "-u", str(_bw_project/"scripts/molab_m1_blackwell.py")],
                         cwd=_bw_project, env=_bw_env)
_bw_result = _bw_extract/"gambit-zero-m1-blackwell-results.zip"
if _bw_result.is_file():
    _bw_download = _bw_parent/_bw_result.name
    shutil.copy2(_bw_result, _bw_download)
    print(f"DOWNLOAD AND SHARE: {_bw_download}")
else:
    print(f"No results ZIP was produced; share the error above. Working directory: {_bw_project}")
print(f"Runner exit code: {_bw_run.returncode}. A nonzero exit means failure/incomplete; still share its ZIP.")
'''.replace('@@ZIP_NAME@@', output.name).replace('@@SHA256@@', checksum)
    cell_path = output.parent/'molab_m1_blackwell_cell.py'
    cell_path.write_text(cell)
    print(json.dumps(dict(path=str(output.resolve()), bytes=output.stat().st_size,
                          sha256=checksum, files=len(paths), notebook_cell=str(cell_path.resolve())), indent=2))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/gambit-zero-m1-blackwell-input.zip')
    build(parser.parse_args().output)
