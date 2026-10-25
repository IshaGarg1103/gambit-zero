"""Bounded archive prefixes and streaming game framing, with immutable provenance."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import urllib.request

import zstandard as zstd


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch_prefix(month, config):
    source = config['source']
    cache = Path(source['cache'])
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f'{month}.pgn.zst.prefix'
    meta_path = path.with_suffix(path.suffix + '.json')
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if sha256_file(path) != meta['prefix_sha256']:
            raise RuntimeError('Cached prefix checksum mismatch; do not silently resume different bytes')
        if meta['url'] != source['url_template'].format(month=month) or meta['prefix_bytes'] != source['compressed_prefix_bytes']:
            raise RuntimeError('Cached source/config mismatch')
        return path, meta
    budget = source['compressed_prefix_bytes']
    existing = sum(p.stat().st_size for p in cache.glob('*.prefix'))
    if existing + budget > source['max_total_download_bytes']:
        raise RuntimeError('Compressed source budget exceeded')
    url = source['url_template'].format(month=month)
    request = urllib.request.Request(url, headers={'Range': f'bytes=0-{budget-1}', 'User-Agent': 'GambitZero-M1-Smoke/1.0'})
    tmp = path.with_suffix(path.suffix + '.partial')
    with urllib.request.urlopen(request, timeout=60) as response, tmp.open('wb') as out:
        headers = response.headers
        content_range = headers.get('Content-Range')
        if response.status == 206 and (not content_range or not content_range.startswith('bytes 0-')):
            raise RuntimeError('Response is not a prefix starting at compressed offset zero')
        size = int(content_range.rsplit('/', 1)[1]) if content_range else int(headers.get('Content-Length', 0)) or None
        remaining = budget
        while remaining:
            chunk = response.read(min(65536, remaining))
            if not chunk:
                break
            out.write(chunk)
            remaining -= len(chunk)
        meta = dict(month=month, url=url, archive_size_bytes=size, etag=headers.get('ETag'),
                    last_modified=headers.get('Last-Modified'), response_status=response.status,
                    content_range=content_range, retrieved_utc=datetime.now(timezone.utc).isoformat(),
                    prefix_bytes=budget-remaining, prefix_start=0, archive_checksum_verified=False,
                    sampling='sequential_prefix_smoke_only', chronology_bias=True)
    if remaining:
        raise RuntimeError('Archive prefix shorter than configured budget')
    tmp.replace(path)
    meta['prefix_sha256'] = sha256_file(path)
    # Published checksum is provenance for the full archive; a prefix cannot verify it.
    sums_path = cache / 'sha256sums.txt'
    if not sums_path.exists():
        with urllib.request.urlopen(source['checksum_url'], timeout=30) as response:
            sums = response.read(1024*1024+1)
            if len(sums) > 1024*1024:
                raise RuntimeError('Checksum index exceeded bounded read')
            sums_path.write_bytes(sums)
    entries = dict((line.split()[1].lstrip('*'), line.split()[0]) for line in sums_path.read_text().splitlines() if len(line.split()) == 2)
    meta['published_archive_sha256'] = entries.get(url.rsplit('/', 1)[1])
    meta['checksum_index_url'] = source['checksum_url']
    meta['checksum_index_sha256'] = sha256_file(sums_path)
    meta_path.write_text(json.dumps(meta, sort_keys=True, indent=2)+'\n')
    return path, meta


@contextmanager
def open_text(path):
    path = Path(path)
    with path.open('rb') as source:
        if path.name.endswith(('.zst', '.zst.prefix')):
            with zstd.ZstdDecompressor().stream_reader(source, read_size=65536) as reader:
                with io.TextIOWrapper(reader, encoding='utf-8', errors='replace') as text:
                    yield text
        else:
            with io.TextIOWrapper(source, encoding='utf-8', errors='replace') as text:
                yield text


def game_blocks(text, max_game_bytes=2**20, max_line_chars=65536):
    """Lichess framing: Event is first header. Resync after malformed games.

    Bound line and game size. Oversized blocks are rejected as one record rather
    than accumulated. This is an archive-specific framer, not all possible PGN.
    """
    lines, size, bad = [], 0, None
    for line in iter(lambda: text.readline(max_line_chars + 1), ''):
        if line.startswith('[Event '):
            if lines or bad:
                yield ''.join(lines), bad
            lines, size, bad = [], 0, None
        if len(line) > max_line_chars:
            bad = 'overlong_line'
            while line and not line.endswith('\n'):
                line = text.readline(max_line_chars + 1)
        size += len(line.encode('utf-8'))
        if size > max_game_bytes:
            bad = 'overlong_game'
        if '\ufffd' in line:
            bad = 'invalid_utf8'
        if not bad:
            lines.append(line)
    if lines or bad:
        yield ''.join(lines), bad
