"""Bounded, read-only source/resource preflight. Never starts archive extraction.

Network probes start at byte zero and are discarded. Partial compressed bytes
cannot verify a full-archive checksum or establish archive-wide seekability.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time
import urllib.request

import psutil
import zstandard as zstd

from tinychess.data.sources import game_blocks, open_text

BASE = 'https://database.lichess.org/standard/'
MONTHS = [f'2026-{i:02}' for i in range(1, 10)]
PROBE_BYTES = 8 * 1024**2


def estimated_seconds(size, rate):
    if size < 0 or rate <= 0:
        raise ValueError('Size must be nonnegative and rate positive')
    return size / rate


def request(url, **kwargs):
    return urllib.request.urlopen(urllib.request.Request(
        url, headers={'User-Agent': 'GambitZero-M1-Preflight/1.0',
                      'Accept-Encoding': 'identity', **kwargs.pop('headers', {})},
        **kwargs), timeout=30)


def source_metadata(month):
    url = BASE + f'lichess_db_standard_rated_{month}.pgn.zst'
    with request(url, method='HEAD') as response:
        # Some HTTP caches return a cached partial response even to HEAD.
        content_range = response.headers.get('Content-Range')
        size = int(content_range.rsplit('/', 1)[1] if content_range
                   else response.headers['Content-Length'])
        return dict(month=month, url=url, archive_size_bytes=size,
                    status=response.status, headers=dict(response.headers))


def throughput_probe(url):
    started = time.perf_counter()
    digest = hashlib.sha256()
    with request(url, headers={'Range': f'bytes=0-{PROBE_BYTES-1}'}) as response:
        if response.status != 206 or not response.headers.get('Content-Range', '').startswith('bytes 0-'):
            raise RuntimeError('Probe requires a bounded offset-zero range response')
        header_seconds = time.perf_counter() - started
        body_started = time.perf_counter()
        size = 0
        while size < PROBE_BYTES:
            chunk = response.read(min(65536, PROBE_BYTES-size))
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        body_seconds = time.perf_counter() - body_started
        if size != PROBE_BYTES:
            raise RuntimeError('Short throughput probe; do not extrapolate as complete')
        return dict(bytes=size, request_and_read_seconds=time.perf_counter()-started,
                    header_seconds=header_seconds, body_seconds=body_seconds,
                    body_bytes_per_second=size/body_seconds,
                    end_to_end_bytes_per_second=size/(time.perf_counter()-started),
                    prefix_sha256=digest.hexdigest(), full_archive_verified=False,
                    content_range=response.headers['Content-Range'])


def inspect_cached_prefix(path):
    started = time.perf_counter()
    games, eligible, strong = 0, 0, 0
    # A fast header-only count for scan-cost diagnostics, not a sampler.
    with open_text(path) as stream:
        for text, bad in game_blocks(stream):
            games += 1
            if bad:
                continue
            header_text = text.split('\n\n', 1)[0]
            headers = dict(re.findall(r'^\[(\w+) "([^"\\]*)"\]\s*$', header_text, re.M))
            try:
                rating = min(int(headers['WhiteElo']), int(headers['BlackElo']))
                tc = re.fullmatch(r'(\d+)(?:\+\d+)?', headers.get('TimeControl', ''))
                ok = (headers.get('Event', '').startswith('Rated ')
                      and headers.get('Variant', 'Standard') == 'Standard'
                      and headers.get('Rated', 'true').lower() not in ['false', '0']
                      and rating >= 1800 and tc and int(tc[1]) >= 300
                      and headers.get('Result') in ['1-0', '0-1', '1/2-1/2']
                      and headers.get('Termination') not in ['Abandoned', 'Rules infraction', 'Aborted']
                      and 'FEN' not in headers and headers.get('SetUp') != '1')
                eligible += bool(ok)
                strong += bool(ok and rating >= 2000)
            except (KeyError, ValueError):
                pass
    # Inspect a prefix for an actual completed frame, without inventing seekability.
    decoder = zstd.ZstdDecompressor().decompressobj()
    consumed = 0
    produced = 0
    with path.open('rb') as source:
        while not decoder.eof:
            chunk = source.read(65536)
            if not chunk:
                break
            consumed += len(chunk)
            produced += len(decoder.decompress(chunk))
    elapsed = time.perf_counter()-started
    return dict(path=str(path), compressed_bytes=path.stat().st_size,
                framed_blocks_including_partial_tail=games,
                header_eligible_candidates_including_possible_partial_tail=eligible,
                header_strong_candidates=strong, elapsed_seconds=elapsed,
                compressed_bytes_per_second=path.stat().st_size/elapsed,
                complete_first_frame_seen=decoder.eof,
                first_frame_decompressed_bytes=produced,
                first_frame_end_compressed_offset=(consumed-len(decoder.unused_data)
                                                   if decoder.eof else None),
                archive_wide_frame_structure_verified=False,
                representative_of_full_month=False)


def main(output):
    started = time.perf_counter()
    result = dict(status='PREFLIGHT_ONLY', retrieved_utc=datetime.now(timezone.utc).isoformat(),
                  platform=platform.platform(), disk=shutil.disk_usage('.')._asdict(),
                  memory=psutil.virtual_memory()._asdict(),
                  network_probe_payload_cap_bytes=3*PROBE_BYTES + 1024**2,
                  full_archive_scans_started=0)
    try:
        quota = subprocess.run(['quota', '-v'], capture_output=True, text=True, timeout=10)
        result['quota'] = dict(returncode=quota.returncode, stdout=quota.stdout, stderr=quota.stderr,
                               authoritative_usable_quota_confirmed=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        result['quota'] = dict(error=str(exc), authoritative_usable_quota_confirmed=False)
    # Read Git identity as files: no Git commands or repository mutations.
    head = Path('.git/HEAD').read_text().strip()
    result['git_head'] = head
    ref = Path('.git') / head.removeprefix('ref: ')
    result['commit'] = ref.read_text().strip() if ref.is_file() else None
    try:
        with request(BASE+'sha256sums.txt') as response:
            sums = response.read(1024**2+1)
            if len(sums) > 1024**2:
                raise RuntimeError('Checksum index exceeds bounded read')
        result['checksum_index'] = dict(url=BASE+'sha256sums.txt', bytes=len(sums),
                                        sha256=hashlib.sha256(sums).hexdigest())
        checksums = {parts[1].lstrip('*'): parts[0] for line in sums.decode().splitlines()
                     if len(parts := line.split()) == 2}
        with ThreadPoolExecutor(max_workers=3) as pool:
            result['sources'] = list(pool.map(source_metadata, MONTHS))
        for source in result['sources']:
            source['published_sha256'] = checksums.get(source['url'].rsplit('/', 1)[1])
            source['full_archive_checksum_verified'] = False
        result['total_source_bytes'] = sum(s['archive_size_bytes'] for s in result['sources'])
        result['probes'] = []
        for i in range(3):
            probe = throughput_probe(result['sources'][i*4]['url'])
            probe['url'] = result['sources'][i*4]['url']
            result['probes'].append(probe)
            print(json.dumps(dict(probe=i+1, **probe)), flush=True)
        rates = [p['end_to_end_bytes_per_second'] for p in result['probes']]
        result['network_estimated_seconds_range'] = [estimated_seconds(result['total_source_bytes'], r)
                                                     for r in [max(rates), min(rates)]]
        result['network_archive_payload_measured_bytes'] = sum(p['bytes'] for p in result['probes'])
        result['network_http_overhead_measured'] = False
        result['cached_prefix_scan'] = [inspect_cached_prefix(p) for p in sorted(
            Path('data/raw/m1_prefixes').glob('*.pgn.zst.prefix'))]
        result['scan_estimated_seconds_range_prefix_only'] = [estimated_seconds(result['total_source_bytes'], r)
            for r in [max(p['compressed_bytes_per_second'] for p in result['cached_prefix_scan']),
                      min(p['compressed_bytes_per_second'] for p in result['cached_prefix_scan'])]] if result['cached_prefix_scan'] else None
        result['status'] = 'MEASURED_RESOURCE_GATE_NOT_CONFIRMED'
    except Exception as exc:
        result['status'] = 'PREFLIGHT_INCOMPLETE'
        result['error'] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        result['elapsed_seconds'] = time.perf_counter()-started
        Path(output).write_text(json.dumps(result, sort_keys=True, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='reports/m1_50k_preflight.json')
    main(parser.parse_args().output)
