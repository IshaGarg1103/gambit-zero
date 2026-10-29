"""Stream a bounded puzzle snapshot and freeze independent evaluation fixtures.

Default is a local resource plan. --execute streams the source; --publish uses
verified HF uploads. No Dataset A data is read or modified by this command.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import urllib.request

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import yaml
import zstandard as zstd

from tinychess.data.archive_scan import canonical
from tinychess.data.frozen_eval import PuzzleSampler, evaluation_positions, select_openings
from tinychess.data.hf_month import read_frozen_d
from tinychess.data.hf_upload import VerifiedUploader


class HashingReader(io.RawIOBase):
    def __init__(self, response, size, deadline):
        self.response, self.size, self.deadline = response, size, deadline
        self.digest, self.count = hashlib.sha256(), 0

    def readable(self):
        return True

    def readinto(self, buffer):
        if time.monotonic() > self.deadline:
            raise RuntimeError('Puzzle-stream deadline reached; no snapshot accepted')
        payload = self.response.read(min(len(buffer), 65536))
        self.count += len(payload)
        if self.count > self.size:
            raise ValueError('Puzzle stream exceeds frozen compressed-byte bound')
        self.digest.update(payload)
        buffer[:len(payload)] = payload
        return len(payload)


def run(args):
    started = time.monotonic()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    cfg = config['puzzles']
    free = shutil.disk_usage('.').free
    plan = dict(puzzle_compressed_bytes=cfg['compressed_bytes'], maximum_source_transfer_bytes=350*2**20,
                planned_peak_local_bytes=64*2**20, free_disk_bytes=free,
                free_disk_reserve_bytes=20*2**30, whole_archive_saved=False,
                source_sha256_policy='computed full compressed-stream fingerprint; no publisher checksum claimed',
                intended_puzzles=len(cfg['rating_band_starts'])*cfg['per_rating_band'],
                intended_openings=sum(config['openings']['quotas'].values()))
    print(json.dumps(plan, indent=2), flush=True)
    if free < 20*2**30 + plan['planned_peak_local_bytes']:
        raise RuntimeError('Insufficient disk reserve')
    if not args.execute:
        return
    if not args.publish:
        raise ValueError('Use --publish: production evaluation fixtures must not remain on this Mac')
    sampler = PuzzleSampler(config)
    headers = {'Accept-Encoding': 'identity', 'User-Agent': 'GambitZero-M1-evaluation-freeze/1'}
    deadline = started + 3600
    with urllib.request.urlopen(urllib.request.Request(cfg['url'], headers=headers), timeout=60) as response:
        if (response.status != 200 or int(response.headers['Content-Length']) != cfg['compressed_bytes']
                or response.headers.get('ETag') != cfg['etag']
                or response.headers.get('Last-Modified') != cfg['last_modified']):
            raise ValueError('Puzzle snapshot changed after resource/selection freeze')
        raw = HashingReader(response, cfg['compressed_bytes'], deadline)
        buffered = io.BufferedReader(raw, buffer_size=65536)
        window = zstd.get_frame_parameters(buffered.peek(18)).window_size
        if window > 64*2**20:
            raise ValueError('Puzzle decoder window exceeds 64 MiB memory plan')
        with zstd.ZstdDecompressor().stream_reader(buffered, closefd=False) as decoder:
            with io.TextIOWrapper(decoder, encoding='utf-8') as text:
                csv.field_size_limit(128*2**10)
                for row in csv.DictReader(text):
                    sampler.add(row)
                    if sampler.counts['source_rows'] % 500000 == 0:
                        print(f'Puzzle rows inspected: {sampler.counts["source_rows"]:,}', flush=True)
        while buffered.read(65536):
            pass
        digest = raw.digest.hexdigest()
        if raw.count != cfg['compressed_bytes']:
            raise ValueError('Incomplete puzzle source; no full-source checksum accepted')
        if cfg['expected_computed_sha256'] and cfg['expected_computed_sha256'] != digest:
            raise ValueError('Puzzle source differs from frozen computed checksum')
    puzzles = sampler.finish()
    print(f'Complete puzzle source SHA256: {digest}; selected {len(puzzles)} legal fixtures', flush=True)
    snapshot_date = datetime.now(timezone.utc).date().isoformat()
    sources = [dict(url=cfg['url'], snapshot_date=snapshot_date, sha256=digest,
                    bytes=raw.count, etag=cfg['etag'], last_modified=cfg['last_modified'],
                    verification='complete compressed-stream computed SHA256; not publisher checksum')]
    payloads = {}
    for letter in sorted(config['openings']['quotas']):
        url = f'https://raw.githubusercontent.com/{config["openings"]["repository"]}/{config["openings"]["revision"]}/{letter}.tsv'
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
            payloads[letter] = response.read(2**20 + 1)
        if len(payloads[letter]) > 2**20:
            raise ValueError('Opening source exceeds 1 MiB per-file limit')
        sources.append(dict(url=url, snapshot_date=snapshot_date,
                            sha256=hashlib.sha256(payloads[letter]).hexdigest(), bytes=len(payloads[letter]),
                            revision=config['openings']['revision']))
        print(f'Pinned opening source {letter}.tsv verified', flush=True)
    openings = select_openings(payloads, config)
    fens = evaluation_positions(puzzles, openings)
    frozen = dict(kind='dataset_d_frozen', version=config['version'], config=config,
                  selection_frozen_before_dataset_a_overlap=True,
                  puzzle_ids=[puzzle['id'] for puzzle in puzzles], puzzles=puzzles,
                  opening_suite=openings, position_fens=fens, sources=sources,
                  warning='Evaluation only. Puzzle solutions never supply human-game training labels. '
                  'Previous repetition history is unknown for puzzle FENs. No model evaluation performed.')
    spool = Path('data/processed/m1_dataset_d_spool')
    uploader = VerifiedUploader('ishagarg1103/gambit-zero', spool, spool_cap_bytes=64*2**20,
                                deadline=deadline, transfer_budget_bytes=64*2**20)
    path = spool / 'frozen.json'
    path.write_text(json.dumps(frozen, sort_keys=True, indent=2)+'\n')
    _, sha = read_frozen_d(path, synthetic=False)
    receipt = uploader.publish(path, f'evaluation/{config["version"]}/{sha}.json')
    report = dict(status='FROZEN_NOT_YET_OVERLAP_CHECKED', plan=plan, frozen_receipt=receipt,
                  version=config['version'], sources=sources, source_counts=dict(sampler.counts),
                  puzzles=len(puzzles), openings=len(openings), unique_evaluation_boards=len(fens),
                  selected_ids_sha256=hashlib.sha256(canonical(frozen['puzzle_ids']).encode()).hexdigest(),
                  dataset_a_modified=False, dataset_a_overlap_checked=False,
                  elapsed_seconds=time.monotonic()-started,
                  peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                  * (1 if sys.platform == 'darwin' else 1024),
                  local_artifact_deleted_after_remote_verification=True)
    Path(args.output).write_text(json.dumps(report, sort_keys=True, indent=2)+'\n')
    shutil.rmtree(spool)
    print(json.dumps({k: report[k] for k in ['status', 'puzzles', 'openings', 'unique_evaluation_boards']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/dataset_d.yaml')
    parser.add_argument('--output', default='reports/m1_dataset_d_freeze.json')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--publish', action='store_true')
    run(parser.parse_args())
