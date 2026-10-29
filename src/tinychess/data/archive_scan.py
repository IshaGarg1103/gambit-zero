"""Bounded PZstandard scanning and UTC-day sampling; no complete archive cache.

Completed months are checkpoints. Interrupted months restart at byte zero, so
SHA256 verification never depends on a fabricated resumable hashlib state.
This module does not build position records or change accepted chess rules.
"""
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
import calendar
import hashlib
import heapq
import io
import json
import os
from pathlib import Path
import re
import shutil
import struct
import time
import urllib.request

import psutil
import zstandard as zstd

ALGORITHM = 'utc-day-rating-bottomk-sha256-v1'
PZSTD_MAGIC = 0x184D2A50


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


class ScanError(RuntimeError):
    pass


class HeaderRejected(ValueError):
    pass


class Shortfall(ScanError):
    pass


class DiskGuard:
    """Reserve free space; optionally enforce a managed-output size budget.

    The worker may explicitly waive output budgets for a disk-reserve-only run.
    Free-space and memory checks remain mandatory in that mode.
    """
    def __init__(self, root, max_bytes, reserve_bytes):
        self.root = Path(root)
        self.max_bytes = int(max_bytes)
        self.reserve_bytes = int(reserve_bytes)
        if min(self.max_bytes, self.reserve_bytes) <= 0:
            raise ValueError('Positive disk budget and reserve required')
        self.root.mkdir(parents=True, exist_ok=True)
        self.check()

    def check(self, additional_bytes=0):
        used = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
        free = shutil.disk_usage(self.root).free
        if (os.environ.get('TINYCHESS_ENFORCE_TEMP_STORAGE_CAPS', '1') != '0'
                and used + additional_bytes > self.max_bytes):
            raise ScanError('Managed disk budget exceeded')
        if free - additional_bytes < self.reserve_bytes:
            raise ScanError('Free-disk reserve would be exhausted')
        if psutil.virtual_memory().available < 512*2**20:
            raise ScanError('Available host memory below 512 MiB safety floor')
        return dict(managed_bytes=used, free_bytes=free, reserve_bytes=self.reserve_bytes)


def header_candidate(text, month):
    """No python-chess parsing. Reject missing, duplicate or malformed headers."""
    headers = {}
    for line in io.StringIO(text):
        line = line.rstrip('\r\n')
        if not line.strip():
            if headers:
                break
            continue
        match = re.fullmatch(r'\[(\w+) ("(?:[^"\\]|\\["\\])*")\]', line)
        if not match or match[1] in headers:
            raise HeaderRejected('malformed_or_duplicate_header')
        try:
            headers[match[1]] = json.loads(match[2])
        except ValueError as exc:
            raise HeaderRejected('malformed_header_escape') from exc
    for key in ['Event', 'WhiteElo', 'BlackElo', 'TimeControl', 'UTCDate', 'Termination', 'Result']:
        if key not in headers:
            raise HeaderRejected('missing_header:' + key)
    if headers.get('Variant', 'Standard').casefold() != 'standard':
        raise HeaderRejected('variant')
    if not headers['Event'].startswith('Rated ') or headers.get('Rated', 'true').casefold() not in ['true', '1']:
        raise HeaderRejected('not_rated')
    if headers.get('FEN') or headers.get('SetUp', '0') != '0':
        raise HeaderRejected('nonstandard_start')
    try:
        if not all(re.fullmatch(r'\d+', headers[k]) for k in ['WhiteElo','BlackElo']):
            raise ValueError
        rating = min(int(headers['WhiteElo']), int(headers['BlackElo']))
    except ValueError as exc:
        raise HeaderRejected('invalid_rating') from exc
    if rating < 1800:
        raise HeaderRejected('rating_below_minimum')
    tc = re.fullmatch(r'(\d+)(?:\+\d+)?', headers['TimeControl'])
    if not tc:
        raise HeaderRejected('invalid_time_control')
    if int(tc[1]) < 300:
        raise HeaderRejected('time_control_below_minimum')
    if headers['Result'] not in ['1-0', '0-1', '1/2-1/2']:
        raise HeaderRejected('invalid_result')
    if headers['Termination'].casefold() in ['abandoned', 'rules infraction', 'aborted']:
        raise HeaderRejected('excluded_termination')
    try:
        if not re.fullmatch(r'\d{4}\.\d{2}\.\d{2}', headers['UTCDate']):
            raise ValueError
        day = date.fromisoformat(headers['UTCDate'].replace('.', '-'))
    except ValueError as exc:
        raise HeaderRejected('invalid_utc_date') from exc
    if day.strftime('%Y-%m') != month:
        raise HeaderRejected('utc_month_mismatch')
    site = re.fullmatch(r'https://lichess.org/([a-zA-Z0-9]{8})(?:/(?:white|black))?', headers.get('Site', ''))
    game_id = site[1] if site else hashlib.sha256(text.encode()).hexdigest()
    return dict(game_id=game_id, day=day.isoformat(), min_rating=rating,
                rating_stratum='strong' if rating >= 2000 else 'regular')


def daily_quotas(month, total):
    first = date.fromisoformat(month+'-01')
    days = calendar.monthrange(first.year, first.month)[1]
    if total < days:
        raise ValueError('Quota must permit at least one accepted game per UTC day')
    return {f'{month}-{day:02}': total//days + (day <= total % days) for day in range(1,days+1)}


def priority(seed, month, game_id):
    # JSON array with a domain/version prefix is unambiguous and UTF-8 encoded.
    return hashlib.sha256(canonical([ALGORITHM, seed, month, game_id]).encode()).hexdigest()


@dataclass
class Entry:
    key: tuple
    record: dict

    def __lt__(self, other):
        return self.key > other.key  # heap root is the worst retained priority


class DaySampler:
    def __init__(self, month, quota, seed, reserve_factor=3, max_candidate_bytes=128*2**20):
        if reserve_factor < 1 or max_candidate_bytes <= 0:
            raise ValueError('Positive reserve factor/cache budget required')
        self.month, self.seed = month, seed
        self.quotas = daily_quotas(month, quota)
        self.reserve_factor = reserve_factor
        self.max_candidate_bytes = max_candidate_bytes
        self.heaps, self.retained = {}, {}
        self.candidate_bytes = 0
        self.counts, self.rejections, self.eligible_by_day = Counter(), Counter(), Counter()
        for day, n in self.quotas.items():
            # Reserve a minimum ceil(60%) strong per day. Keep strong reserves
            # for explicitly reported regular-stratum shortages as well.
            self.heaps[(day,'strong')] = []
            self.heaps[(day,'regular')] = []

    def offer(self, text, ordinal, framing_error=None):
        self.counts['source_game_blocks'] += 1
        try:
            if framing_error:
                raise HeaderRejected(framing_error)
            h = header_candidate(text, self.month)
        except HeaderRejected as exc:
            self.rejections[str(exc)] += 1
            return
        self.counts['header_eligible_occurrences'] += 1
        self.eligible_by_day[(h['day'],h['rating_stratum'])] += 1
        rank = priority(self.seed, self.month, h['game_id'])
        heap = self.heaps[(h['day'],h['rating_stratum'])]
        n = self.quotas[h['day']]
        strong_n = (3*n + 4)//5
        limit = (n if h['rating_stratum']=='strong' else n-strong_n)*self.reserve_factor
        if not limit:
            return
        # Hash the complete PGN only when it could enter the bounded reservoir.
        if len(heap) >= limit and (rank,h['game_id']) > heap[0].key[:2]:
            return
        raw = text.encode('utf-8')
        digest = hashlib.sha256(raw).hexdigest()
        if h['game_id'] in self.retained:
            if self.retained[h['game_id']].record['raw_pgn_sha256'] != digest:
                raise ScanError('Conflicting payload for a retained stable game ID')
            self.counts['duplicate_retained_game_occurrences'] += 1
            return
        record = dict(**h, rank=rank, raw_pgn_sha256=digest, source_ordinal=ordinal, pgn=text)
        entry = Entry((rank,h['game_id'],digest), record)
        evicted = heap[0] if len(heap) == limit else None
        projected = self.candidate_bytes + len(raw) - (len(evicted.record['pgn'].encode()) if evicted else 0)
        if projected > self.max_candidate_bytes:
            raise ScanError('Bounded candidate-memory payload budget exceeded')
        if evicted:
            heapq.heapreplace(heap, entry)
            del self.retained[evicted.record['game_id']]
        else:
            heapq.heappush(heap, entry)
        self.retained[h['game_id']] = entry
        self.candidate_bytes = projected

    def snapshot(self):
        return dict(algorithm=ALGORITHM, month=self.month, seed=self.seed, quotas=self.quotas,
                    reserve_factor=self.reserve_factor, candidate_payload_bytes=self.candidate_bytes,
                    counts=dict(self.counts), header_rejections=dict(sorted(self.rejections.items())),
                    eligible_occurrences_by_day={day:{band:self.eligible_by_day[(day,band)]
                        for band in ['strong','regular']} for day in self.quotas},
                    candidates=[e.record for key in sorted(self.heaps) for e in sorted(self.heaps[key],key=lambda e:e.key)])

    def select(self, validate):
        """validate returns an accepted object; ONLY expected source rejections caught.

        Pass the accepted replay_game callback here. ParityError/ScanError escape
        immediately. No quota redistribution or silent reserve enlargement.
        """
        from tinychess.data.human import RejectedGame
        selected, rejected, coverage = [], Counter(), {}
        for day,n in self.quotas.items():
            required_strong = (3*n+4)//5
            accepted = []
            by_band = {}
            for band in ['strong','regular']:
                by_band[band] = iter(sorted(self.heaps[(day,band)], key=lambda e:e.key))
            def take(band, count):
                if len(accepted) >= count:
                    return
                for entry in by_band[band]:
                    try:
                        value = validate(entry.record)
                    except RejectedGame as exc:
                        rejected[str(exc)] += 1
                        continue
                    accepted.append((entry.record,value))
                    if len(accepted) == count:
                        return
            take('strong', required_strong)
            if len(accepted) < required_strong:
                raise Shortfall(f'{day}: insufficient fully validated 2000+ reserves')
            take('regular', n)
            regular_accepted = len(accepted)-required_strong
            if len(accepted) < n:
                take('strong', n)
            if len(accepted) != n:
                raise Shortfall(f'{day}: insufficient fully validated daily reserves')
            selected.extend(accepted)
            coverage[day] = dict(accepted=n, strong=sum(r['rating_stratum']=='strong' for r,v in accepted),
                                 extra_strong_due_to_regular_shortage=n-required_strong-regular_accepted)
        return selected, dict(rejected), coverage


class PzstdStream(io.RawIOBase):
    """Bounded independent-frame decoder with SHA256 over ORIGINAL compressed bytes.

    Supports only the PZstandard layout actually observed in the cached archives.
    Unknown formats/oversized frames stop; they are not treated as seekable.
    """
    def __init__(self, source, expected_size, expected_sha256, guard=None,
                 max_compressed_frame=8*2**20, max_decoded_frame=64*2**20,
                 max_seconds=86400):
        super().__init__()
        if expected_size <= 0 or not re.fullmatch('[0-9a-f]{64}',expected_sha256) or max_seconds <= 0:
            raise ValueError('Frozen size, SHA256 and runtime bound required')
        self.source, self.expected_size, self.expected_sha256 = source, expected_size, expected_sha256
        self.guard, self.max_compressed_frame, self.max_decoded_frame = guard,max_compressed_frame,max_decoded_frame
        self.max_seconds, self.started = max_seconds,time.monotonic()
        self.digest, self.compressed_bytes, self.frames = hashlib.sha256(),0,0
        self.buffer, self.offset, self.verified = b'',0,False
        self.frame_index = []

    def readable(self):
        return True

    def _read_exact(self, size):
        chunks = []
        remaining = size
        while remaining:
            if time.monotonic()-self.started > self.max_seconds:
                raise ScanError('Runtime budget exhausted')
            chunk = self.source.read(min(65536,remaining))
            if not chunk:
                raise ScanError('Interrupted/incomplete compressed archive')
            self.digest.update(chunk)
            self.compressed_bytes += len(chunk)
            if self.compressed_bytes > self.expected_size:
                raise ScanError('Compressed archive exceeds frozen size')
            chunks.append(chunk)
            remaining -= len(chunk)
        return b''.join(chunks)

    def _next_frame(self):
        if self.guard:
            self.guard.check()
        if self.compressed_bytes == self.expected_size:
            # Force actual transport EOF, rather than trusting a length alone.
            if self.source.read(1):
                raise ScanError('Unexpected bytes beyond frozen archive size')
            if self.digest.hexdigest() != self.expected_sha256:
                raise ScanError('Full compressed archive SHA256 mismatch')
            self.verified = True
            return False
        metadata_offset = self.compressed_bytes
        magic,metadata_size,frame_size = struct.unpack('<III',self._read_exact(12))
        if magic != PZSTD_MAGIC or metadata_size != 4:
            raise ScanError('Unsupported PZstandard frame metadata; stop before assuming seekability')
        if not 0 < frame_size <= self.max_compressed_frame:
            raise ScanError('Compressed frame exceeds bounded buffer')
        if self.compressed_bytes + frame_size > self.expected_size:
            raise ScanError('Advertised frame extends beyond frozen archive size')
        compressed = self._read_exact(frame_size)
        params = zstd.get_frame_parameters(compressed)
        if params.dict_id or params.window_size > self.max_decoded_frame:
            raise ScanError('Unsupported dictionary/window exceeds decoder budget')
        if params.content_size not in [zstd.CONTENTSIZE_UNKNOWN,zstd.CONTENTSIZE_ERROR] and params.content_size > self.max_decoded_frame:
            raise ScanError('Decoded frame exceeds bounded buffer')
        self.buffer = zstd.ZstdDecompressor().decompress(compressed,max_output_size=self.max_decoded_frame,allow_extra_data=False)
        if len(self.buffer) > self.max_decoded_frame:
            raise ScanError('Decoded frame exceeds bounded buffer')
        self.frames += 1
        if self.frames > 20000:
            raise ScanError('Compression-frame index exceeds bounded 20K entries/month')
        self.frame_index.append(dict(metadata_offset=metadata_offset,
            compressed_frame_offset=metadata_offset+12,compressed_frame_bytes=frame_size,
            decoded_bytes=len(self.buffer),compressed_frame_sha256=hashlib.sha256(compressed).hexdigest()))
        self.offset = 0
        return True

    def readinto(self, target):
        while self.offset == len(self.buffer):
            self.buffer = b''
            if self.verified or not self._next_frame():
                return 0
        size = min(len(target),len(self.buffer)-self.offset)
        target[:size] = self.buffer[self.offset:self.offset+size]
        self.offset += size
        return size


def atomic_json(path, value, guard):
    path = Path(path)
    raw = (canonical(value)+'\n').encode()
    guard.check(len(raw))  # Accounts for BOTH old file and temporary replacement.
    temporary = path.with_suffix(path.suffix+'.partial')
    with temporary.open('wb') as stream:
        stream.write(raw)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    temporary.replace(path)


def scan_month(source, root, quota, seed, *, network_budget_bytes, max_seconds,
               max_cache_bytes=2**30, reserve_disk_bytes=20*2**30,
               candidate_memory_bytes=128*2**20, opener=urllib.request.urlopen):
    """Explicit one-month job. NOT called by imports or a default CLI.

    A live interrupted HTTP body can reconnect to its exact validated suffix,
    retaining the same decoder/hash/parser/sampler. A stopped process still
    restarts from zero. Verified month checkpoints need no source transfer.
    """
    from tinychess.data.sources import game_blocks
    from tinychess.data.archive_http import ContinuousArchiveResponse
    month,expected = source['month'],source['archive_size_bytes']
    if (network_budget_bytes is None or max_seconds is None or max_seconds <= 0
            or expected <= 0 or network_budget_bytes < expected):
        raise ScanError('Explicit sufficient transfer and runtime budgets are required before any request')
    if not re.fullmatch('[0-9a-f]{64}',str(source.get('published_sha256',''))):
        raise ScanError('Frozen official archive SHA256 required before any request')
    guard = DiskGuard(root,max_cache_bytes,reserve_disk_bytes)
    destination = Path(root)/f'{month}.selection.json'
    identity = dict(algorithm=ALGORITHM,month=month,seed=seed,quota=quota,
                    url=source['url'],archive_size_bytes=expected,published_sha256=source['published_sha256'])
    if destination.exists():
        value = json.loads(destination.read_text())
        expected_hash = value.pop('checkpoint_sha256',None)
        if hashlib.sha256(canonical(value).encode()).hexdigest() != expected_hash:
            raise ScanError('Month checkpoint integrity mismatch')
        if value['identity'] != identity or not value['archive']['full_archive_checksum_verified']:
            raise ScanError('Incompatible or unverified month checkpoint')
        value['checkpoint_sha256'] = expected_hash
        return value
    # A failure status never authorizes incomplete sampling for downstream use.
    status_path = Path(root)/f'{month}.status.json'
    partial = destination.with_suffix(destination.suffix+'.partial')
    if partial.exists():
        partial.unlink()  # Own abandoned atomic write, never a verified checkpoint.
    atomic_json(status_path,dict(status='INCOMPLETE',identity=identity,restart_policy='restart_current_month_at_zero'),guard)
    sampler = DaySampler(month,quota,seed,max_candidate_bytes=candidate_memory_bytes)
    decoder = None
    started=time.perf_counter()
    try:
        with ContinuousArchiveResponse(source,opener=opener,max_seconds=max_seconds) as response:
            decoder = PzstdStream(response,expected,source['published_sha256'],guard,max_seconds=max_seconds)
            with io.TextIOWrapper(io.BufferedReader(decoder,buffer_size=65536),encoding='utf-8',errors='replace') as text:
                for ordinal,(block,bad) in enumerate(game_blocks(text),1):
                    sampler.offer(block,ordinal,bad)
                    if ordinal % 100000 == 0:
                        atomic_json(status_path,dict(status='INCOMPLETE',identity=identity,
                            compressed_bytes=decoder.compressed_bytes,game_blocks=ordinal,
                            full_archive_checksum_verified=False,http_reconnects=response.reconnects,
                            restart_policy='restart_current_month_at_zero'),guard)
        if not decoder.verified:
            raise ScanError('Decoder did not consume and verify the full compressed archive')
        value = dict(identity=identity,selection=sampler.snapshot(),
                     archive=dict(full_archive_checksum_verified=True,sha256=decoder.digest.hexdigest(),
                                  bytes=decoder.compressed_bytes,frames=decoder.frames,
                                  completed_utc=datetime.now(timezone.utc).isoformat()),
                     kind='bounded_candidate_manifest_not_random_access_index')
        elapsed=time.perf_counter()-started
        value['scan_measurements']=dict(elapsed_seconds=elapsed,
            compressed_bytes=decoder.compressed_bytes,compressed_bytes_per_second=decoder.compressed_bytes/elapsed,
            source_game_blocks=sampler.counts['source_game_blocks'],
            source_games_per_second=sampler.counts['source_game_blocks']/elapsed,
            eligible_occurrences=sampler.counts['header_eligible_occurrences'],
            http_reconnects=response.reconnects,http_reconnection_events=response.events)
        value['compression_frame_index'] = decoder.frame_index
        value['index_scope'] = 'Verified compression boundaries, not PGN/game boundaries or a uniform-game sampling index'
        value['selection_digest'] = hashlib.sha256(canonical(value['selection']).encode()).hexdigest()
        value['checkpoint_sha256'] = hashlib.sha256(canonical(value).encode()).hexdigest()
        atomic_json(destination,value,guard)
        atomic_json(status_path,dict(status='VERIFIED_CANDIDATES',checkpoint_sha256=value['checkpoint_sha256']),guard)
        return value
    except Exception as exc:
        atomic_json(status_path,dict(status='INCOMPLETE',error=type(exc).__name__,message=str(exc),
                    identity=identity,full_archive_checksum_verified=False,
                    compressed_bytes=decoder.compressed_bytes if decoder else 0,
                    restart_policy='restart_current_month_at_zero'),guard)
        raise
