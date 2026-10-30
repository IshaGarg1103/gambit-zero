"""Lossless, bounded human_v3 minibatches; no chess replay or training here.

Only the accepted immutable Dataset A is fetched. Compressed originals are
verified before use; one shard pair is held on disk by default. The shuffle
buffer spans Arrow batches, games and shard boundaries. No malformed row is
filtered: validation happens *before* the explicit eligibility filter.
"""
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil

import numpy as np
import pyarrow.parquet as pq

from tinychess.chess.action_codec import ACTION_MAP_VERSION
from tinychess.chess.pgx_env import RULES_POLICY_VERSION
from tinychess.chess.state_features import ENCODER_VERSION
from tinychess.data.human import split_for_month, RESULTS
from tinychess.data.storage import POSITION_SCHEMA, GAME_SCHEMA, unpack_features

REPO = 'ishagarg1103/gambit-zero'
REVISION = 'b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c'
# Read at REVISION, compared against the accepted M1 catalog receipt.
CATALOG_SHA256 = 'ab791031be6b4aa2f7962ddc527fee9fbb91840223759339686d55ab1ab9daa1'
SPLITS = ('train', 'validation', 'test')
MAX_FILE_BYTES = 32 * 2**20


class RecordError(ValueError):
    """Fail-stop format/integrity/target error, with the source row identity."""


def require(condition, message):
    if not condition:
        raise RecordError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for part in iter(lambda: source.read(2**20), b''):
            digest.update(part)
    return digest.hexdigest()


@dataclass(frozen=True)
class FileSpec:
    path: str
    size: int
    sha256: str

    def __post_init__(self):
        p = PurePosixPath(self.path)
        require(not p.is_absolute() and '..' not in p.parts and '\\' not in self.path,
                'Unsafe dataset path')
        require(type(self.size) is int and 0 < self.size <= MAX_FILE_BYTES, 'Invalid file size')
        require(re.fullmatch('[0-9a-f]{64}', self.sha256) is not None, 'Invalid SHA256')

    @classmethod
    def from_receipt(cls, value):
        return cls(value['path_in_repo'], value['bytes'], value['sha256'])


@dataclass(frozen=True)
class Shard:
    month: str
    split: str
    positions: FileSpec
    games: FileSpec
    rows: int
    game_count: int
    game_ids_sha256: str

    def __post_init__(self):
        require(self.split in SPLITS and split_for_month(self.month) == self.split,
                'Invalid shard month/split')
        require(type(self.rows) is int and 0 < self.rows <= 128 * 512
                and type(self.game_count) is int and 0 < self.game_count <= 128,
                'Shard exceeds accepted bounded format')


class VerifiedStore:
    """Explicit local originals only; no global Hub cache or persistent tensors.

    Local files may be symlinks to existing verified processed shards. A bad
    local checksum is an error, never a reason to overwrite/re-download it.
    Newly downloaded files are atomic; failures remove only their .partial.
    Tokens are read by the Hub SDK and are never written to reports/logs.
    """
    def __init__(self, root, *, download=False, keep_downloads=False,
                 reserve_bytes=20 * 2**30):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.download, self.keep_downloads = download, keep_downloads
        self.reserve_bytes = reserve_bytes
        self.downloaded_bytes = 0
        self.verified = set()

    def _read_remote(self, path):
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers, get_session
        return get_session().stream(
            'GET', hf_hub_url(REPO, path, repo_type='dataset', revision=REVISION),
            headers=build_hf_headers(), follow_redirects=True, timeout=120)

    def _fetch(self, path, limit, digest=None, size=None):
        target = self.root / path
        if target.exists():
            require(target.stat().st_size <= limit, f'Local file exceeds bound: {path}')
            if size is not None:
                require(target.stat().st_size == size, f'Local size mismatch: {path}')
            if digest is not None:
                require(sha256(target) == digest, f'Local SHA256 mismatch: {path}')
            self.verified.add(path)
            return target, False
        require(self.download, f'Missing verified local file (downloads disabled): {path}')
        require(shutil.disk_usage(self.root).free >= self.reserve_bytes + limit,
                'Insufficient free disk; preserve reserve before download')
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.partial')
        require(not temporary.exists(), f'Unfinished download exists: {temporary}')
        count, checksum = 0, hashlib.sha256()
        try:
            with self._read_remote(path) as response:
                response.raise_for_status()
                with temporary.open('xb') as output:
                    for chunk in response.iter_bytes(chunk_size=2**20):
                        count += len(chunk)
                        require(count <= limit, f'Remote file exceeds bound: {path}')
                        checksum.update(chunk)
                        output.write(chunk)
                        self.downloaded_bytes += len(chunk)
            require(size is None or count == size, f'Remote size mismatch: {path}')
            require(digest is None or checksum.hexdigest() == digest, f'Remote SHA256 mismatch: {path}')
            temporary.replace(target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        self.verified.add(path)
        return target, True

    def json(self, path, digest):
        target, _ = self._fetch(path, 2**20, digest)
        try:
            return json.loads(target.read_text())
        except (ValueError, UnicodeError) as exc:
            raise RecordError(f'Malformed JSON: {path}') from exc

    @contextmanager
    def file(self, spec):
        path, acquired = self._fetch(spec.path, spec.size, spec.sha256, spec.size)
        try:
            yield path
        finally:
            # Never delete pre-existing user files. Download failures retain no
            # trusted file; successful newly fetched originals may be ephemeral.
            if acquired and not self.keep_downloads:
                path.unlink()


def inventory(store):
    """Resolve *all* original current shards from pinned, hashed manifests."""
    catalog = store.json('dataset_manifest.json', CATALOG_SHA256)
    require(catalog['repo_id'] == REPO and not catalog['synthetic']
            and catalog['status'] == 'complete-50k-filtered-evaluation-v1'
            and catalog['fully_reproduced'], 'Not the accepted real Dataset A catalog')
    shards, seen_files, seen_months = [], set(), set()
    totals = {s: dict(games=0, positions=0) for s in SPLITS}
    for entry in catalog['months']:
        month = entry['month']
        require(month not in seen_months, 'Duplicate month in catalog')
        seen_months.add(month)
        match = re.search(r'-([0-9a-f]{64})\.json$', entry['manifest_path'])
        require(match is not None, 'Unhashed monthly manifest')
        manifest = store.json(entry['manifest_path'], match[1])
        split = split_for_month(month)
        for key, expected in [('month', month), ('split', split), ('schema', 'human_v3'),
                              ('encoder', ENCODER_VERSION), ('action_map', ACTION_MAP_VERSION),
                              ('rules', RULES_POLICY_VERSION), ('parity_mismatches', 0)]:
            require(manifest[key] == expected, f'Month manifest mismatch: {month} {key}')
        hashes, rows, games = [], 0, 0
        for part in manifest['shards']:
            require((part['month'], part['split']) == (month, split), 'Shard split mismatch')
            files = {f['kind']: f for f in part['files']}
            require(set(files) == {'positions', 'games'} and len(part['files']) == 2,
                    'Invalid shard file inventory')
            for f in part['files']:
                require(f['path_in_repo'] not in seen_files, 'Duplicate dataset file')
                seen_files.add(f['path_in_repo']); hashes.append(f['sha256'])
            require(part['rows'] > 0 and part['games'] > 0, 'Empty shard')
            shards.append(Shard(month, split, FileSpec.from_receipt(files['positions']),
                                FileSpec.from_receipt(files['games']), part['rows'],
                                part['games'], part['game_ids_sha256']))
            rows += part['rows']; games += part['games']
        encoded = json.dumps(hashes, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
        require(hashlib.sha256(encoded).hexdigest() == entry['shard_digest'] == manifest['shard_digest'],
                'Monthly shard digest mismatch')
        require(rows == entry['training_positions'] == manifest['training_positions']
                and games == entry['accepted_games'] == manifest['accepted_games'], 'Month totals mismatch')
        totals[split]['games'] += games; totals[split]['positions'] += rows
    require(seen_months == {f'2026-0{i}' for i in range(1, 10)} and totals == catalog['counts'],
            'Incomplete temporal inventory')
    require({s: v['games'] for s, v in totals.items()} == dict(train=35000, validation=7500, test=7500),
            'Unexpected game quotas')
    return tuple(shards), catalog


def parquet_batches(path, schema, chunk_rows):
    require(type(chunk_rows) is int and 1 <= chunk_rows <= 512, 'chunk_rows must be 1..512')
    with pq.ParquetFile(path) as file:
        require(file.schema_arrow.equals(schema, check_metadata=False)
                and file.schema_arrow.metadata == schema.metadata, f'Parquet schema mismatch: {path}')
        require(all(file.metadata.row_group(i).num_rows <= 512 for i in range(file.num_row_groups)),
                f'Parquet row group exceeds accepted memory bound: {path}')
        for batch in file.iter_batches(batch_size=chunk_rows, use_threads=False):
            # Arrow bounded batches; at most chunk_rows full feature records.
            yield batch.to_pylist()


def validate_game(row, shard):
    require(all(v is not None for v in row.values()), 'Null game field')
    game = json.loads(row['metadata'])
    require((game['game_id'], game['source_month'], game['split']) ==
            (row['game_id'], shard.month, shard.split), 'Game identity/month/split mismatch')
    require(game['schema'] == 'human_v3' and game['encoder_version'] == ENCODER_VERSION
            and game['action_map_version'] == ACTION_MAP_VERSION
            and game['rules_policy'] == RULES_POLICY_VERSION
            and game['label_source'] == 'human_game', 'Game version/label provenance mismatch')
    require(type(game['plies']) is int and 0 < game['plies'] <= 512
            and len(row['actions']) == game['plies'], 'Game length mismatch')
    require(all(type(a) is int and 0 <= a < 4672 for a in row['actions']), 'Invalid game action')
    require(hashlib.sha256(np.asarray(row['actions'], dtype='<u2').tobytes()).hexdigest()
            == game['trajectory_sha256'], 'Game trajectory digest mismatch')
    require(game['result'] in RESULTS and game['outcome_white'] == RESULTS[game['result']],
            'Game outcome mismatch')
    require(game['value_weight'] in (0., 1.) and (game['value_weight'] == 0.
            if 'time' in game['termination_header'].casefold() and
            ('forfeit' in game['termination_header'].casefold() or 'out' in game['termination_header'].casefold())
            else game['value_weight'] == 1.), 'Invalid game value weight')
    require(min(game['white_elo'], game['black_elo']) >= 1800 and
            re.fullmatch(r'\d+(?:\+\d+)?', game['time_control']) is not None and
            int(game['time_control'].split('+')[0]) >= 300, 'Game filter mismatch')
    return game, row['actions']


def validate_row(row, game, actions, shard):
    """Validate original targets/features; no inferred replacement labels."""
    try:
        require(set(row) == set(POSITION_SCHEMA.names) and all(v is not None for v in row.values()),
                'Missing/extra/null position field')
        ply = row['ply']
        require((row['source_month'], row['split']) == (shard.month, shard.split), 'Position split mismatch')
        require(type(ply) is int and 0 <= ply < game['plies'] and row['remaining_plies'] == game['plies'] - ply,
                'Invalid ply/remaining_plies')
        require(row['color'] == ply % 2 and row['label_source'] == 'human_game', 'Color/provenance mismatch')
        expected = game['outcome_white'] * (1 if row['color'] == 0 else -1)
        require(row['outcome_stm'] == expected and row['wdl_class'] == {1: 0, 0: 1, -1: 2}[expected],
                'WDL side-to-move mismatch')
        require(row['value_weight'] == game['value_weight'] and row['policy_weight'] == 1.,
                'Position weights mismatch')
        require(type(row['training_eligible']) is bool, 'Invalid eligibility flag')
        require(len(row['position_hash']) == 32 and 1 <= row['repetition_count'] <= 5, 'Invalid hash/repetition')
        require(len(row['planes']) == 30720 and len(row['context']) == 32 and len(row['legal_mask']) == 584,
                'Invalid feature byte length')
        features = unpack_features(row)
        require(np.isfinite(features['planes']).all() and np.isfinite(features['context']).all(),
                'Nonfinite model features')
        action = row['action']
        require(type(action) is int and 0 <= action < 4672 and action == actions[ply]
                and bool(features['legal_mask'][action]), 'Human action illegal or inconsistent')
        require(re.fullmatch(r'[a-h][1-8][a-h][1-8][qrbn]?', row['human_move_uci']) is not None,
                'Invalid human UCI')
        uci = row['human_move_uci']
        rank = int(uci[1]) - 1
        source = (ord(uci[0]) - ord('a')) * 8 + (rank if row['color'] == 0 else 7-rank)
        require(action // 73 == source, 'Human UCI/action source mismatch')
        own = features['planes'][7-source % 8, source // 8, :6]
        require(np.count_nonzero(own == 1) == 1 and np.count_nonzero(own) == 1,
                'Policy source has no canonical own-piece feature')
        fields = row['fen'].split()
        require(len(fields) == 6 and fields[1] == ('w' if row['color'] == 0 else 'b')
                and int(fields[4]) >= 0 and int(fields[5]) == ply // 2 + 1, 'FEN color/counter mismatch')
        require(features['context'][0] == row['color'] and features['context'][5] == 0
                and features['context'][6] == 0, 'Context color/terminal mismatch')
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise RecordError(f"{shard.positions.path}: game={row.get('game_id')} ply={row.get('ply')}: {exc}") from exc


def iter_records(store, shards, split, *, chunk_rows=128):
    require(split in SPLITS, 'Unknown split')
    seen_games = set()  # <=50K short identifiers; no features retained here.
    for shard in shards:
        require(shard.split == split and split_for_month(shard.month) == split, 'Mixed input splits')
        try:
            with store.file(shard.games) as game_path:
                games = {}
                for batch in parquet_batches(game_path, GAME_SCHEMA, chunk_rows):
                    for row in batch:
                        gid = row['game_id']
                        require(gid not in seen_games, f'Duplicate game: {gid}')
                        seen_games.add(gid)
                        games[gid] = validate_game(row, shard)
            ids = list(games)
            encoded = json.dumps(ids, separators=(',', ':'), ensure_ascii=True).encode()
            require(len(games) == shard.game_count and hashlib.sha256(encoded).hexdigest()
                    == shard.game_ids_sha256, 'Shard game count/order mismatch')
            counts, n = Counter(), 0
            with store.file(shard.positions) as path:
                for batch in parquet_batches(path, POSITION_SCHEMA, chunk_rows):
                    for row in batch:
                        gid = row['game_id']
                        require(gid in games, f'Position has no game: {gid}')
                        game, actions = games[gid]
                        validate_row(row, game, actions, shard)
                        # Dataset D physically removes train rows: gaps are
                        # valid; duplicate/backward plies are never valid.
                        require(row['ply'] > counts.get(gid, -1), f'Duplicate/backward ply: {gid}')
                        counts[gid] = row['ply']; n += 1
                        yield row
            require(n == shard.rows, 'Shard position count mismatch')
            require(set(counts) <= set(games), 'Unknown game in position shard')
        except (ValueError, KeyError, TypeError) as exc:
            raise RecordError(f'{shard.month}/{shard.positions.path}: {exc}') from exc


def shuffled_records(records, capacity, rng):
    """Replacement buffer, followed by random drain: each record exactly once.

    Not a uniform permutation of an entire epoch. Mixing is bounded by buffer
    size; the caller also shuffles shards. Never deduplicates or caps a game.
    """
    require(type(capacity) is int and capacity > 0, 'Shuffle buffer must be positive')
    buffer = []
    for row in records:
        if len(buffer) < capacity:
            buffer.append(row)
        else:
            i = int(rng.integers(len(buffer)))
            yield buffer[i]
            buffer[i] = row
    while buffer:
        i = int(rng.integers(len(buffer)))
        yield buffer[i]
        buffer[i] = buffer[-1]
        buffer.pop()


@dataclass
class TrainingBatch:
    planes: np.ndarray
    context: np.ndarray
    legal_mask: np.ndarray
    action: np.ndarray
    wdl_class: np.ndarray
    outcome_stm: np.ndarray
    policy_weight: np.ndarray
    value_weight: np.ndarray
    training_eligible: np.ndarray
    game_id: tuple
    ply: np.ndarray
    position_hash: tuple
    source_month: tuple
    split: str

    @property
    def max_positions_per_game(self):
        return max(Counter(self.game_id).values(), default=0)


def collate(rows, split):
    n = len(rows)
    return TrainingBatch(
        planes=np.frombuffer(b''.join(r['planes'] for r in rows), '<f4').reshape(n, 8, 8, 120),
        context=np.frombuffer(b''.join(r['context'] for r in rows), '<f4').reshape(n, 8),
        legal_mask=np.unpackbits(np.frombuffer(b''.join(r['legal_mask'] for r in rows), np.uint8)
                                .reshape(n, 584), axis=1, bitorder='little').astype(bool),
        action=np.asarray([r['action'] for r in rows], np.int32),
        wdl_class=np.asarray([r['wdl_class'] for r in rows], np.int32),
        outcome_stm=np.asarray([r['outcome_stm'] for r in rows], np.int8),
        policy_weight=np.asarray([r['policy_weight'] for r in rows], np.float32),
        value_weight=np.asarray([r['value_weight'] for r in rows], np.float32),
        training_eligible=np.asarray([r['training_eligible'] for r in rows], bool),
        game_id=tuple(r['game_id'] for r in rows), ply=np.asarray([r['ply'] for r in rows], np.uint16),
        position_hash=tuple(r['position_hash'] for r in rows),
        source_month=tuple(r['source_month'] for r in rows), split=split)


def iter_batches(store, shards, split, *, batch_size=256, chunk_rows=128,
                 shuffle=None, shuffle_buffer=16384, seed=0, epoch=0):
    """CPU NumPy model inputs; seed+epoch fully determine the training order.

    Only eligible rows are emitted; ineligible rows are still validated first.
    Timeout rows remain for policy with unchanged zero value weights. No padding
    or drop_last: the final batch is short, so every eligible row is accounted
    for. Consume to exhaustion to check shard counts. Stopping an iterator early
    is a sample-only check, never an epoch verification.
    """
    require(split in SPLITS, 'Unknown split')
    require(type(batch_size) is int and 0 < batch_size <= 8192, 'batch_size must be 1..8192')
    require(type(seed) is int and seed >= 0 and type(epoch) is int and epoch >= 0, 'Invalid seed/epoch')
    shuffle = split == 'train' if shuffle is None else shuffle
    require(not shuffle or split == 'train', 'Only training may shuffle')
    if shuffle:
        require(shuffle_buffer >= batch_size, 'Shuffle buffer smaller than batch')
    selected = list(shards)
    require(all(s.split == split for s in selected), 'Mixed input splits')
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, epoch])))
    if shuffle:
        rng.shuffle(selected)
    rows = (r for r in iter_records(store, selected, split, chunk_rows=chunk_rows) if r['training_eligible'])
    if shuffle:
        rows = shuffled_records(rows, shuffle_buffer, rng)
    pending = []
    for row in rows:
        pending.append(row)
        if len(pending) == batch_size:
            yield collate(pending, split)
            pending = []
    if pending:
        yield collate(pending, split)
