"""Synthetic-only loader fixtures; never written to the real Dataset A."""
from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from tinychess.data.human import replay_game
from tinychess.data.storage import GAME_SCHEMA, POSITION_SCHEMA, unpack_features
from tinychess.data.training_loader import (
    FileSpec, Shard, VerifiedStore, RecordError, iter_records, iter_batches,
    shuffled_records, collate, validate_row, inventory, REVISION,
)


@pytest.fixture(scope='module')
def originals():
    config = yaml.safe_load(Path('configs/data_human.yaml').read_text())
    games = []
    for i, result in enumerate(['1-0', '0-1', '1/2-1/2']):
        pgn = f'''[Event "Rated Blitz game"]
[Site "https://lichess.org/TEST000{i}"]
[WhiteElo "2100"]
[BlackElo "2000"]
[TimeControl "300+0"]
[UTCDate "2026.01.01"]
[Termination "{'Time forfeit' if i == 1 else 'Normal'}"]
[Result "{result}"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 {result}
'''
        games.append(replay_game(pgn, '2026-01', i, 'https://example.invalid/synthetic', config))
    return games


def write_shard(root, originals, split='train', n=3, mutate=None, part=0):
    month = {'train': '2026-01', 'validation': '2026-08', 'test': '2026-09'}[split]
    games, rows = [], []
    for i in range(n):
        original = copy.deepcopy(originals[i % 3])
        gid = f'synthetic-{split}-{part}-{i}'
        m = original.game['metadata']
        m.update(game_id=gid, split=split, source_month=month)
        games.append(dict(game_id=gid, metadata=json.dumps(m), actions=original.game['actions']))
        for row in original.positions:
            row.update(game_id=gid, split=split, source_month=month)
            rows.append(row)
    if mutate:
        mutate(rows)
    paths = [Path(root)/f'{split}-{part}.{kind}.parquet' for kind in ('positions', 'games')]
    pq.write_table(pa.Table.from_pylist(rows, schema=POSITION_SCHEMA), paths[0], row_group_size=3)
    pq.write_table(pa.Table.from_pylist(games, schema=GAME_SCHEMA), paths[1])
    specs = [FileSpec(p.name, p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
    digest = hashlib.sha256(json.dumps([r['game_id'] for r in games], separators=(',', ':')).encode()).hexdigest()
    return Shard(month, split, *specs, len(rows), len(games), digest), rows, games


def identities(batches):
    return [(g, int(p)) for b in batches for g, p in zip(b.game_id, b.ply)]


@pytest.mark.parametrize('chunk', [1, 4, 128])
def test_exact_stored_features_and_batches(tmp_path, originals, chunk):
    shard, rows, _ = write_shard(tmp_path, originals)
    batches = list(iter_batches(VerifiedStore(tmp_path), [shard], 'train', batch_size=7,
                               chunk_rows=chunk, shuffle=False))
    assert [len(b.action) for b in batches] == [7, 7, 4]
    offset = 0
    for b in batches:
        n = len(b.action)
        assert b.planes.shape == (n, 8, 8, 120) and b.planes.dtype == np.float32
        assert b.context.shape == (n, 8) and b.context.dtype == np.float32
        assert b.legal_mask.shape == (n, 4672) and b.legal_mask.dtype == bool
        assert b.action.dtype == b.wdl_class.dtype == np.int32
        for i, r in enumerate(rows[offset:offset+n]):
            assert b.planes[i].tobytes() == r['planes']
            assert b.context[i].tobytes() == r['context']
            assert np.packbits(b.legal_mask[i], bitorder='little').tobytes() == r['legal_mask']
            assert b.game_id[i] == r['game_id'] and b.ply[i] == r['ply']
            assert b.action[i] == r['action'] and b.wdl_class[i] == r['wdl_class']
            assert b.outcome_stm[i] == r['outcome_stm']
            assert b.policy_weight[i] == r['policy_weight'] and b.value_weight[i] == r['value_weight']
            assert b.training_eligible[i] == r['training_eligible']
            expected = unpack_features(r)
            np.testing.assert_array_equal(b.planes[i], expected['planes'])
        offset += n
    assert offset == len(rows)


def test_byte_exact_nonbinary_and_negative_zero(originals):
    rows = copy.deepcopy(originals[0].positions[:1])
    planes = np.frombuffer(rows[0]['planes'], '<f4').copy()
    planes[100] = np.float32(-0.0); planes[101] = np.float32(1 / 150)
    rows[0]['planes'] = planes.tobytes()
    b = collate(rows, 'train')
    assert b.planes[0].tobytes() == rows[0]['planes']
    assert np.signbit(b.planes.reshape(-1)[100])


def test_eligibility_and_zero_value_weight(tmp_path, originals):
    def excluded(rows):
        rows[0]['training_eligible'] = False
    shard, rows, _ = write_shard(tmp_path, originals, mutate=excluded)
    batches = list(iter_batches(VerifiedStore(tmp_path), [shard], 'train', batch_size=5,
                               shuffle_buffer=12, seed=4))
    ids = identities(batches)
    assert len(ids) == len(rows)-1 and len(set(ids)) == len(ids)
    assert (rows[0]['game_id'], 0) not in ids
    assert all(b.training_eligible.all() for b in batches)
    clock_rows = [(b, i) for b in batches for i, gid in enumerate(b.game_id) if gid.endswith('-1')]
    assert len(clock_rows) == 6
    assert all(b.value_weight[i] == 0 and b.policy_weight[i] == 1 for b, i in clock_rows)
    assert set(int(b.outcome_stm[i]) for b, i in clock_rows) == {-1, 1}


def test_cross_chunk_game_shard_shuffle_determinism_and_accounting(tmp_path, originals):
    shards, expected = [], []
    for part in range(3):
        shard, rows, _ = write_shard(tmp_path, originals, n=40, part=part)
        shards.append(shard); expected.extend((r['game_id'], r['ply']) for r in rows)
    def run(seed, epoch=0, chunk=7):
        return list(iter_batches(VerifiedStore(tmp_path), shards, 'train', batch_size=64,
                                shuffle_buffer=384, seed=seed, epoch=epoch, chunk_rows=chunk))
    a, b, c, d = run(7), run(7, chunk=1), run(8), run(7, epoch=1)
    assert identities(a) == identities(b)
    assert identities(a) != identities(c) and identities(a) != identities(d)
    assert sorted(identities(a)) == sorted(expected)
    assert len(set(identities(a))) == len(expected)
    assert min(len(set(batch.game_id)) for batch in a[:-1]) >= 25
    assert max(batch.max_positions_per_game for batch in a) < 6
    assert any(len({gid.split('-')[2] for gid in batch.game_id}) > 1 for batch in a)


@pytest.mark.parametrize('split', ['train', 'validation', 'test'])
def test_split_isolation(tmp_path, originals, split):
    shard, rows, _ = write_shard(tmp_path, originals, split)
    output = list(iter_batches(VerifiedStore(tmp_path), [shard], split, batch_size=5, shuffle=False))
    assert identities(output) == [(r['game_id'], r['ply']) for r in rows]
    assert all(b.split == split and set(b.source_month) == {shard.month} for b in output)
    wrong = 'test' if split != 'test' else 'train'
    with pytest.raises(RecordError, match='Mixed input splits'):
        list(iter_batches(VerifiedStore(tmp_path), [shard], wrong))


@pytest.mark.parametrize('field,value,why', [
    ('planes', b'bad', 'byte length'), ('context', bytes(28), 'byte length'),
    ('legal_mask', bytes(584), 'illegal'), ('action', 4672, 'illegal'),
    ('action', None, 'null'), ('wdl_class', 2, 'WDL'), ('outcome_stm', -1, 'WDL'),
    ('value_weight', float('nan'), 'weights'), ('policy_weight', 0., 'weights'),
    ('split', 'test', 'split'), ('source_month', '2026-08', 'split'),
    ('color', 1, 'Color'), ('position_hash', bytes(31), 'hash'),
    ('remaining_plies', 0, 'remaining'), ('label_source', 'stockfish', 'provenance'),
    ('human_move_uci', 'bad', 'UCI'), ('fen', 'invalid', 'FEN'),
    ('planes', np.full((8,8,120), np.nan, '<f4').tobytes(), 'Nonfinite'),
])
def test_malformed_records_fail_even_if_ineligible(tmp_path, originals, field, value, why):
    def corrupt(rows):
        rows[0]['training_eligible'] = False
        rows[0][field] = value
    # Arrow itself rejects malformed fixed-size payloads; direct validator also
    # must reject them, rather than relying on the storage library.
    shard, rows, games = write_shard(tmp_path, originals)
    row = copy.deepcopy(rows[0]); row['training_eligible'] = False; row[field] = value
    game = json.loads(games[0]['metadata'])
    with pytest.raises(RecordError, match=why):
        validate_row(row, game, games[0]['actions'], shard)
    if field not in {'planes', 'context', 'position_hash'}:
        shard, _, _ = write_shard(tmp_path, originals, mutate=corrupt)
        with pytest.raises(RecordError, match=why):
            list(iter_batches(VerifiedStore(tmp_path), [shard], 'train', shuffle=False))


def test_missing_duplicate_rows_schema_and_checksum(tmp_path, originals):
    shard, rows, _ = write_shard(tmp_path, originals)
    with pytest.raises(RecordError, match='count mismatch'):
        list(iter_records(VerifiedStore(tmp_path), [replace(shard, rows=shard.rows+1)], 'train'))
    with pytest.raises(RecordError, match='Duplicate game'):
        list(iter_records(VerifiedStore(tmp_path), [shard, shard], 'train'))
    path = tmp_path/shard.positions.path
    pq.write_table(pa.Table.from_pylist(rows[:-1], schema=POSITION_SCHEMA), path)
    updated = replace(shard, positions=FileSpec(path.name,path.stat().st_size,hashlib.sha256(path.read_bytes()).hexdigest()))
    with pytest.raises(RecordError, match='count mismatch'):
        list(iter_records(VerifiedStore(tmp_path), [updated], 'train'))
    with pytest.raises(RecordError, match='Local (SHA256|size) mismatch'):
        list(iter_records(VerifiedStore(tmp_path), [shard], 'train'))
    bad_schema = POSITION_SCHEMA.with_metadata({b'schema': b'bad'})
    pq.write_table(pa.Table.from_pylist(rows, schema=bad_schema), path)
    updated = replace(shard, positions=FileSpec(path.name,path.stat().st_size,hashlib.sha256(path.read_bytes()).hexdigest()))
    with pytest.raises(RecordError, match='schema mismatch'):
        list(iter_records(VerifiedStore(tmp_path), [updated], 'train'))


def test_empty_single_partial_and_parameters(tmp_path, originals):
    store = VerifiedStore(tmp_path)
    assert list(iter_batches(store, [], 'train')) == []
    shard, rows, _ = write_shard(tmp_path, originals, n=1)
    output = list(iter_batches(store, [shard], 'train', batch_size=64))
    assert [len(b.action) for b in output] == [6]
    for kwargs in [dict(batch_size=0), dict(chunk_rows=0), dict(shuffle_buffer=1),dict(seed=-1)]:
        with pytest.raises(RecordError):
            list(iter_batches(store, [shard], 'train', **kwargs))
    with pytest.raises(RecordError, match='Only training'):
        list(iter_batches(store, [], 'test', shuffle=True))
    assert list(shuffled_records([], 8, np.random.default_rng(0))) == []


def test_local_integrity_offline_reserve_and_path_safety(tmp_path):
    with pytest.raises(RecordError, match='Unsafe'):
        FileSpec('../escape', 1, '0'*64)
    with pytest.raises(RecordError, match='downloads disabled'):
        with VerifiedStore(tmp_path).file(FileSpec('missing', 1, '0'*64)):
            pass
    with pytest.raises(RecordError, match='reserve'):
        with VerifiedStore(tmp_path, download=True, reserve_bytes=2**90).file(FileSpec('missing', 1, '0'*64)):
            pass
    with pytest.raises(RecordError, match='downloads disabled'):
        inventory(VerifiedStore(tmp_path))
    assert REVISION == 'b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c'


@pytest.mark.parametrize('mode', ['ok', 'short', 'hash', 'network', 'oversize'])
def test_download_atomic_integrity_cleanup_and_reuse(tmp_path, monkeypatch, mode):
    payload=b'bounded-original'
    digest=hashlib.sha256(payload).hexdigest()
    spec=FileSpec('data/small.parquet',len(payload),digest)
    class Response:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def raise_for_status(self): pass
        def iter_bytes(self,chunk_size):
            if mode=='network':
                yield payload[:3]
                raise OSError('interrupted')
            yield payload if mode=='ok' else payload[:-1] if mode=='short' else b'x'*len(payload) if mode=='hash' else payload+b'x'
    store=VerifiedStore(tmp_path,download=True,reserve_bytes=0)
    monkeypatch.setattr(store,'_read_remote',lambda path: Response())
    if mode=='ok':
        with store.file(spec) as path:
            assert path.read_bytes()==payload
        assert not path.exists()  # ephemeral original by default
        store.keep_downloads=True
        with store.file(spec) as path: pass
        assert path.exists()
        offline=VerifiedStore(tmp_path)
        with offline.file(spec) as reused: assert reused==path
        assert path.exists()  # existing local original belongs to user
    else:
        with pytest.raises((RecordError,OSError)):
            with store.file(spec): pass
        assert not (tmp_path/spec.path).exists()
        assert not list(tmp_path.rglob('*.partial'))


def test_missing_game_unsupported_metadata_and_duplicate_ply(tmp_path, originals):
    for mutate, message in [
        (lambda rows: rows[0].update(game_id='missing'), 'no game'),
        (lambda rows: rows[1].update(ply=0,action=rows[0]['action'],outcome_stm=1,wdl_class=0,
                                    color=0,remaining_plies=6,fen=rows[0]['fen'],
                                    human_move_uci=rows[0]['human_move_uci'],
                                    planes=rows[0]['planes'],context=rows[0]['context'],legal_mask=rows[0]['legal_mask']),
         'Duplicate/backward'),
    ]:
        shard,_,_=write_shard(tmp_path, originals, mutate=mutate)
        with pytest.raises(RecordError,match=message):
            list(iter_records(VerifiedStore(tmp_path),[shard],'train'))
