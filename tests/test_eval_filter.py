"""Filtering must remove only exact evaluation boards, preserving tensor bytes."""
import copy

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tinychess.data.eval_filter import position_keys, prefetched, verify_retained_rows, write_filter
from tinychess.data.storage import POSITION_SCHEMA


def payload(split='train'):
    rows = [dict(game_id='fixture', source_month='2026-01', split=split, ply=ply,
                 human_move_uci='e2e4', action=1, outcome_stm=1, wdl_class=0,
                 value_weight=1, policy_weight=1, color=0, fen='unit-fixture',
                 repetition_count=1, position_hash=bytes([ply])*32, training_eligible=True,
                 remaining_plies=3-ply, label_source='human_game', planes=bytes(30720),
                 context=bytes(32), legal_mask=bytes(584)) for ply in range(3)]
    output = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(rows, schema=POSITION_SCHEMA), output)
    return output.getvalue().to_pybytes(), rows


def test_exact_filter_reproduces_and_preserves_every_retained_field(tmp_path):
    source, rows = payload()
    excluded = {rows[1]['position_hash']}
    first, second = tmp_path/'a.parquet', tmp_path/'b.parquet'
    assert write_filter(source, excluded, first) == write_filter(source, excluded, second)
    assert verify_retained_rows(source, excluded, first) == 2
    assert pq.read_table(first).to_pylist() == [rows[0], rows[2]]
    assert first.read_bytes() == second.read_bytes()
    changed = copy.deepcopy([rows[0], rows[2]])
    changed[0]['planes'] = bytes([1])*30720
    pq.write_table(pa.Table.from_pylist(changed, schema=POSITION_SCHEMA), first)
    with pytest.raises(ValueError, match='changed'):
        verify_retained_rows(source, excluded, first)


def test_filter_refuses_heldout_data(tmp_path):
    source, _ = payload('validation')
    with pytest.raises(ValueError, match='only apply to training'):
        write_filter(source, set(), tmp_path/'bad.parquet')


def test_empty_and_unchanged_filters_remain_lossless(tmp_path):
    source, rows = payload()
    path = tmp_path/'filtered.parquet'
    assert write_filter(source, set(), path)[0] == 3
    assert verify_retained_rows(source, set(), path) == 3
    excluded = {r['position_hash'] for r in rows}
    assert write_filter(source, excluded, path)[0] == 0
    assert verify_retained_rows(source, excluded, path) == 0


def test_evaluation_key_ignores_clocks_and_only_uncapturable_ep():
    import chess
    board = chess.Board()
    board.push_uci('e2e4')
    with_ep = board.fen(en_passant='fen')
    without_ep = board.fen(en_passant='legal')
    assert position_keys([with_ep, without_ep]) == position_keys([with_ep])
    board.turn = not board.turn
    assert position_keys([board.fen(), with_ep]) != position_keys([with_ep])


def test_prefetch_is_ordered_bounded_and_propagates_failed_source():
    import threading
    import time
    active = peak = 0
    lock = threading.Lock()
    def fetch(item):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.002)
        with lock:
            active -= 1
        if item == 5:
            raise ValueError('source checksum mismatch')
        return item*2
    assert list(prefetched(range(5), fetch, workers=2)) == [(i, i*2) for i in range(5)]
    assert peak <= 2
    with pytest.raises(ValueError, match='checksum'):
        list(prefetched(range(10), fetch, workers=2))
