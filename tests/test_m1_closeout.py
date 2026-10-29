"""Diagnostic indices must never admit a training board or corrupt labels."""
import json

import pytest

from tinychess.data.archive_scan import DiskGuard
from tinychess.data.closeout import CloseoutIndex


def game(gid, month, split, result='1-0', weight=1):
    metadata = dict(game_id=gid, source_month=month, split=split, plies=2,
                    white_elo=2100, black_elo=1900, time_control='300+0', result=result,
                    outcome_white={'1-0': 1, '0-1': -1, '1/2-1/2': 0}[result],
                    value_weight=weight, ending_semantics='time_forfeit' if weight == 0 else 'decisive_nonboard_unspecified',
                    trajectory_sha256=gid, eco='A00', utc_date=month.replace('-', '.')+'.01')
    return dict(game_id=gid, metadata=json.dumps(metadata), actions=[0, 1])


def rows(gid, month, split, keys, result=1, weight=1):
    return [dict(game_id=gid, source_month=month, split=split, ply=ply,
                 position_hash=bytes([key])*32, outcome_stm=result*(-1)**ply,
                 wdl_class={1: 0, 0: 1, -1: 2}[result*(-1)**ply], value_weight=weight,
                 policy_weight=1, training_eligible=True) for ply, key in enumerate(keys)]


@pytest.fixture
def index(tmp_path):
    value = CloseoutIndex(tmp_path, 'fixture', DiskGuard(tmp_path, 2**30, 2**20))
    yield value
    value.close()


def add(index, gid, month, split, keys, result='1-0', weight=1):
    index.begin()
    metadata = index.add_game(game(gid, month, split, result, weight), month, split)
    values = rows(gid, month, split, keys, metadata['outcome_white'], weight)
    index.add_rows(values, {gid: metadata}, month, split, {})
    index.commit()
    return values


def test_disjoint_excludes_all_train_boards_preserves_row_offsets_and_heldout_repeats(index):
    add(index, 't', '2026-01', 'train', [1, 2])
    validation = add(index, 'v', '2026-08', 'validation', [2, 3])
    test = add(index, 'e', '2026-09', 'test', [3, 4])
    selected = list(index.disjoint_rows(validation, 'val.parquet'))
    assert [(r['game_id'], r['ply'], r['row_in_shard']) for r in selected] == [('v', 1, 1)]
    assert selected[0]['source_path'] == 'val.parquet'
    assert len(list(index.disjoint_rows(test, 'test.parquet'))) == 2
    assert index.result()['overlap']['all_plies']['train_validation_shared'] == 1
    assert index.result()['counts']['positions'] == 6


@pytest.mark.parametrize('result,weight', [('1-0', 1), ('0-1', 0), ('1/2-1/2', 1)])
def test_closeout_checks_perspective_and_masks(index, result, weight):
    values = add(index, 'v', '2026-08', 'validation', [1, 2], result, weight)
    assert index.result()['counts']['value_masked_rows'] == 2*(weight == 0)
    assert len(list(index.disjoint_rows(values, 'val.parquet'))) == 2


@pytest.mark.parametrize('field,value,error', [
    ('outcome_stm', -1, 'WDL'), ('wdl_class', 2, 'WDL'),
    ('value_weight', 0, 'WDL'), ('ply', 1, 'ordering'),
    ('source_month', '2026-09', 'month/split'),
    ('training_eligible', False, 'excluded'), ('position_hash', b'x', 'hash length'),
])
def test_closeout_halts_for_invalid_rows(index, field, value, error):
    index.begin()
    metadata = index.add_game(game('t', '2026-01', 'train'), '2026-01', 'train')
    values = rows('t', '2026-01', 'train', [1, 2])
    values[0][field] = value
    with pytest.raises(ValueError, match=error):
        index.add_rows(values, {'t': metadata}, '2026-01', 'train', {})
    index.rollback()


def test_duplicate_game_even_identical_content_stops_closeout(index):
    add(index, 't', '2026-01', 'train', [1, 2])
    index.begin()
    with pytest.raises(ValueError, match='Duplicate'):
        index.add_game(game('t', '2026-01', 'train'), '2026-01', 'train')
    index.rollback()


def test_unindexed_board_cannot_silently_enter_diagnostic(index):
    with pytest.raises(ValueError, match='Unindexed'):
        list(index.disjoint_rows(rows('v', '2026-08', 'validation', [1, 2]), 'val.parquet'))
