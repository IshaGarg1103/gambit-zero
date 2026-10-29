import copy
import io
import json
from pathlib import Path

import chess
import chess.pgn
import jax
import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml
import zstandard as zstd

from tinychess.chess.pgx_env import init, step
from tinychess.chess.state_features import encode_jit
from tinychess.data.human import replay_game, parse_filter, split_for_month, RejectedGame
from tinychess.data.sources import game_blocks, open_text
from tinychess.data.storage import MonthWriter, read_games, unpack_features
from tinychess.data.audits import audit

CONFIG = yaml.safe_load((Path(__file__).parents[1] / 'configs/data_human.yaml').read_text())


def pgn(moves, result='1-0', **headers):
    game = chess.pgn.Game()
    game.headers.update(Event='Rated Blitz game', Site='https://lichess.org/AbCd1234',
                        WhiteElo='2000', BlackElo='2100', TimeControl='300+0',
                        UTCDate='2026.01.01', UTCTime='00:00:01', Termination='Normal', Result=result)
    game.headers.update(headers)
    node = game
    board = game.board()
    for uci in moves:
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves
        node = node.add_variation(move)
        board.push(move)
    node.comment = '[%eval 99.9] must not be used as a target'
    return game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=True)) + '\n\n'


def replay(text):
    return replay_game(text, '2026-01', 1, 'https://example.invalid/archive.pgn.zst', CONFIG)


@pytest.mark.parametrize('result,white,black', [('1-0',1,-1),('0-1',-1,1),('1/2-1/2',0,0)])
def test_side_to_move_outcomes_and_nonboard_endings(result, white, black, monkeypatch):
    monkeypatch.setattr(chess.pgn.GameNode, 'eval', lambda *a: pytest.fail('eval annotations read'))
    game = replay(pgn(['e2e4','e7e5'], result))
    assert [r['outcome_stm'] for r in game.positions] == [white,black]
    assert [r['wdl_class'] for r in game.positions] == [0 if white>0 else 2 if white<0 else 1,0 if black>0 else 2 if black<0 else 1]
    assert game.game['metadata']['final_board_reason'] == 'ongoing'
    assert not game.game['metadata']['actual_checkmate']
    assert '99.9' not in json.dumps(game.game)


def test_actual_checkmate_and_time_forfeit_mask():
    mate = replay(pgn(['f2f3','e7e5','g2g4','d8h4'], '0-1'))
    assert mate.game['metadata']['ending_semantics'] == 'board_checkmate'
    assert mate.game['metadata']['actual_checkmate']
    clock = replay(pgn(['e2e4','e7e5'], '0-1', Termination='Time forfeit'))
    assert all(r['value_weight']==0 and r['policy_weight']==1 for r in clock.positions)
    assert clock.game['metadata']['ending_semantics'] == 'time_forfeit'
    with pytest.raises(RejectedGame, match='checkmate_result'):
        replay(pgn(['f2f3','e7e5','g2g4','d8h4'], '1-0'))


@pytest.mark.parametrize('headers,why', [
    ({'WhiteElo':'?'},'invalid_rating'), ({'BlackElo':'1799'},'rating_below'),
    ({'TimeControl':'60+0'},'time_control_below'), ({'TimeControl':'-'},'invalid_time'),
    ({'Variant':'Chess960'},'variant'), ({'Variant':'Atomic'},'variant'),
    ({'Event':'Casual Blitz game'},'not_rated'), ({'Termination':'Abandoned'},'excluded_termination'),
    ({'Termination':'Rules infraction'},'excluded_termination'), ({'Result':'*'},'invalid_result'),
    ({'FEN':chess.STARTING_FEN},'nonstandard_start'),
])
def test_filters(headers, why):
    with pytest.raises(RejectedGame, match=why):
        replay(pgn(['e2e4','e7e5'], **headers))


def test_corrupted_and_inconsistent_game():
    with pytest.raises(RejectedGame):
        replay(pgn(['e2e4','e7e5']).replace('1. e4', '1. e5'))
    with pytest.raises(RejectedGame, match='movetext_result'):
        replay(pgn(['e2e4','e7e5']).replace(' 1-0', ' 0-1'))


@pytest.mark.parametrize('piece', ['q','r','b','n'])
def test_all_white_black_capture_promotions(piece):
    moves = ['a2a4','h7h5','a4a5','h5h4','a5a6','h4h3','a6b7','h3g2','b7a8'+piece,'g2h1'+piece]
    record = replay(pgn(moves))
    assert [r['human_move_uci'] for r in record.positions][-2:] == moves[-2:]
    assert record.game['metadata']['plies'] == 10


@pytest.mark.parametrize('moves', [
    ['e2e4','a7a6','e4e5','d7d5','e5d6'],
    ['g1f3','g8f6','e2e3','e7e6','f1e2','f8e7','e1g1','e8g8'],
    ['g1f3','g8f6','f3g1','f6g8']*2,
])
def test_replay_history_and_shared_encoder(moves):
    record = replay(pgn(moves, '1/2-1/2'))
    state = init(jax.random.PRNGKey(37))
    for row, action in zip(record.positions, record.game['actions']):
        decoded = unpack_features(row)
        current = jax.device_get(encode_jit(state))
        for key in current:
            np.testing.assert_array_equal(decoded[key], current[key])
        assert decoded['legal_mask'][action]
        state = step(state, action)
    if len(moves)==8 and moves[0]=='g1f3' and moves[-1]=='f6g8':
        assert max(r['repetition_count'] for r in record.positions) == 2


def test_horizon_rejected_not_draw():
    text = pgn(['g1f3','g8f6','f3g1','f6g8']*129, '1/2-1/2')
    with pytest.raises(RejectedGame, match='exceeds_512'):
        replay(text)
    config = copy.deepcopy(CONFIG)
    # 512 plies can end on an automatic rule; the actual truncation branch is
    # tested by the M0 final-adjudication suite. This early fivefold continuation
    # is rejected as an unreplayable source, not converted to a new draw target.
    with pytest.raises(RejectedGame, match='moves_after_board_episode_end'):
        replay(pgn(['g1f3','g8f6','f3g1','f6g8']*5))


def test_month_splits():
    assert [split_for_month(f'2026-0{i}') for i in range(1,10)] == ['train']*7+['validation','test']
    with pytest.raises(ValueError):
        split_for_month('2025-01')


def test_parquet_roundtrip_and_leakage(tmp_path):
    for month, site in [('2026-01','AbCd1234'),('2026-08','EfGh5678'),('2026-09','IjKl9012')]:
        game = replay_game(pgn(['e2e4','e7e5'], Site='https://lichess.org/'+site), month, 1, 'offline-fixture', CONFIG)
        writer = MonthWriter(tmp_path, month)
        writer.append(game)
        writer.close()
        rows = pq.read_table(writer.positions_path).to_pylist()
        assert rows[0]['action'] == game.game['actions'][0]
        np.testing.assert_array_equal(unpack_features(rows[0])['planes'], unpack_features(game.positions[0])['planes'])
    assert len(list(read_games(tmp_path))) == 3
    result = audit(tmp_path)
    assert result['overlap']['ply_ge_0']['validation']['shared_with_train'] == 2
    assert result['disjoint_eval_rows'] == {}
    assert result['duplicate_game_ids'] == 0


def test_stream_framing_bounds_and_recovery(tmp_path):
    good = pgn(['e2e4','e7e5'])
    bad = '[Event "broken"]\n' + 'x'*200 + '\n'
    blocks = list(game_blocks(io.StringIO(bad+good), max_game_bytes=10000, max_line_chars=128))
    assert blocks[0][1] == 'overlong_line'
    assert blocks[1][1] is None
    assert parse_filter(blocks[1][0], CONFIG['filters'])[1]
    path = tmp_path/'sample.pgn.zst'
    path.write_bytes(zstd.ZstdCompressor().compress(good.encode()))
    with open_text(path) as text:
        assert len(list(game_blocks(text))) == 1


def test_stage_and_quota_guard_no_50k_path(tmp_path):
    from tinychess.data.build_human import build
    path = tmp_path/'config.yaml'
    config = copy.deepcopy(CONFIG)
    config['months']['2026-01']['smoke1000'] = 50000
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='quotas'):
        build(path, tmp_path/'never-created', 1000, offline=True)
    assert not (tmp_path/'never-created').exists()
    with pytest.raises(ValueError, match='100/1000'):
        build(path, tmp_path/'never-created', 50000, offline=True)


def test_pipeline_fails_immediately_on_unexplained_parity_error(tmp_path, monkeypatch):
    import hashlib
    import tinychess.data.build_human as module
    from tinychess.validation.parity import ParityError
    config = copy.deepcopy(CONFIG)
    config['months'] = {'2026-01': {'split':'train','smoke100':100}}
    cache = tmp_path/'cache'
    cache.mkdir()
    compressed = zstd.ZstdCompressor().compress(pgn(['e2e4','e7e5']).encode())
    path = cache/'2026-01.pgn.zst.prefix'
    path.write_bytes(compressed)
    path.with_suffix('.prefix.json').write_text(json.dumps(dict(month='2026-01',
        url=config['source']['url_template'].format(month='2026-01'), prefix_bytes=len(compressed),
        prefix_sha256=hashlib.sha256(compressed).hexdigest())))
    config['source']['cache'] = str(cache)
    config['source']['compressed_prefix_bytes'] = len(compressed)
    config_path = tmp_path/'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    def fail(*args, **kwargs):
        raise ParityError('injected legal-action mismatch')
    monkeypatch.setattr(module, 'replay_game', fail)
    with pytest.raises(ParityError, match='injected'):
        module.build(config_path, tmp_path/'output', 100, offline=True)
    evidence = json.loads((tmp_path/'output/FAIL_STOP.json').read_text())
    assert evidence['type'] == 'ParityError'
    assert evidence['counts']['encountered_games'] == 1


def test_frozen_evaluation_exclusion():
    config = copy.deepcopy(CONFIG)
    config['frozen_evaluation_fens'] = [chess.STARTING_FEN]
    record = replay_game(pgn(['e2e4','e7e5']), '2026-01', 1, 'offline-fixture', config)
    assert not record.positions[0]['training_eligible']
    assert record.positions[1]['training_eligible']


def test_compact_reader_rejects_incompatible_encoder_metadata(tmp_path):
    import pyarrow as pa
    from tinychess.data.storage import GAME_SCHEMA
    metadata = dict(GAME_SCHEMA.metadata)
    metadata[b'encoder'] = b'incompatible-v999'
    schema = GAME_SCHEMA.with_metadata(metadata)
    pq.write_table(pa.Table.from_pylist([dict(game_id='x',metadata='{}',actions=[])],schema=schema),tmp_path/'bad.games.parquet')
    with pytest.raises(ValueError,match='metadata'):
        list(read_games(tmp_path))


@pytest.mark.parametrize('termination,result,ending', [
    ('Resignation','1-0','resignation_reported'),
    ('Draw agreement','1/2-1/2','draw_agreement_reported'),
    ('Draw claim','1/2-1/2','draw_claim_reported'),
])
def test_explicit_human_endings_do_not_require_board_terminal(termination,result,ending):
    record = replay(pgn(['e2e4','e7e5'], result, Termination=termination))
    assert record.game['metadata']['ending_semantics'] == ending
    assert record.game['metadata']['final_board_reason'] == 'ongoing'
    assert not record.game['metadata']['actual_checkmate']
    assert all(row['value_weight'] == 1 for row in record.positions)
