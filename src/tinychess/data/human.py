"""Offline parsing, strict filtering and history-preserving Pgx human replay."""
from dataclasses import dataclass
import hashlib
import io
import json
import re

import chess
import chess.pgn
import jax
import numpy as np

from tinychess.chess.action_codec import ACTION_MAP_VERSION
from tinychess.chess.pgx_env import init, step, RULES_POLICY_VERSION, MAX_PLIES, reason
from tinychess.chess.state_features import encode_jit, ENCODER_VERSION
from tinychess.validation.parity import Oracle, ParityError, board_identity

SCHEMA_VERSION = 'human_v3'
RESULTS = {'1-0': 1, '0-1': -1, '1/2-1/2': 0}
WDL_ORDER = ['win', 'draw', 'loss']


class RejectedGame(ValueError):
    pass


class RecordingBuilder(chess.pgn.GameBuilder):
    def visit_result(self, result):
        self.game.movetext_result = result
        super().visit_result(result)

    def handle_error(self, error):
        self.game.errors.append(error)


@dataclass
class Replayed:
    game: dict
    positions: list


def split_for_month(month):
    if not re.fullmatch(r'2026-0[1-9]', month):
        raise ValueError(f'Unconfigured source month: {month}')
    return 'train' if month <= '2026-07' else 'validation' if month == '2026-08' else 'test'


def parse_filter(text, filters):
    try:
        game = chess.pgn.read_game(io.StringIO(text), Visitor=RecordingBuilder)
    except (ValueError, IndexError) as exc:
        raise RejectedGame('parse_error') from exc
    if game is None:
        raise RejectedGame('parse_error')
    if game.errors:
        raise RejectedGame('illegal_move' if any(isinstance(e, chess.IllegalMoveError) for e in game.errors) else 'parse_error')
    h = game.headers
    if h.get('Variant', 'Standard').lower() != 'standard':
        raise RejectedGame('variant')
    if h.get('SetUp') == '1' or h.get('FEN') or game.board().fen() != chess.STARTING_FEN:
        raise RejectedGame('nonstandard_start')
    if not h.get('Event', '').startswith('Rated '):
        raise RejectedGame('not_rated')
    try:
        if min(int(h.get('WhiteElo', '?')), int(h.get('BlackElo', '?'))) < filters['min_rating']:
            raise RejectedGame('rating_below_minimum')
    except ValueError as exc:
        if isinstance(exc, RejectedGame):
            raise
        raise RejectedGame('invalid_rating') from exc
    tc = h.get('TimeControl', '')
    match = re.fullmatch(r'(\d+)(?:\+(\d+))?', tc)
    if not match:
        raise RejectedGame('invalid_time_control')
    if int(match[1]) < filters['min_base_seconds']:
        raise RejectedGame('time_control_below_minimum')
    term = h.get('Termination', '')
    if term.casefold() in {t.casefold() for t in filters['reject_terminations']}:
        raise RejectedGame('excluded_termination:' + term.casefold())
    if h.get('Result') not in RESULTS:
        raise RejectedGame('invalid_result')
    if getattr(game, 'movetext_result', None) != h['Result']:
        raise RejectedGame('missing_or_inconsistent_movetext_result')
    moves = list(game.mainline_moves())
    if not moves:
        raise RejectedGame('empty_game')
    if len(moves) > min(filters['max_plies'], MAX_PLIES):
        raise RejectedGame('exceeds_512_ply_horizon')
    return game, moves


def classify_ending(board, termination, result):
    if board.is_checkmate():
        return 'board_checkmate'
    automatic = board.outcome(claim_draw=False)
    if automatic:
        return 'board_' + automatic.termination.name.lower()
    name = termination.casefold()
    if 'time' in name and ('forfeit' in name or 'out' in name):
        return 'time_forfeit'
    if 'resign' in name:
        return 'resignation_reported'
    if 'claim' in name:
        return 'draw_claim_reported'
    if 'agreement' in name or 'agreed' in name:
        return 'draw_agreement_reported'
    return 'draw_nonboard_unspecified' if result == '1/2-1/2' else 'decisive_nonboard_unspecified'


def replay_game(text, month, ordinal, url, config, materialize=True):
    parsed, moves = parse_filter(text, config['filters'])
    h = parsed.headers
    result = h['Result']
    raw_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
    site = h.get('Site', '')
    match = re.fullmatch(r'https://lichess.org/([a-zA-Z0-9]{8})(?:/(?:white|black))?', site)
    game_id = match[1] if match else raw_hash
    split = split_for_month(month)
    # Use a fixed seed for storage reproducibility; color is always state._x.color,
    # never an assumption that current_player==White.
    state = init(jax.random.PRNGKey(0))
    oracle = Oracle()
    positions, indices = [], []
    time_forfeit = classify_ending(chess.Board(), h.get('Termination', ''), result) == 'time_forfeit'
    value_weight = config['filters']['time_forfeit_value_weight'] if time_forfeit else 1.0
    frozen = {board_identity(chess.Board(fen)) for fen in config.get('frozen_evaluation_fens', [])}
    for ply, move in enumerate(moves):
        actions, host, rep = oracle.check(state)
        if host.terminated or host.truncated:
            raise RejectedGame('moves_after_board_episode_end')
        uci = move.uci()
        if uci not in actions:
            raise ParityError(f'Human move missing from full legal mask: {game_id} {ply} {uci}')
        action = actions[uci]
        indices.append(action)
        outcome = RESULTS[result] * (1 if oracle.board.turn else -1)
        cls = 0 if outcome > 0 else 2 if outcome < 0 else 1
        key = hashlib.sha256(board_identity(oracle.board).encode()).digest()
        record = dict(game_id=game_id, source_month=month, split=split, ply=ply,
                      human_move_uci=uci, action=action, outcome_stm=outcome, wdl_class=cls,
                      value_weight=float(value_weight), policy_weight=1.0, color=int(host._x.color),
                      fen=oracle.board.fen(en_passant='fen'), repetition_count=rep,
                      position_hash=key, training_eligible=not (split == 'train' and board_identity(oracle.board) in frozen),
                      remaining_plies=len(moves)-ply, label_source='human_game')
        if materialize:
            features = jax.device_get(encode_jit(state))
            record.update(planes=np.asarray(features['planes'], dtype='<f4').tobytes(),
                          context=np.asarray(features['context'], dtype='<f4').tobytes(),
                          legal_mask=np.packbits(features['legal_mask'], bitorder='little').tobytes())
        positions.append(record)
        state = step(state, action)
        oracle.push(uci)
    _, host, _ = oracle.check(state)
    if host.truncated:
        raise RejectedGame('512_ply_truncation_no_final_outcome')
    board = oracle.board
    if board.is_checkmate():
        if not host.terminated or reason(state) != 'checkmate':
            raise ParityError('Final actual checkmate disagrees with Pgx')
        if result != board.result(claim_draw=False):
            raise RejectedGame('checkmate_result_inconsistent')
    elif board.outcome(claim_draw=False) is not None and result != board.result(claim_draw=False):
        raise RejectedGame('board_draw_result_inconsistent')
    ending = classify_ending(board, h.get('Termination', ''), result)
    metadata = dict(schema=SCHEMA_VERSION, game_id=game_id, source_month=month, split=split,
                    source_ordinal=ordinal, archive_url=url, raw_pgn_sha256=raw_hash,
                    result=result, outcome_white=RESULTS[result], value_weight=float(value_weight),
                    termination_header=h.get('Termination', ''), ending_semantics=ending,
                    final_board_reason=reason(state), final_board_fen=board.fen(en_passant='fen'),
                    actual_checkmate=board.is_checkmate(), plies=len(moves),
                    white_elo=int(h['WhiteElo']), black_elo=int(h['BlackElo']),
                    time_control=h['TimeControl'], utc_date=h.get('UTCDate', h.get('Date', 'unknown')),
                    utc_time=h.get('UTCTime', 'unknown'), eco=h.get('ECO', 'unknown'), opening=h.get('Opening', 'unknown'),
                    encoder_version=ENCODER_VERSION, action_map_version=ACTION_MAP_VERSION,
                    rules_policy=RULES_POLICY_VERSION, label_source='human_game',
                    trajectory_sha256=hashlib.sha256(np.asarray(indices, dtype='<u2').tobytes()).hexdigest())
    if metadata['utc_date'] != 'unknown':
        metadata['utc_month_differs_from_source'] = metadata['utc_date'][:7].replace('.', '-') != month
    return Replayed(dict(metadata=metadata, actions=indices), positions)
