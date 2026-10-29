"""Stage 0 only: exhaustive legal successors at sampled roots and policy orientation.

python-chess remains an offline oracle. Reuse the accepted history-aware Oracle
rather than adding another production rules/encoder implementation. Each branch
retains the complete parent history; only the explicit sparse FEN fixtures start
with history discarded. No dataset, model, search or training work is performed.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random

import chess
import jax
import numpy as np
import pytest

from tinychess.chess.action_codec import action_to_uci, uci_to_action
from tinychess.chess.pgx_env import ENV, init, step, to_fen
from tinychess.chess.state_features import ENCODER_VERSION, encode_jit
from tinychess.validation.parity import Oracle, ParityError

SEEDS = (0, 1, 7, 19, 37, 73, 101, 2026)
SAMPLED_PLIES = (0, 1, 8, 9, 32, 33, 64, 65, 96, 97, 126, 127)
CYCLE = ('g1f3', 'g8f6', 'f3g1', 'f6g8')


@dataclass
class Case:
    name: str
    seed: int
    state: object
    oracle: Oracle
    moves: tuple
    random_game: bool = False

    def context(self, action=None, uci=None):
        return json.dumps(dict(case=self.name, seed=self.seed, ply=self.oracle.plies,
                               root_fen=self.oracle.board.fen(en_passant='fen'),
                               action=action, uci=uci, moves_uci=self.moves), sort_keys=True)


@pytest.fixture(scope='module')
def coverage():
    result = dict(random_seeds=list(SEEDS), sampled_plies=list(SAMPLED_PLIES),
                  trajectory_positions_checked=0, transition_roots=0, random_transition_roots=0,
                  special_transition_roots=0, successors_checked=0, mapping_roots=0,
                  policy_actions_checked=0, transition_mismatches=0, mapping_mismatches=0,
                  transition_passed=False, mapping_passed=False)
    yield result
    result['status'] = 'PASS' if result['transition_passed'] and result['mapping_passed'] else 'FAIL_OR_INCOMPLETE'
    output = os.environ.get('TINYCHESS_STAGE0_COVERAGE')
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, sort_keys=True, indent=2)+'\n')


@pytest.fixture(scope='module')
def corpus(coverage):
    cases = []
    for seed in SEEDS:
        rng = random.Random(seed)
        oracle, state, moves = Oracle(), init(jax.random.PRNGKey(seed)), []
        for ply in range(128):
            # Check the played path too, including roots not selected for the
            # exhaustive successor audit. Stop instead of discarding a bad game.
            try:
                actions, host, _ = oracle.check(state)
            except ParityError as exc:
                pytest.fail(f'{Case("random-path", seed, state, oracle, tuple(moves)).context()}: {exc}')
            coverage['trajectory_positions_checked'] += 1
            if host.terminated or host.truncated:
                break
            if ply in SAMPLED_PLIES:
                cases.append(Case(f'random-{seed}-{ply}', seed, state, deepcopy(oracle), tuple(moves), True))
            # Choose from the independent python-chess move list, never the Pgx
            # generator alone. Oracle already verified the complete move set.
            uci = rng.choice(sorted(m.uci() for m in oracle.board.legal_moves))
            state = step(state, actions[uci])
            oracle.push(uci)
            moves.append(uci)

    prefixes = {
        'castle-k-white': 'e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 d2d3 f8c5',
        'castle-k-black': 'e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 d2d3 f8c5 a2a3',
        'castle-q-white': 'd2d4 d7d5 b1c3 b8c6 c1f4 c8f5 d1d2 d8d7 e2e3 e7e6',
        'castle-q-black': 'd2d4 d7d5 b1c3 b8c6 c1f4 c8f5 d1d2 d8d7 e2e3 e7e6 g1f3',
        'ep-white': 'e2e4 a7a6 e4e5 d7d5',
        'ep-black': 'a2a3 e7e5 a3a4 e5e4 d2d4',
        'mate-white': 'e2e4 e7e5 d1h5 b8c6 f1c4 g8f6',
        'mate-black': 'f2f3 e7e5 g2g4',
        'evade-black': 'e2e4 d7d5 f1b5',
        'evade-white': 'd2d4 e7e5 a2a3 f8b4',
        'repetition-three': ' '.join(CYCLE+ CYCLE[:3]),
        'repetition-four': ' '.join(CYCLE*2 + CYCLE[:3]),
        'repetition-five': ' '.join(CYCLE*3 + CYCLE[:3]),
        'repetition-irrelevant-ep': ' '.join(('a2a4', 'a7a5') + CYCLE*3 + CYCLE[:3]),
    }
    for number, (name, line) in enumerate(prefixes.items()):
        seed = 3000+number
        oracle, state = Oracle(), init(jax.random.PRNGKey(seed))
        moves = tuple(line.split())
        for uci in moves:
            state = step(state, uci_to_action(jax.device_get(state), uci))
            oracle.push(uci)
        cases.append(Case(name, seed, state, oracle, moves))

    sparse = {
        'promotion-capture': '1r2k3/P7/8/8/8/8/7p/4K1R1 w - - 0 1',
        'stalemate-or-mate': '7k/5K2/8/6Q1/8/8/8/8 w - - 0 1',
        'capture-to-insufficient': '7k/8/8/8/8/8/3n4/2B1K3 w - - 0 1',
        'claimable-fifty': '7k/8/8/8/8/8/8/R3K3 w - - 99 1',
        'automatic-seventyfive': '7k/8/8/8/8/8/8/R3K3 w - - 149 1',
        'mate-precedes-seventyfive': '7k/8/5KQ1/8/8/8/8/8 w - - 149 1',
    }
    for name, fen in sparse.items():
        for mirror in [False, True]:
            board = chess.Board(fen)
            if mirror:
                board = board.mirror()
            assert board.is_valid(), (name, board.fen())
            state = ENV.from_fen(board.fen(en_passant='fen'), history='discard')
            cases.append(Case(f'{name}-{"black" if mirror else "white"}', 4000+int(mirror),
                              state, Oracle(board), ()))
    coverage['case_ids'] = [case.name for case in cases]
    return cases


def test_a_every_legal_action_transition(corpus, coverage):
    events = Counter()
    castling_moves, promotions, root_fens = set(), set(), set()
    digest = hashlib.sha256()
    for case in corpus:
        action = uci = None
        try:
            _, host, _ = case.oracle.check(case.state)
            assert not bool(host.terminated | host.truncated), case.context()
            actions = np.flatnonzero(host.legal_action_mask)
            decoded = [action_to_uci(host, int(a)) for a in actions]
            assert len(set(decoded)) == len(actions), case.context()
            assert set(decoded) == {m.uci() for m in case.oracle.board.legal_moves}, case.context()
            root_fens.add(case.oracle.board.fen(en_passant='fen'))
            coverage['transition_roots'] += 1
            coverage['random_transition_roots' if case.random_game else 'special_transition_roots'] += 1
            digest.update(json.dumps([case.name, case.seed, case.oracle.board.fen(en_passant='fen'),
                                      list(zip(map(int, actions), decoded))], separators=(',', ':')).encode())
            for action, uci in zip(actions, decoded):
                action = int(action)
                context = case.context(action, uci)
                branch = deepcopy(case.oracle)  # Preserve counts, recent boards and the full python-chess stack.
                board = branch.board
                move = chess.Move.from_uci(uci)
                mover = 'white' if board.turn else 'black'
                events['mover_'+mover] += 1
                if board.is_castling(move):
                    castling_moves.add(uci)
                if board.is_en_passant(move):
                    events['en_passant_'+mover] += 1
                if move.promotion:
                    promotions.add(mover+':'+chess.piece_symbol(move.promotion))
                events['check_evasions'] += int(board.is_check())
                branch.push(uci)  # python-chess independently applies the decoded move.
                child = step(case.state, action)
                _, child_host, _ = branch.check(child)  # Existing full FEN, legal-set, history/hash, status/reward oracle.
                assert to_fen(child_host) == board.fen(en_passant='fen'), context
                assert int(child_host._x.halfmove_count) == board.halfmove_clock, context
                assert int(child_host._x.fullmove_count) == board.fullmove_number, context
                assert int(child_host._x.color) == (0 if board.turn else 1), context
                assert int(child_host._x.step_count) == case.oracle.plies+1, context
                np.testing.assert_array_equal(child_host._x.hash_history[1:], host._x.hash_history[:-1], err_msg=context)
                # A second terminal reference uses python-chess's public outcome
                # API with voluntary claims disabled, matching the accepted policy.
                outcome = board.outcome(claim_draw=False)
                assert bool(child_host.terminated) == (outcome is not None), context
                assert not bool(child_host.truncated), context  # Corpus roots are strictly below the 512-ply cap.
                expected = np.zeros(2, np.float32)
                if outcome and outcome.winner is not None:
                    winner = 0 if outcome.winner == chess.WHITE else 1
                    expected[:] = -1
                    expected[int(child_host._player_order[winner])] = 1
                np.testing.assert_array_equal(child_host.rewards, expected, err_msg=context)
                if outcome:
                    events['terminal_'+outcome.termination.name.lower()] += 1
                else:
                    events['claimable_fifty_ongoing'] += int(board.halfmove_clock >= 100)
                events['checks_delivered'] += int(board.is_check())
                events['current_threefold_successors'] += int(board.is_repetition(3))
                events['mate_at_150_halfmoves'] += int(board.is_checkmate() and board.halfmove_clock >= 150)
                coverage['successors_checked'] += 1
        except (AssertionError, ParityError, ValueError) as exc:
            coverage['transition_mismatches'] += 1
            pytest.fail(f'{case.context(locals().get("action"), locals().get("uci"))}\n{exc}')
    assert castling_moves == {'e1g1', 'e1c1', 'e8g8', 'e8c8'}
    assert promotions == {color+':'+p for color in ['white', 'black'] for p in 'qrbn'}
    for name in ['mover_white', 'mover_black', 'en_passant_white', 'en_passant_black',
                 'check_evasions', 'checks_delivered', 'current_threefold_successors',
                 'claimable_fifty_ongoing', 'mate_at_150_halfmoves', 'terminal_checkmate',
                 'terminal_stalemate', 'terminal_insufficient_material',
                 'terminal_fivefold_repetition', 'terminal_seventyfive_moves']:
        assert events[name] > 0, ('Missing intended coverage', name)
    coverage.update(transition_events=dict(events), castling_moves=sorted(castling_moves),
                    promotion_types=sorted(promotions), unique_root_fens=len(root_fens),
                    transition_corpus_sha256=digest.hexdigest(), transition_passed=True)


def test_b_policy_source_index_matches_encoder(corpus, coverage):
    pieces = Counter()
    for case in corpus:
        host = jax.device_get(case.state)
        encoded = jax.device_get(encode_jit(case.state))
        planes = encoded['planes']
        assert ENCODER_VERSION == 'pgx-v2-history8-ep-context8-v1', case.context()
        assert planes.shape == (8, 8, 120) and planes.dtype == np.float32, case.context()
        np.testing.assert_array_equal(encoded['legal_mask'], host.legal_action_mask, err_msg=case.context())
        coverage['mapping_roots'] += 1
        for action in np.flatnonzero(host.legal_action_mask):
            action = int(action)
            uci = None
            try:
                uci = action_to_uci(host, action)
                context = case.context(action, uci)
                # UCI coordinates + python-chess supply the independent square
                # and piece reference. Do not reuse codec _square/FROM_PLANE or
                # derive the expected piece from the encoder/Pgx board itself.
                square = chess.Move.from_uci(uci).from_square
                piece = case.oracle.board.piece_at(square)
                assert piece is not None and piece.color == case.oracle.board.turn, context
                rank = chess.square_rank(square)
                relative_rank = rank if case.oracle.board.turn == chess.WHITE else 7-rank
                expected_source = chess.square_file(square)*8+relative_rank
                source = action//73
                assert source == expected_source, context
                expected = np.zeros(6, np.float32)
                expected[piece.piece_type-1] = 1
                np.testing.assert_array_equal(planes[7-source%8, source//8, 0:6], expected, err_msg=context)
                pieces[chess.piece_name(piece.piece_type)] += 1
                coverage['policy_actions_checked'] += 1
            except (AssertionError, ValueError) as exc:
                coverage['mapping_mismatches'] += 1
                pytest.fail(f'{case.context(action, uci)}\n{exc}')
    assert set(pieces) == {'pawn', 'knight', 'bishop', 'rook', 'queen', 'king'}
    assert coverage['policy_actions_checked'] == coverage['successors_checked']
    coverage.update(mapping_source_pieces=dict(pieces), mapping_passed=True)
