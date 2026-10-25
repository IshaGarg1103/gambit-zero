"""Offline python-chess oracle; never imported by environment/search modules."""
import os
import random

import chess
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pgx._src.games.chess import _is_checked

from tinychess.chess.action_codec import action_to_uci, legal_uci_actions, uci_to_action
from tinychess.chess.pgx_env import ENV, init, step, to_fen, reason, repetition_count

CHECKED = jax.jit(_is_checked)
CLAIMS = jax.jit(ENV.claims)


def assert_position(state, board):
    assert to_fen(state) == board.fen(en_passant="fen")
    assert bool(CHECKED(state._x)) == board.is_check()
    outcome = board.outcome(claim_draw=False)
    if bool(state.truncated):
        assert outcome is None
        assert not bool(state.terminated)
    else:
        assert bool(state.terminated) == (outcome is not None), (to_fen(state), reason(state), outcome)
    if outcome is not None:
        expected = np.zeros(2)
        if outcome.winner is not None:
            winner_color = 0 if outcome.winner == chess.WHITE else 1
            winner_player = int(state._player_order[winner_color])
            expected[:] = -1
            expected[winner_player] = 1
        np.testing.assert_array_equal(state.rewards, expected)
    if not bool(state.terminated | state.truncated):
        assert set(legal_uci_actions(state)) == {m.uci() for m in board.legal_moves}


# Valid orthodox positions covering both colors and every special move branch.
FENS = [
    chess.STARTING_FEN,
    # Petrovic 218-move composition, retrieved via Lichess analysis board.
    "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "4kr2/8/8/8/8/8/8/R3K2R w KQ - 0 1",  # attacked castle transit
    "4k3/8/8/8/8/8/4r3/R3K2R w KQ - 0 1",  # castling in check
    "r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1",
    "k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",  # EP exposes king
    "4k3/8/8/r4pPK/8/8/8/8 w - f6 0 1",  # EP horizontal pin
    "1r2k3/P7/8/8/8/8/7p/4K1R1 w - - 0 1",
    "1r2k3/P7/8/8/8/8/7p/4K1R1 b - - 0 1",
    "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1",  # mate
    "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1",  # stalemate
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
    "4k3/8/8/8/8/8/8/2B1K3 w - - 0 1",
    "4k3/8/8/8/8/8/8/2N1K3 w - - 0 1",
    "4kb2/8/8/8/8/8/8/2B1K3 w - - 0 1",  # same-color bishops
    "2b1k3/8/8/8/8/8/8/2B1K3 w - - 0 1",  # opposite-color bishops
    "4k3/8/8/8/8/8/8/1NN1K3 w - - 0 1",  # not insufficient
]


@pytest.mark.parametrize("fen", FENS)
def test_constructed_positions_and_every_successor(fen):
    board = chess.Board(fen)
    assert board.is_valid(), (fen, board.status())
    state = ENV.from_fen(fen, history="discard")
    assert_position(state, board)
    for uci, action in legal_uci_actions(state).items():
        assert action_to_uci(state, action) == uci
        assert uci_to_action(state, uci) == action
        child = step(state, action)
        oracle = board.copy()
        oracle.push_uci(uci)
        assert_position(child, oracle)


@pytest.mark.parametrize("seed", range(int(os.environ.get("TINYCHESS_RANDOM_GAMES", "4"))))
def test_seeded_random_complete_games(seed):
    rng = random.Random(seed)
    board = chess.Board()
    state = init(jax.random.PRNGKey(seed))
    for ply in range(513):
        assert_position(state, board)
        if bool(state.terminated | state.truncated):
            break
        moves = sorted(legal_uci_actions(state))
        # Compare all successor positions periodically, not only the played move.
        if ply % 64 == 0:
            for uci in moves:
                child = step(state, uci_to_action(state, uci))
                oracle = board.copy()
                oracle.push_uci(uci)
                assert_position(child, oracle)
        uci = rng.choice(moves)
        state = step(state, uci_to_action(state, uci))
        board.push_uci(uci)
    else:
        pytest.fail("No terminal/truncated state by 512 plies")


@pytest.mark.parametrize("color", ["w", "b"])
@pytest.mark.parametrize("clock", [49, 50, 98, 99, 100, 149, 150])
def test_fifty_seventyfive_and_claims(color, clock):
    fen = f"7k/8/8/8/8/8/8/R3K3 {color} - - {clock} 1"
    state = ENV.from_fen(fen, history="discard")
    board = chess.Board(fen)
    assert_position(state, board)
    if not state.terminated:
        claims = np.asarray(CLAIMS(state))
        assert bool(claims[1]) == board.can_claim_fifty_moves()
        assert bool(claims[0]) == board.can_claim_threefold_repetition()
        for uci in legal_uci_actions(state):
            child = step(state, uci_to_action(state, uci))
            oracle = board.copy()
            oracle.push_uci(uci)
            assert_position(child, oracle)


@pytest.mark.parametrize("prefix", [[], ["e2e4"], ["a2a4", "a7a5"]])
def test_repetition_claims_and_fivefold(prefix):
    state = init(jax.random.PRNGKey(0))
    board = chess.Board()
    for uci in prefix:
        state = step(state, uci_to_action(state, uci))
        board.push_uci(uci)
    cycle = ["g1f3", "g8f6", "f3g1", "f6g8"] if board.turn else ["g8f6", "g1f3", "f6g8", "f3g1"]
    for ply in range(17):
        assert_position(state, board)
        if state.terminated:
            assert board.is_fivefold_repetition()
            assert int(repetition_count(state)) == 5
            break
        claims = np.asarray(CLAIMS(state))
        assert bool(claims[0]) == board.can_claim_threefold_repetition(), (prefix, ply)
        assert bool(claims[1]) == board.can_claim_fifty_moves()
        uci = cycle[ply % 4]
        state = step(state, uci_to_action(state, uci))
        board.push_uci(uci)
    else:
        pytest.fail("Fivefold not detected")


def test_checkmate_precedes_150_halfmoves():
    board = chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 149 1")
    state = ENV.from_fen(board.fen(), history="discard")
    board.push_uci("g6g7")
    state = step(state, uci_to_action(state, "g6g7"))
    assert board.is_checkmate()
    assert reason(state) == "checkmate"
    assert_position(state, board)


def test_truncation_and_absorption():
    state = init(jax.random.PRNGKey(3))
    # Inject the episode boundary while retaining an ordinary legal board.
    state = state.replace(_x=state._x._replace(step_count=jnp.int32(511)), _step_count=jnp.int32(511))
    state = step(state, uci_to_action(state, "e2e4"))
    assert bool(state.truncated) and not bool(state.terminated)
    assert reason(state) == "max_plies"
    assert not np.any(state.rewards)
    child = step(state, jnp.int32(0))
    assert to_fen(child) == to_fen(state)
    assert int(child._x.step_count) == 512
    assert not np.any(child.rewards)


def test_fen_requires_explicit_history_policy(play):
    with pytest.raises(TypeError):
        ENV.from_fen(chess.STARTING_FEN)
    with pytest.raises(ValueError):
        ENV.from_fen(chess.STARTING_FEN, history="preserve")
    state = play(["g1f3", "g8f6", "f3g1", "f6g8"])
    imported = ENV.from_fen(to_fen(state), history="discard")
    assert int(repetition_count(imported)) == 1
    assert int(repetition_count(state)) == 2


def test_invalid_codec_and_illegal_forfeit():
    state = init(jax.random.PRNGKey(0))
    for move in ["e2e5", "e2e4q", "e1g1", "garbage"]:
        with pytest.raises(ValueError):
            uci_to_action(state, move)
    for action in [-1, 4672, 0]:
        with pytest.raises(ValueError):
            action_to_uci(state, action)
    child = step(state, jnp.int32(0))
    assert child.terminated
    assert float(child.rewards[state.current_player]) == -1
    assert to_fen(child) == to_fen(state)
    absorbed = step(child, jnp.int32(0))
    assert not np.any(absorbed.rewards)


def test_terminal_absorption(play):
    state = play(["f2f3", "e7e5", "g2g4", "d8h4"])
    assert reason(state) == "checkmate"
    child = step(state, jnp.int32(0))
    assert to_fen(child) == to_fen(state)
    assert child.terminated
    assert not np.any(child.rewards)


@pytest.mark.parametrize("fen,ep_matters", [
    ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", True),
    ("k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", False),
    ("4k3/8/8/r4pPK/8/8/8/8 w - f6 0 1", False),
])
def test_repetition_identity_only_counts_legal_ep(fen, ep_matters):
    from tinychess.chess.pgx_env import position_hash
    state = ENV.from_fen(fen, history="discard")
    fields = fen.split()
    fields[3] = "-"
    other = ENV.from_fen(" ".join(fields), history="discard")
    equal = np.array_equal(position_hash(state._x), position_hash(other._x))
    assert equal != ep_matters
    assert chess.Board(fen).has_legal_en_passant() == ep_matters


def test_repetition_identity_preserves_castling_and_ignores_clocks():
    from tinychess.chess.pgx_env import position_hash
    fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    state = ENV.from_fen(fen, history="discard")
    rights_lost = ENV.from_fen(fen.replace("KQkq", "-"), history="discard")
    clocks = ENV.from_fen(fen.replace("0 1", "80 99"), history="discard")
    assert not np.array_equal(position_hash(state._x), position_hash(rights_lost._x))
    np.testing.assert_array_equal(position_hash(state._x), position_hash(clocks._x))


def test_castling_rights_not_restored_by_rook_return(play):
    fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    moves = ["h1h2", "h8h7", "h2h1", "h7h8"]
    state = play(moves, fen=fen)
    board = chess.Board(fen)
    for move in moves:
        board.push_uci(move)
    assert_position(state, board)
    assert "e1g1" not in legal_uci_actions(state)
    assert "e1c1" in legal_uci_actions(state)
