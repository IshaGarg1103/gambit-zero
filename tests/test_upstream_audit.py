"""Assert the native Pgx behaviors requiring our explicit corrections.

These passing tests reproduce upstream mismatches. They do NOT certify native
Pgx parity; only the wrapped environment is a candidate for the project gate.
"""
import chess
import jax
import jax.numpy as jnp
import numpy as np
from pgx.experimental.chess import from_fen

from tinychess.chess.action_codec import uci_to_action
from tinychess.chess.pgx_env import ENV, position_hash


RAW_STEP = jax.jit(ENV.raw.step)


def test_raw_candidate_overflow():
    fen = "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1"
    raw = from_fen(fen)
    assert chess.Board(fen).legal_moves.count() == 218
    assert int(raw.legal_action_mask.sum()) == 200
    wrapped = ENV.from_fen(fen, history="discard")
    assert int(wrapped.legal_action_mask.sum()) == 218


def test_raw_100_halfmove_automatic_draw_and_512_termination():
    state = from_fen("7k/8/8/8/8/8/8/R3K3 w - - 100 1")
    assert state.terminated and not state.truncated
    assert chess.Board("7k/8/8/8/8/8/8/R3K3 w - - 100 1").outcome() is None
    state = ENV.raw.init(jax.random.PRNGKey(0))
    state = state.replace(_x=state._x._replace(step_count=jnp.int32(511)))
    child = RAW_STEP(state, uci_to_action(state, "e2e4"))
    assert child.terminated and not child.truncated
    assert np.all(child.legal_action_mask)  # Pgx core terminal sentinel


def test_raw_threefold_is_automatic():
    state = ENV.raw.init(jax.random.PRNGKey(0))
    board = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2 + ["g1f3"]:
        state = RAW_STEP(state, uci_to_action(state, uci))
        board.push_uci(uci)
    assert state.terminated
    assert board.can_claim_threefold_repetition()
    assert board.outcome(claim_draw=False) is None


def test_raw_ep_repetition_hash_is_not_canonical():
    from pgx._src.games.chess import _zobrist_hash
    # e3 is not capturable by Black in this position, so it is not part of a
    # repetition identity. Pgx's raw hash nevertheless includes it.
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    with_ep = from_fen(fen)
    without_ep = from_fen(fen.replace(" e3 ", " - "))
    assert not np.array_equal(_zobrist_hash(with_ep._x), _zobrist_hash(without_ep._x))
    np.testing.assert_array_equal(position_hash(with_ep._x), position_hash(without_ep._x))


def test_vendored_mask_is_only_buffer_change():
    """Fail if an upstream upgrade or local drift changes the audited patch."""
    import inspect
    from pgx._src.games.chess import _legal_action_mask
    from tinychess.chess._pgx_legal import legal_action_mask
    # Compare executable Python syntax, ignoring comments and function name.
    import ast
    original = ast.parse(inspect.getsource(_legal_action_mask))
    patched = ast.parse(inspect.getsource(legal_action_mask))
    original.body[0].name = "legal_action_mask"
    for node in ast.walk(original):
        if isinstance(node, ast.keyword) and node.arg == "size" and isinstance(node.value, ast.Constant) and node.value.value == 200:
            node.value.value = 434
    assert ast.dump(original) == ast.dump(patched)


def test_no_python_chess_in_hot_path():
    import ast
    from pathlib import Path
    root = Path(__file__).parents[1] / "src/tinychess"
    for folder in ["chess", "search", "selfplay", "model", "train"]:
        for path in (root / folder).glob("**/*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    assert all(a.name != "chess" and not a.name.startswith("chess.") for a in node.names), path
                if isinstance(node, ast.ImportFrom):
                    assert node.module != "chess" and not (node.module or "").startswith("chess."), path


def test_raw_initial_hash_is_stale():
    from pgx._src.games.chess import _zobrist_hash
    state = ENV.raw.init(jax.random.PRNGKey(0))
    assert not np.array_equal(state._x.hash_history[0], _zobrist_hash(state._x))
    wrapped = ENV.init(jax.random.PRNGKey(0))
    np.testing.assert_array_equal(wrapped._x.hash_history[0], position_hash(wrapped._x))
