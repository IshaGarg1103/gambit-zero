import jax
import pytest

from tinychess.chess.pgx_env import ENV, init, step


@pytest.fixture(scope="session")
def env():
    return ENV


@pytest.fixture(scope="session")
def play():
    from tinychess.chess.action_codec import uci_to_action

    def run(moves, fen=None, seed=0):
        state = init(jax.random.PRNGKey(seed)) if fen is None else ENV.from_fen(fen, history="discard")
        for move in moves:
            state = step(state, uci_to_action(state, move))
        return state
    return run
