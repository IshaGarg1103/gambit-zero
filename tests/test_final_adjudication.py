"""Review condition: no WDL target can be derived from a truncated episode."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tinychess.chess.action_codec import uci_to_action
from tinychess.chess.adjudication import final_wdl
from tinychess.chess.pgx_env import ENV, init, step


def test_ongoing_has_no_final_label():
    with pytest.raises(ValueError, match="Ongoing"):
        final_wdl(init(jax.random.PRNGKey(0)))


@pytest.mark.parametrize("terminated", [False, True])
def test_truncation_never_becomes_draw_or_final_label(terminated):
    state = init(jax.random.PRNGKey(0))
    state = state.replace(_x=state._x._replace(step_count=jnp.int32(511)))
    child = step(state, uci_to_action(state, "e2e4"))
    assert child.truncated and not child.terminated
    assert not np.any(child.rewards)  # A zero reward is not a draw label.
    child = child.replace(terminated=jnp.bool_(terminated))
    with pytest.raises(ValueError, match="Truncated"):
        final_wdl(child)


def test_genuine_draw_has_draw_labels():
    state = ENV.from_fen("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1", history="discard")
    np.testing.assert_array_equal(final_wdl(state), [[0, 1, 0], [0, 1, 0]])


@pytest.mark.parametrize("seed", [0, 1])
def test_checkmate_labels_player_ids_and_rejects_absorbed_result(play, seed):
    state = play(["f2f3", "e7e5", "g2g4", "d8h4"], seed=seed)
    labels = final_wdl(state)
    np.testing.assert_array_equal(labels[int(state._player_order[0])], [0, 0, 1])
    np.testing.assert_array_equal(labels[int(state._player_order[1])], [1, 0, 0])
    with pytest.raises(ValueError, match="absorbed"):
        final_wdl(step(state, jnp.int32(0)))
