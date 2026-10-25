import inspect

import jax
import jax.numpy as jnp
import mctx
import numpy as np
import pytest

from tinychess.chess.action_codec import uci_to_action
from tinychess.chess.pgx_env import ENV, init, to_fen
from tinychess.search.mctx_recurrent import recurrent_fn, search


def batch(states):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *states)


def test_installed_api_contract():
    assert list(inspect.signature(mctx.RootFnOutput).parameters) == ["prior_logits", "value", "embedding"]
    assert list(inspect.signature(mctx.RecurrentFnOutput).parameters) == ["reward", "discount", "prior_logits", "value"]
    for fn in [mctx.muzero_policy, mctx.gumbel_muzero_policy]:
        params = inspect.signature(fn).parameters
        for name in ["params", "rng_key", "root", "recurrent_fn", "num_simulations", "invalid_actions"]:
            assert name in params
    assert list(inspect.signature(ENV.raw.step).parameters) == ["state", "action", "key"]


def test_batched_environment_smoke():
    keys = jax.random.split(jax.random.PRNGKey(42), 8)
    states = jax.jit(jax.vmap(ENV.init))(keys)
    actions = jnp.argmax(states.legal_action_mask, axis=-1)
    child = jax.jit(jax.vmap(ENV.step))(states, actions)
    jax.block_until_ready(child)
    assert child.observation.shape == (8, 8, 8, 119)
    np.testing.assert_array_equal(child._x.step_count, np.ones(8))
    assert child.legal_action_mask.shape == (8, 4672)
    for i in range(8):
        s = jax.tree.map(lambda x: x[i], states)
        expected = ENV.step(s, actions[i])
        np.testing.assert_array_equal(child._x.board[i], expected._x.board)


@pytest.mark.parametrize("algorithm", ["puct", "gumbel"])
def test_batched_search(algorithm):
    states = jax.jit(jax.vmap(ENV.init))(jax.random.split(jax.random.PRNGKey(1), 2))
    result = search(states, jax.random.PRNGKey(2), algorithm=algorithm, num_simulations=4)
    jax.block_until_ready(result)
    actions = np.asarray(result.action)
    mask = np.asarray(states.legal_action_mask)
    weights = np.asarray(result.action_weights)
    assert mask[np.arange(2), actions].all()
    assert np.isfinite(weights).all() and (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(-1), 1, atol=1e-6)
    assert np.all(weights[~mask] == 0)
    tree = result.search_tree
    assert np.all(np.asarray(tree.node_visits[:, 0]) == 5)  # root + four simulations
    if algorithm == "puct":
        np.testing.assert_allclose(weights, tree.summary().visit_probs)
    # Check every expanded edge, including interior legality (root invalid_actions
    # alone cannot mask interior nodes; recurrent logits must also do so).
    for b, node, action in np.argwhere(np.asarray(tree.children_index) >= 0):
        assert bool(tree.embeddings.legal_action_mask[b, node, action])


@pytest.mark.parametrize("seed", range(4))
def test_recurrent_perspectives_terminal_and_absorbing(play, seed):
    ongoing = init(jax.random.PRNGKey(seed))
    before_mate = play(["f2f3", "e7e5", "g2g4"], seed=seed)
    terminal = play(["f2f3", "e7e5", "g2g4", "d8h4"], seed=seed)
    stale = ENV.from_fen("7k/8/6QK/8/8/8/8/8 w - - 0 1", history="discard")
    states = batch([ongoing, before_mate, terminal, stale])
    actions = jnp.array([uci_to_action(ongoing, "e2e4"), uci_to_action(before_mate, "d8h4"),
                         0, uci_to_action(stale, "g6f7")])
    out, child = jax.jit(recurrent_fn)((), jax.random.PRNGKey(0), actions, states)
    jax.block_until_ready(out)
    np.testing.assert_array_equal(out.reward, [0, 1, 0, 0])
    np.testing.assert_array_equal(out.discount, [-1, 0, 0, 0])
    np.testing.assert_array_equal(out.value, [0.25, 0, 0, 0])
    assert out.reward[0] + out.discount[0] * out.value[0] == -0.25
    assert np.isfinite(out.prior_logits).all()
    assert to_fen(jax.tree.map(lambda x: x[2], child)) == to_fen(terminal)


def test_truncation_cuts_bootstrap():
    state = init(jax.random.PRNGKey(0))
    state = state.replace(_x=state._x._replace(step_count=jnp.int32(511)))
    out, child = jax.jit(recurrent_fn)((), jax.random.PRNGKey(0),
                                    jnp.array([uci_to_action(state, "e2e4")]), batch([state]))
    assert child.truncated[0] and not child.terminated[0]
    assert out.discount[0] == 0 and out.reward[0] == 0 and out.value[0] == 0
