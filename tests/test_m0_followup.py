import chess
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import runpy
from pathlib import Path

# Load the repository-owned probe explicitly; Molab can omit cwd from sys.path.
_probe = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/probe_mctx_tactics.py'))
run_search = _probe['run_search']
zero_recurrent = _probe['zero_recurrent']
winning_moves = _probe['winning_moves']
from tinychess.chess.action_codec import uci_to_action, action_to_uci
from tinychess.chess.pgx_env import ENV, init, step
from tinychess.chess.state_features import encode, encode_jit, ENCODER_VERSION
from tinychess.validation.parity import Oracle, ParityError


@pytest.mark.parametrize('black', [False, True])
@pytest.mark.parametrize('algorithm', ['puct', 'gumbel'])
def test_zero_value_mate_one_search_and_positive_root_q(black, algorithm):
    board = chess.Board('7k/8/5KQ1/8/8/8/8/8 w - - 0 1')
    if black:
        board = board.mirror()
    state = ENV.from_fen(board.fen(), history='discard')
    states = jax.tree.map(lambda x: x[None], state)
    result = run_search(states, jax.random.PRNGKey(0), 64, algorithm)
    selected = action_to_uci(jax.device_get(state), int(result.action[0]))
    assert selected in winning_moves(board, 1)
    q = result.search_tree.qvalues(jnp.array([0]))
    assert float(q[0, result.action[0]]) == pytest.approx(1.0)


@pytest.mark.parametrize('black', [False, True])
def test_three_edge_mate_return_signs(black):
    board = chess.Board('8/5Q2/8/8/8/1K6/8/3k4 w - - 0 1')
    if black:
        board = board.mirror()
    assert not winning_moves(board, 1)
    first = winning_moves(board, 3)[0]
    state = ENV.from_fen(board.fen(), history='discard')
    rewards, discounts = [], []
    for i in range(3):
        uci = first if i == 0 else sorted(m.uci() for m in board.legal_moves)[0] if i == 1 else winning_moves(board, 1)[0]
        action = uci_to_action(jax.device_get(state), uci)
        states = jax.tree.map(lambda x: x[None], state)
        out, child = jax.jit(zero_recurrent)((), jax.random.PRNGKey(0), jnp.array([action]), states)
        rewards.append(float(out.reward[0]))
        discounts.append(float(out.discount[0]))
        state = jax.tree.map(lambda x: x[0], child)
        board.push_uci(uci)
    assert board.is_checkmate()
    assert rewards == [0, 0, 1]
    assert discounts == [-1, -1, 0]
    value = 0
    values = []
    for reward, discount in zip(reversed(rewards), reversed(discounts)):
        value = reward + discount * value
        values.append(value)
    assert values == [1, -1, 1]


def test_history_oracle_detects_corruption():
    state = init(jax.random.PRNGKey(0))
    oracle = Oracle()
    oracle.check(state)
    corrupt = state.replace(_x=state._x._replace(board_history=state._x.board_history.at[1, 0].set(1)))
    with pytest.raises(ParityError, match='history'):
        oracle.check(corrupt)


def test_encoder_same_after_replay_and_live_transitions():
    live = init(jax.random.PRNGKey(0))
    replay = init(jax.random.PRNGKey(37))
    oracle = Oracle()
    for uci in ['e2e4', 'd7d5', 'e4e5', 'f7f5']:
        actions, _, _ = oracle.check(live)
        live = step(live, actions[uci])
        replay = step(replay, actions[uci])
        oracle.push(uci)
    a = jax.device_get(encode_jit(live))
    b = jax.device_get(encode_jit(replay))
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert a['planes'].shape == (8, 8, 120)
    assert a['context'].shape == (8,)
    assert a['legal_mask'].shape == (4672,)
    assert a['planes'][:, :, 119].sum() == 1
    assert ENCODER_VERSION == 'pgx-v2-history8-ep-context8-v1'
    batched = jax.jit(jax.vmap(encode))(jax.tree.map(lambda x: jnp.stack([x, x]), live))
    np.testing.assert_array_equal(batched['planes'][0], a['planes'])
