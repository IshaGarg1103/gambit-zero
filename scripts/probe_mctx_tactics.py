"""Deterministic zero-value/uniform-prior tactical diagnostics, not self-play."""
import argparse
from functools import partial
import json
from pathlib import Path
import time

import chess
import jax
import jax.numpy as jnp
import mctx
import numpy as np

from tinychess.chess.action_codec import uci_to_action, action_to_uci
from tinychess.chess.pgx_env import ENV
from tinychess.search.mctx_recurrent import recurrent_fn


def zero_recurrent(params, key, action, state):
    out, child = recurrent_fn(params, key, action, state)
    return out.replace(value=jnp.zeros_like(out.value)), child


@partial(jax.jit, static_argnames=('simulations', 'algorithm'))
def run_search(states, key, simulations, algorithm):
    done = states.terminated | states.truncated
    logits = jnp.where(states.legal_action_mask | done[:, None], 0.0, -1e30)
    root = mctx.RootFnOutput(prior_logits=logits, value=jnp.zeros(logits.shape[0]), embedding=states)
    kwargs = dict(params=(), rng_key=key, root=root, recurrent_fn=zero_recurrent,
                  num_simulations=simulations, invalid_actions=~states.legal_action_mask, max_depth=8)
    if algorithm == 'puct':
        return mctx.muzero_policy(**kwargs, dirichlet_fraction=0.0, temperature=0.0)
    return mctx.gumbel_muzero_policy(**kwargs, gumbel_scale=0.0, max_num_considered_actions=64)


def winning_moves(board, plies):
    """Offline exact tiny-horizon oracle. Return moves forcing mate in <=plies."""
    attacker = board.turn

    def forced(b, left):
        if b.is_checkmate():
            return b.turn != attacker
        if left == 0 or b.is_game_over(claim_draw=False):
            return False
        results = []
        for move in list(b.legal_moves):
            b.push(move)
            results.append(forced(b, left-1))
            b.pop()
        return any(results) if b.turn == attacker else all(results)

    result = []
    for move in list(board.legal_moves):
        board.push(move)
        if forced(board, plies-1):
            result.append(move.uci())
        board.pop()
    return sorted(result)


def probe(output, simulations=(64, 128)):
    # The mate-in-two fixture is verified by the offline oracle before use.
    mate1 = chess.Board('7k/8/5KQ1/8/8/8/8/8 w - - 0 1')
    mate2 = chess.Board('8/5Q2/8/8/8/1K6/8/3k4 w - - 0 1')
    fixtures = [('mate_in_one', mate1), ('mate_in_one', mate1.mirror()),
                ('mate_in_two', mate2), ('mate_in_two', mate2.mirror())]
    rows = []
    for label, board in fixtures:
        assert board.is_valid(), board.fen()
        winning = winning_moves(board, 1 if label == 'mate_in_one' else 3)
        assert winning, (label, board.fen())
        if label == 'mate_in_two':
            assert not winning_moves(board, 1)
        state = ENV.from_fen(board.fen(en_passant='fen'), history='discard')
        states = jax.tree.map(lambda x: x[None], state)
        # Explicit per-edge parent reward, child value and discount checks.
        if label == 'mate_in_one':
            action = uci_to_action(jax.device_get(state), winning[0])
            out, child = jax.jit(zero_recurrent)((), jax.random.PRNGKey(0), jnp.array([action]), states)
            np.testing.assert_array_equal(out.reward, [1.0])
            np.testing.assert_array_equal(out.discount, [0.0])
            np.testing.assert_array_equal(out.value, [0.0])
        for algorithm in ['puct', 'gumbel']:
            for count in simulations:
                started = time.perf_counter()
                result = jax.block_until_ready(run_search(states, jax.random.PRNGKey(0), count, algorithm))
                cold = time.perf_counter()-started
                started = time.perf_counter()
                repeat = jax.block_until_ready(run_search(states, jax.random.PRNGKey(0), count, algorithm))
                warm = time.perf_counter()-started
                assert int(result.action[0]) == int(repeat.action[0])
                selected = action_to_uci(jax.device_get(state), int(result.action[0]))
                assert chess.Move.from_uci(selected) in board.legal_moves
                found = selected in winning
                row = dict(fixture=label, fen=board.fen(), algorithm=algorithm, simulations=count,
                           winning_moves=winning, selected=selected, found_mate=found,
                           cold_seconds=cold, warm_seconds=warm)
                rows.append(row)
                print(json.dumps(row), flush=True)
                if label == 'mate_in_one' and not found:
                    raise AssertionError(f'Mate-in-one gate failed: {row}')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(status='PASS', max_depth=8, prior='uniform_legal', value=0.0,
                                     gumbel_scale=0.0, rows=rows, devices=[str(d) for d in jax.devices()],
                                     mate_two_policy='diagnostic only; finite-budget discovery is not guaranteed'), indent=2)+'\n')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('reports/m0_tactics.json'))
    parser.add_argument('--simulations', type=int, nargs='+', default=[64, 128])
    args = parser.parse_args()
    probe(args.output, tuple(args.simulations))
