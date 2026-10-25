"""M0-only dummy policy/value search. No training or self-play actors."""
from functools import partial

import jax
import jax.numpy as jnp
import mctx

from tinychess.chess.pgx_env import ENV


def dummy_policy_value(states):
    """Uniform legal prior and constant nonzero STM value for sign tests."""
    logits = jnp.where(states.legal_action_mask, 0.0, -1e30)
    done = states.terminated | states.truncated
    # Terminal embeddings are absorbing; a finite dummy distribution avoids NaNs.
    logits = jnp.where(done[:, None], jnp.zeros_like(logits), logits)
    return logits, jnp.where(done, 0.0, 0.25).astype(jnp.float32)


def recurrent_fn(params, rng_key, action, embedding):
    """mctx 0.0.6: (params,key,[B] action,embedding) -> (output,embedding).

    Reward is for the parent player; value is for the child player. Thus the
    nonterminal discount is -1. Truncations cut bootstrap but carry no outcome.
    """
    del params, rng_key
    child = jax.vmap(ENV.step)(embedding, action)
    reward = jnp.take_along_axis(child.rewards, embedding.current_player[:, None], axis=1)[:, 0]
    done = child.terminated | child.truncated
    logits, value = dummy_policy_value(child)
    output = mctx.RecurrentFnOutput(reward=reward, discount=jnp.where(done, 0.0, -1.0),
                                    prior_logits=logits, value=value)
    return output, child


@partial(jax.jit, static_argnames=("num_simulations", "algorithm"))
def search(states, key, *, num_simulations=4, algorithm="puct"):
    """Tiny API smoke only; roots must be active and have legal moves."""
    logits, value = dummy_policy_value(states)
    root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=states)
    kwargs = dict(params=(), rng_key=key, root=root, recurrent_fn=recurrent_fn,
                  num_simulations=num_simulations, invalid_actions=~states.legal_action_mask)
    if algorithm == "puct":
        return mctx.muzero_policy(**kwargs, dirichlet_fraction=0.0)
    if algorithm == "gumbel":
        return mctx.gumbel_muzero_policy(**kwargs, max_num_considered_actions=4, gumbel_scale=0.0)
    raise ValueError(algorithm)
