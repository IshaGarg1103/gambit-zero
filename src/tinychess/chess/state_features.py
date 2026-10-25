"""Canonical model-input contract, shared by human replay and future self-play.

No model is implemented. Inputs are entirely derived from corrected Pgx states.
"""
import jax
import jax.numpy as jnp

from tinychess.chess.pgx_env import MAX_PLIES, repetition_count

ENCODER_VERSION = 'pgx-v2-history8-ep-context8-v1'
PLANES_SHAPE = (8, 8, 120)
CONTEXT_SHAPE = (8,)
CONTEXT_NAMES = ('black_to_move', 'fullmove_over_256', 'halfmove_over_150',
                 'episode_ply_over_512', 'repetition_over_5', 'terminated',
                 'truncated', 'available_history_over_8')


def encode(state):
    """Unbatched pure-JAX encoder; batch with jax.vmap.

    0:119 are the installed Pgx chess-v2 observation (including eight history
    boards, repetition planes and castling/counter features). Channel 119 is
    the raw EP target, oriented exactly as Pgx's board planes. Context supplies
    exact normalized counters and current repetition frequency. No clipping.
    Rules/search must retain the full Pgx state; this short neural input is not
    asserted to encode every past repetition identity beyond the history stack.
    """
    ep = state._x.en_passant
    ep_plane = jnp.zeros(64, jnp.float32).at[jnp.clip(ep, 0, 63)].set((ep >= 0).astype(jnp.float32))
    ep_plane = jnp.rot90(ep_plane.reshape(8, 8), k=1)[:, :, None]
    planes = jnp.concatenate((state.observation.astype(jnp.float32), ep_plane), axis=-1)
    context = jnp.array([state._x.color, state._x.fullmove_count / 256.0,
                         state._x.halfmove_count / 150.0, state._x.step_count / MAX_PLIES,
                         repetition_count(state) / 5.0, state.terminated, state.truncated,
                         jnp.minimum(state._x.step_count + 1, 8) / 8.0], dtype=jnp.float32)
    return dict(planes=planes, context=context, legal_mask=state.legal_action_mask)


encode_jit = jax.jit(encode)
