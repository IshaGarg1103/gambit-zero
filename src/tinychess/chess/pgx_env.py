"""Version-pinned Pgx rules adapter, with explicit automatic-draw semantics.

Policy: players never voluntarily claim draws. Fivefold/150-halfmove draws
are automatic; checkmate takes precedence. At 512 plies truncate, never draw.
FEN imports explicitly discard prior history; trajectories must replay moves.
All rules/transition functions are pure JAX. Private Pgx APIs are isolated here.
"""
from importlib.metadata import version

import jax
import jax.numpy as jnp
import pgx
from pgx._src.games.chess import (
    _is_checked, _zobrist_hash, has_insufficient_pieces, _apply_move,
    _flip, _update_history, Action,
)
from pgx.experimental.chess import from_fen as pgx_from_fen, to_fen

from tinychess.chess._pgx_legal import legal_action_mask

RULES_POLICY_VERSION = "tinychess-auto-fide-pgx-v1"
MAX_PLIES = 512


def position_hash(x):
    """FIDE repetition ignores EP unless an EP capture is legal (incl. pins)."""
    sources = jnp.array([x.en_passant - 9, x.en_passant + 7])
    # Ordinary move planes for +9 and -7 in Pgx's file-major coordinates.
    from pgx._src.games.chess import TO_PLANE
    safe_sources = jnp.clip(sources, 0, 63)
    safe_target = jnp.clip(x.en_passant, 0, 63)
    labels = safe_sources * 73 + TO_PLANE[safe_sources, safe_target]
    valid = (x.en_passant >= 0) & (sources >= 0) & (sources < 64)
    valid &= (x.board[safe_sources] == 1) & x.legal_action_mask[labels]
    ep = jnp.where(valid.any(), x.en_passant, -1)
    return _zobrist_hash(x._replace(en_passant=ep))


def repetition_count(state):
    return jnp.all(state._x.hash_history == position_hash(state._x), axis=1).sum()


def terminal_flags(state):
    """Independent flags; reason() supplies precedence for reporting."""
    no_moves = ~state._x.legal_action_mask.any()
    checked = _is_checked(state._x)
    return {
        "checkmate": no_moves & checked,
        "stalemate": no_moves & ~checked,
        "insufficient_material": has_insufficient_pieces(state._x),
        "fivefold_repetition": repetition_count(state) >= 5,
        "seventyfive_moves": state._x.halfmove_count >= 150,
    }


def reason(state):
    """Host-only diagnostic. Truncations are ineligible for outcome labels."""
    for name, flag in terminal_flags(state).items():
        if bool(flag):
            return name
    if bool(state.terminated):
        return "illegal_action"
    return "max_plies" if bool(state.truncated) else "ongoing"


class ChessEnv:
    def __init__(self):
        if version("pgx") != "2.6.0":
            raise RuntimeError("Adapter requires pgx==2.6.0; revalidate private APIs")
        self.raw = pgx.make("chess")
        if self.raw.version != "v2":
            raise RuntimeError("Adapter requires Pgx chess-v2")

    def _finish(self, state):
        flags = terminal_flags(state)
        terminated = jnp.any(jnp.stack(list(flags.values())))
        truncated = (state._x.step_count >= MAX_PLIES) & ~terminated
        rewards = jnp.where(flags["checkmate"], self.raw.game.rewards(state._x)[state._player_order],
                            jnp.zeros(2, jnp.float32))
        # Keep the actual chess mask, unlike Pgx core's all-True terminal mask.
        state = state.replace(terminated=terminated, truncated=truncated,
                              rewards=rewards, legal_action_mask=state._x.legal_action_mask)
        return state.replace(observation=self.raw.observe(state))

    def init(self, key):
        state = self.raw.init(key)
        x = state._x
        x = x._replace(hash_history=jnp.zeros_like(x.hash_history).at[0].set(position_hash(x)))
        return self._finish(state.replace(_x=x))

    def from_fen(self, fen: str, *, history: str):
        """Offline boundary. history='discard' acknowledges FEN's missing history.

        Only orthodox, valid six-field FEN is supported; validate external input
        offline before calling. Clock fields are retained, episode ply resets.
        """
        if history != "discard":
            raise ValueError("Replay a full trajectory to preserve repetition; FEN has no history")
        state = pgx_from_fen(fen)
        x = state._x
        x = x._replace(legal_action_mask=jax.jit(legal_action_mask)(x))
        x = x._replace(
            hash_history=jnp.zeros_like(x.hash_history).at[0].set(position_hash(x)),
            board_history=jnp.zeros_like(x.board_history).at[0].set(x.board),
        )
        return self._finish(state.replace(_x=x))

    def step(self, state, action, key=None):
        """Legal-action contract, matching mctx's masks. Illegal moves forfeit.

        Actions must be scalar integers in [0,4672). Terminal/truncated states
        absorb transitions with zero reward, even for an invalid chess action.
        """
        def advance(s):
            def legal(_):
                # Pgx Game.step sequence, using the audited full-buffer mask.
                x = _update_history(_flip(_apply_move(s._x, Action._from_label(action))))
                x = x._replace(legal_action_mask=legal_action_mask(x), step_count=x.step_count + 1)
                # Replace Pgx's raw hash that Game.step appended, retaining the
                # prior *canonical* hashes and board history in the embedding.
                x = x._replace(hash_history=x.hash_history.at[0].set(position_hash(x)))
                nxt = s.replace(_x=x, _step_count=s._step_count + 1,
                                current_player=s._player_order[x.color])
                return self._finish(nxt)

            def illegal(_):
                reward = jnp.ones(2, jnp.float32).at[s.current_player].set(-1)
                return s.replace(terminated=jnp.bool_(True), rewards=reward)

            return jax.lax.cond(s.legal_action_mask[action], legal, illegal, None)

        return jax.lax.cond(state.terminated | state.truncated,
                            lambda s: s.replace(rewards=jnp.zeros_like(s.rewards)), advance, state)

    def claims(self, state):
        """JAX diagnostic for present OR intended-move claims; no claim action.

        This intentionally expensive audit helper is not used in search. A claim
        does not occur in this policy: players play on to automatic termination.
        """
        current_three = repetition_count(state) >= 3
        current_fifty = (state._x.halfmove_count >= 100) & state._x.legal_action_mask.any()
        labels = jnp.nonzero(state.legal_action_mask, size=218, fill_value=-1)[0]

        def inspect_move(i, claims):
            label = labels[i]
            def check(_):
                nxt = self.step(state, label)
                three = repetition_count(nxt) >= 3
                fifty = (nxt._x.halfmove_count >= 100) & nxt._x.legal_action_mask.any()
                return claims | jnp.array([three, fifty])
            return jax.lax.cond(label >= 0, check, lambda _: claims, None)

        claims = jax.lax.fori_loop(0, 218, inspect_move, jnp.array([current_three, current_fifty]))
        return claims & ~(state.terminated | state.truncated)


ENV = ChessEnv()
init = jax.jit(ENV.init)
step = jax.jit(ENV.step)
from_fen = ENV.from_fen
