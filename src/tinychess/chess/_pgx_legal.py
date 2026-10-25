# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Vendored from pgx==2.6.0 pgx/_src/games/chess.py.
# Upstream SHA256: 2770bf1ba2aae5d8cf6f2c8b4bab24f1e4f820d8aee2ed063b1891d7ae0f6323
# Change: candidate selection size 200 -> 434, with no global monkeypatch.
# All other move-generation semantics are unchanged; see parity tests.
import jax
import jax.numpy as jnp
from jax import Array, lax
from pgx._src.games.chess import (
    GameState, Action, PAWN, ROOK, KING, EMPTY, BETWEEN, CAN_MOVE,
    LEGAL_DEST, _is_checked, _apply_move, _is_attacked,
)


def legal_action_mask(state: GameState) -> Array:
    def legal_normal_moves(from_):
        piece = state.board[from_]

        def legal_label(to):
            ok = (from_ >= 0) & (piece > 0) & (to >= 0) & (state.board[to] <= 0)
            between_ixs = BETWEEN[from_, to]
            ok &= CAN_MOVE[piece, from_, to] & ((between_ixs < 0) | (state.board[between_ixs] == EMPTY)).all()
            c0, c1 = from_ // 8, to // 8
            pawn_should = ((c1 == c0) & (state.board[to] == EMPTY)) | ((c1 != c0) & (state.board[to] < 0))
            ok &= (piece != PAWN) | pawn_should
            return lax.select(ok, Action(from_=from_, to=to)._to_label(), -1)

        return jax.vmap(legal_label)(LEGAL_DEST[piece, from_])

    def legal_en_passants():
        to = state.en_passant

        def legal_labels(from_):
            ok = (from_ >= 0) & (from_ < 64) & (to >= 0) & (state.board[from_] == PAWN) & (state.board[to - 1] == -PAWN)
            a = Action(from_=from_, to=to)
            return lax.select(ok, a._to_label(), -1)

        return jax.vmap(legal_labels)(jnp.int32([to - 9, to + 7]))

    def is_not_checked(label):
        a = Action._from_label(label)
        return ~_is_checked(_apply_move(state, a))

    def legal_underpromotions(mask):
        def legal_labels(label):
            a = Action._from_label(label)
            ok = (state.board[a.from_] == PAWN) & (a.to >= 0)
            ok &= mask[Action(from_=a.from_, to=a.to)._to_label()]
            return lax.select(ok, label, -1)

        labels = jnp.int32([from_ * 73 + i for i in range(9) for from_ in [6, 14, 22, 30, 38, 46, 54, 62]])
        return jax.vmap(legal_labels)(labels)

    # normal move and en passant
    possible_piece_positions = jnp.nonzero(state.board > 0, size=16, fill_value=-1)[0]
    a1 = jax.vmap(legal_normal_moves)(possible_piece_positions).flatten()
    a2 = legal_en_passants()
    actions = jnp.hstack((a1, a2))  # include -1
    # Keep all 16 * 27 + 2 candidate slots, including EP. A 200-slot
    # cap drops legal moves in high-mobility positions (218-move regression).
    ixs = jnp.nonzero(actions >= 0, size=434, fill_value=0)[0]
    actions = actions[ixs]  # bounded by the full candidate array
    # filter ignoring checks and suicides
    actions = jnp.where(jax.vmap(is_not_checked)(actions), actions, -1)
    mask = jnp.zeros(64 * 73 + 1, dtype=jnp.bool_)  # +1 for sentinel
    mask = mask.at[actions].set(True)

    # castling
    b = state.board
    can_castle_queen_side = state.castling_rights[0, 0]
    can_castle_queen_side &= (b[0] == ROOK) & (b[8] == EMPTY) & (b[16] == EMPTY) & (b[24] == EMPTY) & (b[32] == KING)
    can_castle_king_side = state.castling_rights[0, 1]
    can_castle_king_side &= (b[32] == KING) & (b[40] == EMPTY) & (b[48] == EMPTY) & (b[56] == ROOK)
    not_checked = ~jax.vmap(_is_attacked, in_axes=(None, 0))(state, jnp.int32([16, 24, 32, 40, 48]))
    mask = mask.at[2364].set(mask[2364] | (can_castle_queen_side & not_checked[:3].all()))
    mask = mask.at[2367].set(mask[2367] | (can_castle_king_side & not_checked[2:].all()))

    # set underpromotions
    actions = legal_underpromotions(mask)
    mask = mask.at[actions].set(True)

    return mask[:-1]

