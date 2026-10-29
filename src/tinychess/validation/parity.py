"""Fail-fast, history-aware offline oracle for random games and human replay."""
from collections import Counter

import chess
import jax
import numpy as np
from pgx._src.games.chess import _is_checked

from tinychess.chess.action_codec import legal_uci_actions, action_to_uci
from tinychess.chess.pgx_env import MAX_PLIES, position_hash, to_fen


class ParityError(RuntimeError):
    """Unexplained rules mismatch: stop extraction, never reject and continue."""


inspect_state = jax.jit(lambda state: (position_hash(state._x), _is_checked(state._x)))


def board_identity(board):
    # Capturable EP, side to move, castling and pieces; clocks excluded.
    return " ".join(board.fen(en_passant="legal").split()[:4])


def absolute_board(board):
    values = np.zeros(64, np.int32)
    for square, piece in board.piece_map().items():
        values[chess.square_file(square) * 8 + chess.square_rank(square)] = piece.piece_type * (1 if piece.color else -1)
    return values


class Oracle:
    def __init__(self, board=None):
        self.board = chess.Board() if board is None else board.copy(stack=False)
        self.history = [absolute_board(self.board)]
        self.counts = Counter([board_identity(self.board)])
        self.plies = 0

    def push(self, uci):
        self.board.push_uci(uci)
        self.plies += 1
        self.history.insert(0, absolute_board(self.board))
        self.history = self.history[:8]
        self.counts[board_identity(self.board)] += 1

    def check(self, state):
        computed, checked = jax.device_get(inspect_state(state))
        host = jax.device_get(state)
        board = self.board

        def require(ok, detail):
            if not ok:
                raise ParityError(f"ply={self.plies} {detail}; oracle={board.fen(en_passant='fen')}")

        require(to_fen(host) == board.fen(en_passant="fen"), "full FEN mismatch")
        require(bool(checked) == board.is_check(), "check mismatch")
        legal = {move.uci() for move in board.legal_moves}
        # Compare actual rule mask even at drawn terminal states; public codec
        # intentionally exposes no playable actions once an episode ends.
        actions = {action_to_uci(host, int(a), require_legal=False): int(a)
                   for a in np.flatnonzero(host._x.legal_action_mask)}
        require(set(actions) == legal, f"legal actions mismatch missing={legal - actions.keys()} extra={actions.keys() - legal}")
        count = self.counts[board_identity(board)]
        require(np.array_equal(computed, host._x.hash_history[0]), "current canonical hash mismatch")
        actual_count = int(np.all(host._x.hash_history == computed, axis=1).sum())
        require(actual_count == count, f"repetition mismatch {actual_count}!={count}")
        expected_history = np.zeros((8, 64), np.int32)
        expected_history[:len(self.history)] = self.history
        if not board.turn:
            expected_history = -np.flip(expected_history.reshape(8, 8, 8), axis=2).reshape(8, 64)
        require(np.array_equal(host._x.board_history, expected_history), "board history mismatch")
        require(int(host._x.step_count) == self.plies, "episode ply mismatch")
        color = 0 if board.turn else 1
        require(int(host.current_player) == int(host._player_order[color]), "player/color mapping mismatch")
        terminal = (not legal or board.is_insufficient_material() or count >= 5 or board.halfmove_clock >= 150)
        truncated = self.plies >= MAX_PLIES and not terminal
        require(bool(host.terminated) == terminal, "terminal status mismatch")
        require(bool(host.truncated) == truncated, "truncation status mismatch")
        rewards = np.zeros(2, np.float32)
        if not legal and board.is_check():
            rewards[:] = 1
            rewards[int(host.current_player)] = -1
        require(np.array_equal(host.rewards, rewards), "reward mismatch")
        return actions, host, count
