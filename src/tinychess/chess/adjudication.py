"""Host-side final-label boundary for completed environment episodes.

This is metadata validation outside MCTS. Never infer a draw merely because a
transition reward is zero: ongoing and truncated episodes have zero reward too.
"""
import numpy as np

from tinychess.chess.pgx_env import reason


def final_wdl(state) -> np.ndarray:
    """Return [player_id, win/draw/loss] one-hot labels, or reject no outcome.

    Consume the terminal transition before any absorbing step clears rewards.
    A 512-ply truncation, even with all-zero rewards, never produces labels.
    """
    if bool(state.truncated):
        raise ValueError("Truncated episodes have no final WDL outcome")
    if not bool(state.terminated):
        raise ValueError("Ongoing episodes have no final WDL outcome")
    rewards = np.asarray(state.rewards)
    cause = reason(state)
    draw_causes = {"stalemate", "insufficient_material", "fivefold_repetition", "seventyfive_moves"}
    decisive = cause in {"checkmate", "illegal_action"}
    if decisive:
        if not np.array_equal(np.sort(rewards), [-1.0, 1.0]):
            raise ValueError("Terminal win/loss rewards missing; do not label absorbed states")
    elif cause not in draw_causes or not np.array_equal(rewards, [0.0, 0.0]):
        raise ValueError("Inconsistent final outcome")
    classes = np.where(rewards > 0, 0, np.where(rewards < 0, 2, 1))
    return np.eye(3, dtype=np.int8)[classes]
