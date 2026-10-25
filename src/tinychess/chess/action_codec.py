"""Host-side UCI boundary for Pgx 2.6.0 chess-v2; no python-chess dependency.

Pgx indexes squares file-major (a1=0, a2=1), flips ranks for Black,
and reserves planes 0..8 for rook/bishop/knight underpromotions.
Ordinary pawn moves to the eighth rank imply queen promotion.
"""
import numpy as np
from pgx._src.games.chess import FROM_PLANE

ACTION_MAP_VERSION = "pgx-2.6.0-chess-v2-uci-v1"
NUM_ACTIONS = 4672
_DEST = np.asarray(FROM_PLANE)


def _square(index: int, color: int) -> str:
    rank = index % 8 if color == 0 else 7 - index % 8
    return "abcdefgh"[index // 8] + str(rank + 1)


def action_to_uci(state, action: int, *, require_legal: bool = True) -> str:
    """Decode a legal move. Terminal states have no playable UCI actions."""
    action = int(action)
    if not 0 <= action < NUM_ACTIONS:
        raise ValueError(f"Action out of range: {action}")
    if require_legal and (bool(state.terminated | state.truncated) or
                          not bool(state.legal_action_mask[action])):
        raise ValueError(f"Action is not playable: {action}")
    source, plane = divmod(action, 73)
    target = int(_DEST[source, plane])
    if target < 0:
        raise ValueError(f"Off-board action: {action}")
    color = int(state._x.color)
    suffix = "rbn"[plane // 3] if plane < 9 else ""
    if plane >= 9 and int(state._x.board[source]) == 1 and target % 8 == 7:
        suffix = "q"
    return _square(source, color) + _square(target, color) + suffix


def legal_uci_actions(state) -> dict[str, int]:
    if bool(state.terminated | state.truncated):
        return {}
    pairs = [(action_to_uci(state, a), int(a))
             for a in np.flatnonzero(np.asarray(state.legal_action_mask))]
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Action codec is not injective")
    return result


def uci_to_action(state, uci: str) -> int:
    try:
        return legal_uci_actions(state)[uci]
    except KeyError as exc:
        raise ValueError(f"Illegal or noncanonical UCI: {uci}") from exc
