# Pgx 2.6.0 chess-v2 → accepted Gambit Zero adapter: documented delta

Pinned upstream commit: `87278d2d6e677fd87248c457207b59cfa42e578d`.
`pgx_m1_pin.json` records exact upstream/adapter SHA256s. M1 does not change this
accepted rules adapter or action codec. No site-packages edits are made.

The only vendored upstream function is `_legal_action_mask`; its actual unified
function diff is `pgx_legal_buffer.patch`. Other changes are a **wrapper**, so
the following is an explicit code/behavior comparison, not an applyable patch
claiming to modify the installed package.

| Upstream location / expression | Accepted wrapper counterpart | Purpose |
|---|---|---|
| `_src/games/chess.py::_legal_action_mask`: `nonzero(... size=200)` | `chess/_pgx_legal.py::legal_action_mask`: size=434 | Retain all 16×27+2 candidate slots; avoid losing 18 moves in 218-move regression |
| `GameState.hash_history[0] = INIT_ZOBRIST_HASH` | `ChessEnv.init`: zeros history, at[0] = `position_hash(x)` | Compute initial hash rather than retain stale hard-coded value |
| `_zobrist_hash`: `ZOBRIST_EN_PASSANT[state.en_passant]` | `position_hash`: substitute -1 unless the current legal mask permits an actual EP capture; then call upstream hash | Standard repetition identity, including pinned EP cases |
| `Game.step`: apply, flip, update raw history, compute mask, increment step | `ChessEnv.step`: same sequence with full-buffer mask, then overwrite newest hash with canonical EP-aware hash | Preserve canonical prior history and append correct current identity |
| `experimental.chess.from_fen`: inherited default histories, `_update_history(x)` | `ChessEnv.from_fen(history='discard')`: clear arrays, seed only imported current board/hash, recompute full-buffer mask | Do not invent prior initial board/history; explicitly acknowledge missing FEN history |
| `Game.is_terminal`: no moves, halfmove≥100, insufficient material, repetition≥3, ply≥512 | `terminal_flags` / `_finish`: no moves, insufficient material, repetition≥5, halfmove≥150; genuine terminal takes precedence, otherwise ply≥512 is truncation | Adopted auto-only convention; not a claim that Pgx's convention is an accidental bug |
| `core.Env.step`: terminal mask becomes all-True sentinel | `_finish`: retain actual rule mask; public codec returns no playable moves after either end flag | Distinguish internal finite search priors from actual playable chess actions |
| `_step`/core API player-order handling | `ChessEnv.step` and `_finish`: same player-order mapping, genuine mate reward indexed to player IDs | Preserve correct reward/side-to-move perspective |

Source references: original installed `pgx/_src/games/chess.py`,
`pgx/chess.py`, `pgx/core.py`, `pgx/experimental/chess.py`; accepted implementation
`src/tinychess/chess/pgx_env.py`, `action_codec.py`, `_pgx_legal.py`. Exact source
locations may move on upstream upgrades; SHA/version checks and differential
coverage must be rerun rather than silently adapting private APIs.

Short upstream-report notes:

1. Move-candidate truncation: explicit 218-vs-200 reproduction and complete move-set regression.
2. Initial hash: computed hash differs on the pinned JAX stack; two knight cycles miss the initial occurrence under native automatic-threefold behavior.
3. EP identity: a target with no legal capture changes native repetition hash; include pinned EP regressions.
4. FEN history: an unrelated prior initial board is seeded despite no supplied history; clear defaults and document the limitation.
5. Draw thresholds/horizon/terminal sentinel: convention/API choices, not unconditional upstream bug reports.
6. Official AlphaZero example: unavailable chess baseline and terminated-only handling of a downstream explicit truncation; no mctx library defect established.

The Python oracle is offline only. No python-chess calls occur in corrected
Pgx transitions, model encoding or mctx recurrence.
