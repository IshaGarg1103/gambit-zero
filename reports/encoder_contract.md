# Canonical model-input contract — M1 v1

Version: `pgx-v2-history8-ep-context8-v1`.
Implementation: `tinychess.chess.state_features.encode`; pure JAX, unbatched;
use `jax.vmap(encode)` for batches. The same function must be called on both
human-game replay states and future live self-play states. No transformer or
training loop is implemented.

| Field | Unbatched shape | dtype | Meaning |
|---|---|---|---|
| `planes` | `(8,8,120)` | float32 | Exact corrected-state Pgx chess-v2 observation plus one EP-target plane |
| `context` | `(8,)` | float32 | Explicit state counters/flags below |
| `legal_mask` | `(4672,)` | bool | Corrected native Pgx action order, separate from strategic supervision |

Batch shape prepends B to each field. Spatial orientation matches Pgx's
`rot90(file_major_board.reshape(8,8), k=1)` from the current side's perspective.
Black reverses ranks and piece ownership, not files. Action indexing remains
file-major source×73+plane in the existing versioned codec.

Channels 0–111 are eight 14-channel history groups: own pawn/knight/bishop/rook/
queen/king, opponent's six piece channels, and two repetition indicators supplied
by Pgx. Channels 112–118 are Pgx's color, episode ply/512, four relative castling
rights (own queen/king then opponent queen/king), and halfmove/100 planes. Channel
119 is a one-hot raw en-passant target in the same spatial orientation, all zero
if absent. Non-capturable EP targets can be present as a feature; repetition
identity separately includes EP only when a legal capture exists.

Context order (no clipping):

1. Black-to-move indicator (`state._x.color`).
2. Fullmove number / 256.
3. Halfmove clock / 150.
4. Episode ply / 512.
5. Current position's occurrence count / 5, using corrected canonical history.
6. `terminated` indicator.
7. `truncated` indicator.
8. Number of available recent boards, min(episode ply+1,8) / 8.

The complete Pgx state, including its 513-entry hash history, remains the rules
and future search embedding. This short model input is not asserted to contain
all past repetition identities beyond its eight-board stack. A standalone FEN
cannot substitute for replay history. M1 accepted games always start from the
standard initial board and replay every action; no per-position FEN import is
used. FEN import remains limited to explicit tests/probes.

Materialized Parquet stores planes/context as little-endian float32 byte arrays
(30,720 and 32 bytes before compression) and the legal mask as 584 packed bytes,
with little bit order. Compact game rows store uint16 action lists; encoder
features are regenerated with this same function. Any layout/semantic change
requires a new encoder version and regenerated/validated dataset metadata.

Tests compare replay and live transitions under different random player-order
seeds, scalar and batched encode, EP features, and packed/unpacked records. The
storage benchmark compares the entire sequential materialized feature stream to
replayed features, as well as deterministic random minibatches. New cross-device
encoder parity is not claimed unless separately executed on the GPU; IEEE float32
counter normalization can require numerical tolerances across compiler targets.
