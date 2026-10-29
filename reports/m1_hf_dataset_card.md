---
license: cc0-1.0
language:
- en
tags:
- chess
- jax
- policy-value
- human-games
configs:
- config_name: smoke_games
  default: false
  data_files:
  - split: train
    path:
    - smoke/hf-smoke-v1/games/train/2026-01/00000-aee5e0d786f472ba0cf138d2c945045266a4d6a70de91fe3f05c6c61fb8ad849.games.parquet
- config_name: smoke_human_v3
  default: true
  data_files:
  - split: train
    path:
    - smoke/hf-smoke-v1/positions/train/2026-01/00000-5a0cc5d5c5737ed9d97e3bae5e55e469da2ea43d754d6670b1d5854f63a33966.positions.parquet
---

# Gambit Zero — Human Chess Dataset A

**Status: synthetic-smoke-only.** All uploaded games in this test are synthetic fixtures. No real 50K dataset exists yet. Do not train or evaluate on the smoke fixtures.

Repository visibility is public, as requested by the owner.
Run: `hf-smoke-v1`. This repository stores dataset artifacts, not model weights.
No M2 model or self-play training is included.

## Size and splits actually published in this catalog

| Split | Games | Training positions after Dataset D exclusion |
|---|---:|---:|
| train | 31 | 38 |
| validation | 0 | 0 |
| test | 0 | 0 |

Final real-data targets are 35,000 training games (5,000 each January–July 2026),
7,500 validation games (August), and 7,500 test games (September). A game never
crosses splits. The table above reports this catalog, not promised target counts.

## Sources, selection and verification

Real sources: public CC0 Lichess monthly standard-rated PGN archives,
https://database.lichess.org/. Exact URLs, published hashes and verification
status are in `source_provenance.json` and each completed month manifest.
Published archive hashes alone do not mean the complete sources were verified.
Synthetic smoke artifacts are explicitly marked and are not Lichess samples.

Streaming consumes original compressed bytes in bounded frames without retaining
full archives. Month checkpoints are accepted only after byte-count/EOF and full
SHA256 verification against the official monthly checksum. Sampling uses
`utc-day-rating-bottomk-sha256-v1`, seed 20261025, full UTC calendar-day strata,
deterministic SHA256 priorities and bounded reserves. At least 60% of accepted
games per day have both player ratings >=2000; all remaining games are >=1800.
The lower rating defines the band. Exhausted reserves/coverage gaps stop the run.

## Quality and targets

Standard rated chess from the initial position, base clock >=300 seconds
independent of increment, valid complete results and legal moves. Exclude
variants, abandoned/aborted/rules-infraction games, malformed metadata and games
that cannot be completely replayed under the 512-ply horizon. Time forfeits
retain policy targets and default value weight 0. Generic Normal endings are not
misrepresented as proven resignation/agreement. Labels are noisy human outcomes,
not strategic ground truth. No Stockfish/[%eval] labels are consumed.

Replay every accepted trajectory from the initial corrected Pgx state, preserving
history. Full legal actions, position/history, terminal status and reward parity
are checked against python-chess offline at every ply. Actual final checkmate is
checked against Pgx; non-board PGN results do not require environment termination.
Rules: `tinychess-auto-fide-pgx-v1`, no voluntary claims, fivefold/150-halfmove
automatic draws; truncations never supply WDL labels. Remaining dead-position and
hash-collision limitations are inherited from the accepted rules contract.

## Schema and encoder

`human_v3`; `schema.json` defines every physical field. Policy uses Pgx's native
4,672-action order, codec `pgx-2.6.0-chess-v2-uci-v1`. Encoder
`pgx-v2-history8-ep-context8-v1`: planes float32 (8,8,120), context float32 (8,),
legal mask bool (4672,). Feature bytes are little-endian float32; masks are
584 bytes, little bit order. See `encoder_contract.md` for channel/context order.
WDL class order is win/draw/loss (0/1/2), scalar +1/0/-1 from side to move.
Game metadata/action sequences are provenance; the primary training inputs are
the materialized positions configuration.

## Evaluation integrity and limitations

Before production upload, freeze versioned Dataset D puzzle IDs/source snapshot
and exact opening positions. Match by pieces, side to move, castling rights and
legally capturable en passant; ignore clocks/history conservatively. Exact Dataset
D positions are physically absent from training position shards. Game-level
provenance retains complete actions; do not turn it into an unfiltered training
reader. FEN cannot establish full repetition history. Report overlap of unique
boards between splits overall and at ply >=10 using the compact exact audit index;
repeated board positions are not necessarily duplicate games. Frozen evaluation
snapshots/hashes, exclusion counts and coverage are stored in monthly manifests.

## Upload, reproducibility and recovery

Small game-aligned Parquet shards are grouped by kind, split and source month.
Each immutable filename includes its full SHA256. Only a completed month is
listed in the catalog. Uploads use authenticated huggingface_hub; files are read
back at an immutable revision, streaming without a download cache, and compared
by SHA256 and byte count before the Mac copy is removed. An upload/read-back
failure retains the local file and halts generation of further shards.

A second feature build can consume the verified compact selected-PGN input from
Hugging Face, one bounded selection file/game at a time, and compare regenerated
shard hashes without a second local dataset. Reproduction is accepted only when
actually run; completed reproduction manifests record results. Store the upload
journal and completed-month receipts for recovery. Interrupted archive months
restart from byte zero; verified completed months are reused. No blind compressed
byte-range game access is assumed.

## Usage without downloading the whole dataset

Authenticate using `hf auth login`; never embed a token in a notebook or commit.
Install `datasets` in your consumer environment, then use its streaming mode:

```python
from datasets import load_dataset
# Use the configuration listed in this card; smoke data is synthetic only.
rows = load_dataset("ishagarg1103/gambit-zero", name="smoke_human_v3",
                    split="train", streaming=True)
row = next(iter(rows))
```

Decode a position row (NumPy):

```python
import numpy as np
planes = np.frombuffer(row["planes"], dtype="<f4").reshape(8, 8, 120)
context = np.frombuffer(row["context"], dtype="<f4")
legal = np.unpackbits(np.frombuffer(row["legal_mask"], dtype=np.uint8),
                      bitorder="little").astype(bool)
assert legal[int(row["action"])]
```

Use policy_weight/value_weight separately, retain perspective, and pin the Hub
revision in any experiment. Outcome value loss must respect the timeout mask.
This dataset makes no claim of optimal moves, model strength or frontier-GPT wins.
