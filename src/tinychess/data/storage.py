"""Two lossless Parquet prototypes: compact games and materialized positions."""
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tinychess.chess.action_codec import ACTION_MAP_VERSION
from tinychess.chess.pgx_env import RULES_POLICY_VERSION
from tinychess.chess.state_features import ENCODER_VERSION

META = {b'schema': b'human_v3', b'encoder': ENCODER_VERSION.encode(),
        b'action_map': ACTION_MAP_VERSION.encode(), b'rules': RULES_POLICY_VERSION.encode(),
        b'wdl_order': b'win,draw,loss', b'legal_mask_bitorder': b'little', b'feature_dtype': b'little_endian_float32'}
GAME_SCHEMA = pa.schema([('game_id', pa.string()), ('metadata', pa.string()), ('actions', pa.list_(pa.uint16()))], metadata=META)
POSITION_SCHEMA = pa.schema([
    ('game_id', pa.string()), ('source_month', pa.string()), ('split', pa.string()), ('ply', pa.uint16()),
    ('human_move_uci', pa.string()), ('action', pa.uint16()), ('outcome_stm', pa.int8()), ('wdl_class', pa.uint8()),
    ('value_weight', pa.float32()), ('policy_weight', pa.float32()), ('color', pa.uint8()),
    ('fen', pa.string()), ('repetition_count', pa.uint8()), ('position_hash', pa.binary(32)),
    ('training_eligible', pa.bool_()), ('remaining_plies', pa.uint16()), ('label_source', pa.string()),
    ('planes', pa.binary(8*8*120*4)), ('context', pa.binary(8*4)), ('legal_mask', pa.binary(584)),
], metadata=META)


class MonthWriter:
    """One position row group/game; bounded <=512 rows and ~16 MiB features."""
    def __init__(self, folder, month):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.games_path = folder / f'{month}.games.parquet'
        self.positions_path = folder / f'{month}.positions.parquet'
        self.writer = pq.ParquetWriter(self.positions_path, POSITION_SCHEMA, compression='zstd', compression_level=3)
        self.games = []
        self.index = []

    def append(self, replayed):
        record = replayed.game
        m = record['metadata']
        self.writer.write_table(pa.Table.from_pylist(replayed.positions, schema=POSITION_SCHEMA), row_group_size=512)
        self.index.append(dict(game_id=m['game_id'], game_file=self.games_path.name, game_row=len(self.games),
                               position_file=self.positions_path.name, position_row_group=len(self.games), plies=m['plies']))
        self.games.append(dict(game_id=m['game_id'], metadata=json.dumps(m, sort_keys=True, separators=(',', ':')), actions=record['actions']))

    def close(self):
        self.writer.close()
        pq.write_table(pa.Table.from_pylist(self.games, schema=GAME_SCHEMA), self.games_path,
                       compression='zstd', compression_level=3, row_group_size=256)


def read_games(folder):
    for path in sorted(Path(folder).glob('*.games.parquet')):
        parquet = pq.ParquetFile(path)
        metadata = parquet.schema_arrow.metadata or {}
        if any(metadata.get(key) != value for key, value in META.items()):
            raise ValueError(f'Unsupported dataset/encoder/action/rules metadata: {path}')
        for batch in parquet.iter_batches(batch_size=128):
            for row in batch.to_pylist():
                yield dict(metadata=json.loads(row['metadata']), actions=row['actions'])


def unpack_features(row):
    return dict(planes=np.frombuffer(row['planes'], dtype='<f4').reshape(8, 8, 120),
                context=np.frombuffer(row['context'], dtype='<f4'),
                legal_mask=np.unpackbits(np.frombuffer(row['legal_mask'], np.uint8), bitorder='little').astype(bool))
