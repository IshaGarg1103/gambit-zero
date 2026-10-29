"""Bounded game-aligned Parquet outbox; flush/upload blocks further generation."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tinychess.data.archive_scan import canonical,ScanError
from tinychess.data.human import RESULTS,SCHEMA_VERSION,split_for_month
from tinychess.data.storage import POSITION_SCHEMA,GAME_SCHEMA
from tinychess.data.sources import sha256_file


def validate_rows(replayed,month,*,materialized):
    m=replayed.game['metadata']
    if m['schema']!=SCHEMA_VERSION or m['source_month']!=month or m['split']!=split_for_month(month):
        raise ScanError('Game/schema/month/split mismatch before sharding')
    if len(replayed.positions)!=m['plies'] or len(replayed.game['actions'])!=m['plies']:
        raise ScanError('Incomplete game rows/actions')
    if m['plies']>512 or 'truncat' in m['final_board_reason']:
        raise ScanError('Truncation cannot provide supervised WDL labels')
    for ply,(row,action) in enumerate(zip(replayed.positions,replayed.game['actions'])):
        outcome=RESULTS[m['result']]*(1 if row['color']==0 else -1)
        cls=0 if outcome>0 else 2 if outcome<0 else 1
        if (row['game_id']!=m['game_id'] or row['ply']!=ply or row['action']!=action
                or row['split']!=m['split'] or row['outcome_stm']!=outcome or row['wdl_class']!=cls
                or row['label_source']!='human_game' or row['value_weight']!=m['value_weight']
                or row['policy_weight']!=1.0):
            raise ScanError('Invalid human target/action/perspective metadata')
        if materialized:
            mask=np.unpackbits(np.frombuffer(row['legal_mask'],np.uint8),bitorder='little')
            if len(mask)!=4672 or not mask[action]:
                raise ScanError('Materialized human action missing from Pgx mask')
            if len(row['planes'])!=30720 or len(row['context'])!=32:
                raise ScanError('Incorrect canonical feature byte dimensions')
            if not all(np.isfinite(np.frombuffer(row[k],dtype='<f4')).all() for k in ['planes','context']):
                raise ScanError('Nonfinite canonical model inputs')


class ShardOutbox:
    """At most one open positions/games pair; no month-sized features retained.

    on_shard must finish publishing/verifying before a new pair can be opened.
    Rows excluded by frozen Dataset D are physically absent from training shards.
    Game action sequences remain provenance, never an unfiltered training reader.
    """
    def __init__(self,root,month,guard,on_shard,*,max_file_bytes=32*2**20,max_games=128,namespace=''):
        if max_file_bytes<2**20 or max_games<1:
            raise ValueError('Shard cap >=1 MiB and positive game cap required')
        if namespace and not re.fullmatch('[A-Za-z0-9_-]+',namespace):
            raise ValueError('Unsafe shard namespace')
        self.root,self.month,self.guard,self.on_shard=Path(root),month,guard,on_shard
        self.split=split_for_month(month)
        self.max_file_bytes,self.max_games=max_file_bytes,max_games
        self.namespace=namespace
        self.part,self.games,self.rows,self.excluded=0,[],0,0
        self.position_writer=self.position_sink=None
        self.receipts=[]
        self.closed=False

    def _open(self):
        prefix=(self.namespace+'-' if self.namespace else '')+f'{self.month}-{self.part:05}'
        self.position_path=self.root/f'{prefix}.positions.parquet.partial'
        self.game_path=self.root/f'{prefix}.games.parquet.partial'
        # Unfinished files are not silently overwritten on restart.
        if self.position_path.exists() or self.game_path.exists():
            raise ScanError('Unsealed shard remains; inspect/recover it before restart')
        self.guard.check(4*2**20)
        self.position_sink=pa.OSFile(str(self.position_path),'wb')
        self.position_writer=pq.ParquetWriter(self.position_sink,POSITION_SCHEMA,compression='zstd',compression_level=3)

    def append(self,replayed):
        if self.closed:
            raise ScanError('Outbox closed or failed; stop generating more shards')
        validate_rows(replayed,self.month,materialized=True)
        eligible=[row for row in replayed.positions if self.split!='train' or row['training_eligible']]
        table=pa.Table.from_pylist(eligible,schema=POSITION_SCHEMA)
        # Uncompressed next-game payload + 25% expansion and 1 MiB footer
        # reserve. Oversized games fail before writing or uploading them.
        estimate=int(table.nbytes*1.25)+2**20
        if estimate>self.max_file_bytes:
            raise ScanError('One validated game cannot fit the configured shard bound')
        if self.position_writer and (len(self.games)>=self.max_games or self.position_sink.tell()+estimate>self.max_file_bytes):
            self.flush()
        if not self.position_writer:
            self._open()
        self.guard.check(estimate)
        self.position_writer.write_table(table,row_group_size=512)
        m=replayed.game['metadata']
        self.games.append(dict(game_id=m['game_id'],metadata=canonical(m),actions=replayed.game['actions']))
        self.rows+=len(eligible)
        self.excluded+=len(replayed.positions)-len(eligible)

    def flush(self):
        if not self.position_writer:
            return
        self.position_writer.close()
        self.position_sink.close()
        self.position_writer=self.position_sink=None
        games=pa.Table.from_pylist(self.games,schema=GAME_SCHEMA)
        self.guard.check(int(games.nbytes*1.25)+2**20)
        pq.write_table(games,self.game_path,compression='zstd',compression_level=3,row_group_size=128)
        files=[]
        for kind,path in [('positions',self.position_path),('games',self.game_path)]:
            if path.stat().st_size>self.max_file_bytes:
                self.closed=True
                raise ScanError('Sealed Parquet exceeded file cap; retained for inspection, not uploaded')
            sealed=path.with_suffix('')
            if sealed.exists():
                if sha256_file(sealed)!=sha256_file(path):
                    self.closed=True
                    raise ScanError('Existing sealed shard differs; no overwrite')
                path.unlink()
            else:
                path.replace(sealed)
            files.append(dict(kind=kind,path=sealed,sha256=sha256_file(sealed),bytes=sealed.stat().st_size))
        descriptor=dict(month=self.month,split=self.split,part=self.part,games=len(self.games),
                        rows=self.rows,excluded_dataset_d_rows=self.excluded,files=files,
                        game_ids_sha256=hashlib.sha256(canonical([g['game_id'] for g in self.games]).encode()).hexdigest())
        try:
            self.receipts.append(self.on_shard(descriptor))
        except Exception:
            self.closed=True  # Upload outage exerts backpressure: stop writing.
            raise
        self.part+=1
        self.games,self.rows,self.excluded=[],0,0

    def close(self):
        if not self.closed:
            self.flush()
        self.closed=True
        return self.receipts
