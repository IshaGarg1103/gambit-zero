"""Bounded exact board-overlap metadata; no feature tensors/PGNs retained."""
import json
from pathlib import Path
import sqlite3

from tinychess.data.archive_scan import ScanError,canonical

BANDS={'train':('ta','tl'),'validation':('va','vl'),'test':('ea','el')}


class AuditIndex:
    def __init__(self,root,signature,guard):
        self.root,self.guard=Path(root),guard
        self.root.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.root/'overlap.sqlite')
        self.db.execute('PRAGMA journal_mode=DELETE')
        self.db.execute('PRAGMA temp_store=MEMORY')
        # 512 MiB DB + at most ~512 MiB rollback journal: 1 GiB disk plan.
        page_size=self.db.execute('PRAGMA page_size').fetchone()[0]
        self.db.execute(f'PRAGMA max_page_count={512*2**20//page_size}')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS metadata(k TEXT PRIMARY KEY,v TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS games(id TEXT PRIMARY KEY,month TEXT,split TEXT,sha TEXT) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS positions(k BLOB PRIMARY KEY,ta INTEGER DEFAULT 0,tl INTEGER DEFAULT 0,
                va INTEGER DEFAULT 0,vl INTEGER DEFAULT 0,ea INTEGER DEFAULT 0,el INTEGER DEFAULT 0) WITHOUT ROWID;
        ''')
        existing=self.db.execute("SELECT v FROM metadata WHERE k='signature'").fetchone()
        if existing and existing[0]!=signature:
            raise ScanError('Audit index belongs to a different run/config/evaluation snapshot')
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES('signature',?)",(signature,))
        self.db.commit()

    def begin(self):
        self.guard.check()
        self.db.execute('BEGIN')

    def add(self,replayed,logical_sha):
        m=replayed.game['metadata']
        old=self.db.execute('SELECT month,split,sha FROM games WHERE id=?',(m['game_id'],)).fetchone()
        if old:
            if old!=(m['source_month'],m['split'],logical_sha):
                raise ScanError('Whole-game split/content overlap detected')
            return  # Deterministic repeated build: do not double-count metadata.
        self.db.execute('INSERT INTO games VALUES(?,?,?,?)',(m['game_id'],m['source_month'],m['split'],logical_sha))
        all_col,late_col=BANDS[m['split']]
        for row in replayed.positions:
            if m['split']=='train' and not row['training_eligible']:
                continue
            self.db.execute(f'''INSERT INTO positions(k,{all_col},{late_col}) VALUES(?,?,?)
                ON CONFLICT(k) DO UPDATE SET {all_col}={all_col}+1,{late_col}={late_col}+excluded.{late_col}''',
                (row['position_hash'],1,int(row['ply']>=10)))
        self.guard.check()

    def commit(self):
        self.guard.check()
        self.db.commit()

    def rollback(self):
        self.db.rollback()

    def summary(self,frozen_fens=None):
        out={}
        for label,columns in [('all_plies',['ta','va','ea']),('ply_ge_10',['tl','vl','el'])]:
            t,v,e=columns
            row=self.db.execute(f'''SELECT count(*) FILTER(WHERE {t}>0), count(*) FILTER(WHERE {v}>0),
                count(*) FILTER(WHERE {e}>0),count(*) FILTER(WHERE {t}>0 AND {v}>0),
                count(*) FILTER(WHERE {t}>0 AND {e}>0),count(*) FILTER(WHERE {t}>0 AND {v}>0 AND {e}>0),
                coalesce(sum({t}),0),coalesce(sum({v}),0),coalesce(sum({e}),0) FROM positions''').fetchone()
            out[label]=dict(zip(['train_unique','validation_unique','test_unique','train_validation_shared',
                'train_test_shared','all_three_shared','train_rows','validation_rows','test_rows'],row))
            out[label]['train_validation_shared_percent_of_validation']=100*row[3]/row[1] if row[1] else 0.0
            out[label]['train_test_shared_percent_of_test']=100*row[4]/row[2] if row[2] else 0.0
            if frozen_fens is not None:
                import chess
                import hashlib
                from tinychess.validation.parity import board_identity
                keys={hashlib.sha256(board_identity(chess.Board(fen)).encode()).digest() for fen in frozen_fens}
                matches=[self.db.execute(f'SELECT {t},{v},{e} FROM positions WHERE k=?',(key,)).fetchone() for key in keys]
                out[label]['dataset_d_unique']=len(keys)
                for index,split in enumerate(['train','validation','test']):
                    out[label][split+'_dataset_d_shared_unique']=sum(bool(r and r[index]>0) for r in matches)
        out['games_by_split']=dict(self.db.execute('SELECT split,count(*) FROM games GROUP BY split'))
        out['equivalence']='SHA256(pieces, side to move, castling, legally capturable EP); clocks/history excluded'
        return out

    def close(self):
        self.db.close()
