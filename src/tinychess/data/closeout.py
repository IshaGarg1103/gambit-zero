"""Exact, bounded diagnostics over already verified Dataset A metadata.

This module never imports positions into Pgx, changes training tensors, or
generates labels. Published hashes identify boards, including legal EP rights;
they deliberately ignore clocks/history for conservative overlap reporting.
"""
from collections import Counter
import hashlib
import json

from tinychess.data.hf_audit import AuditIndex, BANDS


class CloseoutIndex(AuditIndex):
    def __init__(self, root, signature, guard):
        super().__init__(root, signature, guard)
        self.db.execute('CREATE TABLE IF NOT EXISTS trajectories '
                        '(sha TEXT PRIMARY KEY, n INTEGER NOT NULL) WITHOUT ROWID')
        self.db.commit()
        self.stats = Counter()
        self.distributions = {}

    def count(self, name, value):
        self.distributions.setdefault(name, Counter())[str(value)] += 1

    def add_game(self, game, month, split):
        metadata = json.loads(game['metadata'])
        gid = game['game_id']
        if (metadata['game_id'], metadata['source_month'], metadata['split']) != (gid, month, split):
            raise ValueError('Game identity/month/split mismatch')
        if not 0 < metadata['plies'] <= 512 or len(game['actions']) != metadata['plies']:
            raise ValueError('Game action length/horizon mismatch')
        if any(not 0 <= action < 4672 for action in game['actions']):
            raise ValueError('Game action outside codec range')
        rating = min(metadata['white_elo'], metadata['black_elo'])
        base = int(metadata['time_control'].split('+')[0])
        if rating < 1800 or base < 300:
            raise ValueError('Rating/time-control filter violated')
        result = {'1-0': 1, '0-1': -1, '1/2-1/2': 0}[metadata['result']]
        if metadata['outcome_white'] != result or metadata['value_weight'] not in (0, 1):
            raise ValueError('Game result/value-weight mismatch')
        if metadata['ending_semantics'] == 'time_forfeit' and metadata['value_weight'] != 0:
            raise ValueError('Time-forfeit value loss was not masked')
        # Unlike the generation index, even identical repeated games fail here:
        # the published inventory must contain each source game exactly once.
        try:
            self.db.execute('INSERT INTO games VALUES(?,?,?,?)',
                            (gid, month, split, metadata['trajectory_sha256']))
        except Exception as exc:
            import sqlite3
            if isinstance(exc, sqlite3.IntegrityError):
                raise ValueError('Duplicate published game identity') from None
            raise
        self.db.execute('INSERT INTO trajectories VALUES(?,1) '
                        'ON CONFLICT(sha) DO UPDATE SET n=n+1', (metadata['trajectory_sha256'],))
        for name, value in [('game_split', split), ('ending', metadata['ending_semantics']),
                            ('result', metadata['result']), ('eco', metadata['eco']),
                            ('time_control', metadata['time_control']),
                            ('rating_band_lower_player', rating // 200 * 200),
                            ('utc_day', metadata['utc_date'])]:
            self.count(name, value)
            self.count(month + ':' + name, value)
        self.stats['games'] += 1
        self.stats['time_forfeit_games'] += int(metadata['ending_semantics'] == 'time_forfeit')
        self.stats['strong_games'] += int(rating >= 2000)
        return metadata

    def add_rows(self, rows, games, month, split, seen_plies):
        all_col, late_col = BANDS[split]
        updates = []
        for row in rows:
            game = games[row['game_id']]
            ply = row['ply']
            if row['split'] != split or row['source_month'] != month:
                raise ValueError('Position month/split mismatch')
            if ply != seen_plies.get(row['game_id'], 0) or ply >= game['plies']:
                raise ValueError('Position ordering/duplicate/missing ply')
            seen_plies[row['game_id']] = ply + 1
            sign = game['outcome_white'] * (1 if ply % 2 == 0 else -1)
            if (row['outcome_stm'], row['wdl_class'], row['value_weight']) != (
                    sign, {1: 0, 0: 1, -1: 2}[sign], game['value_weight']):
                raise ValueError('Position WDL/perspective/value-mask mismatch')
            if not row['training_eligible'] or row['policy_weight'] != 1:
                raise ValueError('Unexpected excluded/weighted row in original catalog')
            if len(row['position_hash']) != 32:
                raise ValueError('Invalid board hash length')
            updates.append((row['position_hash'], 1, int(ply >= 10)))
            self.stats['positions'] += 1
            self.stats['value_masked_rows'] += int(row['value_weight'] == 0)
            for name, value in [('position_split', split), ('wdl_class', row['wdl_class']),
                                ('side_to_move', 'white' if ply % 2 == 0 else 'black'),
                                ('phase_by_ply', 'opening' if ply < 20 else 'middlegame' if ply < 80 else 'late')]:
                self.count(name, value)
                self.count(month + ':' + name, value)
        self.db.executemany(f'INSERT INTO positions(k,{all_col},{late_col}) VALUES(?,?,?) '
                            f'ON CONFLICT(k) DO UPDATE SET {all_col}={all_col}+1,'
                            f'{late_col}={late_col}+excluded.{late_col}', updates)

    def disjoint_rows(self, rows, source_path):
        """All held-out rows absent from train at *every* ply, not just late plies.

        Validation and test may overlap each other. Multiple held-out rows at the
        same unseen board remain separate observations; this is not deduplication.
        """
        for offset, row in enumerate(rows):
            if row['split'] not in ('validation', 'test'):
                raise ValueError('Disjoint evaluation index only accepts held-out rows')
            match = self.db.execute('SELECT ta FROM positions WHERE k=?',
                                    (row['position_hash'],)).fetchone()
            if match is None:
                raise ValueError('Unindexed position in diagnostic pass')
            if match[0] == 0:
                yield dict(game_id=row['game_id'], split=row['split'], ply=row['ply'],
                           position_hash=row['position_hash'], source_path=source_path,
                           row_in_shard=offset)

    def result(self):
        return dict(overlap=self.summary(), counts=dict(self.stats),
                    distributions={key: dict(sorted(value.items()))
                                   for key, value in sorted(self.distributions.items())},
                    repeated_trajectory_groups=self.db.execute(
                        'SELECT count(*) FROM trajectories WHERE n>1').fetchone()[0])


def payload_sha(payload):
    return hashlib.sha256(payload).hexdigest()
