"""Offline evaluation-fixture selection; never supplies Dataset A targets."""
from collections import Counter
import csv
import hashlib
import heapq
import io

import chess
import chess.pgn

from tinychess.data.archive_scan import canonical
from tinychess.validation.parity import board_identity


def priority(seed, kind, identity):
    return int.from_bytes(hashlib.sha256(canonical([seed, kind, identity]).encode()).digest(), 'big')


class PuzzleSampler:
    def __init__(self, config):
        self.config = config
        self.heaps = {start: [] for start in config['puzzles']['rating_band_starts']}
        self.retained_ids = {start: set() for start in self.heaps}
        self.counts = Counter()

    def add(self, row):
        self.counts['source_rows'] += 1
        cfg = self.config['puzzles']
        try:
            rating = int(row['Rating'])
            if (rating < 0 or int(row['Popularity']) < cfg['min_popularity']
                    or int(row['NbPlays']) < cfg['min_plays']
                    or int(row['RatingDeviation']) > cfg['max_rating_deviation']):
                self.counts['quality_filtered'] += 1
                return
            band = max(start for start in self.heaps if start <= rating)
        except (KeyError, ValueError):
            raise ValueError('Malformed puzzle metadata; source snapshot rejected') from None
        identity = row['PuzzleId']
        rank = priority(self.config['seed'], 'puzzle', identity)
        heap = self.heaps[band]
        item = (-rank, identity, row)
        # A duplicate candidate with an identical ID would otherwise compare
        # dictionaries in heapq. Reject it in the bounded retained reservoir.
        if identity in self.retained_ids[band]:
            raise ValueError('Duplicate retained puzzle identity')
        if len(heap) < cfg['per_rating_band']:
            heapq.heappush(heap, item)
            self.retained_ids[band].add(identity)
        elif item[:2] > heap[0][:2]:
            removed = heapq.heapreplace(heap, item)
            self.retained_ids[band].remove(removed[1])
            self.retained_ids[band].add(identity)
        self.counts['eligible_band_'+str(band)] += 1

    def finish(self):
        output = []
        for band, heap in sorted(self.heaps.items()):
            if len(heap) != self.config['puzzles']['per_rating_band']:
                raise ValueError('Insufficient puzzles for frozen rating quota')
            for _, _, row in sorted(heap, key=lambda value: (-value[0], value[1])):
                output.append(validate_puzzle(row, band))
        if len({row['id'] for row in output}) != len(output):
            raise ValueError('Duplicate frozen puzzle identity')
        return output


def validate_puzzle(row, band):
    board = chess.Board(row['FEN'])
    if not board.is_valid():
        raise ValueError('Invalid puzzle FEN')
    moves = row['Moves'].split()
    if len(moves) < 2:
        raise ValueError('Puzzle lacks opponent setup move and solver solution')
    # The official FEN is BEFORE the opponent's setup move, not the solver root.
    fens = []
    for index, uci in enumerate(moves):
        if index > 0:
            fens.append(board.fen(en_passant='fen'))
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError('Illegal frozen puzzle solution: '+row['PuzzleId'])
        board.push(move)
    return dict(id=row['PuzzleId'], source_row=dict(row), rating_band=band,
                source_row_sha256=hashlib.sha256(canonical(row).encode()).hexdigest(),
                initial_fen=row['FEN'], setup_uci=moves[0], root_fen=fens[0],
                solution_uci=moves[1:], overlap_fens=fens,
                history_status='FEN clocks known; prior repetition history unavailable',
                evaluation_only=True)


def select_openings(payloads, config):
    output = []
    for letter, quota in sorted(config['openings']['quotas'].items()):
        families = {}
        for row in csv.DictReader(io.StringIO(payloads[letter].decode()), delimiter='\t'):
            game = chess.pgn.read_game(io.StringIO(row['pgn']))
            if game is None or game.errors:
                raise ValueError('Invalid official opening PGN')
            moves = list(game.mainline_moves())
            if not config['openings']['min_plies'] <= len(moves) <= config['openings']['max_plies']:
                continue
            board = game.end().board()
            if board.is_game_over(claim_draw=False):
                continue
            identity = canonical([row['eco'], row['name'], row['pgn']])
            rank = priority(config['seed'], 'opening', identity)
            family = row['name'].split(':')[0]
            candidate = (rank, identity, dict(eco=row['eco'], name=row['name'], pgn=row['pgn'],
                         moves_uci=[m.uci() for m in moves], fen=board.fen(en_passant='fen'),
                         play_colors=['white', 'black'], history_status='complete from standard start'))
            if family not in families or candidate[:2] < families[family][:2]:
                families[family] = candidate
        selected = sorted(families.values(), key=lambda value: value[:2])[:quota]
        if len(selected) != quota:
            raise ValueError('Insufficient distinct opening families in '+letter)
        output.extend(value[2] for value in selected)
    # Avoid allowing two evaluation fixtures with the same root to inflate counts.
    if len({board_identity(chess.Board(entry['fen'])) for entry in output}) != len(output):
        raise ValueError('Duplicate selected opening roots; revise selection policy explicitly')
    return output


def evaluation_positions(puzzles, openings):
    fens = [fen for puzzle in puzzles for fen in puzzle['overlap_fens']]
    fens.extend(opening['fen'] for opening in openings)
    # One representative FEN per conservative board identity; clocks/history do
    # not expand the exclusion set. Never exclude whole opening families.
    unique = {}
    for fen in fens:
        unique.setdefault(board_identity(chess.Board(fen)), fen)
    return [unique[key] for key in sorted(unique)]
