"""Synthetic source rows test evaluation-only freeze semantics, not chess skill."""
import copy
import io
from pathlib import Path
import runpy
import time

import chess
import pytest
import yaml

from tinychess.data.frozen_eval import PuzzleSampler, evaluation_positions, validate_puzzle


def row(identity='fixture', rating=1500):
    return dict(PuzzleId=identity, FEN=chess.STARTING_FEN, Moves='e2e4 e7e5 g1f3',
                Rating=str(rating), Popularity='50', NbPlays='200', RatingDeviation='60')


def test_solver_root_is_after_opponents_first_move():
    frozen = validate_puzzle(row(), 1500)
    board = chess.Board()
    board.push_uci('e2e4')
    assert frozen['root_fen'] == board.fen(en_passant='fen')
    assert frozen['solution_uci'] == ['e7e5', 'g1f3']
    assert frozen['root_fen'] != frozen['initial_fen']
    assert len(frozen['overlap_fens']) == 2
    assert frozen['evaluation_only']
    assert len(evaluation_positions([frozen, frozen], [])) == 2


def test_illegal_frozen_solution_is_blocking():
    source = row()
    source['Moves'] = 'e2e4 e7e4'
    with pytest.raises(ValueError, match='Illegal'):
        validate_puzzle(source, 1500)


def test_difficulty_balanced_selection_is_order_independent_and_bounded():
    config = yaml.safe_load((Path(__file__).parents[1] / 'configs/dataset_d.yaml').read_text())
    config['puzzles']['rating_band_starts'] = [0, 2000]
    config['puzzles']['per_rating_band'] = 2
    source = [row(str(i), 1500 if i % 2 else 2100) for i in range(30)]
    selected = []
    for records in [source, list(reversed(source))]:
        sampler = PuzzleSampler(copy.deepcopy(config))
        for record in records:
            sampler.add(record)
        assert sum(map(len, sampler.heaps.values())) == 4
        selected.append([p['id'] for p in sampler.finish()])
    assert selected[0] == selected[1]
    assert len(set(selected[0])) == 4


def test_hashing_reader_rejects_excess_bytes_and_preserves_complete_source_hash():
    import hashlib
    cls = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/freeze_dataset_d.py'))['HashingReader']
    raw = cls(io.BytesIO(b'abcdef'), 6, time.monotonic()+10)
    assert io.BufferedReader(raw).read() == b'abcdef'
    assert raw.count == 6 and raw.digest.hexdigest() == hashlib.sha256(b'abcdef').hexdigest()
    with pytest.raises(ValueError, match='exceeds'):
        io.BufferedReader(cls(io.BytesIO(b'abcdef'), 5, time.monotonic()+10)).read()


def test_quality_shortfall_does_not_silently_relax_evaluation_policy():
    config = yaml.safe_load((Path(__file__).parents[1] / 'configs/dataset_d.yaml').read_text())
    sampler = PuzzleSampler(config)
    source = row()
    source['NbPlays'] = '1'
    sampler.add(source)
    assert sampler.counts['quality_filtered'] == 1
    with pytest.raises(ValueError, match='Insufficient'):
        sampler.finish()
