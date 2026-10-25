"""Staged random-position audit, with genuine repeated/100/150-halfmove traces."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import chess
import jax
import numpy as np

from tinychess.chess.pgx_env import ENV, init, step, RULES_POLICY_VERSION
from tinychess.validation.parity import Oracle, ParityError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.positions < 1:
        parser.error("positions must be positive")
    started = time.perf_counter()
    stats = Counter()
    identities = set()
    sequence = hashlib.sha256()
    game = 0
    moves = []

    def check(state, oracle):
        actions, host, count = oracle.check(state)
        stats['checked_positions'] += 1
        stats['check_positions'] += int(oracle.board.is_check())
        stats['threefold_positions'] += int(count >= 3)
        stats['100_halfmove_positions'] += int(oracle.board.halfmove_clock >= 100)
        stats['150_halfmove_positions'] += int(oracle.board.halfmove_clock >= 150)
        identities.add(oracle.board.fen(en_passant='legal'))
        sequence.update(oracle.board.fen(en_passant='fen').encode())
        return actions, host

    try:
        while stats['random_positions'] < args.positions:
            seed = args.seed + game
            rng = random.Random(seed)
            oracle = Oracle()
            state = init(jax.random.PRNGKey(seed))
            moves = []
            while stats['random_positions'] < args.positions:
                actions, host = check(state, oracle)
                stats['random_positions'] += 1
                if host.terminated or host.truncated:
                    stats['completed_games' if host.terminated else 'truncated_games'] += 1
                    break
                uci = rng.choice(sorted(actions))
                moves.append(uci)
                state = step(state, actions[uci])
                oracle.push(uci)
                if stats['random_positions'] % 5000 == 0:
                    print(dict(stats), flush=True)
            game += 1
        # Explicit trajectory tests supplement random coverage, not fake random games.
        oracle = Oracle()
        state = init(jax.random.PRNGKey(0))
        moves = []
        for uci in ['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 4:
            actions, _ = check(state, oracle)
            state = step(state, actions[uci])
            oracle.push(uci)
            moves.append(uci)
        check(state, oracle)
        assert state.terminated and oracle.board.is_fivefold_repetition()
        # Start from a known sparse legal board, then genuinely play 150 quiet
        # plies, avoiding automatic repetition/mate before the 75-move boundary.
        fen = '7k/8/8/8/8/8/8/R3K3 w - - 0 1'
        oracle = Oracle(chess.Board(fen))
        state = ENV.from_fen(fen, history='discard')
        rng = random.Random(143)
        moves = []
        for ply in range(150):
            actions, _ = check(state, oracle)
            candidates = sorted(actions)
            rng.shuffle(candidates)
            for uci in candidates:
                move = chess.Move.from_uci(uci)
                if oracle.board.is_capture(move):
                    continue
                nxt = oracle.board.copy()
                nxt.push(move)
                if ply < 149 and nxt.outcome(claim_draw=False) is not None:
                    continue
                break
            else:
                raise RuntimeError('Quiet-trajectory generator exhausted candidates')
            state = step(state, actions[uci])
            oracle.push(uci)
            moves.append(uci)
        check(state, oracle)
        assert state.terminated and oracle.board.is_seventyfive_moves()
    except Exception as exc:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(status='FAIL_STOP', error=str(exc), stats=dict(stats), seed=args.seed + game,
                                               moves_uci=moves), indent=2)+'\n')
        raise
    report = dict(status='PASS', rules_policy=RULES_POLICY_VERSION, seed=args.seed,
                  requested_random_positions=args.positions, stats=dict(stats),
                  unique_full_fens=len(identities), trajectory_sha256=sequence.hexdigest(),
                  elapsed_seconds=time.perf_counter()-started, mismatches=0,
                  devices=[str(d) for d in jax.devices()],
                  limitations='Sampled seeded games plus two constructed histories; not exhaustive legality or dead-position proof.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
