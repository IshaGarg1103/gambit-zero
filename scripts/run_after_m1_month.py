"""Wait for a verified M1 month, then run exactly one configured next month."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from tinychess.data.evidence import read_json_evidence


def predecessor_ready(state, month, expected):
    if state.get('status') in ['STOPPED', 'CANCELLED_CLEANED'] or state.get('error'):
        raise RuntimeError('Generation stopped; inspect its error and recovery files before continuing')
    return (state.get('status') == 'MONTH_COMPLETE'
            and state.get('session_months') == [month]
            and state.get('months', {}).get(month, {}).get('cloud_reproduced_games') == expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    months = config.get('session_months', [])
    if len(months) != 1 or months[0] <= args.after:
        raise RuntimeError('Expected one successor month after the prerequisite')
    if not config.get('resource_budget_verified'):
        raise RuntimeError('Successor resource allowance is not enabled')
    target = months[0]
    expected = config['months'][args.after]['accepted_games']
    target_games = config['months'][target]['accepted_games']
    # Prevent two copies of this waiting command from launching duplicates.
    with (ROOT / 'reports/.m1_waiting_launcher.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another waiting month launcher is already active') from None
        print(f'Waiting for {args.after}: MONTH_COMPLETE and {expected:,} verified games.', flush=True)
        while True:
            state = read_json_evidence('reports/m1_50k_run_status.json', root=ROOT)
            if (state.get('status') in ['MONTH_COMPLETE', 'COMPLETE']
                    and state.get('months', {}).get(target, {}).get('cloud_reproduced_games') == target_games):
                print(f'{target} already verified; no duplicate run started.', flush=True)
                return 0
            if predecessor_ready(state, args.after, expected):
                break
            print('Still waiting; current worker status:', state.get('status'), flush=True)
            time.sleep(30)
        worker = config_path.parents[1] / 'scripts/run_dataset_a_pilot.py'
        print(f'Starting {target} only: {target_games:,} games.', flush=True)
        result = subprocess.run([sys.executable, '-u', str(worker), '--config',
                                 str(config_path), '--project', str(ROOT)], cwd=ROOT)
        if result.returncode:
            return result.returncode
        state = read_json_evidence('reports/m1_50k_run_status.json', root=ROOT)
        if (state.get('status') not in ['MONTH_COMPLETE', 'COMPLETE']
                or state.get('months', {}).get(target, {}).get('cloud_reproduced_games') != target_games):
            raise RuntimeError('Worker exited without verified target-month completion')
        return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nWaiting launcher interrupted. If the successor was running, it may also have been interrupted.')
        raise SystemExit(130)
