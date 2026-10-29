"""Read-only M1 progress viewer. Ctrl+C never signals the dataset worker."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from tinychess.data.evidence import read_json_evidence


def display(months):
    run = read_json_evidence('reports/m1_50k_run_status.json', root=ROOT)
    print('\033[2J\033[H', end='')
    print('Worker:', run['status'])
    print('Updated (UTC):', run['updated_utc'])
    updated = datetime.fromisoformat(run['updated_utc'])
    print('Status age:', round((datetime.now(timezone.utc)-updated).total_seconds()), 'seconds')
    print('Total completed games:', f"{run.get('cloud_reproduced_games', 0):,}")
    for month in months:
        entry = run.get('months', {}).get(month, {})
        target = entry.get('target_games', 7500 if month in ['2026-08', '2026-09'] else 5000)
        path = ROOT / 'data/raw/m1_month_candidates' / (month + '.status.json')
        if entry.get('cloud_reproduced_games', 0) == target:
            message = f"COMPLETE: {target:,} games; {entry.get('positions', 0):,} positions"
        elif path.exists():
            scan = json.loads(path.read_text())
            total = scan.get('identity', {}).get('archive_size_bytes', 0)
            done = scan.get('compressed_bytes', 0)
            if scan.get('status') == 'VERIFIED_CANDIDATES':
                message = 'Archive verified; validation/upload/reproduction not yet complete'
            elif total:
                message = f'Scanned {done/1e9:.2f}/{total/1e9:.2f} GB ({100*done/total:.1f}%)'
                if scan.get('error'):
                    message += ' — ' + str(scan.get('message', scan['error']))
            else:
                message = scan.get('status', 'Waiting')
        else:
            message = 'Not started'
        print(month + ': ' + message)
        print('  Validated:', entry.get('validated_games', 0), '/', f'{target:,}')
    print('Error:', run.get('error'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--months', nargs='+', default=['2026-05', '2026-06'])
    parser.add_argument('--interval', type=float, default=10)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    try:
        while True:
            try:
                display(args.months)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                print('Waiting for status update:', exc, flush=True)
            if args.once:
                break
            time.sleep(max(1, args.interval))
    except KeyboardInterrupt:
        print('\nViewer stopped. Dataset worker was not interrupted.')
