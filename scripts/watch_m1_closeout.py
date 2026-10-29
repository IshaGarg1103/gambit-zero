"""Read-only viewer for M1 metadata/index checks and Dataset D exclusions."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from tinychess.data.evidence import read_json_evidence


def read(name):
    return read_json_evidence('reports/'+name, root=ROOT, required=False) or {}


def snapshot():
    audit = read('m1_50k_diagnostics.json')
    exclusions = read('m1_dataset_d_exclusions.json')
    block = read('m1_dataset_d_upload_diagnostic.json')
    result = dict(metadata_audit=audit.get('status', 'NOT_STARTED'),
                  months_audited=len(audit.get('months', {})),
                  diagnostic_shards_verified=len(audit.get('receipts', [])),
                  dataset_d_exclusions=exclusions.get('status', 'WAITING_FOR_AUDIT'),
                  months_filtered=len(exclusions.get('months', {})),
                  train_rows_excluded=sum(m['excluded_positions'] for m in exclusions.get('months', {}).values()))
    if (exclusions.get('status') == 'FAIL' and block.get('http_status') == 429
            and (exclusions.get('http_status') == 429 or 'http_status' not in exclusions)):
        result['stop_reason'] = 'HTTP 429; uploads stopped; no chess mismatch reported'
        result['resume_not_before_ist'] = block.get('resume_not_before_ist')
    result['verified_staging_shards'] = len(exclusions.get('rewritten_shards', []))
    for path in (ROOT/'reports').glob('m1-eval-filter-*/overlap.sqlite'):
        try:
            with sqlite3.connect(path.as_uri()+'?mode=ro', uri=True, timeout=1) as db:
                result['games_checked_for_dataset_d_live'] = db.execute('SELECT count(*) FROM games').fetchone()[0]
        except sqlite3.Error:
            pass
    for path in (ROOT/'reports').glob('m1-closeout-*/overlap.sqlite'):
        try:
            with sqlite3.connect(path.as_uri()+'?mode=ro', uri=True, timeout=1) as db:
                result['games_audited_live'] = db.execute('SELECT count(*) FROM games').fetchone()[0]
        except sqlite3.Error:
            pass  # A brief commit lock or completed cleanup is not a run failure.
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=float, default=10)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    try:
        while True:
            state = snapshot()
            print(json.dumps(state, sort_keys=True, indent=2), flush=True)
            if args.once or state['dataset_d_exclusions'] in ['PASS', 'FAIL'] or state['metadata_audit'] == 'FAIL':
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('Viewer stopped; audit workers continue.')
