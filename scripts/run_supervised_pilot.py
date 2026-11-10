"""Stage 3B preparation/preflight by default; execution needs explicit approval flag."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/stage3b_25m.json'))
    p.add_argument('--store',default=str(ROOT/'data/processed/m2_training_originals'))
    p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['sweep','pilot'],required=True)
    p.add_argument('--lr-index',type=int,choices=range(3))
    p.add_argument('--lr',type=float)
    p.add_argument('--download',action='store_true',help='Allow pinned original-shard acquisition; retained persistently')
    p.add_argument('--execute-approved',action='store_true',help='Run ONLY after owner approval of this experiment')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--wandb-mode',choices=['online','offline','disabled'],default='online')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    from tinychess.train.supervised_runner import run
    try:
        result = run(args, ROOT)
    except Exception as exc:
        print('Stage 3B stopped safely:',type(exc).__name__,file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(result,indent=2,sort_keys=True))
    raise SystemExit(0 if result['status'] in ('PREFLIGHT_ONLY','COMPLETED','EARLY_STOPPED') else 2)
