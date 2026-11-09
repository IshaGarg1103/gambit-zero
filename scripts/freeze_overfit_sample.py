"""Freeze Stage 3A inputs from existing local verified originals (no downloads)."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tinychess.train.overfit_sample import build_sample
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--store',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=20261031);p.add_argument('--size',type=int,default=1024)
    a=p.parse_args();t=time.perf_counter();m=build_sample(a.store,a.output,size=a.size,seed=a.seed)
    print(json.dumps({k:m[k] for k in ('positions','games','months','coverage','value_masked_rows','sample_bytes','logical_sha256','downloaded_bytes')},indent=2))
    print('sample_build_seconds',time.perf_counter()-t)
