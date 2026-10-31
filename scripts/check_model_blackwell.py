"""Single fresh-process Stage 2B worker; always requires the requested GPU."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tinychess.validation.model_blackwell import run_case

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=['25m','100m'],required=True)
    p.add_argument('--compute',choices=['float32','bfloat16'],required=True)
    p.add_argument('--batch',type=int,choices=[4,32],required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    try:result=run_case(a.profile,a.compute,a.batch)
    except BaseException as e:
        target.write_text(json.dumps(dict(status='FAILED',error_type=type(e).__name__),indent=2)+'\n');raise
    target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result.get(k) for k in ['status','profile','compute','batch','parameter_count','parameter_updates','reason']},indent=2))
    raise SystemExit(1 if result['status']=='FAILED' else 0)
