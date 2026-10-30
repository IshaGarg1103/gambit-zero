"""Reproducible full training-split audit; persistent originals, no PGN scan."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tinychess.data.training_loader import VerifiedStore, inventory, DEFAULT_SHUFFLE_BUFFER
from tinychess.data.loader_audit import preflight, audit_training_split


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='data/processed/m2_training_originals')
    p.add_argument('--output',default='data/processed/m2_stage1_followup/full_train.json')
    p.add_argument('--download',action='store_true')
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--shuffle-buffer',type=int,default=DEFAULT_SHUFFLE_BUFFER)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--chunk-rows',type=int,default=128)
    p.add_argument('--seed',type=int,default=20261030)
    p.add_argument('--epochs',type=int,default=2)
    a=p.parse_args()
    store=VerifiedStore(a.root,download=a.download)  # persistent default, never audit cleanup
    shards,catalog=inventory(store)
    train=[s for s in shards if s.split=='train']
    if a.preflight_only:
        print(json.dumps(preflight(store,train,shuffle_buffer=a.shuffle_buffer,
                                  batch_size=a.batch_size,chunk_rows=a.chunk_rows),indent=2))
        return
    result=audit_training_split(store,train,catalog['counts']['train'],a.output,
        shuffle_buffer=a.shuffle_buffer,batch_size=a.batch_size,chunk_rows=a.chunk_rows,
        seed=a.seed,epochs=a.epochs)
    print(result['status'])


if __name__=='__main__':main()
