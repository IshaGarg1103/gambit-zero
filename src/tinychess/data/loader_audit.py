"""Full-split exact-once audit using a bounded, disposable SQLite identity index.

Only reads original pinned Parquet. Persistent originals are never deleted.
No model, chess replay, source PGNs, derived tensor cache or data publication.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import shutil
import sqlite3
import sys
import tempfile
import time

import numpy as np
import psutil
from filelock import FileLock

from tinychess.data.training_loader import (
    REVISION, REPO, CATALOG_SHA256, DEFAULT_SHUFFLE_BUFFER, check_shuffle_memory,
    iter_records, iter_batches, require, sha256,
)


def preflight(store, shards, *, shuffle_buffer=DEFAULT_SHUFFLE_BUFFER,
              batch_size=256, chunk_rows=128, state_root=None):
    require(store.keep_downloads, 'Full-split audit requires persistent originals')
    require(shards and all(s.split == 'train' for s in shards), 'Audit only accepts train shards')
    paths = [f.path for s in shards for f in (s.positions, s.games)]
    require(len(paths) == len(set(paths)), 'Duplicate shard file in audit inventory')
    missing = sum(f.size for s in shards for f in (s.positions, s.games)
                  if not (store.root / f.path).exists())
    free = shutil.disk_usage(store.root).free
    # Identity DB + rollback journal + metrics and one incomplete shard.
    work_allowance = 1024 * 2**20
    required = missing + work_allowance + store.reserve_bytes
    require(free >= required, f'Full audit needs {required:,} free bytes; available {free:,}')
    if state_root is not None:
        Path(state_root).mkdir(parents=True, exist_ok=True)
        require(shutil.disk_usage(state_root).free >= work_allowance + store.reserve_bytes,
                'Insufficient audit-state disk reserve')
    estimate = check_shuffle_memory(sum(s.rows for s in shards), shuffle_buffer, batch_size, chunk_rows)
    return dict(original_files=len(paths), original_bytes=sum(f.size for s in shards for f in (s.positions,s.games)),
                missing_download_bytes=missing, free_disk_bytes=free, disk_reserve_bytes=store.reserve_bytes,
                extra_work_allowance_bytes=work_allowance, estimated_shuffle_bytes=estimate,
                host_ram_bytes=psutil.virtual_memory().total, host_available_ram_bytes=psutil.virtual_memory().available,
                logical_cpus=psutil.cpu_count(), shuffle_buffer=shuffle_buffer)


def audit_training_split(store, shards, expected, output, *, shuffle_buffer=DEFAULT_SHUFFLE_BUFFER,
                         batch_size=256, chunk_rows=128, seed=20261030, epochs=2):
    """Baseline identity stream vs every emitted row, for each whole epoch.

    Compare row sets exactly in SQLite, not with a probabilistic sum/hash.
    Source row validation/checksums are still performed on every traversal.
    Per-batch mixing is streamed to JSONL; only small histogram summaries stay
    in RAM. Original Parquet is retained on PASS, FAIL and interruption alike.
    Restart builds a fresh identity index and reuses verified original files;
    it does not claim to resume at an arbitrary row inside a shuffled epoch.
    """
    require(type(epochs) is int and epochs > 0, 'epochs must be positive')
    require(type(seed) is int and seed >= 0, 'seed must be a nonnegative integer')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + '.lock', timeout=1):
        return _audit(store, tuple(shards), expected, output, shuffle_buffer,
                      batch_size, chunk_rows, seed, epochs)


def _audit(store, shards, expected, output, capacity, batch_size, chunk_rows, seed, epochs):
    started = time.monotonic()
    baseline_downloads = store.downloaded_bytes
    report = dict(status='RUNNING', repo=REPO, revision=REVISION, catalog_sha256=CATALOG_SHA256,
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  environment=dict(python=platform.python_version(), platform=platform.platform()),
                  parameters=dict(shuffle_buffer=capacity, batch_size=batch_size, chunk_rows=chunk_rows,
                                  seed=seed, epochs=epochs), epochs=[],
                  scope='complete training split identity accounting; no fresh PGN/Pgx replay',
                  persistent_originals=True, originals_deleted=0)
    def save():
        report['elapsed_seconds'] = time.monotonic()-started
        report['downloaded_bytes_this_audit'] = store.downloaded_bytes-baseline_downloads
        report['peak_process_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform=='darwin' else 1024)
        tmp=output.with_name(output.name+'.partial')
        tmp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');tmp.replace(output)
    db=None;work=None
    metrics=output.with_suffix('.batches.jsonl')
    partial=metrics.with_name(metrics.name+'.partial')
    try:
        report['preflight']=preflight(store, shards, shuffle_buffer=capacity,
                                      batch_size=batch_size,chunk_rows=chunk_rows,state_root=output.parent)
        require(sum(s.rows for s in shards)==expected['positions']
                and sum(s.game_count for s in shards)==expected['games'], 'Training inventory/manifest counts mismatch')
        print('PREFLIGHT '+json.dumps(report['preflight'],sort_keys=True),flush=True)
        save()
        work=Path(tempfile.mkdtemp(prefix='loader-accounting-',dir=output.parent))
        report['accounting_work_directory']=str(work)
        db=sqlite3.connect(work/'accounting.sqlite')
        db.execute('PRAGMA journal_mode=DELETE')
        db.execute('PRAGMA cache_size=-16384')  # 16 MiB, regardless of full epoch length
        db.execute('PRAGMA temp_store=FILE')
        db.execute('PRAGMA max_page_count=131072')  # <=512 MiB at default 4KiB pages
        db.execute('CREATE TABLE expected(gid TEXT,ply INTEGER,month TEXT,seen INTEGER NOT NULL DEFAULT 0,'
                   'PRIMARY KEY(gid,ply)) WITHOUT ROWID')
        games={};counts=Counter();month_rows=Counter();month_eligible=Counter();pending=[]
        original_start=time.monotonic()
        for row in iter_records(store,shards,'train',chunk_rows=chunk_rows):
            gid=row['game_id'];month=row['source_month']
            require(games.setdefault(gid,month)==month,'Game crosses source months/splits')
            counts['rows']+=1;month_rows[month]+=1
            if row['training_eligible']:
                counts['eligible']+=1;month_eligible[month]+=1
                pending.append((gid,row['ply'],month))
            else:
                counts['ineligible']+=1
            if len(pending)>=4096:
                db.executemany('INSERT INTO expected(gid,ply,month) VALUES(?,?,?)',pending);db.commit();pending=[]
            if counts['rows']%100000==0:
                report['reference_progress_rows']=counts['rows'];save()
                print(f"REFERENCE {counts['rows']:,}/{expected['positions']:,}",flush=True)
        db.executemany('INSERT INTO expected(gid,ply,month) VALUES(?,?,?)',pending);db.commit()
        require(counts['rows']==expected['positions'] and len(games)==expected['games'], 'Full baseline manifest counts mismatch')
        for month in {s.month for s in shards}:
            require(month_rows[month]==sum(s.rows for s in shards if s.month==month), 'Month baseline count mismatch')
        report['reference']=dict(rows=counts['rows'],eligible=counts['eligible'],ineligible=counts['ineligible'],
            games=len(games),rows_by_month=dict(month_rows),eligible_by_month=dict(month_eligible),
            seconds=time.monotonic()-original_start,sqlite_bytes=(work/'accounting.sqlite').stat().st_size)
        save()
        with partial.open('w') as stream:
            for epoch in range(epochs):
                db.execute('UPDATE expected SET seen=0');db.commit()
                downloaded_before=store.downloaded_bytes;start=time.monotonic()
                total=0;hist={k:Counter() for k in ['max_positions_one_game','distinct_games','distinct_source_months']}
                order=hashlib.sha256();full_hist={k:Counter() for k in hist};last=None;n=0
                for n,batch in enumerate(iter_batches(store,shards,'train',batch_size=batch_size,chunk_rows=chunk_rows,
                                                     shuffle_buffer=capacity,seed=seed,epoch=epoch),1):
                    require(batch.split=='train' and bool(batch.training_eligible.all()), 'Wrong split or ineligible batch row')
                    identities=list(zip(batch.game_id,batch.ply.tolist(),batch.source_month))
                    cur=db.executemany('UPDATE expected SET seen=1 WHERE gid=? AND ply=? AND month=? AND seen=0',identities)
                    require(cur.rowcount==len(identities), 'Duplicate, unknown, wrong-month or ineligible emitted identity')
                    if n % 16 == 0:
                        db.commit()
                    item=dict(epoch=epoch,batch=n,size=len(identities),
                        max_positions_one_game=batch.max_positions_per_game,
                        distinct_games=len(set(batch.game_id)),distinct_source_months=len(set(batch.source_month)))
                    stream.write(json.dumps(item,sort_keys=True,separators=(',',':'))+'\n')
                    for k in hist:
                        hist[k][item[k]]+=1
                        if len(identities)==batch_size:full_hist[k][item[k]]+=1
                    for gid,ply,month in identities:
                        order.update(json.dumps([gid,ply,month],separators=(',',':')).encode()+b'\n')
                    total+=len(identities);last=item
                    if n%1000==0:
                        report['epoch_progress']=dict(epoch=epoch,batches=n,rows=total);save()
                        print(f'EPOCH {epoch}: {total:,}/{counts["eligible"]:,}',flush=True)
                db.commit()
                missing=db.execute('SELECT count(*) FROM expected WHERE seen!=1').fetchone()[0]
                require(total==counts['eligible'] and missing==0, 'Missing eligible positions at epoch end')
                require(store.downloaded_bytes==downloaded_before, 'Persistent epoch unexpectedly downloaded data')
                seconds=time.monotonic()-start
                report['epochs'].append(dict(epoch=epoch,rows=total,batches=n,last_batch=last,
                    missing_positions=missing,duplicate_positions=0,split_violations=0,
                    downloaded_bytes=store.downloaded_bytes-downloaded_before,seconds=seconds,
                    positions_per_second=total/seconds,order_sha256=order.hexdigest(),
                    mixing_histograms={k:dict(sorted(h.items())) for k,h in hist.items()},
                    full_batch_mixing_histograms={k:dict(sorted(h.items())) for k,h in full_hist.items()}))
                stream.flush();save()
        partial.replace(metrics)
        report['per_batch_metrics']=dict(path=str(metrics),sha256=sha256(metrics),bytes=metrics.stat().st_size,
                                         records=sum(e['batches'] for e in report['epochs']))
        report['status']='PASS_FULL_TRAIN'
    except BaseException as exc:
        report['status']='FAIL';report['error_type']=type(exc).__name__
        # Do not record network error bodies, headers or signed redirect URLs.
        raise
    finally:
        if db is not None:db.close()
        if work is not None and report['status']=='PASS_FULL_TRAIN':
            shutil.rmtree(work);report['accounting_index_removed']=True
        save()
    return report
