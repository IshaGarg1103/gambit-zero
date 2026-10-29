"""One-shot Dataset A worker with bounded progress, report, verification/cleanup.

Run from a frozen source snapshot. No scheduling, training or Dataset D creation.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import plistlib
import subprocess
import signal
import runpy
import shutil
import sys
import threading
import time
import zipfile

os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
os.environ['JAX_PLATFORMS']='cpu'
SNAPSHOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SNAPSHOT/'src'))
os.chdir(SNAPSHOT)

import psutil
import yaml
# Operational interlock: a disarmed launch must not overwrite the active or
# completed attempt's report. Applies only to a new process, not a live scan.
if __name__ == '__main__' and '--config' in sys.argv:
    _arg_index = sys.argv.index('--config') + 1
    if _arg_index < len(sys.argv):
        _launch_config = yaml.safe_load(Path(sys.argv[_arg_index]).read_text())
        if _launch_config.get('resource_budget_verified') is False:
            raise SystemExit(0)
from tinychess.data.archive_scan import ScanError,atomic_json
from tinychess.data.hf_upload import VerifiedUploader,UploadError


def now():return datetime.now(timezone.utc).isoformat()


def external_power_status():
    if sys.platform!='darwin':
        return dict(required=False,external_connected=True)
    result=subprocess.run(['/usr/sbin/ioreg','-a','-r','-c','AppleSmartBattery'],
                          capture_output=True,check=True,timeout=10)
    batteries=plistlib.loads(result.stdout)
    if not batteries:
        raise ScanError('Unable to confirm MacBook external power')
    battery=batteries[0]
    return dict(required=True,external_connected=battery.get('ExternalConnected') is True,
        charging=battery.get('IsCharging'),percent=battery.get('CurrentCapacity'),
        remaining_minutes=battery.get('TimeRemaining'))


def startup_power_status(config):
    # Restore the charger-required default at the user's request. An explicit
    # false remains available for deliberately configured battery runs.
    if not config.get('require_external_power',True):
        return dict(required=False,monitoring_enabled=False)
    power=external_power_status()
    if not power['external_connected']:
        raise ScanError('External power was explicitly required for this run but is not connected')
    return power


def main(config_path,project):
    config=yaml.safe_load(Path(config_path).read_text())
    enforce_storage_caps=bool(config.get('enforce_temporary_storage_caps',True))
    os.environ['TINYCHESS_ENFORCE_TEMP_STORAGE_CAPS']='1' if enforce_storage_caps else '0'
    hf=yaml.safe_load(Path(config['destination']['config']).read_text())
    project=Path(project).resolve()
    reports=project/'reports'
    started=time.monotonic()
    state=dict(status='RUNNING',started_utc=now(),dataset_d_status='not_checked_deferred',
               full_target_games=50000,phase='starting',cleanup_completed=False,
               external_power_required=bool(config.get('require_external_power',True)))
    state['temporary_storage_caps_enforced']=enforce_storage_caps
    stop=threading.Event()
    lock=threading.Lock()
    spool=Path(hf['spool'])
    run_id=hf['run_id']
    preflight=json.loads(Path(config['source_manifest']).read_text())
    sources={s['month']:s for s in preflight['sources']}
    last_catalog=None
    def terminate(signum,frame):
        raise ScanError(state.get('stop_reason','Worker termination requested; recovery data retained'))
    signal.signal(signal.SIGTERM,terminate)

    def collect():
        months={}
        journal_path=spool/'upload_journal.json'
        try:journal=json.loads(journal_path.read_text())['files'] if journal_path.exists() else {}
        except (ValueError,OSError):journal={}
        for month,spec in config['months'].items():
            entry=dict(split=spec['split'],target_games=spec['accepted_games'],validated_games=0,
                cloud_reproduced_games=0,positions=0,source_verified=False)
            approval=spool/f'{run_id}-{month}.approval.json'
            complete=spool/f'{run_id}-{month}.complete.json'
            if approval.exists():
                value=json.loads(approval.read_text())
                entry.update(validated_games=value['accepted_games'],source_verified=True,
                    source_scan_measurements=value.get('source_scan_measurements'),
                    header_rejections=value['source_header_rejections'],validation_rejections=value['validation_rejections'],
                    coverage=value['coverage'],strong_games=value['strong_games'],parity_states=value['parity_states'],
                    parity_mismatches=value['parity_mismatches'],source=value['source'])
            base=f'data/{run_id}/manifests/{month}'
            if complete.exists() and all(journal.get(base+s,{}).get('status') in ['verified','deleted'] for s in ['.json','.reproduction.json']):
                value=json.loads(complete.read_text())
                entry.update(cloud_reproduced_games=value['accepted_games'],positions=value['training_positions'],
                    shard_digest=value['shard_digest'],artifact_bytes=value['artifact_bytes'],
                    source=value['source'],manifest_receipt=journal[base+'.json'])
            scan_status=Path(config['cache'])/f'{month}.status.json'
            if scan_status.exists():entry['scanner_status']=json.loads(scan_status.read_text())
            measurements=spool/f'{run_id}-{month}.measurements.json'
            if measurements.exists():entry['construction_measurements']=json.loads(measurements.read_text())
            months[month]=entry
        with lock:result=dict(state)
        result.update(updated_utc=now(),elapsed_seconds=time.monotonic()-started,months=months,
            cloud_reproduced_games=sum(m['cloud_reproduced_games'] for m in months.values()),
            training_positions=sum(m['positions'] for m in months.values()),
            free_disk_bytes=shutil.disk_usage(project).free,
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
            cpu_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,
            cpu_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,
            repo_url='https://huggingface.co/datasets/'+hf['repo_id'],
            archive_payload_allowance_bytes=config['network_budget_bytes'],
            hub_payload_allowance_bytes=hf['transfer_budget_bytes'],max_runtime_hours=config['max_runtime_hours'],
            storage_budget_bytes=2684354560,free_disk_reserve_bytes=hf['free_disk_reserve_bytes'])
        return result

    def write_report(value):
        target=reports/'m1_50k_run_status.json'
        temp=target.with_suffix('.json.partial')
        temp.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');temp.replace(target)
        lines=['# Dataset A — real 50K M1 pilot', '',f"**Status: {value['status']}**. Updated {value['updated_utc']} (UTC).",'',
            'Dataset D is deferred: **overlap NOT checked; no evaluation-position exclusions applied**.',
            'No M2, model training, self-play, Git branch/commit/push/merge is performed.',
            '',f"Repository: {value['repo_url']}",
            f"Cloud-verified/reproduced games: **{value['cloud_reproduced_games']:,} / 50,000**; positions: **{value['training_positions']:,}**.",
            f"Elapsed: {value['elapsed_seconds']/3600:.3f} hours; peak process RSS: {value['peak_process_rss_bytes']:,} bytes.",
            '', '| Month | Split | Target | Fully validated | Cloud + second build | Position rows |',
            '|---|---|---:|---:|---:|---:|']
        for month,m in value['months'].items():
            lines.append(f"| {month} | {m['split']} | {m['target_games']} | {m['validated_games']} | {m['cloud_reproduced_games']} | {m['positions']} |")
        lines+=['', 'Source transfer is streamed without a whole compressed or decoded archive on disk.',
            'Month-wide deterministic day/rating sampling; source SHA256 and every accepted Pgx state/action/history checked.',
            'Policy action and side-to-move WDL targets retain timeout value masking. Synthetic smoke objects are separate.',
            'Checksums, rejection counts, per-day coverage, timing and immutable Hub receipts are in the companion status JSON and monthly Hub manifests.',
            '',f"Archive payload allowance: {value['archive_payload_allowance_bytes']:,} bytes (one pass; no automatic retry).",
            f"Hub logical payload allowance: {value['hub_payload_allowance_bytes']:,} bytes; HTTP/TLS overhead is not metered.",
            f"Runtime cap: {value['max_runtime_hours']} hours. Local conservative estimate 2.5 GiB, with 20 GiB kept free.",
            f"Temporary-output size caps enforced: {value.get('temporary_storage_caps_enforced',True)}. Free-disk and memory safeguards remain active.",
            'Parse/replay/parity/encode timing is measured as a combined phase, not falsely reported as pure kernel/parser speed.',
            'Source scanning overlaps network and header parsing; full source hashes are verified only at exact EOF.',
            '', 'Large-batch Blackwell benchmarks remain separate/pending and do not change this CPU dataset job.',
            'Private credentials are never written into reports or snapshots.']
        if value.get('phase'):lines+=['',f"Current phase: {value['phase']}"]
        if value.get('error'):lines+=['',f"Failure: `{value['error']['type']}` — {value['error']['message']}"]
        if value.get('final_revision'):lines+=['',f"Verified Hub catalog revision: `{value['final_revision']}`"]
        if 'verified_artifacts' in value:lines+=[f"Verified artifacts: {value['verified_artifacts']}; bytes: {value['verified_artifact_bytes']:,}."]
        if value.get('cleanup_completed'):lines+=['','Generated data/cache directories were deleted after remote verification. Small reports and recovery manifests remain in this project.']
        lines+=['','The final pilot is NOT accepted until status is COMPLETE; partial progress is not a 50K result.']
        temp=reports/'m1_50k_pilot.md.partial';temp.write_text('\n'.join(lines)+'\n');temp.replace(reports/'m1_50k_pilot.md')

    def monitor():
        while not stop.is_set():
            try:
                if state['external_power_required'] and state.get('power_confirmed'):
                    power=external_power_status()
                    with lock:state['power']=power
                    if not power['external_connected']:
                        with lock:state['stop_reason']='External power disconnected; job stopped to preserve recovery state'
                        os.kill(os.getpid(),signal.SIGTERM)
                        return
                write_report(collect())
            except Exception as exc:
                print('Progress report retry:',type(exc).__name__,flush=True)
            stop.wait(30)

    monitor_thread=threading.Thread(target=monitor,daemon=True)
    monitor_thread.start()
    try:
        power=startup_power_status(config)
        state['power']=power
        state['power_confirmed']=power['required']
        execute=runpy.run_path(str(SNAPSHOT/'scripts/scan_m1_archives.py'))['execute']
        catalog=execute(config)
        if config.get('session_months') and catalog and catalog['manifest']['status']!='complete-50k':
            finished=collect()
            for month in config['session_months']:
                if finished['months'][month]['cloud_reproduced_games']!=config['months'][month]['accepted_games']:
                    raise ScanError('Requested one-month session has incomplete remote receipts')
            stop.set();monitor_thread.join(timeout=35)
            finished.update(status='MONTH_COMPLETE',phase='One-month session finished; no next-month scan started',
                session_months=config['session_months'],finished_utc=now(),
                final_revision=catalog['revision'],cleanup_completed=False)
            write_report(finished)
            (reports/f"m1_session_{config['session_months'][0]}.json").write_text(json.dumps(finished,indent=2)+'\n')
            print('MONTH_COMPLETE',json.dumps(catalog),flush=True)
            return
        if not catalog or catalog['manifest']['status']!='complete-50k':
            raise ScanError('Final dataset catalog is incomplete')
        with lock:state['phase']='verifying final Hub inventory'
        up=VerifiedUploader(hf['repo_id'],spool,private=hf['private'],spool_cap_bytes=hf['spool_cap_bytes'],
            reserve_disk_bytes=hf['free_disk_reserve_bytes'],transfer_budget_bytes=hf['transfer_budget_bytes'])
        transport=up._transport()
        journal=json.loads(up.journal.read_text())
        receipts=list(journal['files'].values())
        revision=catalog['revision']
        for index in range(0,len(receipts),100):
            batch=receipts[index:index+100]
            entries=transport.api.get_paths_info(hf['repo_id'],[r['path_in_repo'] for r in batch],repo_type='dataset',revision=revision)
            by_path={item.path:item for item in entries}
            for receipt in batch:
                item=by_path.get(receipt['path_in_repo'])
                if item is None or item.size!=receipt['bytes']:
                    raise ScanError('Final immutable Hub snapshot is missing an expected artifact')
                if item.lfs:
                    if item.lfs.sha256!=receipt['sha256']:raise ScanError('Final Hub LFS checksum mismatch')
                elif transport.hash_remote(receipt['path_in_repo'],revision,receipt['bytes'])!=(receipt['sha256'],receipt['bytes']):
                    raise ScanError('Final remote payload mismatch')
        completed=collect()
        if completed['cloud_reproduced_games']!=50000:
            raise ScanError('Local completion receipts do not sum to 50K')
        with lock:
            state.update(phase='backing up recovery metadata before cleanup',final_revision=revision,
                verified_artifacts=len(receipts),verified_artifact_bytes=sum(r['bytes'] for r in receipts))
        recovery=spool/'recovery.json'
        atomic_json(recovery,dict(catalog=catalog,status=completed,upload_receipts=journal),up.guard)
        recovery_receipt=up.publish(recovery,f'recovery/{run_id}/recovery.json')
        source_zip=spool/'generation-source.zip'
        up.guard.check(8*2**20)
        with zipfile.ZipFile(source_zip,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
            for folder in ['src/tinychess','scripts','tests','configs','licenses','benchmarks']:
                for path in sorted((SNAPSHOT/folder).rglob('*')):
                    if path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts:
                        bundle.write(path,path.relative_to(SNAPSHOT))
            for name in ['README.md','pyproject.toml','uv.lock','requirements-lock.txt','requirements-cuda-lock.txt']:
                bundle.write(SNAPSHOT/name,name)
        source_receipt=up.publish(source_zip,f'recovery/{run_id}/generation-source.zip')
        final=collect()
        final.update(final_revision=revision,verified_artifacts=len(receipts),
            verified_artifact_bytes=sum(r['bytes'] for r in receipts),
            recovery_receipt=recovery_receipt,source_snapshot_receipt=source_receipt)
        (reports/'m1_50k_recovery.json').write_text(json.dumps(final,sort_keys=True,indent=2)+'\n')
        # Stop reporting before deleting data roots. Final report retains the
        # captured counts; no background reader races with directory removal.
        stop.set();monitor_thread.join(timeout=35)
        allow_raw={'m1_prefixes','m1_month_candidates'}
        allow_processed={'m1_smoke100','m1_smoke1000_run1','m1_smoke1000_run2','m1_final_run1','m1_final_run2',
            'm1_hf_live_smoke','m1_hf_spool','m1_hf_audit'}
        removed=[]
        for parent,names in [(project/'data/raw',allow_raw),(project/'data/processed',allow_processed)]:
            for name in sorted(names):
                path=parent/name
                if not path.exists():continue
                if path.is_symlink() or not path.resolve().is_relative_to(parent.resolve()):
                    raise ScanError('Cleanup refuses a path outside managed generated data')
                size=sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
                shutil.rmtree(path)
                removed.append(dict(path=str(path.relative_to(project)),bytes=size))
        marker=SNAPSHOT/'.dataset-a-runtime.json'
        if (SNAPSHOT!=project and SNAPSHOT.is_relative_to(project/'data/processed') and marker.is_file()
                and json.loads(marker.read_text()).get('project')==str(project)):
            os.chdir(project)
            shutil.rmtree(SNAPSHOT)
            removed.append(dict(path=str(SNAPSHOT.relative_to(project)),kind='verified backed-up source runtime'))
        final.update(status='COMPLETE',phase='finished; generated dataset caches removed',
            cleanup_completed=True,removed_directories=removed,finished_utc=now(),
            elapsed_seconds=time.monotonic()-started,free_disk_bytes=shutil.disk_usage(project).free)
        write_report(final)
        print('COMPLETE',json.dumps({k:final[k] for k in ['cloud_reproduced_games','training_positions','final_revision','cleanup_completed']}),flush=True)
    except Exception as exc:
        stop.set();monitor_thread.join(timeout=35)
        with lock:
            state.update(status='STOPPED',phase='stopped safely; local recovery data retained',error=dict(type=type(exc).__name__,message=str(exc)),finished_utc=now())
        write_report(collect())
        print('STOPPED:',type(exc).__name__,str(exc),flush=True)
        raise
    finally:
        stop.set()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--project',required=True)
    args=parser.parse_args()
    main(args.config,args.project)
