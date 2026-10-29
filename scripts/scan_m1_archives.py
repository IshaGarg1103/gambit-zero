"""Default: local storage/resource plan only. Explicit gated execution is separate.

A full run creates VERIFIED CANDIDATE manifests, not an accepted 50K dataset.
Candidate validation/exclusion/materialization remains a subsequent M1 gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import psutil
import yaml

from tinychess.data.archive_scan import ALGORITHM, DiskGuard, ScanError, atomic_json, scan_month


def session_sources(config,sources):
    requested=config.get('session_months')
    if requested is None:
        return sorted(sources,key=lambda s:s['month'])
    if not isinstance(requested,list) or len(requested)!=1 or requested[0] not in config['months']:
        raise ScanError('One-month sessions require exactly one configured source month')
    return [source for source in sources if source['month']==requested[0]]


def plan(config):
    sources = json.loads(Path(config['source_manifest']).read_text())['sources']
    if {s['month'] for s in sources} != set(config['months']):
        raise ScanError('Frozen source months do not match configuration')
    if config['sampling']['algorithm'] != ALGORITHM or config['sampling']['reserve_factor'] != 3:
        raise ScanError('Unsupported sampling algorithm/reserve configuration')
    if config['sampling']['min_strong_fraction'] != 0.6:
        raise ScanError('Implementation pins at least 60% strong per day')
    targets = config['months']
    if sum(s['accepted_games'] for s in targets.values()) != 50000:
        raise ScanError('Expected exactly 50K target games')
    for month,spec in targets.items():
        expected_split = 'train' if month <= '2026-07' else 'validation' if month == '2026-08' else 'test'
        expected_games = 5000 if expected_split == 'train' else 7500
        if spec != dict(split=expected_split,accepted_games=expected_games):
            raise ScanError('Month split/quota differs from the approved plan')
    s = config['storage']
    if s['max_compressed_frame_bytes'] != 8*2**20 or s['max_decoded_frame_bytes'] != 64*2**20:
        raise ScanError('Frame bounds must match the tested decoder configuration')
    hf_mode=config.get('destination',{}).get('mode')=='huggingface'
    if hf_mode:
        hf=yaml.safe_load(Path(config['destination']['config']).read_text())
        peak=(s['candidate_cache_cap_bytes']+hf['spool_cap_bytes']+hf['audit_cap_bytes']
              +s['other_temporary_cap_bytes']+hf['metadata_allowance_bytes'])
    else:
        peak = (s['candidate_cache_cap_bytes'] + s['final_build_count']*s['per_final_build_cap_bytes']
                + s['other_temporary_cap_bytes'] + s['metadata_cap_bytes'])
    available = shutil.disk_usage('.').free
    memory = psutil.virtual_memory().available
    return dict(mode='PLAN_ONLY_NO_NETWORK',total_archive_payload_bytes=sum(v['archive_size_bytes'] for v in sources),
                session_months=config.get('session_months'),
                session_archive_payload_bytes=sum(v['archive_size_bytes'] for v in session_sources(config,sources)),
                free_disk_bytes=available,available_memory_bytes=memory,
                peak_additional_disk_budget_bytes=peak,free_disk_reserve_bytes=s['free_disk_reserve_bytes'],
                minimum_free_disk_required_bytes=peak+s['free_disk_reserve_bytes'],
                storage_gate_passed=available >= peak+s['free_disk_reserve_bytes'],
                memory_planning_gate_passed=memory >= 2*2**30,
                minimum_available_memory_for_planning_bytes=2*2**30,
                full_archive_cache_bytes=0,decompressed_archive_disk_bytes=0,
                network_budget_allocated=config['network_budget_bytes'] is not None,
                runtime_budget_allocated=config['max_runtime_hours'] is not None,
                persistent_host_confirmed=False,
                dataset_d_frozen=config['frozen_dataset_d_manifest'] is not None,
                execution_enabled=config['execution_enabled'],
                destination='huggingface' if hf_mode else 'local',
                complete_local_dataset_required=not hf_mode,
                note='Conservative component caps, not measured 50K output. HF mode uploads/verifies/removes each shard and reproduces from cloud selection without a second local dataset.')


def execute(config):
    status = plan(config)
    if not (config['execution_enabled'] and config['resource_budget_verified']
            and status['storage_gate_passed'] and status['memory_planning_gate_passed']):
        raise ScanError('Full scan disabled or resource preflight not approved')
    budget,hours = config['network_budget_bytes'],config['max_runtime_hours']
    if budget is None or hours is None or hours <= 0 or budget < status['total_archive_payload_bytes']:
        raise ScanError('Explicit adequate transfer/runtime allowances required')
    frozen = config['frozen_dataset_d_manifest']
    if config.get('dataset_d_policy')!='deferred' and (not frozen or hashlib.sha256(Path(frozen).read_bytes()).hexdigest() != config['frozen_dataset_d_sha256']):
        raise ScanError('Dataset D must be frozen/checksummed before extraction')
    sources = json.loads(Path(config['source_manifest']).read_text())['sources']
    root = Path(config['cache'])
    guard = DiskGuard(root,config['storage']['candidate_cache_cap_bytes'],config['storage']['free_disk_reserve_bytes'])
    ledger_path = root/'network_budget_ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(reserved_attempt_payload_bytes=0,attempts=[])
    deadline = time.monotonic()+hours*3600
    destination=None
    if config.get('destination',{}).get('mode')=='huggingface':
        from tinychess.data.hf_destination import HfMonthlyDestination
        destination=HfMonthlyDestination(config,deadline=deadline)
    try:
        for source in session_sources(config,sources):
            month = source['month']
            checkpoint = root/f'{month}.selection.json'
            if destination and destination.resume_completed(source):
                destination.retire_completed_checkpoint(checkpoint,month)
                continue
            if not checkpoint.exists():
                # Charge a whole-month allowance BEFORE opening the socket. Crashes
                # cannot erase network spending; failed attempts are conservative.
                charged = ledger['reserved_attempt_payload_bytes'] + source['archive_size_bytes']
                if charged > budget:
                    raise ScanError('Cumulative transfer allowance exhausted; no automatic retry')
                ledger['reserved_attempt_payload_bytes'] = charged
                ledger['attempts'].append(dict(month=month,reserved_payload_bytes=source['archive_size_bytes']))
                atomic_json(ledger_path,ledger,guard)
            remaining_seconds = deadline-time.monotonic()
            if remaining_seconds <= 0:
                raise ScanError('Invocation runtime allowance exhausted')
            value = scan_month(source,root,config['months'][month]['accepted_games'],config['seed'],
                        network_budget_bytes=source['archive_size_bytes'],max_seconds=remaining_seconds,
                        max_cache_bytes=config['storage']['candidate_cache_cap_bytes'],
                        reserve_disk_bytes=config['storage']['free_disk_reserve_bytes'],
                        candidate_memory_bytes=config['storage']['candidate_payload_memory_cap_bytes'])
            print(json.dumps(dict(month=month,status='VERIFIED_CANDIDATES',selection_digest=value['selection_digest'])),flush=True)
            if destination:
                destination.publish_checkpoint(checkpoint)
        if destination:
            if config.get('session_months'):
                return destination.finish_session()
            return destination.finalize()
    finally:
        if destination:
            destination.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/data_human_50k_stream.yaml')
    parser.add_argument('--output',default='reports/m1_streaming_storage_plan.json')
    parser.add_argument('--execute-full-scan',action='store_true')
    args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text())
    if args.execute_full_scan:
        execute(config)
    else:
        result=plan(config)
        Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
