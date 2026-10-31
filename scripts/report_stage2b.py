"""Render actual Stage 2B evidence, including setup/worker failure, without JAX."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tinychess.validation.stage2b_plan import EXPECTED_COUNTS,planned_cases


def assess(report):
    if report.get('status')!='COMPLETE' or not report.get('protected_sources_unchanged'):
        return 'FAIL',{},False
    if report.get('regression_returncode')!=0 or report.get('regression_passed',0)<=0 or report.get('host',{}).get('status')!='PASS':
        return 'FAIL',{},False
    rows=report.get('cases',[])
    keys=[(r.get('profile'),r.get('compute'),r.get('batch')) for r in rows]
    expected=[(r['profile'],r['compute'],r['batch']) for r in planned_cases()]
    if keys!=expected:return 'FAIL',{},False
    successful={}
    for r in rows:
        if r.get('status') not in ('PASS','SKIPPED'):return 'FAIL',{},False
        if r['status']=='PASS':
            if not(r.get('gpu_measured') and r.get('parameter_count')==EXPECTED_COUNTS[r['profile']]
                   and r.get('parameter_updates')==1 and r.get('master_params_fp32')
                   and r.get('optimizer_fp32') and r.get('gradients_finite')
                   and r.get('gradient_squared_norm',0)>0
                   and r.get('parameter_delta_squared',0)>0):return 'FAIL',{},False
            successful[(r['profile'],r['compute'],r['batch'])]=r
    required=[(p,'bfloat16',4) for p in EXPECTED_COUNTS]
    if not all(k in successful for k in required):return 'FAIL',{},False
    rec={p:max(r['batch'] for k,r in successful.items() if k[0]==p and k[1]=='bfloat16') for p in EXPECTED_COUNTS}
    complete=all((p,'float32',4) in successful for p in EXPECTED_COUNTS)
    memory=all(isinstance(r.get('jax_peak_bytes_in_use'),int) and r['jax_peak_bytes_in_use']>0 for r in successful.values())
    return ('PASS' if complete and memory else 'CONDITIONAL PASS'),(rec if memory else {}),complete and memory


def render(root=ROOT):
    root=Path(root);report=json.loads((root/'reports/stage2b_blackwell/run.json').read_text())
    gate,recommended,ready=assess(report)
    report['gate']=gate;report['recommended_bf16_microbatch']=recommended
    report['stage3a_technical_readiness']=ready
    (root/'reports/stage2b_blackwell/run.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    lines=['# M2 Stage 2B — returned Blackwell model validation','',f'**Recommendation: {gate}.**',
      'Stage 3 was not executed. Readiness below is a technical result, not authorization.','',
      f"Source base: `{report.get('source_base_commit')}`. Input ZIP SHA256: `{report.get('input_zip_sha256')}`.",
      f"Exact source manifest SHA256: `{report.get('source_manifest_sha256')}`.",
      'Dataset A remains pinned; no dataset, loader benchmark, model redesign, MCTS or self-play run.',
      '',f"Started UTC: {report.get('started_utc')}; completed UTC: {report.get('completed_utc')}."]
    host=report.get('host',{})
    lines += ['', '## Hardware and pinned environment','',
      f"GPU: {host.get('gpu_kind','not established')}. Packages/runtime/driver/VRAM/host metadata: `stage2b_blackwell/host.json`.",
      f"Host available RAM: {host.get('ram_available_bytes')}; logical CPUs: {host.get('logical_cpus')}; disk preflight: {report.get('disk_preflight')}.",
      f"Physical GPUs/VRAM/driver: `{json.dumps(host.get('physical_gpus',[]),sort_keys=True)}`.",
      f"Backend: `{host.get('backend_platform_version')}`; Python: `{host.get('python')}`.",
      f"Installed versions: `{json.dumps(host.get('packages',{}),sort_keys=True)}`.",
      'nvidia-smi driver capability, system nvcc and pip CUDA runtime are recorded separately, not conflated.',
      '', '## Actual isolated cases','',
      '| Model | Compute | B | Status | Parameters | Updates | JAX peak bytes | Sampled physical used bytes |',
      '|---|---|---:|---|---:|---:|---:|---:|']
    for r in report.get('cases',[]):
        lines.append(f"| {r.get('profile')} | {r.get('compute')} | {r.get('batch')} | {r.get('status')} | {r.get('parameter_count','uninitialized')} | {r.get('parameter_updates',0)} | {r.get('jax_peak_bytes_in_use')} | {r.get('physical_sampled_peak_used_bytes')} |")
        if r.get('reason'):lines += ['',f"Case limitation: {r['reason']}",'']
    lines += ['', '## Phase timings and numerical checks','',
      '| Model / compute / B | Phase | Trace s | Compile s | Synchronized execution s |',
      '|---|---|---:|---:|---:|']
    for r in report.get('cases',[]):
        for phase,t in r.get('phases',{}).items():
            lines.append(f"| {r.get('profile')} / {r.get('compute')} / {r.get('batch')} | {phase} | {t['trace_seconds']:.6f} | {t['compile_seconds']:.6f} | {t['synchronized_execution_seconds']:.6f} |")
    lines += ['', '| Model / compute / B | Loss before / after | Gradient norm squared | Parameter delta squared | Host RSS bytes |',
              '|---|---|---:|---:|---:|']
    for r in report.get('cases',[]):
        lines.append(f"| {r.get('profile')} / {r.get('compute')} / {r.get('batch')} | {r.get('loss_before')} / {r.get('loss_after')} | {r.get('gradient_squared_norm')} | {r.get('parameter_delta_squared')} | {r.get('peak_host_rss_bytes')} |")
    lines += ['', 'Each successful case uses a fresh initialization and exactly one AdamW update;',
      'no worker trains a trajectory beyond one step. Up to six independent updates total.',
      'Synthetic labels are applied to four scripted canonical Pgx positions; no human-data acquisition.',
      'AdamW smoke constants: LR=1e-4, beta=(0.9,0.95), weight_decay=0.01; these are not tuned pilot settings.',
      'Every case JSON records before/after losses, parameter delta, fp32 parameter/gradient/optimizer checks,',
      'separate tracing/compilation and synchronized execution times for init, forward, backward, update and post-forward.',
      'JAX peak covers its allocator in that fresh worker. nvidia-smi is sampled every ~0.2s and may miss short peaks;',
      'sampled physical use includes other processes (maximum across physical devices), not just this model.',
      'Host RSS is recorded per case. Estimates are preflight safeguards, not measured-memory replacements.',
      '', '## Tests and recommendation','',
      f"Relevant CUDA regression: return code {report.get('regression_returncode')}, passed {report.get('regression_passed')}.",
      f"Safe measured starting bf16 microbatches: `{recommended}`. No effective-batch or accumulation claim.",
      f"Stage 3A technical readiness established: **{ready}**; owner review/authorization still required.",
      'If any essential smoke/test/source-integrity check failed, do not start overfit training.',
      '', '## Reproduction','',
      'The exact commands are in `stage2b_blackwell/commands.json`; logs and per-case JSON are in the same directory.',
      'Reproduce by uploading the matching input ZIP and running its generated Molab cell in a fresh GPU notebook.',
      'The runner uses uv 0.10.9, Python 3.11.15, unchanged locked test/data/CUDA extras.',
      'Missing dependencies/interpreter may require network; no Lichess/HF dataset transfer is allowed.',
      'The results ZIP includes checksummed evidence and BLACKWELL_BUNDLE.json, including failure diagnostics.']
    if report.get('failure'):lines += ['',f"Failure: {report['failure']}"]
    (root/'reports/m2_stage2b.md').write_text('\n'.join(lines)+'\n')
    return gate


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    print(render(p.parse_args().root))
