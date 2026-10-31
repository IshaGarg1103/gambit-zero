"""Render only measured Stage 3A evidence. Plotting is an optional offline step."""
import argparse
import json
from pathlib import Path


def render(results,output):
    data=json.loads(Path(results).read_text());r=data['training'];m=r['sample_manifest'];v=data['verification']
    initial=r['initial'];final=r['final'];memory=r['jax_memory_stats']
    verdict=r['status']
    lines=['# M2 Stage 3A — 1K-position overfit','',f'**Recommendation: {verdict}.**',
        'This is a bounded memorization diagnostic, not a supervised pilot or chess-strength result.','',
        '## Source, scope and environment','',
        f"Branch: `m2/stage-3a-overfit`; merged Stage 2A/2B base: `{r['source_base_commit']}`.",
        f"Exact source overlay manifest SHA256: `{r['config']['source_manifest_sha256']}`.",
        'The owner merged Stage 2A/2B before implementation. Main, origin/main and the Stage 3A branch',
        'were checked via local refs and agreed on that base. No Git command was executed by the agent.',
        f"GPU: {r['gpu']}. Versions: `{json.dumps(r['versions'],sort_keys=True)}`.",
        'Hardware: 97,887 MiB physical VRAM, driver 595.71.05; host-visible 160 GiB RAM and 20 logical CPUs.',
        'GPU/driver/CUDA and host details are retained in the compact JSON results and original evidence ZIP.',
        '', '## Frozen real-data sample','',
        f"Dataset A revision: `{m['dataset_revision']}`; training split only; seed {m['seed']}.",
        f"{m['positions']} eligible real positions from **{m['games']} different games**, one position per game.",
        f"Months: `{json.dumps(m['months'],sort_keys=True)}`.",
        f"Original WDL class counts: `{json.dumps(m['wdl_classes'],sort_keys=True)}` (win=0/draw=1/loss=2).",
        f"Zero-value-weight time-forfeit rows: **{m['value_masked_rows']}**; these still contribute to policy loss.",
        f"Sampling: {m['algorithm']}.",
        f"Scanned/validated training rows per sample build: {m['coverage']['scanned_positions']:,}; two builds.",
        f"Duplicate encoded inputs excluded from this derived sample: {m['coverage']['duplicate_inputs_excluded']};",
        f"of those, {m['coverage']['conflicting_duplicate_inputs_excluded']} had conflicting policy/weighted-WDL targets.",
        'Both minimum-position ranking and candidate-game ranking are seed/SHA256 based, independent of traversal order.',
        'This diversity filter only makes the tiny memorization task unambiguous; it does not modify Dataset A',
        'or prescribe deduplication for a future full-data pilot. Original feature bytes, history, masks and labels are preserved.',
        f"Logical sample SHA256: `{m['logical_sha256']}`.",
        f"NPZ SHA256: `{m['sample_sha256']}`; {m['sample_bytes']:,} bytes compressed.",
        'Two independent full-training-source sample builds produced identical manifests and compressed sample bytes.',
        'All original shard sizes/SHA256, schemas, actions, WDL perspective, weights and source counts were validated.',
        'No PGNs were replayed again; this reuses the previously accepted M1 parity evidence.',
        'Zero dataset bytes downloaded for sample selection. Validation/test and Dataset D were not used for optimization.',
        '', '## Model, loss and optimizer','',
        f"25M only: **{r['model_parameters']:,} parameters**, accepted architecture unchanged; bf16 compute, fp32 masters/moments.",
        'Explicit run initialization: random backbone, zero final policy_head and value_head kernels/biases.',
        'This preserves the accepted architecture while making initial probabilities uniform for an exact loss sanity check.',
        'It is an initialization choice for this diagnostic, not a model-source or encoder modification.',
        'Masked policy CE + separately value-weighted WDL CE (coefficient 1). Illegal actions stay masked.',
        'AdamW beta=(0.9,0.95), epsilon=1e-8, zero weight decay; global gradient clip 1.0.',
        'Microbatch 32; 100-step linear warmup from 1e-5 to constant LR 1e-3; no LR sweep or fallback run.',
        '0% history dropout. Every 32 updates is an exact-once permutation of the same 1,024 positions.',
        'Compiled updates reject nonfinite gradients/state. Full-sample metrics are weighted aggregates over 32 minibatches.',
        '', '## Initial loss sanity','',
        f"Policy loss **{initial['policy_loss']:.8f}**, versus mean ln(legal-count) **{initial['uniform_policy_baseline']:.8f}**.",
        f"WDL loss **{initial['wdl_loss']:.8f}**, versus ln(3) **{initial['uniform_wdl_baseline']:.8f}**.",
        'Both agreed within 1e-4 before training. Weighted WDL accuracy excludes zero-weight rows.',
        '', '## Learning result','',
        '| Metric | Initial | Final |','|---|---:|---:|',
        f"| Masked policy CE | {initial['policy_loss']:.6f} | {final['policy_loss']:.6f} |",
        f"| Policy top-1 accuracy | {initial['policy_accuracy']:.4%} | {final['policy_accuracy']:.4%} |",
        f"| Weighted WDL CE | {initial['wdl_loss']:.6f} | {final['wdl_loss']:.6f} |",
        f"| Weighted WDL accuracy | {initial['wdl_accuracy']:.4%} | {final['wdl_accuracy']:.4%} |",
        '', f"Stopped after **{r['steps']:,} updates**; no later training started.",
        'Predeclared code gate: policy accuracy >=99%, policy CE <=0.05, WDL accuracy >=95%,',
        'both component losses below initial, finite values, within 3,000 updates.',
        'The WDL 95% and policy CE 0.05 thresholds explicitly quantify “rise sharply” and “near zero”.',
        '', '![Measured full-sample learning curves](m2_stage3a_learning_curve.svg)','',
        '| Step | Policy CE | Policy accuracy | WDL CE | WDL accuracy |',
        '|---:|---:|---:|---:|---:|']
    for row in r['learning_curve']:
        if row['step']%500==0 or row['step']==r['steps']:
            lines.append(f"| {row['step']} | {row['policy_loss']:.5f} | {row['policy_accuracy']:.2%} | {row['wdl_loss']:.5f} | {row['wdl_accuracy']:.2%} |")
    lines += ['', '## Checkpoint and tracking evidence','',
        f"Exact checkpoint save/restore checks: `{json.dumps(r['checkpoint_checks'],sort_keys=True)}`.",
        f"Restored optimizer/step state used for continued training: **{r['checkpoint_resumed_and_continued']}**.",
        'The CPU test independently verifies the next update is identical after restore, and corruption/metadata mismatch reject.',
        f"W&B experiment: [{r['wandb_url']}]({r['wandb_url']}).",
        'Logged minibatch metrics every 10 updates; complete 1K-sample evaluation every 100 updates, including step zero.',
        'LR, gradient norm, policy/WDL losses and accuracies, run configuration and system metrics were recorded.',
        'API keys are read only from environment by the reusable tracker; no secret values or environment dumps are recorded.',
        'Tracker supports explicit online/offline/disabled modes; offline behavior is covered by the focused test.',
        'The final model+optimizer checkpoint was logged as an explicit checkpoint artifact; remote digests match.',
        'With explicit owner approval, the exact source/sample/results archives were also saved as',
        '`ishagarg-research/gambit-zero/stage3a-25m-provenance:v0`; all four remote file digests match.',
        'Artifact names, immutable version digests and per-file hashes are recorded in the results JSON.',
        'A separate CPU-only SDK test verified offline logging without an API key; it performed no model updates.',
        '', '## Measured resources and timing','',
        f"Training loop wall time (includes evaluations/checkpoints): **{r['training_wall_seconds']:.3f}s**.",
        f"Synchronized compiled-update execution sum: **{r['synchronized_training_execution_seconds']:.3f}s**.",
        f"Initialization including compilation/execution: **{r['phases']['initialize_compile_and_execute_seconds']:.3f}s**;",
        f"training-step tracing + compilation: **{r['phases']['training_compile_seconds']:.3f}s**.",
        f"Total worker wall time including setup inside worker/tracking/upload: **{r['total_wall_seconds']:.3f}s**.",
        f"JAX peak bytes in use: **{memory['peak_bytes_in_use']:,}** ({memory['peak_bytes_in_use']/2**30:.3f} GiB).",
        f"Sampled physical device-wide peak: **{r['sampled_device_peak_bytes']:,}** ({r['sampled_device_peak_bytes']/2**30:.3f} GiB).",
        f"Peak host worker RSS: **{r['peak_host_rss_bytes']:,}** ({r['peak_host_rss_bytes']/2**30:.3f} GiB).",
        'Physical memory was sampled approximately every 0.2s and includes other processes; brief peaks may be missed.',
        'These are tiny-set training measurements, not full-loader/pilot/self-play throughput.',
        'Molab reported a near-2^63 free-disk sentinel; actual usable disk capacity is not established by that value.',
        'The isolated job completed its bounded I/O. The Mac retains only small derived samples/evidence, not trained weights.',
        '', '## Verification, tests and remaining blocker','',
        f"Complete CPU regression: **{v['cpu_tests']} passed in {v['cpu_seconds']:.2f}s**.",
        f"Focused CPU tests: **{v['focused_tests']} passed locally**; **{v['remote_focused_tests']} passed in Molab's pinned environment**.",
        'Tests cover uniform initial losses, finite updates, exact zero-weight WDL masking, deterministic epoch accounting,',
        'checkpoint integrity/continuation, explicit offline tracking, artifact file selection and sample hashing/deduplication.',
        f"Results ZIP SHA256: `{v['results_zip_sha256']}`; all indexed members verified.",
        'Source bundle/retained source hashes were checked; original model/rules/encoder/codec/loader/core locks remain unchanged.',
        '', '**Blocking result:** the model did not memorize the tiny sample within the agreed bound.',
        'Loss reduction alone does not satisfy the gate. No numerical failure or checkpoint discrepancy was observed,',
        'but the learning/implementation root cause is unresolved. Exit code 1 and W&B failed status',
        'deliberately reflect the unmet learning gate; the model worker reported no runtime exception.',
        'Treat this as a blocking diagnostic, not evidence',
        'for proceeding to a pilot or selecting a different learning rate. No second run or sweep was launched.',
        'Next action requires focused review/debugging of this failure before Stage 3B. No chess-strength claim is made.',
        '', '## Reproduction and manual Git workflow','',
        'See [exact commands and recovery limits](../docs/m2_stage3a_run.md).',
        'Stage 2A/2B is already merged; no repeat merge is needed. Stage 3A changes remain uncommitted for owner review.',
        'No history dropout, openings_v2, MCTS, self-play, full pilot or dataset expansion was implemented or executed.']
    Path(output).write_text('\n'.join(lines)+'\n')


def plot(results,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=json.loads(Path(results).read_text())['training'];rows=r['learning_curve'];steps=[x['step'] for x in rows]
    plt.rcParams['svg.hashsalt']='gambit-zero-stage3a-v1'
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for k,label in [('policy_loss','Policy CE'),('wdl_loss','Weighted WDL CE')]:
        axes[0].plot(steps,[x[k] for x in rows],label=label)
    for k,label in [('policy_accuracy','Policy top-1'),('wdl_accuracy','Weighted WDL')]:
        axes[1].plot(steps,[100*x[k] for x in rows],label=label)
    axes[1].axhline(99,color='gray',ls='--',lw=1,label='99% policy gate')
    axes[0].set_ylabel('Cross-entropy');axes[1].set_ylabel('Accuracy (%)');axes[1].set_ylim(0,102)
    for ax in axes:ax.set_xlabel('Optimizer updates');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Stage 3A: 25M / 1,024 real training positions — '+r['status'])
    fig.savefig(output,metadata={'Date':None});plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--output',required=True);p.add_argument('--plot')
    a=p.parse_args();render(a.results,a.output)
    if a.plot:plot(a.results,a.plot)
