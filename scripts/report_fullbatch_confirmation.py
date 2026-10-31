"""Render the one-run confirmation, all missed identities and measured subgroups."""
import argparse
import json
from pathlib import Path


def render(path,output):
    data=json.loads(Path(path).read_text());r=data['run'];s=r['stratification'];v=data['verification']
    lines=['# Stage 3A — single full-batch confirmation — 31 October 2026','',
        f"**Result: {r['status']}.** Policy gate: accuracy >=99% and masked policy CE <=0.05, checked after every update.",
        'This report supplements the unchanged original failed run and failure diagnosis; it does not overwrite their evidence.',
        'No Stage 3B, full-data pilot, model change or parameter sweep was started.','',
        '## Configuration and execution','',
        'Same frozen 1,024 eligible training positions, same 25,622,092-parameter model, same seed 20261031,',
        'same random backbone / zero output heads. bf16 compute with fp32 masters and optimizer moments.',
        'Constant peak LR 3e-4, 100-update warmup, AdamW beta=(0.9,0.95), epsilon=1e-8, zero weight decay, clip 1.0.',
        '0% history dropout; all 1,024 rows contribute to every update. No minibatch sampling or shuffle noise.',
        f"Execution mode: **{r['execution_mode']}**. Effective batch 1,024; maximum 2,000 updates; exactly one run launched.",
        'The direct graph was compiled and checked against a conservative GPU memory budget before any update.',
        'Direct backward temporaries were 14,409,274,216 bytes; doubled buffer safety exceeded the 24 GiB guard.',
        'No direct optimizer update was executed. The selected 32x32 accumulation sums global-normalized policy/WDL gradients and clips once after summation;',
        'unit tests cover unequal value counts, all-zero value microbatches and unequal policy weights.',
        'Policy denominator is 1,024; WDL denominator is 852. The 172 forfeits contribute zero WDL gradient.',
        'The current instruction defines acceptance using policy only; WDL metrics are reported separately.',
        '', 'Frozen sample SHA256: `cc00bda4d5feb8a980be6e32208712f50eb94bb197ae60d01aeaff6e47ced590`.',
        'Logical SHA256: `f2dca7915a44569801b824b2815c04be04703a69827f46b9139f3b7cce1bb498`.',
        f"Initial parameter SHA256: `{r['initial_parameter_sha256']}`.",
        'Dataset A remains pinned to `b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c`; no source datasets, encoder, codec or chess rules changed.',
        '', '## Actual result','', '| Metric | Initial | Final |','|---|---:|---:|']
    for key,label in [('policy_accuracy','Policy top-1 accuracy'),('policy_loss','Masked policy CE'),('wdl_accuracy','Weighted WDL accuracy'),('wdl_loss','Weighted WDL CE')]:
        fmt=lambda x:f'{x:.4%}' if 'accuracy' in key else f'{x:.6f}'
        lines.append(f"| {label} | {fmt(r['initial'][key])} | {fmt(r['final'][key])} |")
    lines += ['', f"Stopped at **{r['steps']} updates**; **{r['examples_seen']:,} position presentations**, or {r['steps']} full passes.",
        f"Clip-active updates: **{r['clip_active_count']}/{r['steps']} ({r['clip_active_fraction']:.2%})**.",
        'The fraction uses all updates, not sampled logging. Norms before/after clipping are retained in raw evidence.',
        'The earlier minibatch runs saw 96,000 presentations in 3,000 updates (93.75 passes). This run changes both',
        'gradient noise and presentations per update; it does not isolate their individual causal contributions.',
        '', f"W&B: [{r['wandb_url']}]({r['wandb_url']}). Every update logs losses, accuracy, LR and pre/post clipping norms.",
        '', '![Measured full-batch learning](m2_stage3a_fullbatch_confirmation_curve.svg)','',
        '## Final policy error breakdown','',
        'Ply is the original zero-based pre-move ply. Bounds are inclusive. Legal counts come from the original Pgx legal mask.',
        'Board hash is the existing Dataset A SHA256 of pieces, side to move, castling and legal en-passant FEN fields;',
        'clocks and board history are excluded. Full encoded inputs are distinct even when current board hashes match.',
        'These are descriptive diagnostics, not a new split, filter or causal test.', '']
    for title,key in [('Ply buckets','ply_buckets'),('Number of legal moves','legal_move_buckets'),('Shared board hash','board_hash_groups')]:
        lines += [f'### {title}','','| Group | Rows | Correct | Missed | Accuracy | Error rate |','|---|---:|---:|---:|---:|---:|']
        for name,g in s[key].items():
            acc='n/a' if g['accuracy'] is None else f"{g['accuracy']:.2%}"
            err='n/a' if g['accuracy'] is None else f"{1-g['accuracy']:.2%}"
            lines.append(f"| {name} | {g['rows']} | {g['correct']} | {g['missed']} | {acc} | {err} |")
        lines.append('')
    lines += ['### Concentration assessment','',data['concentration_assessment'],'',
        'Only two rows share a board hash (one group), so that stratum cannot establish a general near-duplicate effect.',
        'No approximate board-similarity metric was added; current-board hash equality is narrower than “near duplicate”.',
        'Evenly spread errors would be consistent with budget/noise; early-ply concentration would warrant position-discrimination',
        'review. Neither pattern alone proves its mechanism, especially if very few misses remain.',
        '', '### All missed rows','',
        '| game_id | Ply | Legal moves | Shares board hash |','|---|---:|---:|---|']
    if not s['missed_rows']:lines.append('| None | — | — | — |')
    for row in s['missed_rows']:
        lines.append(f"| {row['game_id']} | {row['ply']} | {row['legal_moves']} | {row['shared_board_hash']} |")
    lines += ['', 'All missed identities are listed above without truncation; indices, board hashes and target/predicted actions are also in the JSON.',
        '', '## Earlier D comparison and limitations','',
        'The earlier D run (microbatch 32, LR 3e-4) ended at 77.34% / policy CE 0.7080.',
        'Its checkpoint was not retained locally or in a W&B checkpoint artifact; only metrics/history remain.',
        'Therefore the D per-row breakdown is **unavailable**. It was not reconstructed by rerunning D, and the original',
        'LR-1e-3 checkpoint was not silently substituted. Current per-row predictions and checkpoint are retained for future read-only review.',
        '', '## Runtime, memory and verification','',
        f"Training wall time including evaluation: **{r['training_wall_seconds']:.3f}s**; synchronized update execution: **{r['synchronized_update_seconds']:.3f}s**.",
        f"Tracing/compilation phases: `{json.dumps(r['phases'],sort_keys=True)}`.",
        f"Total worker wall time including initialization, compile, checkpoint and tracking completion: **{r['total_wall_seconds']:.3f}s**.",
        f"JAX allocator peak: **{r['jax_memory_stats']['peak_bytes_in_use']/2**30:.3f} GiB**; physical sampled peak: **{r['physical_sampled_peak_bytes']/2**30:.3f} GiB**.",
        f"Host peak RSS: **{r['host_peak_rss_bytes']/2**30:.3f} GiB**. Physical sampling includes other processes and may miss brief peaks.",
        'These are full-batch tiny-sample measurements, not full-dataset training throughput or generalization.',
        f"Tests: **{v['focused_passed']} focused passed; {v['regression_passed']} full CPU regression passed in {v['regression_seconds']:.2f}s**.",
        'Final checkpoint save/restore reproduces parameter fingerprint and evaluation metrics exactly; weights remain outside Git.',
        'Original source, frozen sample and old evidence hashes were verified unchanged.',
        f"Raw results archive SHA256: `{v['zip_sha256']}`; all indexed members verified.",
        'The complete per-update curve is retained in the raw archive and W&B; the [compact results JSON](m2_stage3a_fullbatch_confirmation_results.json)',
        'contains the 10-update curve plus the exact final step, all final predictions, subgroup counts and hashes.',
        '', '## Interpretation and stopping point','',data['interpretation'],'',
        'No further tuning, model redesign, Stage 3B, commit, push or merge was performed.',
        '', '## Reproduction','',
        'Use the existing pinned Molab environment and unchanged frozen `input/` sample. Start from a fresh guarded output directory',
        'only if a future repetition is explicitly authorized. No second run was executed here.',
        '', '```bash',
        'JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_fullbatch_confirmation.py -q',
        'JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q',
        '# Exact authorized command inside the existing Molab source directory:',
        'JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest \\',
        '  .venv/bin/python scripts/confirm_stage3a_fullbatch.py',
        '```']
    Path(output).write_text('\n'.join(lines)+'\n')


def plot(path,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt']='fullbatch-confirmation-v1'
    data=json.loads(Path(path).read_text());r=data['run'];rows=r['learning_curve'];x=[v['step'] for v in rows]
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    for k,label in [('policy_loss','Policy CE'),('wdl_loss','Weighted WDL CE')]:axes[0].plot(x,[v[k] for v in rows],label=label)
    for k,label in [('policy_accuracy','Policy top-1'),('wdl_accuracy','Weighted WDL')]:axes[1].plot(x,[100*v[k] for v in rows],label=label)
    axes[0].axhline(.05,color='gray',ls='--',lw=1);axes[1].axhline(99,color='gray',ls='--',lw=1,label='99% policy gate')
    axes[0].set_ylabel('Cross-entropy');axes[1].set_ylabel('Accuracy (%)');axes[1].set_ylim(0,102)
    for ax in axes:ax.set_xlabel('Full-batch optimizer updates');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('One full-batch 1,024-position confirmation — '+r['status']);fig.savefig(output,metadata={'Date':None});plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--output',required=True);p.add_argument('--plot')
    a=p.parse_args();render(a.results,a.output)
    if a.plot:plot(a.results,a.plot)
