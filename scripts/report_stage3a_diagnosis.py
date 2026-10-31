"""Render the bounded diagnosis without rewriting the original failed evidence."""
import argparse
import json
from pathlib import Path


def render(path,output):
    r=json.loads(Path(path).read_text());runs=r['cases'];audit=r['checkpoint_audit']
    d=runs['D-bfloat16-0.0003-v2'];a=runs['A-bfloat16-0.001']
    lines=['# Stage 3A failure diagnosis — 31 October 2026','',
        '**Root cause: UNRESOLVED. Stage 3A remains FAIL; Stage 3B is blocked.**','',
        'Learning-rate sensitivity is confirmed. No mapping, masking, pairing, state-reset or optimizer-sign bug',
        'was found in the audited scope. The lower LR substantially improves the 1K result but does not meet',
        'the original acceptance gate. Optimization settings are a plausible contributor, not an exclusive proven root cause.',
        '', '## Scope and preserved provenance','',
        'Branch: `m2/stage-3a-overfit`, inherited accepted Stage 2A/2B base `68e8381bd3457cd2dcbbb0a76588fc07df8192e7`.',
        'Original Stage 3A report, JSON, plot, source/results ZIPs, model, losses, optimizer implementation and frozen',
        'sample files remain byte-identical. No original evidence or failed result was replaced.',
        'The new diagnostics use the same 25,622,092-parameter model, zero-output-head initialization, seed 20261031,',
        '100-step LR warmup, AdamW beta=(0.9,0.95), epsilon=1e-8, zero weight decay and global clipping at 1.0.',
        'Compute is bf16, with fp32 master parameters, optimizer state, losses and sensitive reductions.',
        'The original pinned Molab Python 3.11.15 / JAX 0.7.2 / Flax 0.12.0 / Optax 0.2.6 environment was reused.',
        'No package/dependency lock change, model redesign, history dropout, openings_v2, full-data pilot, MCTS or self-play.',
        '', 'Frozen sample NPZ SHA256: `cc00bda4d5feb8a980be6e32208712f50eb94bb197ae60d01aeaff6e47ced590`.',
        'Logical SHA256: `f2dca7915a44569801b824b2815c04be04703a69827f46b9139f3b7cce1bb498`.',
        'All 1,024 rows are eligible, targets are legal and encoded inputs are distinct. 852 rows have value weight 1;',
        '172 forfeits have value weight 0. Direct bf16 conversion also yields 1,024 distinct input pairs, with no input-cast aliases.',
        'Dataset A/D and the immutable HF revision `b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c` were not changed.',
        '', '## Code audit findings','',
        '| Check | Evidence and finding |','|---|---|',
        '| Model policy mapping | Intercepted the **actual 25M model** at `Dense(policy_head)`, installed known nonzero output weights and perturbed each of 64 final board tokens. Exactly the corresponding 73 logits changed; all 4,672 mappings passed. Max error 3.81e-6. |',
        '| Interpretation of locality | Perturbation was after the backbone, immediately before the pointwise head. Perturbing an input board square would intentionally influence other tokens through bidirectional attention and is not a valid head-locality test. |',
        '| Masks/action targets | All 1,024 targets are in their stored legal masks; original checkpoint probabilities are zero on illegal actions and normalize. Independently calculated masked NLL matches the original result. |',
        '| Pairing/shuffle | Every field uses the same index vector. Runtime probes compare selected tensors byte-for-byte with original rows. Seed/epoch permutations have exact-once unit tests. A/B use fixed indices, D repeats the same 1,024 rows with the original order algorithm. |',
        '| Train/eval inputs and parameters | Same-parameter, same-minibatch comparisons and input-parameter fingerprints pass. Small bf16 compiler differences are quantified below, not hidden. |',
        '| Train/eval mode | The model has no dropout, BatchNorm, train/eval boolean, mutable batch statistics or stochastic forward RNG. |',
        '| State persistence | TrainState, Adam count and schedule count increment together; at the end of D both optimizer counters equal 3,000. Inputs are not mutated; no unintended reinitialization appears in either loop. Original checkpoint restores optimizer/step state. |',
        '| Gradient reach | At zero-output-head initialization, backbone/attention gradients are zero as expected. They become nonzero after the first update. Backbone, attention, policy and WDL groups receive gradients and parameter deltas thereafter. |',
        '| Update sign/LR | First-step update dot gradient is negative. A separate prescribed-gradient test checks the complete first Adam update against its bias-corrected closed form, including LR, clipping and epsilon. |',
        '| Loss normalization | Policy is sum(weight*NLL)/sum(policy weight); WDL is sum(value_weight*CE)/sum(value_weight), zero when no value weight. The 0-weight rows contribute exactly zero WDL gradient; all retain policy supervision. Independent nonuniform-logit NumPy references pass. |',
        '', '### Original checkpoint and loss accounting','',
        f"Restored original step-3000 checkpoint `{audit['checkpoint_sha256']}`. No optimizer update was applied.",
        f"Independent NumPy evaluation reproduces the original final metrics within **{audit['max_metric_absolute_error']:.3g}** absolute error.",
        f"At that same checkpoint, its final-step minibatch policy CE is **{audit['final_step_last_minibatch']['policy_loss']:.6f}** versus full-sample **2.688546**;",
        f"minibatch WDL CE is **{audit['final_step_last_minibatch']['wdl_loss']:.6f}** versus full-sample **0.802800**.",
        'The minibatch/full-population difference is expected composition variance. Each matches its own independent labels/weights reference.',
        'Also, the original logger labels minibatch loss with the newly completed update number although that loss was computed',
        '**before** the update; full-sample evaluation is **after** it. These are not identical parameter snapshots.',
        'This is a metric-timing/documentation issue, not an observed optimization-state or pairing bug.',
        '', '### Gradient propagation and clipping','',
        'D gradient group norms at parameter step 1: attention 0.001248, backbone 0.001755, policy head 3.5584, WDL head 5.2907.',
        'At parameter step 2,999: attention 8.1414, backbone 12.5057, policy head 2.6693, WDL head 1.1411.',
        'Both Adam/schedule counters were [2,2] and [3000,3000], respectively. No backbone disconnection was observed.',
        '', '| Run | Updates recorded | Clip-active fraction | Pre-clip min / max | Post-clip maximum |',
        '|---|---:|---:|---:|---:|']
    for name in ['A-bfloat16-0.001','B-bfloat16-0.001','B-bfloat16-0.0003-v2','B-bfloat16-0.0001-v2','D-bfloat16-0.0003-v2']:
        x=runs[name];g=x['recorded_gradient_statistics']
        lines.append(f"| {name} | {g['count']} | {g['clip_activations']/g['count']:.1%} | {g['pre_clip_min']:.4f} / {g['pre_clip_max']:.4f} | {g['post_clip_max']:.7f} |")
    lines += ['', 'Original run: all **300 logged** norms exceeded 1 (min 2.514, max 51.463). Not all 3,000 old updates were logged,',
        'so an exact original clip frequency is not claimed. New D records all 3,000, and clipping activates on 100%.',
        'This is evidence to investigate stochastic gradients/clipping, not proof clipping is wrong or should be removed.',
        '', '## Experiments A–D','',
        'A budget: <=100 updates /120 training seconds. B and conditional C: <=500 updates /180 training seconds each.',
        'D: <=3,000 updates /240 training seconds. Compilation/initialization/W&B time is outside training budgets.',
        'Stop early at the tiny-set gate. A requires 100% policy/WDL accuracy and both CE<0.05; B/C require policy>=99%,',
        'WDL>=95% and both CE<=0.05. D retains the original Stage 3A criterion unchanged.',
        'A selects the first value-weighted real row with >1 legal move. All B runs use frozen indices 0–31 (29 value-weighted rows).',
        'All completed diagnostics have exactly the same initial parameter SHA256:',
        '`1d8de194d0049605769735f46a23b78a51acf4508f78cc7f649135b7fee0ea5b`.',
        '', '| Diagnostic | LR | Updates | Policy accuracy / CE | Weighted WDL accuracy / CE | Training seconds | Result |',
        '|---|---:|---:|---:|---:|---:|---|']
    for name in ['A-bfloat16-0.001','B-bfloat16-0.001','B-bfloat16-0.0003-v2','B-bfloat16-0.0001-v2','D-bfloat16-0.0003-v2']:
        x=runs[name];f=x['final'];lines.append(f"| {name} | {x['config']['lr']:g} | {x['steps']} | {f['policy_accuracy']:.2%} / {f['policy_loss']:.6f} | {f['wdl_accuracy']:.2%} / {f['wdl_loss']:.6f} | {x['training_seconds']:.3f} | {x['status']} |")
    lines += ['', '**C was skipped:** all three bf16 B runs passed. The read-only fp32 numerical checks below applied no optimizer',
        'update and are not a hidden fp32 training experiment.',
        '**Best B LR = 3e-4**, selected by the recorded lowest final policy CE + weighted WDL CE among passing runs,',
        'with lower-LR tie break. B stops early at its gate, so this is not a matched-step convergence benchmark or a claim of optimal pilot LR.',
        '', '![Original versus selected-LR full-sample learning](m2_stage3a_diagnosis_curve.svg)','',
        '**D still failed.** Policy accuracy improved from 25.39% to 77.34%, policy CE from 2.6885 to 0.7080;',
        'WDL accuracy from 59.27% to 84.51%, WDL CE from 0.8028 to 0.3722. No run beyond the single D rerun was launched.',
        '', '### Diagnostic guard correction and preserved aborted attempt','',
        'The first B/3e-4 attempt aborted at the 100th attempted update (99 committed/logged) because the new audit demanded',
        'same-step losses agree within 1e-4. It is retained as an ERROR attempt, not treated as a model-fit failure.',
        'At identical parameters/inputs, the repeated check measured bf16 evaluation versus AD loss difference **0.000246733**;',
        'using the same masters and inputs in read-only fp32 reduced this to **8.94e-8**. Parameter hashes were unchanged,',
        'and every input/target field matched. This supports compiled bf16 arithmetic/fusion rounding, not a pairing/reset bug.',
        'The audit now records the discrepancy, confirms fp32 loss agreement, and rejects discrepancies beyond an explicit',
        'bf16 bound (max(0.005,1% of total loss)). This is an instrumentation correction, not a change to gradients, optimizer,',
        'accepted model, masks, labels or loss formulas. Exactly one B/3e-4 retry ran from the same initialization.',
        'Maximum same-step component-loss discrepancies: B/1e-3 6.18e-5, B/3e-4 2.47e-4, B/1e-4 1.07e-4, D 4.73e-4.',
        '', '## Classification and next step','',
        '- **Implementation bug:** none demonstrated in the audited mapping, masks, pairing, optimizer arithmetic or state.',
        '- **Incorrect targets/masks:** no evidence found in the frozen sample or independent loss/legality checks. Human labels remain noisy strategic targets.',
        '- **Optimization settings:** LR sensitivity is confirmed; lowering LR materially helps but does not solve the gate. Clip/batch-noise interactions remain hypotheses.',
        '- **Initialization:** zero heads temporarily block backbone gradients only at initialization; A/B show learning is possible. No alternative-initialization control was authorized or run.',
        '- **Precision:** no bf16 input-cast aliases; small compiler differences characterized. Full fp32 1K fitting was not tested; it cannot be ruled out as a contributing factor.',
        '- **Overall: UNRESOLVED.** No model or production training-source fix is justified from this evidence.',
        '', 'Minimal applied change: diagnostic instrumentation and tests only. Proposed recipe adjustment: `--lr 0.0003` improves',
        'the diagnostic, but must not be described as a validated overfit fix. Original failed run and defaults were preserved.',
        'Recommend reviewing the evidence, then authorizing **one** bounded same-sample gradient-noise/clipping control',
        '(e.g. microbatch-32 accumulation to a full-sample gradient, with correct weight normalization). This is proposed,',
        'not implemented or run. Do not start Stage 3B or an open-ended LR sweep.',
        '', '## W&B and evidence','',
        'All runs belong to `ishagarg-research/gambit-zero`, group **m2-overfit-diagnostics**. SDK credentials come from',
        'the environment only. No API keys are stored in code, reports or artifacts. Terminal cloud metrics match local evidence.', '']
    for name,x in runs.items():lines.append(f"- {name}: [{x['status']}]({x['wandb_url']})")
    lines += ['', f"Raw diagnosis archive SHA256: `{r['evidence']['zip_sha256']}`; {r['evidence']['zip_bytes']:,} bytes; all {r['evidence']['indexed_files']} indexed members verified.",
        'The raw archive excludes checkpoints and credentials, and retains both audit versions and the aborted attempt.',
        'Compact results: [m2_stage3a_diagnosis_results.json](m2_stage3a_diagnosis_results.json).',
        'Original Stage 3A source/sample/report hashes remain unchanged and are recorded in that JSON.',
        '', '## Tests and reproduction','',
        f"**{r['regression']['focused_passed']} focused tests passed in {r['regression']['focused_seconds']:.2f}s; {r['regression']['passed']} full CPU tests passed in {r['regression']['seconds']:.2f}s.**",
        'New tests cover actual head-token mapping, original/instrumented-step parity, backbone gradient reach, paired shuffling,',
        'independent Adam first-update arithmetic and weighted nonuniform-loss references.',
        '', '```bash',
        'JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_stage3a_diagnosis.py -q',
        'JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q',
        '# In the original pinned Molab bundle, one explicit diagnostic at a time:',
        'JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest \\',
        '  .venv/bin/python scripts/diagnose_stage3a.py --stage B --lr 0.0003 --steps 500 --seconds 180 \\',
        '  --sample input --output data/processed/fresh_B_diagnostic',
        '# Read-only original checkpoint audit, after training processes have stopped:',
        '.venv/bin/python scripts/audit_overfit_checkpoint.py',
        '```',
        'Remote versions are named `diagnose_stage3a.py` (v1) and `diagnose_stage3a_v2.py` (corrected audit);',
        'the local `scripts/diagnose_stage3a.py` contains the corrected version. `configs/stage3a_diagnosis.json` records budgets and selection.',
        'Initialization/compile/wall times, all gradient probes, checkpoint reference and W&B verification are in the JSON.',
        'The diagnostic `--output` must be fresh; existing evidence is never overwritten. The CLI only accepts approved stages/LRs/budgets.',
        'No Git commit/push/merge or later-stage work was performed.']
    Path(output).write_text('\n'.join(lines)+'\n')


def plot(path,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt']='stage3a-diagnosis-v1'
    r=json.loads(Path(path).read_text());base=json.loads(Path('reports/m2_stage3a_results.json').read_text())['training']['learning_curve']
    new=r['cases']['D-bfloat16-0.0003-v2']['learning_curve']
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    for points,label in [(base,'Original LR 1e-3'),(new,'Diagnostic D: LR 3e-4')]:
        axes[0].plot([p['step'] for p in points],[p['policy_loss'] for p in points],label=label)
        axes[1].plot([p['step'] for p in points],[100*p['policy_accuracy'] for p in points],label=label)
    axes[0].set_ylabel('Masked policy cross-entropy');axes[1].set_ylabel('Policy top-1 accuracy (%)')
    axes[1].axhline(99,ls='--',color='gray',lw=1,label='99% gate');axes[1].set_ylim(0,102)
    for ax in axes:ax.grid(alpha=.2);ax.set_xlabel('Optimizer updates');ax.legend(fontsize=8)
    fig.suptitle('Same frozen 1,024 positions — LR helps; overfit gate still FAIL')
    fig.savefig(output,metadata={'Date':None});plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',default='reports/m2_stage3a_diagnosis_results.json');p.add_argument('--output',default='reports/m2_stage3a_diagnosis.md');p.add_argument('--plot')
    a=p.parse_args();render(a.results,a.output)
    if a.plot:plot(a.results,a.plot)
