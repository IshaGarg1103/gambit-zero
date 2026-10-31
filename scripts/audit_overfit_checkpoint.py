"""Read-only original checkpoint audit; no optimizer update, labels or dataset changes."""
from dataclasses import replace
import json
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import jax
import numpy as np
from tinychess.model.transformer import ChessTransformer,PROFILES
from tinychess.model.objectives import validate_batch,supervised_loss
from tinychess.train.overfit_core import create_state,restore_checkpoint,indices
from tinychess.train.overfit_sample import load_sample


def audit():
    arrays,manifest=load_sample(ROOT/'input')
    raw=ROOT/'data/processed/stage3a_run/training'
    evidence=json.loads((raw/'result.json').read_text())
    model=ChessTransformer(replace(PROFILES['25m'],compute='bfloat16'))
    def batch(ix):return validate_batch({k:v[ix] for k,v in arrays.items()})
    s=jax.jit(lambda key:create_state(model,key,batch(np.arange(32)),1e-3))(jax.random.PRNGKey(20261031))
    s,receipt=restore_checkpoint(raw/'checkpoint',s,evidence['config'])
    forward=jax.jit(lambda p,b:model.apply({'params':p},b['planes'],b['context']))
    policy=[];wdl=[]
    for start in range(0,1024,32):
        out=jax.device_get(forward(s.params,batch(np.arange(start,start+32))))
        policy.append(out['policy_logits']);wdl.append(out['wdl_logits'])
    p=np.concatenate(policy).astype(np.float64);v=np.concatenate(wdl).astype(np.float64)
    mask=arrays['legal_mask'];targets=arrays['action'];labels=arrays['wdl_class'];pw=arrays['policy_weight'];vw=arrays['value_weight']
    masked=np.where(mask,p,-np.inf);maximum=masked.max(-1,keepdims=True)
    logp=masked-maximum-np.log(np.exp(masked-maximum).sum(-1,keepdims=True))
    vv=v-v.max(-1,keepdims=True);logv=vv-np.log(np.exp(vv).sum(-1,keepdims=True))
    pnll=-logp[np.arange(1024),targets];vnll=-logv[np.arange(1024),labels]
    independent=dict(policy_loss=float(np.sum(pnll*pw)/pw.sum()),wdl_loss=float(np.sum(vnll*vw)/vw.sum()),
        policy_accuracy=float(np.sum((masked.argmax(-1)==targets)*pw)/pw.sum()),wdl_accuracy=float(np.sum((v.argmax(-1)==labels)*vw)/vw.sum()))
    errors={k:abs(val-evidence['final'][k]) for k,val in independent.items()};assert max(errors.values())<1e-5
    probability=np.exp(logp);assert np.all(probability[~mask]==0) and np.allclose(probability.sum(-1),1.,atol=1e-6)
    assert mask[np.arange(1024),targets].all()
    ix=indices(20261031,2999,1024,32);b=batch(ix)
    outputs=forward(s.params,b);total,m=supervised_loss(outputs,b)
    p_ref=float(np.sum(pnll[ix]*pw[ix])/pw[ix].sum());v_ref=float(np.sum(vnll[ix]*vw[ix])/vw[ix].sum())
    assert abs(float(m['policy_loss'])-p_ref)<1e-5 and abs(float(m['value_loss'])-v_ref)<1e-5
    result=dict(checkpoint_step=int(s.step),checkpoint_sha256=receipt['sha256'],optimizer_updates=0,
        independent_numpy_final=independent,original_final=evidence['final'],max_metric_absolute_error=max(errors.values()),
        final_step_last_minibatch=dict(policy_loss=float(m['policy_loss']),wdl_loss=float(m['value_loss']),
            numpy_policy_loss=p_ref,numpy_wdl_loss=v_ref),
        unique_encoded_inputs=len({hashlib.sha256(p.tobytes()+c.tobytes()).digest() for p,c in zip(arrays['planes'],arrays['context'])}),legal_targets=1024,illegal_policy_probabilities_zero=True,
        value_masked_rows=int((vw==0).sum()),sample_sha256=manifest['sample_sha256'])
    target=ROOT/'data/processed/stage3a_diagnostics/checkpoint_audit.json';target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':audit()
