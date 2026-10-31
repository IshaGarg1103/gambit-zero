"""One full-population objective, clipped once per optimizer update."""
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import optax

from tinychess.train.overfit_core import finite_tree,metric_sums
from tinychess.model.objectives import supervised_loss,masked_policy


def complete_gradient(state,batch):
    def objective(p):
        outputs=state.apply_fn({'params':p},batch['planes'],batch['context'])
        return supervised_loss(outputs,batch)
    (loss,metrics),gradient=jax.value_and_grad(objective,has_aux=True)(state.params)
    return gradient,dict(metrics,loss=loss)


def accumulated_gradient(state,batch,microbatch=32):
    """Sum global-normalized contributions, never average local WDL means."""
    n=batch['action'].shape[0]
    if n%microbatch:raise ValueError('Full population must divide microbatch')
    chunks=jax.tree_util.tree_map(lambda x:x.reshape((n//microbatch,microbatch)+x.shape[1:]),batch)
    psum=jnp.sum(batch['policy_weight']);vsum=jnp.sum(batch['value_weight'])
    pden=jnp.where(psum>0,psum,1.)
    vden=jnp.where(vsum>0,vsum,1.)
    def scan(carry,part):
        total,loss_sum=carry
        def f(p):
            out=state.apply_fn({'params':p},part['planes'],part['context'])
            _,m=supervised_loss(out,part)
            contribution=(m['policy_loss']*jnp.sum(part['policy_weight'])/pden
                          +m['value_loss']*jnp.sum(part['value_weight'])/vden)
            return jnp.where(m['valid'],contribution,jnp.nan)
        value,g=jax.value_and_grad(f)(state.params)
        return (jax.tree_util.tree_map(jnp.add,total,g),loss_sum+value),None
    (gradient,loss),_=jax.lax.scan(scan,(jax.tree_util.tree_map(jnp.zeros_like,state.params),jnp.float32(0)),chunks)
    return gradient,dict(loss=loss,valid=jnp.isfinite(loss))


def apply_full_gradient(state,gradient,metrics):
    # state.tx already contains the single global clip followed by AdamW.
    candidate=state.apply_gradients(grads=gradient)
    clipped,_=optax.clip_by_global_norm(1.).update(gradient,optax.EmptyState())
    valid=metrics['valid'] & finite_tree(gradient) & finite_tree(candidate.params) & finite_tree(candidate.opt_state)
    nxt=jax.lax.cond(valid,lambda:candidate,lambda:state)
    return nxt,dict(metrics,finite=valid,grad_norm=optax.global_norm(gradient),post_clip_norm=optax.global_norm(clipped))


@jax.jit
def direct_step(state,batch):
    return apply_full_gradient(state,*complete_gradient(state,batch))


@jax.jit
def accumulation_step(state,batch):
    return apply_full_gradient(state,*accumulated_gradient(state,batch,32))


@jax.jit
def evaluate_rows(state,batch):
    out=state.apply_fn({'params':state.params},batch['planes'],batch['context'])
    p=masked_policy(out['policy_logits'],batch['legal_mask'])
    m=metric_sums(out,batch)
    return dict(m,policy_prediction=jnp.argmax(p['logits'],axis=-1),wdl_prediction=jnp.argmax(out['wdl_logits'],axis=-1))


def strata(identities,actions,predictions,legal_counts):
    """Counts/rates, not just missed-row shares; exact existing board hashes."""
    if not(len(identities)==len(actions)==len(predictions)==len(legal_counts)):
        raise ValueError('Unpaired analysis rows')
    counts=Counter(r['position_hash'] for r in identities)
    correct=np.asarray(actions)==np.asarray(predictions)
    plies=np.array([r['ply'] for r in identities])
    shared=np.array([counts[r['position_hash']]>1 for r in identities])
    legal_counts=np.asarray(legal_counts)
    def group(mask):
        n=int(np.sum(mask));c=int(np.sum(correct & mask))
        return dict(rows=n,correct=c,missed=n-c,accuracy=c/n if n else None)
    ply={name:group(mask) for name,mask in [
        ('0-10',plies<=10),('11-30',(plies>=11)&(plies<=30)),
        ('31-60',(plies>=31)&(plies<=60)),('61+',plies>=61)]}
    legal={name:group(mask) for name,mask in [
        ('1-10',legal_counts<=10),('11-20',(legal_counts>=11)&(legal_counts<=20)),
        ('21-30',(legal_counts>=21)&(legal_counts<=30)),('31-40',(legal_counts>=31)&(legal_counts<=40)),
        ('41+',legal_counts>=41)]}
    missed=[dict(index=i,game_id=r['game_id'],ply=r['ply'],legal_moves=int(legal_counts[i]),
                 board_hash=r['position_hash'],shared_board_hash=bool(shared[i]),
                 target_action=int(actions[i]),predicted_action=int(predictions[i]))
            for i,r in enumerate(identities) if not correct[i]]
    return dict(ply_buckets=ply,legal_move_buckets=legal,
        board_hash_groups={'shared':group(shared),'unique':group(~shared)},
        distinct_board_hashes=len(counts),shared_board_hash_count=sum(n>1 for n in counts.values()),
        missed_rows=missed,rows=len(identities),correct=int(correct.sum()),accuracy=float(correct.mean()))
