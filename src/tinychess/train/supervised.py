"""Stage 3B primitives; reuse the accepted model, losses and accumulation math."""
from dataclasses import dataclass
from functools import partial
import hashlib
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tinychess.data.training_loader import REVISION
from tinychess.model.objectives import validate_batch, masked_policy, wdl_predictions
from tinychess.train.overfit_core import create_state, metric_sums
from tinychess.train.fullbatch_confirmation import accumulated_gradient, apply_full_gradient

FIELDS = ('planes', 'context', 'legal_mask', 'action', 'wdl_class',
          'policy_weight', 'value_weight', 'training_eligible')


@dataclass(frozen=True)
class PilotConfig:
    format: str = 'gambit-zero-stage3b-plan-v1'
    dataset_revision: str = REVISION
    train_positions: int = 2654850
    validation_positions: int = 569124
    model: str = '25m'
    compute: str = 'bfloat16'
    seed: int = 20261109
    microbatch: int = 32
    accumulation_steps: int = 32
    shuffle_buffer: int = 32768
    history_dropout: float = 0.
    learning_rates: tuple = (1e-4, 3e-4, 6e-4)
    weight_decay: float = .01
    warmup_steps: int = 1000
    min_lr_ratio: float = .1
    sweep_epochs: int = 1
    pilot_epochs: int = 20
    patience_epochs: int = 2
    min_delta: float = 1e-4
    checkpoint_every_steps: int = 500
    log_every_steps: int = 25
    sweep_gpu_hours: float = 1.
    pilot_gpu_hours: float = 12.
    disk_reserve_gib: int = 20

    def __post_init__(self):
        if (self.format != 'gambit-zero-stage3b-plan-v1' or self.dataset_revision != REVISION
                or self.model != '25m' or self.compute != 'bfloat16' or self.history_dropout != 0.
                or self.train_positions != 2654850 or self.validation_positions != 569124):
            raise ValueError('Stage 3B model/dataset/history contract changed')
        for name in ('seed', 'microbatch', 'accumulation_steps', 'shuffle_buffer', 'warmup_steps',
                     'sweep_epochs', 'pilot_epochs', 'patience_epochs', 'checkpoint_every_steps',
                     'log_every_steps', 'disk_reserve_gib'):
            v = getattr(self, name)
            if type(v) is not int or v < (0 if name == 'seed' else 1):
                raise ValueError('Invalid integer: ' + name)
        if (self.microbatch > 32 or self.batch_size > 1024 or self.shuffle_buffer != 32768
                or self.pilot_epochs > 20 or self.sweep_epochs != 1
                or self.shuffle_buffer < self.batch_size or self.disk_reserve_gib < 20):
            raise ValueError('Outside the bounded Stage 3B preparation profile')
        lr = self.learning_rates
        if (len(lr) != 3 or len(set(lr)) != 3
                or not all(isinstance(x, (int, float)) and np.isfinite(x) and 0 < x <= .001 for x in lr)):
            raise ValueError('Exactly three finite learning rates required')
        for name in ('weight_decay', 'min_delta', 'min_lr_ratio', 'sweep_gpu_hours', 'pilot_gpu_hours'):
            if not isinstance(getattr(self, name), (float, int)) or not np.isfinite(getattr(self, name)):
                raise ValueError('Invalid finite setting: ' + name)
        if not (0 <= self.weight_decay <= .05 and 0 <= self.min_delta < 1
                and 0 < self.min_lr_ratio <= 1 and 0 < self.sweep_gpu_hours <= 1
                and 0 < self.pilot_gpu_hours <= 12):
            raise ValueError('Invalid optimizer or budget setting')
        if self.warmup_steps >= self.steps_per_epoch * self.sweep_epochs:
            raise ValueError('Warmup must finish within sweep')

    @property
    def batch_size(self):
        return self.microbatch * self.accumulation_steps

    @property
    def steps_per_epoch(self):
        return math.ceil(self.train_positions / self.batch_size)

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text())
        values['learning_rates'] = tuple(values['learning_rates'])
        return cls(**values)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def schedule(step, peak, warmup, total_steps, floor=.1):
    """Step-zero warmup is nonzero; first decay value equals peak, final equals floor."""
    step = jnp.asarray(step, jnp.float32)
    warm = peak * jnp.minimum((step + 1) / warmup, 1.)
    progress = jnp.clip((step - warmup) / max(total_steps - warmup - 1, 1), 0., 1.)
    decay = peak * (floor + (1 - floor) * .5 * (1 + jnp.cos(jnp.pi * progress)))
    return jnp.where(step < warmup, warm, decay).astype(jnp.float32)


def kernel_decay_mask(params):
    # Decay Dense/attention kernels; exclude biases, LayerNorm and positional embeddings.
    return jax.tree_util.tree_map_with_path(
        lambda path, x: getattr(path[-1], 'key', None) == 'kernel' and x.ndim >= 2, params)


def initial_state(model, batch, config, lr, epochs):
    if lr not in config.learning_rates:
        raise ValueError('LR must be one of the reviewed three points')
    rate = lambda s: schedule(s, lr, config.warmup_steps,
                              config.steps_per_epoch * epochs, config.min_lr_ratio)
    return create_state(model, jax.random.PRNGKey(config.seed), batch, lr,
                        schedule=rate, weight_decay=config.weight_decay,
                        decay_mask=kernel_decay_mask)


def padded_batch(batch, size):
    """Check every real row, then pad with legal copies carrying ZERO loss weights."""
    checked = validate_batch(batch)
    n = checked['action'].shape[0]
    if n > size:
        raise ValueError('Batch exceeds configured size')
    return {k: jnp.concatenate([x, jnp.repeat(x[:1], size-n, axis=0)], axis=0)
            if n < size else x for k, x in checked.items()} | {
                k: jnp.pad(checked[k], (0, size-n)) for k in ('policy_weight', 'value_weight')}


@partial(jax.jit, static_argnames=('microbatch',))
def train_step(state, batch, microbatch=32):
    return apply_full_gradient(state, *accumulated_gradient(state, batch, microbatch))


def _output_sums(out, batch):
    m = metric_sums(out, batch)
    pw, vw = batch['policy_weight'], batch['value_weight']
    top = jax.lax.top_k(masked_policy(out['policy_logits'], batch['legal_mask'])['logits'], 3)[1]
    probs = wdl_predictions(out['wdl_logits'])['probabilities']
    brier = jnp.sum((probs-jax.nn.one_hot(batch['wdl_class'], 3))**2, axis=-1)
    return dict(valid=m['valid'], policy_nll=m['policy_loss']*m['policy_weight'],
                wdl_nll=m['wdl_loss']*m['value_weight'], policy_weight=m['policy_weight'],
                value_weight=m['value_weight'], policy_correct=m['policy_correct'],
                wdl_correct=m['wdl_correct'], policy_top3=jnp.sum(jnp.any(top == batch['action'][:, None], axis=1)*pw),
                wdl_brier=jnp.sum(brier*vw))


@jax.jit
def evaluation_sums(state, batch):
    out = state.apply_fn({'params': state.params}, batch['planes'], batch['context'])
    return _output_sums(out, batch)


@jax.jit
def evaluation_groups(state, batch, early_mask):
    # One model pass; grouped metrics only reweight the existing loss primitives.
    out = state.apply_fn({'params': state.params}, batch['planes'], batch['context'])
    def group(mask):
        weighted = dict(batch, policy_weight=batch['policy_weight']*mask,
                        value_weight=batch['value_weight']*mask)
        return _output_sums(out, weighted)
    return _output_sums(out, batch), group(early_mask), group(1-early_mask)


class Metrics:
    """Sum numerators and denominators, never average batch means."""
    def __init__(self):
        self.sums = {}

    def add(self, values):
        values = jax.device_get(values)
        if not bool(values['valid']) or not all(np.isfinite(x).all() for x in values.values()):
            raise FloatingPointError('Invalid/nonfinite evaluation')
        for k, v in values.items():
            if k != 'valid': self.sums[k] = self.sums.get(k, 0.) + float(v)

    def result(self):
        p, v = self.sums.get('policy_weight', 0), self.sums.get('value_weight', 0)
        if p <= 0: raise ValueError('Empty policy evaluation')
        return dict(policy_loss=self.sums['policy_nll']/p, policy_accuracy=self.sums['policy_correct']/p,
                    policy_top3=self.sums['policy_top3']/p,
                    wdl_loss=self.sums['wdl_nll']/v if v else None,
                    wdl_accuracy=self.sums['wdl_correct']/v if v else None,
                    wdl_brier=self.sums['wdl_brier']/v if v else None,
                    policy_weight=p, value_weight=v)


@dataclass
class EarlyStop:
    best: float = math.inf
    stale: int = 0

    def observe(self, nll, *, patience=2, min_delta=1e-4):
        if not np.isfinite(nll): raise FloatingPointError('Nonfinite validation NLL')
        improved = nll < self.best - min_delta
        if improved: self.best, self.stale = float(nll), 0
        else: self.stale += 1
        return improved, self.stale >= patience
