"""Tiny inference/optimizer API check, not model training."""
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax


def test_flax_optax_compatibility():
    layer = nn.Dense(3)
    x = jnp.ones((2, 4))
    params = layer.init(jax.random.PRNGKey(0), x)
    y = jax.jit(layer.apply)(params, x)
    assert y.shape == (2, 3)
    assert jnp.isfinite(y).all()
    optimizer = optax.adamw(1e-3)
    state = optimizer.init(params)
    assert len(jax.tree.leaves(state)) > 0
