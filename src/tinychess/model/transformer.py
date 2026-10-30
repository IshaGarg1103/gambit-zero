"""Bidirectional pre-LN board transformer with the frozen Pgx policy mapping."""
from dataclasses import asdict, dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp

MODEL_VERSION = 'tinychess-transformer-v1'


@dataclass(frozen=True)
class ModelConfig:
    layers: int
    width: int
    heads: int
    ff_width: int
    compute: str = 'float32'

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in (self.layers,self.width,self.heads,self.ff_width)):
            raise ValueError('Model dimensions must be positive integers')
        if self.width % self.heads:
            raise ValueError('width must be divisible by heads')
        if self.compute not in ('float32','bfloat16'):
            raise ValueError('Only fp32 or bf16 compute with fp32 parameters is supported')

    @property
    def dtype(self):
        return jnp.float32 if self.compute == 'float32' else jnp.bfloat16

    def metadata(self):
        return dict(version=MODEL_VERSION,config=asdict(self),param_dtype='float32',
                    input_planes=[8,8,120],input_context=[8],tokens=65,
                    policy_order='source=col*8+7-row; action=source*73+plane',
                    wdl_order=['win','draw','loss'])


PROFILES = {
    '25m': ModelConfig(layers=8,width=512,heads=8,ff_width=2048),
    '100m': ModelConfig(layers=14,width=768,heads=12,ff_width=3072),
}


def policy_grid_to_actions(grid):
    """(B,row,col,73) -> (B,source*73+plane), explicitly reversing spatial rows.

    Board tokens remain spatial row-major inside the transformer. Pgx sources
    are file-major: source=col*8+(7-row). This is a permutation, not a new codec.
    """
    if grid.ndim != 4 or grid.shape[1:] != (8,8,73):
        raise ValueError('Policy grid must have shape (B,8,8,73)')
    return jnp.transpose(jnp.flip(grid,axis=1),(0,2,1,3)).reshape(grid.shape[0],4672)


def dense(features, *, dtype, name):
    return nn.Dense(features,dtype=dtype,param_dtype=jnp.float32,name=name)


def norm(name):
    return nn.LayerNorm(dtype=jnp.float32,param_dtype=jnp.float32,
                        use_fast_variance=False,force_float32_reductions=True,name=name)


class Attention(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self,x):
        c=self.config;b,s,_=x.shape;d=c.width//c.heads
        qkv=dense(3*c.width,dtype=c.dtype,name='qkv')(x).reshape(b,s,3,c.heads,d)
        q,k,v=qkv[:,:,0],qkv[:,:,1],qkv[:,:,2]
        # BF16 operands where requested, fp32 dot accumulation and softmax.
        scores=jnp.einsum('bshd,bthd->bhst',q,k,preferred_element_type=jnp.float32)
        scores=scores*jnp.asarray(d**-0.5,jnp.float32)
        weights=jax.nn.softmax(scores.astype(jnp.float32),axis=-1)
        attended=jnp.einsum('bhst,bthd->bshd',weights.astype(c.dtype),v,
                           preferred_element_type=jnp.float32).reshape(b,s,c.width)
        return dense(c.width,dtype=c.dtype,name='out')(attended).astype(jnp.float32)


class Block(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self,x):
        c=self.config
        x=x+Attention(c,name='attention')(norm('attention_norm')(x))
        h=dense(c.ff_width,dtype=c.dtype,name='ff_in')(norm('ff_norm')(x))
        h=jax.nn.gelu(h.astype(jnp.float32))
        return x+dense(c.width,dtype=c.dtype,name='ff_out')(h).astype(jnp.float32)


class ChessTransformer(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self,planes,context):
        if planes.ndim!=4 or planes.shape[1:]!=(8,8,120):
            raise ValueError('planes must be (B,8,8,120)')
        if context.shape!=(planes.shape[0],8):
            raise ValueError('context must be (B,8)')
        if planes.dtype!=jnp.float32 or context.dtype!=jnp.float32:
            raise ValueError('Canonical model inputs must remain float32')
        c=self.config;b=planes.shape[0]
        board=dense(c.width,dtype=c.dtype,name='board_embedding')(planes.reshape(b,64,120))
        global_input=dense(c.width,dtype=c.dtype,name='context_embedding')(context)[:,None,:]
        context_token=self.param('context_token',nn.initializers.normal(0.02),(1,1,c.width),jnp.float32)
        positions=self.param('positions',nn.initializers.normal(0.02),(1,65,c.width),jnp.float32)
        x=jnp.concatenate([global_input.astype(jnp.float32)+context_token,board.astype(jnp.float32)],axis=1)+positions
        for i in range(c.layers):
            x=Block(c,name=f'block_{i}')(x)
        x=norm('final_norm')(x)
        # Heads deliberately project in fp32, even with a bf16 backbone.
        grid=dense(73,dtype=jnp.float32,name='policy_head')(x[:,1:]).reshape(b,8,8,73)
        value_hidden=jax.nn.gelu(dense(c.width,dtype=jnp.float32,name='value_hidden')(x[:,0]))
        wdl=dense(3,dtype=jnp.float32,name='value_head')(value_hidden)
        return dict(policy_logits=policy_grid_to_actions(grid),wdl_logits=wdl)


def parameter_count(params):
    """Count actual initialized arrays or eval_shape leaves, not a size label."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
