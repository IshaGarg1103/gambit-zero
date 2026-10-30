"""Stage 2A model contracts and gradient smoke; no optimizer updates."""
from dataclasses import replace

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tinychess.model.transformer import ModelConfig,PROFILES,ChessTransformer,policy_grid_to_actions,parameter_count
from tinychess.model.objectives import masked_policy,policy_cross_entropy,wdl_predictions,supervised_loss,validate_batch,InvalidBatch
from tinychess.chess.state_features import encode_jit
from tinychess.chess.action_codec import action_to_uci
from tinychess.data.training_loader import collate


@pytest.fixture(scope='module')
def inputs():
    rng=np.random.default_rng(21);n=2
    mask=np.zeros((n,4672),bool);mask[:,[13,201,400]]=True
    return dict(planes=jnp.asarray(rng.normal(size=(n,8,8,120)),jnp.float32),
        context=jnp.asarray(rng.normal(size=(n,8)),jnp.float32),legal_mask=jnp.asarray(mask),
        action=jnp.array([13,201],jnp.int32),wdl_class=jnp.array([0,2],jnp.int32),
        policy_weight=jnp.ones(n,jnp.float32),value_weight=jnp.array([1.,0.],jnp.float32),
        training_eligible=jnp.ones(n,bool))


@pytest.mark.parametrize('profile,count',[('25m',25622092),('100m',100031308)])
def test_actual_flax_parameter_shapes_and_counts(profile,count):
    model=ChessTransformer(PROFILES[profile])
    variables=jax.eval_shape(model.init,jax.random.PRNGKey(0),
        jax.ShapeDtypeStruct((1,8,8,120),jnp.float32),jax.ShapeDtypeStruct((1,8),jnp.float32))
    assert parameter_count(variables['params'])==count
    assert all(x.dtype==jnp.float32 for x in jax.tree_util.tree_leaves(variables['params']))


def test_all_policy_indices_are_explicit_spatial_permutation():
    grid=jnp.arange(4672,dtype=jnp.float32).reshape(1,8,8,73)
    actual=np.asarray(policy_grid_to_actions(grid))[0]
    for row in range(8):
        for col in range(8):
            for plane in range(73):
                action=(col*8+7-row)*73+plane
                assert actual[action]==grid[0,row,col,plane]
    assert len(np.unique(actual))==4672


@pytest.mark.parametrize('moves',[
    [],['e2e4'],['e2e4','a7a6','e4e5','d7d5'],
    ['a2a3','e7e5','a3a4','e5e4','d2d4'],
    ['a2a4','h7h5','a4a5','h5h4','a5a6','h4h3','a6b7','h3g2'],
])
def test_policy_permutation_composes_with_pgx_encoder_and_uci(play,moves):
    state=play(moves);features=jax.device_get(encode_jit(state));color=int(state._x.color)
    # Encode each current own-piece feature into an identifiable per-square
    # value, then expand it to all action planes, independently of action order.
    codes=np.sum(features['planes'][:,:,:6]*np.arange(1,7,dtype=np.float32),axis=-1)
    grid=np.repeat(codes[None,:,:,None],73,axis=-1)
    mapped=np.asarray(policy_grid_to_actions(jnp.asarray(grid)))[0]
    for action in np.flatnonzero(features['legal_mask']):
        uci=action_to_uci(state,int(action));rank=int(uci[1])-1;col=ord(uci[0])-ord('a')
        row=7-rank if color==0 else rank
        assert action//73==col*8+7-row
        assert mapped[action]==codes[row,col] and mapped[action]>0


@pytest.mark.parametrize('compute',['float32','bfloat16'])
def test_model_determinism_precision_gradients_and_roundtrip(inputs,compute):
    c=ModelConfig(2,32,4,64,compute);model=ChessTransformer(c);key=jax.random.PRNGKey(17)
    params=model.init(key,inputs['planes'],inputs['context'])['params']
    repeated=model.init(key,inputs['planes'],inputs['context'])['params']
    for a,b in zip(jax.tree_util.tree_leaves(params),jax.tree_util.tree_leaves(repeated)):
        np.testing.assert_array_equal(a,b);assert a.dtype==jnp.float32
    call=jax.jit(model.apply)
    output=call({'params':params},inputs['planes'],inputs['context'])
    again=call({'params':params},inputs['planes'],inputs['context'])
    assert output['policy_logits'].shape==(2,4672) and output['wdl_logits'].shape==(2,3)
    for name,x in output.items():
        assert x.dtype==jnp.float32 and np.isfinite(x).all()
        np.testing.assert_array_equal(x,again[name])
    def loss(p):return supervised_loss(model.apply({'params':p},inputs['planes'],inputs['context']),inputs)
    (value,metrics),gradient=jax.jit(jax.value_and_grad(loss,has_aux=True))(params)
    assert np.isfinite(value) and bool(metrics['valid'])
    assert all(g.dtype==jnp.float32 and np.isfinite(g).all() for g in jax.tree_util.tree_leaves(gradient))
    assert any(np.any(g!=0) for g in jax.tree_util.tree_leaves(gradient))
    restored=flax.serialization.from_bytes(params,flax.serialization.to_bytes(params))
    for a,b in zip(jax.tree_util.tree_leaves(params),jax.tree_util.tree_leaves(restored)):np.testing.assert_array_equal(a,b)
    # Verify the intended arithmetic, rather than inferring it from a dtype flag.
    graph=jax.make_jaxpr(lambda p:model.apply({'params':p},inputs['planes'],inputs['context']))(params)
    dots=[e for e in graph.jaxpr.eqns if e.primitive.name=='dot_general']
    assert dots and any(e.invars[0].aval.dtype==c.dtype for e in dots)
    assert all(e.invars[0].aval.dtype==jnp.float32 for e in graph.jaxpr.eqns if e.primitive.name=='exp')
    assert all(e.invars[0].aval.dtype==jnp.float32 for e in graph.jaxpr.eqns if e.primitive.name in ('reduce_sum','reduce_max'))
    # Demonstrate non-causal cross-square and context conditioning.
    changed=call({'params':params},inputs['planes'].at[:,7,7,:].add(1.),inputs['context'])
    assert not np.array_equal(changed['policy_logits'][:,:73],output['policy_logits'][:,:73])
    changed=call({'params':params},inputs['planes'],inputs['context'].at[:,0].add(1.))
    assert not np.array_equal(changed['wdl_logits'],output['wdl_logits'])


@pytest.mark.parametrize('dtype',[jnp.float32,jnp.bfloat16])
def test_masking_normalization_and_finite_gradients(inputs,dtype):
    logits=jnp.linspace(-3,4,9344,dtype=jnp.float32).reshape(2,4672).astype(dtype)
    mask=inputs['legal_mask'];p=masked_policy(logits,mask)
    assert p['logits'].dtype==p['probabilities'].dtype==jnp.float32
    assert np.all(np.asarray(p['probabilities'])[~np.asarray(mask)]==0)
    np.testing.assert_allclose(p['probabilities'].sum(-1),1.,atol=1e-6)
    def objective(x):return policy_cross_entropy(x,mask,inputs['action'])[0].sum()
    value,gradient=jax.value_and_grad(objective)(logits)
    assert np.isfinite(value) and np.isfinite(gradient).all()
    assert np.all(np.asarray(gradient)[~np.asarray(mask)]==0)


def test_mask_sentinel_does_not_dominate_very_negative_legal_logits(inputs):
    x=jnp.full((2,4672),jnp.float32(1e8)).at[:,13].set(-2e9).at[:,201].set(-3e9)
    p=masked_policy(x,inputs['legal_mask'].at[:,400].set(False))
    np.testing.assert_array_equal(p['probabilities'][:,13],1.)
    assert np.all(np.asarray(p['probabilities'])[~np.asarray(inputs['legal_mask'])]==0)


def test_terminal_all_illegal_is_zero_and_rejected_for_training(inputs):
    mask=jnp.zeros((2,4672),bool);p=masked_policy(jnp.zeros((2,4672)),mask)
    assert not np.asarray(p['has_legal_actions']).any()
    assert np.all(np.asarray(p['probabilities'])==0) and np.isfinite(p['logits']).all()
    with pytest.raises(InvalidBatch,match='all-illegal'):validate_batch(dict(inputs,legal_mask=mask))
    loss,valid=policy_cross_entropy(jnp.zeros((2,4672)),mask,inputs['action'])
    assert not np.asarray(valid).any() and np.isnan(loss).all()


@pytest.mark.parametrize('action',[-1,4672,14])
def test_illegal_target_fails_host_and_compiled_loss(inputs,action):
    bad=dict(inputs,action=inputs['action'].at[0].set(action))
    with pytest.raises(InvalidBatch):validate_batch(bad)
    loss,valid=jax.jit(policy_cross_entropy)(jnp.zeros((2,4672)),inputs['legal_mask'],bad['action'])
    assert not bool(valid[0]) and np.isnan(loss[0])


def test_wdl_sign_order_and_zero_weight_value_gradients(inputs):
    logits=jnp.array([[5.,0.,-5.],[-5.,0.,5.],[0.,5.,0.]],jnp.float32)
    p=wdl_predictions(logits)
    assert p['value'][0]>0 and p['value'][1]<0 and p['value'][2]==0
    np.testing.assert_allclose(p['probabilities'].sum(-1),1.,atol=1e-6)
    out=dict(policy_logits=jnp.zeros((2,4672)),wdl_logits=logits[:2])
    for weights in [jnp.array([1.,0.]),jnp.zeros(2),jnp.array([0.1,0.1])]:
        batch=dict(inputs,value_weight=weights)
        def f(v):return supervised_loss(dict(out,wdl_logits=v),batch)[0]
        value,grad=jax.value_and_grad(f)(out['wdl_logits'])
        assert np.isfinite(value) and np.isfinite(grad).all()
        if weights[1]==0:np.testing.assert_array_equal(grad[1],0.)
        if not np.any(weights):
            np.testing.assert_array_equal(grad,0.)
            assert supervised_loss(out,batch)[1]['value_loss']==0


def test_stage1_batch_contract_and_offboard_mask(play):
    rows=[]
    for state in [play([]),play(['e2e4'])]:
        f=jax.device_get(encode_jit(state));a=int(np.flatnonzero(f['legal_mask'])[0])
        rows.append(dict(planes=np.asarray(f['planes'],'<f4').tobytes(),context=np.asarray(f['context'],'<f4').tobytes(),
            legal_mask=np.packbits(f['legal_mask'],bitorder='little').tobytes(),action=a,wdl_class=0,
            outcome_stm=1,policy_weight=1.,value_weight=0.,training_eligible=True,
            game_id='synthetic',ply=0,position_hash=bytes(32),source_month='2026-01'))
    batch=validate_batch(collate(rows,'train'));model=ChessTransformer(ModelConfig(1,16,4,32))
    p=model.init(jax.random.PRNGKey(7),batch['planes'],batch['context'])
    output=model.apply(p,batch['planes'],batch['context']);probs=masked_policy(output['policy_logits'],batch['legal_mask'])
    assert np.all(np.asarray(probs['probabilities'])[~np.asarray(batch['legal_mask'])]==0)
    assert np.isfinite(supervised_loss(output,batch)[0])


@pytest.mark.parametrize('dimensions',[(0,32,4,64),(1,33,4,64),(1,32,4,0)])
def test_invalid_config(dimensions):
    with pytest.raises(ValueError):ModelConfig(*dimensions)


def test_input_contract_errors(inputs):
    model=ChessTransformer(ModelConfig(1,16,4,32));key=jax.random.PRNGKey(1)
    with pytest.raises(ValueError,match='planes'):model.init(key,inputs['planes'][:,0],inputs['context'])
    with pytest.raises(ValueError,match='float32'):model.init(key,inputs['planes'].astype(jnp.bfloat16),inputs['context'])


@pytest.mark.parametrize('fen',[
    'rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3',
    '7k/5Q2/6K1/8/8/8/8/8 b - - 0 1',
])
def test_terminal_encoded_state_keeps_finite_value_and_zero_policy(play,fen):
    state=play([],fen=fen);features=jax.device_get(encode_jit(state))
    assert bool(state.terminated) and not np.any(features['legal_mask'])
    model=ChessTransformer(ModelConfig(1,16,4,32));planes=jnp.asarray(features['planes'][None]);context=jnp.asarray(features['context'][None])
    variables=model.init(jax.random.PRNGKey(3),planes,context)
    output=model.apply(variables,planes,context)
    assert all(np.isfinite(x).all() for x in output.values())
    p=masked_policy(output['policy_logits'],jnp.asarray(features['legal_mask'][None]))
    assert not p['has_legal_actions'][0] and np.all(np.asarray(p['probabilities'])==0)
    assert np.isfinite(wdl_predictions(output['wdl_logits'])['value']).all()


@pytest.mark.parametrize('field,bad',[
    ('planes',lambda x:x.at[0,0,0,0].set(jnp.nan)),
    ('training_eligible',lambda x:x.at[0].set(False)),
    ('value_weight',lambda x:x.at[0].set(-1.)),
    ('wdl_class',lambda x:x.at[0].set(3)),
    ('context',lambda x:x[:,:7]),
])
def test_host_boundary_rejects_malformed_learning_inputs(inputs,field,bad):
    with pytest.raises(InvalidBatch):validate_batch(dict(inputs,**{field:bad(inputs[field])}))


def test_precision_modes_share_fp32_params_and_close_outputs(inputs):
    c=ModelConfig(2,32,4,64);model=ChessTransformer(c);bf=ChessTransformer(replace(c,compute='bfloat16'))
    params=model.init(jax.random.PRNGKey(5),inputs['planes'],inputs['context'])
    fp=model.apply(params,inputs['planes'],inputs['context']);low=bf.apply(params,inputs['planes'],inputs['context'])
    for k in fp:np.testing.assert_allclose(fp[k],low[k],rtol=0.05,atol=0.05)


def test_loss_shape_guards_and_masked_ce_reference(inputs):
    out=dict(policy_logits=jnp.zeros((2,4672)),wdl_logits=jnp.zeros((2,3)))
    loss,_=policy_cross_entropy(out['policy_logits'],inputs['legal_mask'],inputs['action'])
    np.testing.assert_allclose(loss,np.log(3.),atol=1e-6)
    with pytest.raises(ValueError,match='Loss expects'):
        supervised_loss(out,dict(inputs,value_weight=jnp.ones((2,1))))
