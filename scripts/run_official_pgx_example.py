"""Run the pinned official example's real routines with a bounded smoke config.

No TinyChess trainer is defined here. The sole AST deletion is the unsupported
chess_v0 baseline assignment; evaluate() and upstream __main__ are never called.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

PROJECT = Path(__file__).resolve().parents[1]
UPSTREAM = PROJECT / "benchmarks/official_pgx/upstream"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["native", "wrapped"], default="native")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT / "reports/official_pgx_cpu.json")
    args = parser.parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")
    import jax
    import jax.numpy as jnp
    import numpy as np
    import pgx

    if args.require_gpu and not any(d.platform == "gpu" for d in jax.devices()):
        raise SystemExit("GPU required; CPU fallback refused")
    if jax.local_device_count() != 1:
        raise SystemExit("This bounded audit requires exactly one visible JAX device")
    manifest = json.loads((UPSTREAM / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        actual = hashlib.sha256((UPSTREAM / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Official source checksum mismatch: {name}")
    config = ["env_id=chess", "seed=0", "max_num_iters=1", "num_channels=8", "num_layers=1",
              "selfplay_batch_size=2", "num_simulations=4", "max_num_steps=4",
              "training_batch_size=8", "eval_interval=1"]
    # Molab may enable PYTHONSAFEPATH, which omits the script directory from
    # sys.path. The unchanged upstream CLI imports its sibling `network.py`.
    # Supply only the verified upstream directory, not notebook PYTHONPATH.
    envvars = {**os.environ, "WANDB_MODE": "disabled", "PYTHONPATH": str(UPSTREAM)}
    exact = subprocess.run([sys.executable, str(UPSTREAM / "train.py"), *config],
                           env=envvars, text=True, capture_output=True, timeout=120)
    print("Unmodified official CLI return code:", exact.returncode, flush=True)
    if exact.returncode == 0 or "AssertionError" not in exact.stderr or "make_baseline_model" not in exact.stderr:
        raise RuntimeError("Unmodified CLI behavior changed; inspect before bypassing anything\n" + exact.stderr)

    tree = ast.parse((UPSTREAM / "train.py").read_text())
    matches = [node for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "baseline" for t in node.targets)]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one baseline assignment")
    tree.body.remove(matches[0])
    ast.fix_missing_locations(tree)
    module = types.ModuleType("official_pgx_alphazero_audit")
    module.__file__ = str(UPSTREAM / "train.py")
    sys.modules[module.__name__] = module
    sys.path.insert(0, str(UPSTREAM))
    sys.argv = [str(UPSTREAM / "train.py"), *config]
    exec(compile(tree, module.__file__, "exec"), module.__dict__)
    if args.backend == "wrapped":
        sys.path.insert(0, str(PROJECT / "src"))
        from tinychess.chess.pgx_env import ENV, RULES_POLICY_VERSION
        module.env = types.SimpleNamespace(init=ENV.init, step=ENV.step,
                                            num_actions=4672, version=RULES_POLICY_VERSION)
    else:
        RULES_POLICY_VERSION = "native-pgx-chess-v2-NOT-PARITY-APPROVED"
    dummy = jax.vmap(module.env.init)(jax.random.split(jax.random.PRNGKey(0), 2))
    model = module.forward.init(jax.random.PRNGKey(0), dummy.observation)
    opt_state = module.optimizer.init(model[0])
    parameter_count = sum(x.size for x in jax.tree.leaves(model[0]))
    model, opt_state = jax.device_put_replicated((model, opt_state), jax.local_devices())
    keys = jax.random.split(jax.random.PRNGKey(1), 1)
    start = time.perf_counter()
    data = jax.block_until_ready(module.selfplay(model, keys))
    cold = time.perf_counter() - start
    start = time.perf_counter()
    data = jax.block_until_ready(module.selfplay(model, keys))
    warm = time.perf_counter() - start
    assert data.obs.shape[:3] == (1, 4, 2)
    weights = np.asarray(data.action_weights)
    assert np.isfinite(weights).all() and (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(-1), 1, atol=1e-5)
    # Check the upstream backward return/mask routine with a known terminal
    # event; the four-ply chess trace itself generally has no completed game.
    reward = jnp.zeros_like(data.reward).at[:, 1, :].set(1.0)
    terminated = jnp.zeros_like(data.terminated).at[:, 1, :].set(True)
    discount = -jnp.ones_like(data.discount).at[:, 1, :].set(0.0)
    synthetic = data._replace(reward=reward, terminated=terminated, discount=discount)
    known = jax.block_until_ready(module.compute_loss_input(synthetic))
    np.testing.assert_allclose(known.value_tgt[0, :, 0], [-1, 1, 0, 0])
    np.testing.assert_array_equal(known.mask[0, :, 0], [True, True, False, False])
    truncation_discount = None
    if args.backend == "wrapped":
        near_cap = dummy.replace(_x=dummy._x._replace(step_count=jnp.full((2,), 511, jnp.int32)))
        actions = jnp.argmax(near_cap.legal_action_mask, axis=-1)
        recurrence, capped = jax.jit(module.recurrent_fn)(
            jax.tree.map(lambda x: x[0], model), jax.random.PRNGKey(0), actions, near_cap)
        assert np.asarray(capped.truncated).all() and not np.asarray(capped.terminated).any()
        # An observed upstream limitation, not a passing truncation correctness gate.
        np.testing.assert_array_equal(recurrence.discount, [-1.0, -1.0])
        truncation_discount = np.asarray(recurrence.discount).tolist()
    samples = jax.block_until_ready(module.compute_loss_input(data))
    samples = jax.tree.map(lambda x: x.reshape((1, -1) + x.shape[3:]), samples)
    start = time.perf_counter()
    _, _, policy_loss, value_loss = jax.block_until_ready(module.train(model, opt_state, samples))
    update = time.perf_counter() - start
    assert np.isfinite(policy_loss).all() and np.isfinite(value_loss).all()
    report = dict(
        upstream=manifest, backend=args.backend, rules_policy=RULES_POLICY_VERSION,
        packages={p: importlib.metadata.version(p) for p in
                  ["jax", "jaxlib", "pgx", "mctx", "optax", "dm-haiku", "wandb", "omegaconf", "pydantic"]},
        devices=[dict(kind=d.device_kind, platform=d.platform) for d in jax.devices()],
        unmodified_cli=dict(returncode=exact.returncode, stdout=exact.stdout, stderr=exact.stderr),
        modifications=["delete module-level unsupported chess_v0 baseline assignment",
                       "do not execute upstream __main__/evaluate/W&B/checkpoint code",
                       "override env with TinyChess adapter" if args.backend == "wrapped" else "retain native chess environment"],
        config=config, parameter_count=parameter_count,
        selfplay_compile_and_first_run_seconds=cold, selfplay_warm_seconds=warm,
        training_compile_and_one_update_seconds=update,
        transient_positions=8, saved_dataset_positions=0, saved_checkpoints=0,
        completed_episode_endings=int(np.asarray(data.terminated).sum()),
        valid_value_targets=int(np.asarray(samples.mask).sum()),
        policy_loss=float(np.asarray(policy_loss).mean()), value_loss=float(np.asarray(value_loss).mean()),
        synthetic_return_and_mask_check="passed",
        upstream_discount_on_wrapped_truncation=truncation_discount,
        status="bounded_official_routines_passed",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["status", "backend", "parameter_count", "policy_loss", "value_loss", "valid_value_targets"]}), flush=True)
    print(args.output)


if __name__ == "__main__":
    main()
