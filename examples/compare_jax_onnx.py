#!/usr/bin/env python3
"""Compares actions from a JAX PPO checkpoint and an exported ONNX policy."""

import sys
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort
from brax.training.agents.ppo import checkpoint
from etils import epath
from ml_collections import config_dict

# Set these paths before running the script.
JAX_CHECKPOINT_PATH = epath.Path(
    "results/LeapCubeReorient-20251116-102751/checkpoints/000024903680"
).resolve()
ONNX_PATH = epath.Path("results/LeapCubeReorient-20251116-102751/policy.onnx").resolve()
OBS_KEY = "state"
NUM_SAMPLES = 1
SEED = 0


def _shape_tuple(spec: Any, key: str | None = None) -> tuple[int, ...]:
    if isinstance(spec, config_dict.ConfigDict):
        spec = spec.get("shape")
    if spec is None:
        raise ValueError(
            f"Observation spec for '{key or 'obs'}' has no shape information."
        )
    if isinstance(spec, (list, tuple)):
        return tuple(int(s) for s in spec)
    return (int(spec),)


def _make_random_obs(
    observation_spec: Any, rng: np.random.Generator
) -> Dict[str, np.ndarray] | np.ndarray:
    batch_shape = (NUM_SAMPLES,)
    if isinstance(observation_spec, config_dict.ConfigDict):
        observation_spec = dict(observation_spec)
    if isinstance(observation_spec, dict):
        obs = {}
        for key, spec in observation_spec.items():
            spec_tuple = _shape_tuple(spec, key=key)
            obs[key] = rng.standard_normal(batch_shape + spec_tuple).astype(np.float32)
        return obs
    spec_tuple = _shape_tuple(observation_spec)
    return rng.standard_normal(batch_shape + spec_tuple).astype(np.float32)


def main():
    if not JAX_CHECKPOINT_PATH.exists():
        print(f"Checkpoint path not found: {JAX_CHECKPOINT_PATH}", file=sys.stderr)
        sys.exit(1)
    if not ONNX_PATH.exists():
        print(f"ONNX file not found: {ONNX_PATH}", file=sys.stderr)
        sys.exit(1)

    config = checkpoint.load_config(JAX_CHECKPOINT_PATH)
    observation_spec = config.observation_size
    obs_mapping = (
        dict(observation_spec)
        if isinstance(observation_spec, config_dict.ConfigDict)
        else observation_spec
    )
    rng = np.random.default_rng(SEED)
    full_obs = _make_random_obs(observation_spec, rng)

    if isinstance(obs_mapping, dict):
        if OBS_KEY not in obs_mapping:
            raise ValueError(
                f"Observation key '{OBS_KEY}' not found in observation spec "
                f"{list(obs_mapping.keys())}"
            )
        state_batch = full_obs[OBS_KEY]
    else:
        state_batch = full_obs

    flat_state = state_batch.reshape(NUM_SAMPLES, -1).astype(np.float32)

    policy_fn = checkpoint.load_policy(JAX_CHECKPOINT_PATH)
    if isinstance(full_obs, dict):
        jax_obs = jax.tree_util.tree_map(jnp.asarray, full_obs)
    else:
        jax_obs = jnp.asarray(full_obs)
    jax_actions = policy_fn(jax_obs, jax.random.PRNGKey(SEED + 1))[0]
    jax_actions = np.asarray(jax.device_get(jax_actions))

    session = ort.InferenceSession(
        ONNX_PATH.as_posix(), providers=["CPUExecutionProvider"]
    )
    onnx_actions = session.run(None, {"obs": flat_state})[0]

    diff = np.abs(jax_actions - onnx_actions)
    print("JAX actions:\n", jax_actions)
    print("ONNX actions:\n", onnx_actions)
    print("Max absolute difference:", float(diff.max()))


if __name__ == "__main__":
    main()
