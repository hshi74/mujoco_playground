# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train a PPO agent using JAX on the specified environment."""

import os

os.environ["USE_JAX"] = "true"

import datetime
import functools
import json
import time
import warnings
from collections import OrderedDict
from typing import Any, Dict, List

import jax
import mediapy as media
import mujoco
import numpy as np
import tensorboardX
import torch
from absl import app, flags, logging
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath
from ml_collections import config_dict

import mujoco_playground
import wandb
from mujoco_playground import registry, wrapper
from mujoco_playground.config import (
    dm_control_suite_params,
    locomotion_params,
    manipulation_params,
)

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)

# Suppress warnings

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")


_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "LeapCubeRotateZAxis",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_IMPL = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation")
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string(
    "load_checkpoint_path", None, "Path to load checkpoint from"
)
_NOTE = flags.DEFINE_string("note", "", "Optional comma-separated notes for the run")
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "If true, only play with the model and do not train"
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb",
    True,
    "Use Weights & Biases for logging (ignored in play-only mode)",
)
_USE_TB = flags.DEFINE_boolean(
    "use_tb", False, "Use TensorBoard for logging (ignored in play-only mode)"
)
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", True, "Use domain randomization"
)
_USE_COMPLIANCE = flags.DEFINE_boolean(
    "use_compliance", False, "Use compliance control"
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps", 200_000_000, "Number of timesteps"
)
_NUM_TRAIN_VIDEOS = flags.DEFINE_integer(
    "num_train_videos", 10, "Number of rollout videos to record during training."
)
_NUM_EVAL_VIDEOS = flags.DEFINE_integer(
    "num_eval_videos", 1, "Number of rollout videos to record after training."
)
_NUM_EVALS = flags.DEFINE_integer("num_evals", 5, "Number of evaluations")
_REWARD_SCALING = flags.DEFINE_float("reward_scaling", 0.1, "Reward scaling")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 1000, "Episode length")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean(
    "normalize_observations", True, "Normalize observations"
)
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "Action repeat")
_UNROLL_LENGTH = flags.DEFINE_integer("unroll_length", 10, "Unroll length")
_NUM_MINIBATCHES = flags.DEFINE_integer("num_minibatches", 8, "Number of minibatches")
_NUM_UPDATES_PER_BATCH = flags.DEFINE_integer(
    "num_updates_per_batch", 8, "Number of updates per batch"
)
_DISCOUNTING = flags.DEFINE_float("discounting", 0.97, "Discounting")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 5e-4, "Learning rate")
_ENTROPY_COST = flags.DEFINE_float("entropy_cost", 5e-3, "Entropy cost")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 1024, "Number of environments")
_NUM_EVAL_ENVS = flags.DEFINE_integer(
    "num_eval_envs", 128, "Number of evaluation environments"
)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 256, "Batch size")
_MAX_GRAD_NORM = flags.DEFINE_float("max_grad_norm", 1.0, "Max grad norm")
_CLIPPING_EPSILON = flags.DEFINE_float(
    "clipping_epsilon", 0.2, "Clipping epsilon for PPO"
)
_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "policy_hidden_layer_sizes",
    [64, 64, 64],
    "Policy hidden layer sizes",
)
_VALUE_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "value_hidden_layer_sizes",
    [64, 64, 64],
    "Value hidden layer sizes",
)
_POLICY_OBS_KEY = flags.DEFINE_string("policy_obs_key", "state", "Policy obs key")
_VALUE_OBS_KEY = flags.DEFINE_string("value_obs_key", "state", "Value obs key")
_RSCOPE_ENVS = flags.DEFINE_integer(
    "rscope_envs",
    None,
    "Number of parallel environment rollouts to save for the rscope viewer",
)
_DETERMINISTIC_RSCOPE = flags.DEFINE_boolean(
    "deterministic_rscope",
    True,
    "Run deterministic rollouts for the rscope viewer",
)
_RUN_EVALS = flags.DEFINE_boolean(
    "run_evals",
    True,
    "Run evaluation rollouts between policy updates.",
)
_LOG_TRAINING_METRICS = flags.DEFINE_boolean(
    "log_training_metrics",
    False,
    "Whether to log training metrics and callback to progress_fn. Significantly"
    " slows down training if too frequent.",
)
_TRAINING_METRICS_STEPS = flags.DEFINE_integer(
    "training_metrics_steps",
    1_000_000,
    "Number of steps between logging training metrics. Increase if training"
    " experiences slowdown.",
)


def load_jax_ckpt_to_torch(
    jax_params: Any, layer_dims: List[int]
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Convert JAX model parameters to PyTorch format for cross-framework compatibility."""
    if jax_params is None or len(layer_dims) < 2:
        return {"model_state_dict": {}}

    actor_params = {}
    if isinstance(jax_params, (list, tuple)) and len(jax_params) > 1:
        actor_params = jax_params[1].get("params", {})

    model_state: Dict[str, torch.Tensor] = {}
    total_linears = len(layer_dims) - 1

    for idx in range(total_linears):
        key = f"hidden_{idx}"
        tensors = actor_params.get(key)
        if not tensors:
            logging.debug("Missing actor layer '%s' in JAX params.", key)
            continue

        weight = np.array(tensors.get("kernel"))
        bias = np.array(tensors.get("bias"))
        expected_shape = (int(layer_dims[idx + 1]), int(layer_dims[idx]))
        transposed = weight.T
        if weight.shape != expected_shape and transposed.shape == expected_shape:
            weight = transposed
        elif weight.shape != expected_shape:
            logging.debug(
                "Unexpected kernel shape for %s: %s (expected %s)",
                key,
                weight.shape,
                expected_shape,
            )

        if idx == total_linears - 1:
            name = "body.linear_out"
        else:
            name = f"body.linear_{idx}"

        model_state[f"{name}.weight"] = torch.tensor(weight, dtype=torch.float32)
        model_state[f"{name}.bias"] = torch.tensor(bias, dtype=torch.float32)

    return {"model_state_dict": model_state}


def export_onnx(
    params,
    eval_env,
    logdir: epath.Path,
    wandb_run,
    hidden_layer_sizes: List[int],
    policy_obs_key: str,
) -> None:
    obs_spec = eval_env.observation_size
    if isinstance(obs_spec, dict):
        obs_spec = obs_spec.get(policy_obs_key) or next(iter(obs_spec.values()))
    if isinstance(obs_spec, tuple):
        obs_dim = int(np.prod(obs_spec))
    else:
        obs_dim = int(obs_spec)
    action_dim = eval_env.action_size

    stats = getattr(params[0], "mean", {})
    std_stats = getattr(params[0], "std", {})
    obs_mean = stats.get(policy_obs_key)
    obs_std = std_stats.get(policy_obs_key)
    if obs_mean is None or obs_std is None:
        logging.warning(
            "Observation statistics missing; ONNX export will be unnormalized."
        )
        obs_mean = np.zeros(obs_dim, dtype=np.float32)
        obs_std = np.ones(obs_dim, dtype=np.float32)
    obs_mean = np.asarray(obs_mean).reshape(-1).astype(np.float32)
    obs_std = np.asarray(obs_std).reshape(-1).astype(np.float32)
    obs_std = np.where(obs_std > 0, obs_std, 1.0)

    class NormalizeLayer(torch.nn.Module):
        def __init__(self, mean_vec, std_vec):
            super().__init__()
            self.register_buffer("mean", torch.from_numpy(mean_vec))
            self.register_buffer("inv_std", torch.from_numpy(1.0 / std_vec))

        def forward(self, x):
            return (x - self.mean) * self.inv_std

    layer_dims: List[int] = (
        [obs_dim] + [int(h) for h in hidden_layer_sizes] + [action_dim * 2]
    )

    layer_dict: OrderedDict[str, torch.nn.Module] = OrderedDict()
    layer_dict["normalize"] = NormalizeLayer(obs_mean, obs_std)
    last_dim = obs_dim
    for idx, hidden in enumerate(hidden_layer_sizes):
        layer_dict[f"linear_{idx}"] = torch.nn.Linear(last_dim, int(hidden))
        layer_dict[f"act_{idx}"] = torch.nn.SiLU()
        last_dim = int(hidden)
    layer_dict["linear_out"] = torch.nn.Linear(last_dim, action_dim * 2)

    class ActorModel(torch.nn.Module):
        def __init__(self, layers: OrderedDict[str, torch.nn.Module], act_dim: int):
            super().__init__()
            self.body = torch.nn.Sequential(layers)
            self.action_dim = act_dim

        def forward(self, x):
            logits = self.body(x)
            mean, _ = torch.split(logits, [self.action_dim, self.action_dim], dim=-1)
            return torch.tanh(mean)

    actor_network = ActorModel(layer_dict, action_dim)
    actor_network.eval()

    state_dict = load_jax_ckpt_to_torch(params, layer_dims).get("model_state_dict", {})
    if state_dict:
        actor_network.load_state_dict(state_dict, strict=False)

    dummy_input = torch.zeros((1, obs_dim), dtype=torch.float32)
    onnx_path = logdir / "policy.onnx"
    torch.onnx.export(
        actor_network,
        dummy_input,
        str(onnx_path),
        input_names=["obs"],
        output_names=["action"],
    )
    logging.info("Policy exported to ONNX at %s", onnx_path)
    if not onnx_path.exists():
        logging.warning("ONNX file was not created!")
        return

    if wandb_run is None:
        return

    artifact = wandb.Artifact(name="policy", type="model", metadata={})
    artifact.add_file(str(onnx_path))
    artifact.add_file(str(logdir / "policy.onnx.data"))
    for filename in ("env_config.json", "train_config.json", "args.json"):
        file_path = logdir / filename
        if file_path.exists():
            artifact.add_file(str(file_path))

    wandb_run.log_artifact(artifact, aliases=["latest", os.path.basename(logdir)])
    logging.info("ONNX artifact logged to wandb.")


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
    if env_name in mujoco_playground.manipulation._envs:
        return manipulation_params.brax_ppo_config(env_name, _IMPL.value)
    elif env_name in mujoco_playground.locomotion._envs:
        return locomotion_params.brax_ppo_config(env_name, _IMPL.value)
    elif env_name in mujoco_playground.dm_control_suite._envs:
        return dm_control_suite_params.brax_ppo_config(env_name, _IMPL.value)

    raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")


def print_metrics(
    metrics: Dict[str, Any],
    time_elapsed: float,
    num_steps: int,
    num_total_steps: int,
    notes: str = "",
    width: int = 80,
    pad: int = 35,
) -> None:
    """Pretty-prints training metrics similar to toddlerbot's MJX trainer."""
    log_string = f"""{"#" * width}\n"""
    if num_steps >= 0 and num_total_steps > 0:
        title = f" \033[1m Learning steps {num_steps}/{num_total_steps} \033[0m "
        log_string += f"""{title.center(width, " ")}\n"""

    for key, value in metrics.items():
        words = key.split("/")
        if (
            words[0].startswith("episode")
            and len(words) > 1
            and "sum_reward" not in words[1]
            and "length" not in words[1]
        ):
            log_string += f"""{f"Mean {'/'.join(words[1:])}:":>{pad}} {value:.4f}\n"""

    log_string += (
        f"""{"-" * width}\n"""
        f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(time_elapsed))}\n"""
    )
    if "episode/sum_reward" in metrics:
        log_string += (
            f"""{"Mean reward:":>{pad}} {metrics["episode/sum_reward"]:.2f}\n"""
        )
    if "episode/length" in metrics:
        log_string += (
            f"""{"Mean episode length:":>{pad}} {metrics["episode/length"]:.2f}\n"""
        )

    if num_steps > 0 and num_total_steps > 0 and time_elapsed > 0:
        eta = max((time_elapsed / num_steps) * (num_total_steps - num_steps), 0.0)
        log_string += (
            f"""{"Computation:":>{pad}} {num_steps / time_elapsed:.0f} steps/s\n"""
            f"""{"ETA:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(eta))}\n"""
            f"""{"Notes:":>{pad}} {notes}\n"""
        )

    print(log_string)


def format_metrics(metrics: Dict[str, float], step: int) -> Dict[str, float]:
    """Formats metrics to match the wandb schema used by train_mjx."""
    grouped = {"env_step": int(step)}
    for key, value in metrics.items():
        prefix = key.split("/")[0]
        if "sum_reward" in key:
            tag = "Train" if prefix.startswith("episode") else "Eval"
            grouped[f"{tag}/mean_reward"] = value
        elif "length" in key:
            tag = "Train" if prefix.startswith("episode") else "Eval"
            grouped[f"{tag}/mean_episode_length"] = value
        elif "loss" in key:
            grouped[f"Loss/{key.split('/')[-1]}"] = value
        elif "sps" in key:
            grouped["Perf/total_fps"] = value
        elif key.startswith("eval/"):
            name = key.split("/")[-1]
            if name.startswith("episode_"):
                name = name.replace("episode_", "")
            grouped[f"Eval/{name}"] = value
        elif key.startswith("episode/"):
            grouped[f"Episode/{key.split('/')[-1]}"] = value
        else:
            parts = key.split("/", 1)
            if len(parts) == 2:
                grouped[f"{parts[0].capitalize()}/{parts[1]}"] = value
            else:
                grouped[parts[0].capitalize()] = value
    return grouped


def main(argv):
    """Run training and evaluation for the specified environment."""

    del argv

    # Load environment configuration
    env_cfg = registry.get_default_config(_ENV_NAME.value)
    env_cfg["impl"] = _IMPL.value
    if _USE_COMPLIANCE.value:
        env_cfg["use_compliance"] = True

    ppo_params = get_rl_config(_ENV_NAME.value)

    if _NUM_TIMESTEPS.present:
        ppo_params.num_timesteps = _NUM_TIMESTEPS.value
    if _PLAY_ONLY.present:
        ppo_params.num_timesteps = 0
    if _NUM_EVALS.present:
        ppo_params.num_evals = _NUM_EVALS.value
    if _REWARD_SCALING.present:
        ppo_params.reward_scaling = _REWARD_SCALING.value
    if _EPISODE_LENGTH.present:
        ppo_params.episode_length = _EPISODE_LENGTH.value
    if _NORMALIZE_OBSERVATIONS.present:
        ppo_params.normalize_observations = _NORMALIZE_OBSERVATIONS.value
    if _ACTION_REPEAT.present:
        ppo_params.action_repeat = _ACTION_REPEAT.value
    if _UNROLL_LENGTH.present:
        ppo_params.unroll_length = _UNROLL_LENGTH.value
    if _NUM_MINIBATCHES.present:
        ppo_params.num_minibatches = _NUM_MINIBATCHES.value
    if _NUM_UPDATES_PER_BATCH.present:
        ppo_params.num_updates_per_batch = _NUM_UPDATES_PER_BATCH.value
    if _DISCOUNTING.present:
        ppo_params.discounting = _DISCOUNTING.value
    if _LEARNING_RATE.present:
        ppo_params.learning_rate = _LEARNING_RATE.value
    if _ENTROPY_COST.present:
        ppo_params.entropy_cost = _ENTROPY_COST.value
    if _NUM_ENVS.present:
        ppo_params.num_envs = _NUM_ENVS.value
    if _NUM_EVAL_ENVS.present:
        ppo_params.num_eval_envs = _NUM_EVAL_ENVS.value
    if _BATCH_SIZE.present:
        ppo_params.batch_size = _BATCH_SIZE.value
    if _MAX_GRAD_NORM.present:
        ppo_params.max_grad_norm = _MAX_GRAD_NORM.value
    if _CLIPPING_EPSILON.present:
        ppo_params.clipping_epsilon = _CLIPPING_EPSILON.value
    if _POLICY_HIDDEN_LAYER_SIZES.present:
        ppo_params.network_factory.policy_hidden_layer_sizes = list(
            map(int, _POLICY_HIDDEN_LAYER_SIZES.value)
        )
    if _VALUE_HIDDEN_LAYER_SIZES.present:
        ppo_params.network_factory.value_hidden_layer_sizes = list(
            map(int, _VALUE_HIDDEN_LAYER_SIZES.value)
        )
    if _POLICY_OBS_KEY.present:
        ppo_params.network_factory.policy_obs_key = _POLICY_OBS_KEY.value
    if _VALUE_OBS_KEY.present:
        ppo_params.network_factory.value_obs_key = _VALUE_OBS_KEY.value
    env = registry.load(_ENV_NAME.value, config=env_cfg)
    if _RUN_EVALS.present:
        ppo_params.run_evals = _RUN_EVALS.value
    if _LOG_TRAINING_METRICS.present:
        ppo_params.log_training_metrics = _LOG_TRAINING_METRICS.value
    if _TRAINING_METRICS_STEPS.present:
        ppo_params.training_metrics_steps = _TRAINING_METRICS_STEPS.value

    print(f"Environment Config:\n{env_cfg}")
    print(f"PPO Training Parameters:\n{ppo_params}")

    # Generate unique experiment name
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    exp_name = f"{_ENV_NAME.value}-{timestamp}"
    print(f"Experiment name: {exp_name}")
    note_list = []
    if _NOTE.value:
        note_list.extend(
            [entry.strip() for entry in _NOTE.value.split(",") if entry.strip()]
        )
    notes = ", ".join(note_list)

    # Set up logging directory
    logdir = epath.Path("results").resolve() / exp_name
    logdir.mkdir(parents=True, exist_ok=True)
    video_dir = logdir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logs are being stored in: {logdir}")

    args_dict = flags.FLAGS.flag_values_dict()
    env_config_dict = json.loads(env_cfg.to_json())
    train_config_dict = json.loads(ppo_params.to_json())
    with (logdir / "args.json").open("w", encoding="utf-8") as fp:
        json.dump(args_dict, fp, indent=2)
    with (logdir / "env_config.json").open("w", encoding="utf-8") as fp:
        json.dump(env_config_dict, fp, indent=2)
    with (logdir / "train_config.json").open("w", encoding="utf-8") as fp:
        json.dump(train_config_dict, fp, indent=2)

    # Initialize Weights & Biases if required
    wandb_run = None
    defined_metrics = set()
    if _USE_WANDB.value and not _PLAY_ONLY.value:
        try:
            wandb_run = wandb.init(
                project="leap_hand",
                entity="toddlerbot",
                name=exp_name,
                notes=notes or None,
                job_type="train",
                config={
                    "args": args_dict,
                    "train": train_config_dict,
                    "env": env_config_dict,
                },
            )
            wandb.define_metric("env_step")
            defined_metrics.add("env_step")
        except Exception as err:  # pylint: disable=broad-except
            wandb_run = None
            print(f"Failed to initialise wandb: {err}")

    # Initialize TensorBoard if required
    if _USE_TB.value and not _PLAY_ONLY.value:
        writer = tensorboardX.SummaryWriter(logdir)

    # Handle checkpoint loading
    if _LOAD_CHECKPOINT_PATH.value is not None:
        # Convert to absolute path
        ckpt_path = epath.Path(_LOAD_CHECKPOINT_PATH.value).resolve()
        if ckpt_path.is_dir():
            latest_ckpts = list(ckpt_path.glob("*"))
            latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
            latest_ckpts.sort(key=lambda x: int(x.name))
            latest_ckpt = latest_ckpts[-1]
            restore_checkpoint_path = latest_ckpt
            print(f"Restoring from: {restore_checkpoint_path}")
        else:
            restore_checkpoint_path = ckpt_path
            print(f"Restoring from checkpoint: {restore_checkpoint_path}")
    else:
        print("No checkpoint path provided, not restoring from checkpoint")
        restore_checkpoint_path = None

    # Set up checkpoint directory
    ckpt_path = logdir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint path: {ckpt_path}")

    # Save environment configuration
    with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
        json.dump(env_cfg.to_dict(), fp, indent=4)

    training_params = dict(ppo_params)
    if "network_factory" in training_params:
        del training_params["network_factory"]

    network_fn = ppo_networks.make_ppo_networks
    if hasattr(ppo_params, "network_factory"):
        network_factory = functools.partial(network_fn, **ppo_params.network_factory)
    else:
        network_factory = network_fn

    if _DOMAIN_RANDOMIZATION.value:
        training_params["randomization_fn"] = registry.get_domain_randomizer(
            _ENV_NAME.value
        )

    num_eval_envs = ppo_params.get("num_eval_envs", 128)

    if "num_eval_envs" in training_params:
        del training_params["num_eval_envs"]

    train_fn = functools.partial(
        ppo.train,
        **training_params,
        network_factory=network_factory,
        seed=_SEED.value,
        restore_checkpoint_path=restore_checkpoint_path,
        save_checkpoint_path=ckpt_path,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=num_eval_envs,
        log_training_metrics=True,
        run_evals=False,
    )

    # Load evaluation environment.
    eval_env = registry.load(_ENV_NAME.value, config=env_cfg)

    rng = jax.random.PRNGKey(_SEED.value)
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    def record_rollouts(
        policy_fn, prefix: str, num_videos: int, render_stride: int = 2
    ):
        """Runs rollouts with the provided policy and saves rendered videos."""
        if num_videos <= 0:
            return
        nonlocal rng
        rng, rollout_seed = jax.random.split(rng)
        rollout_keys = jax.random.split(rollout_seed, num_videos)

        inference_fn = policy_fn
        jit_inference_fn = jax.jit(inference_fn)

        def do_rollout(rng_key, state):
            empty_data = state.data.__class__(
                **{k: None for k in state.data.__annotations__}
            )
            empty_traj = state.__class__(**{k: None for k in state.__annotations__})
            empty_traj = empty_traj.replace(data=empty_data)

            def step(carry, _):
                rollout_state, key = carry
                key, act_key = jax.random.split(key)
                action = jit_inference_fn(rollout_state.obs, act_key)[0]
                rollout_state = eval_env.step(rollout_state, action)
                traj_data = empty_traj.tree_replace(
                    {
                        "data.qpos": rollout_state.data.qpos,
                        "data.qvel": rollout_state.data.qvel,
                        "data.time": rollout_state.data.time,
                        "data.ctrl": rollout_state.data.ctrl,
                        "data.mocap_pos": rollout_state.data.mocap_pos,
                        "data.mocap_quat": rollout_state.data.mocap_quat,
                        "data.xfrc_applied": rollout_state.data.xfrc_applied,
                    }
                )
                return (rollout_state, key), traj_data

            _, traj = jax.lax.scan(
                step, (state, rng_key), None, length=_EPISODE_LENGTH.value
            )
            return traj

        reset_states = jax.jit(jax.vmap(eval_env.reset))(rollout_keys)
        traj_stacked = jax.jit(jax.vmap(do_rollout))(rollout_keys, reset_states)

        trajectories = [None] * num_videos
        for i in range(num_videos):
            t = jax.tree_util.tree_map(lambda x, idx=i: x[idx], traj_stacked)
            trajectories[i] = [
                jax.tree_util.tree_map(lambda x, j=j: x[j], t)
                for j in range(_EPISODE_LENGTH.value)
            ]

        fps = 1.0 / eval_env.dt / max(render_stride, 1)
        for vid_idx, rollout in enumerate(trajectories):
            traj = rollout[::render_stride]
            frames = eval_env.render(
                traj, height=480, width=640, scene_option=scene_option, camera="side"
            )
            suffix = "" if num_videos == 1 else str(vid_idx)
            video_path = video_dir / f"{prefix}{suffix}_rollout.mp4"
            media.write_video(str(video_path), frames, fps=fps)
            print(f"Rollout video saved to '{video_path}'.")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        f"Videos/{prefix}{suffix}": wandb.Video(
                            str(video_path), format="mp4"
                        )
                    },
                    commit=False,
                )

    times = [time.monotonic()]
    best_episode_reward = float("-inf")
    best_ckpt_step = 0
    last_ckpt_step = 0
    render_interval = (
        max(
            ppo_params.num_timesteps // max(_NUM_TRAIN_VIDEOS.value or 1, 1),
            _EPISODE_LENGTH.value,
        )
        if _NUM_TRAIN_VIDEOS.value > 0
        else float("inf")
    )
    last_render_step = 0

    # Progress function for logging
    def progress(num_steps, metrics):
        nonlocal defined_metrics, best_episode_reward, best_ckpt_step, last_ckpt_step
        times.append(time.monotonic())

        # Log to Weights & Biases
        if wandb_run is not None:
            grouped = format_metrics(metrics, num_steps)
            for key in grouped.keys():
                if key not in defined_metrics:
                    wandb.define_metric(key, step_metric="env_step")
                    defined_metrics.add(key)
            wandb.log(grouped)

        # Log to TensorBoard
        if _USE_TB.value and not _PLAY_ONLY.value:
            for key, value in metrics.items():
                writer.add_scalar(key, value, num_steps)
            writer.flush()

        is_episode_metrics = any("episode" in key for key in metrics)
        if is_episode_metrics:
            last_ckpt_step = num_steps
            reward = float(metrics.get("episode/sum_reward", 0.0))
            if reward > best_episode_reward:
                best_episode_reward = reward
                best_ckpt_step = num_steps
            print_metrics(
                metrics,
                times[-1] - times[0],
                num_steps,
                ppo_params.num_timesteps,
                notes,
            )
        throughput = num_steps / max(times[-1] - times[0], 1e-6)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "Train/last_episode_step": float(last_ckpt_step),
                    "Train/best_episode_step": float(best_ckpt_step),
                    "Train/best_episode_reward": float(best_episode_reward),
                    "Perf/steps_per_sec": throughput,
                },
                commit=False,
            )

    def policy_params_fn(current_step, make_policy, params):
        nonlocal last_render_step
        if eval_env is None or _NUM_TRAIN_VIDEOS.value <= 0:
            return
        if current_step - last_render_step < render_interval:
            return

        eval_policy = make_policy(params, deterministic=True)
        record_rollouts(eval_policy, prefix=f"step_{current_step}", num_videos=1)
        last_render_step = current_step

    # Train or load the model
    try:
        train_fn(  # pylint: disable=no-value-for-parameter
            environment=env,
            progress_fn=progress,
            policy_params_fn=policy_params_fn,
            eval_env=eval_env,
        )
    except (KeyboardInterrupt, jax.errors.JaxRuntimeError) as e:
        if isinstance(e, KeyboardInterrupt) or "KeyboardInterrupt" in str(e):
            print("Training interrupted by user.")
        else:
            raise e

    print(f"Loading best checkpoint from step {best_ckpt_step}...")

    if _PLAY_ONLY.value:
        load_path = restore_checkpoint_path
    else:
        available_ckpts = {}
        for path in ckpt_path.iterdir():
            if path.is_dir() and path.name.isdigit():
                available_ckpts[int(path.name)] = path.name

        load_path = None
        if available_ckpts:
            # Find closest checkpoint to best_ckpt_step
            closest_step = min(
                available_ckpts.keys(), key=lambda x: abs(x - best_ckpt_step)
            )
            print(f"Closest checkpoint to best step {best_ckpt_step} is {closest_step}")
            load_path = ckpt_path / available_ckpts[closest_step]

    if load_path:
        print(f"Loading policy from {load_path}...")
        final_policy = checkpoint.load_policy(load_path)
        params = checkpoint.load(load_path)
    else:
        print("No checkpoints found. Exiting.")
        return

    print("Done training.")
    if len(times) > 1:
        print(f"Time to JIT compile: {times[1] - times[0]}")
        print(f"Time to train: {times[-1] - times[1]}")

    print("Starting inference...")
    if eval_env is not None and _NUM_EVAL_VIDEOS.value > 0:
        record_rollouts(
            final_policy,
            prefix="final",
            num_videos=_NUM_EVAL_VIDEOS.value,
            render_stride=2,
        )

    hidden_sizes = []
    policy_obs_key = "state"
    if hasattr(ppo_params, "network_factory"):
        hidden_sizes = list(
            getattr(ppo_params.network_factory, "policy_hidden_layer_sizes", [])
        )
        policy_obs_key = getattr(
            ppo_params.network_factory, "policy_obs_key", policy_obs_key
        )

    export_onnx(
        params=params,
        eval_env=eval_env,
        logdir=logdir,
        wandb_run=wandb_run,
        hidden_layer_sizes=hidden_sizes,
        policy_obs_key=policy_obs_key,
    )


if __name__ == "__main__":
    app.run(main)
