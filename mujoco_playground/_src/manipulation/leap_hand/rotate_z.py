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
"""Rotate-z with leap hand."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import numpy as np
import yaml
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env, reward
from mujoco_playground._src.manipulation.leap_hand import base as leap_hand_base
from mujoco_playground._src.manipulation.leap_hand import (
    compliance_control,
    motor_control,
)
from mujoco_playground._src.manipulation.leap_hand import leap_hand_constants as consts


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.01,
        action_scale=0.5,
        action_repeat=1,
        episode_length=1000,
        history_len=1,
        obs_noise=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.05,
            ),
        ),
        use_compliance=False,
        compliance_config=config_dict.create(
            normal_pos_stiffness=80.0,
            tangent_pos_stiffness=400.0,
            normal_rot_stiffness=20.0,
            tangent_rot_stiffness=40.0,
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                position=1.0,
                angvel=1.0,
                linvel=0.0,
                pose=0.0,
                torques=0.0,
                energy=-0.001,
                termination=-100.0,
                action_rate=-0.01,
            ),
        ),
        impl="jax",
        nconmax=30 * 8192,
        njmax=128,
    )


class CubeRotateZAxis(leap_hand_base.LeapHandEnv):
    """Rotate a cube around the z-axis as fast as possible wihout dropping it."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=consts.CUBE_XML.as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:
        home_key = self._mj_model.keyframe("home")
        self._init_q = jp.array(home_key.qpos)
        self._lowers = self._mj_model.actuator_ctrlrange[:, 0]
        self._uppers = self._mj_model.actuator_ctrlrange[:, 1]
        self._hand_qids = mjx_env.get_qpos_ids(self.mj_model, consts.JOINT_NAMES)
        self._hand_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.JOINT_NAMES)
        self._cube_qids = mjx_env.get_qpos_ids(self.mj_model, ["cube_freejoint"])
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._cube_geom_id = self._mj_model.geom("cube").id
        self._default_pose = self._init_q[self._hand_qids]

        with open(consts.ROBOT_CONFIG_PATH, "r", encoding="utf-8") as handle:
            robot_config = yaml.safe_load(handle) or {}

        motor_ordering = list(robot_config.get("motors").keys())
        motor_models = [
            robot_config["motors"][name].get("motor") for name in motor_ordering
        ]
        motor_kp_real = [
            float(robot_config["motors"][name].get("kp", 0.0))
            for name in motor_ordering
        ]
        motor_kd_real = [
            float(robot_config["motors"][name].get("kd", 0.0))
            for name in motor_ordering
        ]
        actuator_cfg = robot_config.get("actuators", {})
        kp_ratio = float(actuator_cfg.get("kp_ratio", 1.0))
        kd_ratio = float(actuator_cfg.get("kd_ratio", 1.0))
        motor_kp_sim = [kp / kp_ratio for kp in motor_kp_real]
        motor_kd_sim = [kd / kd_ratio for kd in motor_kd_real]
        passive_active_ratio = robot_config["actuators"]["passive_active_ratio"]

        motor_tau_max = []
        motor_q_dot_max = []
        motor_tau_q_dot_max = []
        motor_q_dot_tau_max = []
        motor_tau_brake_max = []
        motor_kd_min = []
        for model_name in motor_models:
            motor_model = actuator_cfg.get(model_name, {})
            motor_tau_max.append(float(motor_model.get("tau_max", 0.0)))
            motor_q_dot_max.append(float(motor_model.get("q_dot_max", 0.0)))
            motor_tau_q_dot_max.append(float(motor_model.get("tau_q_dot_max", 0.0)))
            motor_q_dot_tau_max.append(float(motor_model.get("q_dot_tau_max", 0.0)))
            motor_tau_brake_max.append(float(motor_model.get("tau_brake_max", 0.0)))
            motor_kd_min.append(float(motor_model.get("kd_min", 0.0)))

        self._motor_control_kwargs = dict(
            kp=jp.array(motor_kp_sim, dtype=jp.float32),
            kd=jp.array(motor_kd_sim, dtype=jp.float32),
            tau_max=jp.array(motor_tau_max, dtype=jp.float32),
            q_dot_max=jp.array(motor_q_dot_max, dtype=jp.float32),
            tau_q_dot_max=jp.array(motor_tau_q_dot_max, dtype=jp.float32),
            q_dot_tau_max=jp.array(motor_q_dot_tau_max, dtype=jp.float32),
            tau_brake_max=jp.array(motor_tau_brake_max, dtype=jp.float32),
            kd_min=jp.array(motor_kd_min, dtype=jp.float32),
            passive_active_ratio=jp.array(passive_active_ratio, dtype=jp.float32),
        )

        self._fingertip_site_ids = np.array(
            [self._mj_model.site(name).id for name in consts.FINGERTIP_NAMES]
        )
        self._arm_rows = np.arange(len(consts.ACTUATOR_NAMES))

    def get_stiffness_damping(
        self, site_mats: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        cfg = self._config.compliance_config

        # Construct diagonal stiffness matrix in local frame
        k_local = jp.diag(
            jp.array(
                [
                    cfg.tangent_pos_stiffness,
                    cfg.tangent_pos_stiffness,
                    cfg.normal_pos_stiffness,
                    cfg.tangent_rot_stiffness,
                    cfg.tangent_rot_stiffness,
                    cfg.normal_rot_stiffness,
                ]
            )
        )

        # Expand to batch size (num_sites)
        k_local = jp.broadcast_to(k_local, (len(self._fingertip_site_ids), 6, 6))

        # Construct rotation matrix for 6D spatial vector
        # [R, 0]
        # [0, R]
        rot_6d = jp.zeros((len(self._fingertip_site_ids), 6, 6))
        rot_6d = rot_6d.at[:, :3, :3].set(site_mats)
        rot_6d = rot_6d.at[:, 3:, 3:].set(site_mats)

        # Rotate stiffness to world frame: K_world = R * K_local * R^T
        k_world = rot_6d @ k_local @ jp.swapaxes(rot_6d, -1, -2)

        # Extract 3x3 blocks for pos and rot
        kp_pos = k_world[:, :3, :3]
        kp_rot = k_world[:, 3:, 3:]

        # Compute damping matrices
        kd_pos = compliance_control.get_damping_matrix(
            kp_pos, 1.0
        )  # Assuming unit mass for now
        kd_rot = compliance_control.get_damping_matrix(
            kp_rot, 1.0
        )  # Assuming unit inertia for now

        return kp_pos, kd_pos, kp_rot, kd_rot

    def reset(self, rng: jax.Array) -> mjx_env.State:
        # Randomize hand qpos and qvel.
        rng, pos_rng, vel_rng = jax.random.split(rng, 3)
        q_hand = jp.clip(
            self._default_pose + 0.1 * jax.random.normal(pos_rng, (consts.NQ,)),
            self._lowers,
            self._uppers,
        )
        v_hand = 0.0 * jax.random.normal(vel_rng, (consts.NV,))

        # Randomize cube qpos and qvel.
        rng, p_rng, quat_rng = jax.random.split(rng, 3)
        start_pos = jp.array([0.1, 0.0, 0.05]) + jax.random.uniform(
            p_rng, (3,), minval=-0.01, maxval=0.01
        )
        start_quat = leap_hand_base.uniform_quat(quat_rng)
        q_cube = jp.array([*start_pos, *start_quat])
        v_cube = jp.zeros(6)

        qpos = jp.concatenate([q_hand, q_cube])
        qvel = jp.concatenate([v_hand, v_cube])
        data = mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            ctrl=q_hand,
            qvel=qvel,
            mocap_pos=jp.array([-100.0, -100.0, -100.0]),  # Hide goal for task.
            impl=self._mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )

        info = {
            "rng": rng,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": data.ctrl,
            "qpos_error_history": jp.zeros(self._config.history_len * 16),
            "x_prev": jp.zeros(
                (len(self._fingertip_site_ids), 6)
            ),  # TODO:  use default pose
            "v_prev": jp.zeros((len(self._fingertip_site_ids), 6)),
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)  # pylint: disable=redefined-outer-name
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        delta = action * self._config.action_scale
        # motor_targets = state.data.ctrl + delta
        motor_targets = self._default_pose + delta
        motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)

        if self._config.use_compliance:
            data = state.data
            site_xmat = data.site_xmat[self._fingertip_site_ids]
            site_mats = site_xmat.reshape(-1, 3, 3)

            kp_pos, kd_pos, kp_rot, kd_rot = self.get_stiffness_damping(site_mats)

            motor_targets, x_next, v_next = compliance_control.compliance_control(
                model=self.mjx_model,
                data=data,
                motor_target=motor_targets,
                motor_torque=data.qfrc_actuator,
                x_prev=state.info["x_prev"],
                v_prev=state.info["v_prev"],
                kp_pos=kp_pos,
                kd_pos=kd_pos,
                kp_rot=kp_rot,
                kd_rot=kd_rot,
                arm_rows=self._arm_rows,
                site_ids=self._fingertip_site_ids,
                dt=self.dt,
                qpos_indices=self._hand_qids,
            )
            state.info["x_prev"] = x_next
            state.info["v_prev"] = v_next

        def pipeline_step(data: mjx.Data, action: jax.Array):
            def f(data, _):
                ctrl = motor_control.step(
                    data.qpos[self._hand_qids],
                    data.qvel[self._hand_dqids],
                    data.qacc[self._hand_dqids],
                    action,
                    **self._motor_control_kwargs,
                )
                data = data.replace(ctrl=ctrl)
                data = mjx.step(self.mjx_model, data)
                return data, None

            return jax.lax.scan(f, data, (), self.n_substeps)[0]

        data = pipeline_step(state.data, motor_targets)
        # data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)
        state.info["motor_targets"] = motor_targets

        done = self._get_termination(data)
        obs = self._get_obs(data, state.info)

        rewards = self._get_reward(data, action, state.info, state.metrics, done)
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        reward = sum(rewards.values()) * self.dt  # pylint: disable=redefined-outer-name

        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v

        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        fall_termination = self.get_cube_position(data)[2] < -0.05
        nans = jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))
        return fall_termination | nans

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> mjx_env.Observation:
        # Hand joint angles.
        joint_angles = data.qpos[self._hand_qids]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = (
            joint_angles
            + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
            * self._config.obs_noise.level
            * self._config.obs_noise.scales.joint_pos
        )

        # Joint position error history.
        qpos_error_history = (
            jp.roll(info["qpos_error_history"], 16)
            .at[:16]
            .set(noisy_joint_angles - info["motor_targets"])
        )
        info["qpos_error_history"] = qpos_error_history

        state = jp.concatenate(
            [
                noisy_joint_angles,  # 16
                qpos_error_history,  # 16 * history_len
                info["last_act"],  # 16
            ]
        )

        cube_pos = self.get_cube_position(data)
        palm_pos = self.get_palm_position(data)
        cube_pos_error = palm_pos - cube_pos

        privileged_state = jp.concatenate(
            [
                state,
                data.qpos[self._hand_qids],
                data.qvel[self._hand_dqids],
                self.get_fingertip_positions(data),
                cube_pos_error,
                self.get_cube_orientation(data),
                self.get_cube_linvel(data),
                self.get_cube_angvel(data),
                data.actuator_force,
            ]
        )

        return {
            "state": state,
            "privileged_state": privileged_state,
        }

    # Reward terms.

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
    ) -> dict[str, jax.Array]:
        del done, metrics  # Unused.

        cube_pos = self.get_cube_position(data)
        palm_pos = self.get_palm_position(data)
        cube_pose_mse = jp.linalg.norm(palm_pos - cube_pos)
        cube_pos_reward = reward.tolerance(
            cube_pose_mse, (0, 0.02), margin=0.05, sigmoid="linear"
        )
        cube_angvel = self.get_cube_angvel(data)
        cube_linvel = self.get_cube_linvel(data)

        terminated = self._get_termination(data)

        return {
            "angvel": self._reward_angvel(cube_angvel),
            "linvel": self._cost_linvel(cube_linvel),
            "position": cube_pos_reward,
            "termination": terminated,
            "action_rate": self._cost_action_rate(
                action, info["last_act"], info["last_last_act"]
            ),
            "pose": self._cost_pose(data.qpos[self._hand_qids]),
            "torques": self._cost_torques(data.actuator_force),
            "energy": self._cost_energy(
                data.qvel[self._hand_dqids], data.actuator_force
            ),
        }

    def _cost_torques(self, torques: jax.Array) -> jax.Array:
        return jp.sum(jp.square(torques))

    def _cost_energy(self, qvel: jax.Array, qfrc_actuator: jax.Array) -> jax.Array:
        return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

    def _cost_linvel(self, cube_linvel: jax.Array) -> jax.Array:
        return jp.linalg.norm(cube_linvel, ord=1, axis=-1)

    def _reward_angvel(self, cube_angvel: jax.Array) -> jax.Array:
        # Unconditionally maximize angvel in the z-direction.
        return cube_angvel @ jp.array([0.0, 0.0, 1.0])

    def _cost_action_rate(
        self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
    ) -> jax.Array:
        del last_last_act  # Unused.
        return jp.sum(jp.square(act - last_act))

    def _cost_pose(self, joint_angles: jax.Array) -> jax.Array:
        return jp.sum(jp.square(joint_angles - self._default_pose))


def domain_randomize(model: mjx.Model, rng: jax.Array):
    mj_model = CubeRotateZAxis().mj_model
    # cube_geom_id = mj_model.geom("cube").id
    cube_body_id = mj_model.body("cube").id
    hand_qids = mjx_env.get_qpos_ids(mj_model, consts.JOINT_NAMES)
    hand_body_names = [
        "palm",
        "if_bs",
        "if_px",
        "if_md",
        "if_ds",
        "mf_bs",
        "mf_px",
        "mf_md",
        "mf_ds",
        "rf_bs",
        "rf_px",
        "rf_md",
        "rf_ds",
        "th_mp",
        "th_bs",
        "th_px",
        "th_ds",
    ]
    hand_body_ids = np.array([mj_model.body(n).id for n in hand_body_names])
    fingertip_geoms = ["th_tip", "if_tip", "mf_tip", "rf_tip"]
    fingertip_geom_ids = [mj_model.geom(g).id for g in fingertip_geoms]

    @jax.vmap
    def rand(rng):
        rng, key = jax.random.split(rng)
        # Fingertip friction: =U(0.5, 1.0).
        fingertip_friction = jax.random.uniform(key, (1,), minval=0.5, maxval=1.0)
        geom_friction = model.geom_friction.at[fingertip_geom_ids, 0].set(
            fingertip_friction
        )

        # Scale cube mass: *U(0.8, 1.2).
        rng, key1, key2 = jax.random.split(rng, 3)
        dmass = jax.random.uniform(key1, minval=0.8, maxval=1.2)
        body_inertia = model.body_inertia.at[cube_body_id].set(
            model.body_inertia[cube_body_id] * dmass
        )
        dpos = jax.random.uniform(key2, (3,), minval=-5e-3, maxval=5e-3)
        body_ipos = model.body_ipos.at[cube_body_id].set(
            model.body_ipos[cube_body_id] + dpos
        )

        # Jitter qpos0: +U(-0.05, 0.05).
        rng, key = jax.random.split(rng)
        qpos0 = model.qpos0
        qpos0 = qpos0.at[hand_qids].set(
            qpos0[hand_qids]
            + jax.random.uniform(key, shape=(16,), minval=-0.05, maxval=0.05)
        )

        # Scale static friction: *U(0.9, 1.1).
        rng, key = jax.random.split(rng)
        frictionloss = model.dof_frictionloss[hand_qids] * jax.random.uniform(
            key, shape=(16,), minval=0.5, maxval=2.0
        )
        dof_frictionloss = model.dof_frictionloss.at[hand_qids].set(frictionloss)

        # Scale armature: *U(1.0, 1.05).
        rng, key = jax.random.split(rng)
        armature = model.dof_armature[hand_qids] * jax.random.uniform(
            key, shape=(16,), minval=1.0, maxval=1.05
        )
        dof_armature = model.dof_armature.at[hand_qids].set(armature)

        # Scale all link masses: *U(0.9, 1.1).
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(
            key, shape=(len(hand_body_ids),), minval=0.9, maxval=1.1
        )
        body_mass = model.body_mass.at[hand_body_ids].set(
            model.body_mass[hand_body_ids] * dmass
        )

        # Joint stiffness: *U(0.8, 1.2).
        rng, key = jax.random.split(rng)
        kp = model.actuator_gainprm[:, 0] * jax.random.uniform(
            key, (model.nu,), minval=0.8, maxval=1.2
        )
        actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
        actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp)

        # Joint damping: *U(0.8, 1.2).
        rng, key = jax.random.split(rng)
        kd = model.dof_damping[hand_qids] * jax.random.uniform(
            key, (16,), minval=0.8, maxval=1.2
        )
        dof_damping = model.dof_damping.at[hand_qids].set(kd)

        return (
            geom_friction,
            body_mass,
            body_inertia,
            body_ipos,
            qpos0,
            dof_frictionloss,
            dof_armature,
            dof_damping,
            actuator_gainprm,
            actuator_biasprm,
        )

    (
        geom_friction,
        body_mass,
        body_inertia,
        body_ipos,
        qpos0,
        dof_frictionloss,
        dof_armature,
        dof_damping,
        actuator_gainprm,
        actuator_biasprm,
    ) = rand(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace(
        {
            "geom_friction": 0,
            "body_mass": 0,
            "body_inertia": 0,
            "body_ipos": 0,
            "qpos0": 0,
            "dof_frictionloss": 0,
            "dof_armature": 0,
            "dof_damping": 0,
            "actuator_gainprm": 0,
            "actuator_biasprm": 0,
        }
    )

    model = model.tree_replace(
        {
            "geom_friction": geom_friction,
            "body_mass": body_mass,
            "body_inertia": body_inertia,
            "body_ipos": body_ipos,
            "qpos0": qpos0,
            "dof_frictionloss": dof_frictionloss,
            "dof_armature": dof_armature,
            "dof_damping": dof_damping,
            "actuator_gainprm": actuator_gainprm,
            "actuator_biasprm": actuator_biasprm,
        }
    )

    return model, in_axes
