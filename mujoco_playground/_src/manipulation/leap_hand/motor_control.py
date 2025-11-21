"""Standalone motor control function without toddlerbot dependencies."""

from __future__ import annotations

import os
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as _np

USE_JAX = os.getenv("USE_JAX", "false").lower() == "true"
np = jnp if USE_JAX else _np
ArrayType = jax.Array | _np.ndarray


def step(
    q: ArrayType,
    q_dot: ArrayType,
    q_dot_dot: ArrayType,
    a: ArrayType,
    kp: ArrayType,
    kd: ArrayType,
    tau_max: ArrayType,
    q_dot_max: ArrayType,
    tau_q_dot_max: ArrayType,
    q_dot_tau_max: ArrayType,
    tau_brake_max: ArrayType,
    kd_min: ArrayType,
    passive_active_ratio: ArrayType,
    noise: Optional[Dict[str, ArrayType | float]] = None,
) -> ArrayType:
    """
    Compute torque commands for Dynamixel-style motors with asymmetric saturation.

    This mirrors `MotorController.step` from `motor_control.py` but is a pure function
    without toddlerbot dependencies. All gain and limit parameters are provided as
    inputs instead of being pulled from a robot instance.

    Args:
        q: Joint positions (rad or m).
        q_dot: Joint velocities (rad/s or m/s).
        q_dot_dot: Joint accelerations.
        a: Desired action (reference position or torque proxy).
        kp: Proportional gain.
        kd: Derivative gain.
        tau_max: Maximum torque for acceleration side.
        q_dot_max: Velocity beyond which torque is zeroed.
        tau_q_dot_max: Torque at `q_dot_max`.
        q_dot_tau_max: Velocity where tapering begins.
        tau_brake_max: Maximum braking torque.
        kd_min: Minimum derivative gain.
        passive_active_ratio: Ratio applied when decelerating.
        noise: Optional multiplicative noise terms keyed by parameter name.

    Returns:
        Torque command after asymmetric saturation.
    """
    noise = {} if noise is None else noise

    kp = np.asarray(kp, dtype=np.float32) * noise.get("kp", 1.0)
    kd = np.asarray(kd, dtype=np.float32) * noise.get("kd", 1.0)
    tau_max = np.asarray(tau_max, dtype=np.float32) * noise.get("tau_max", 1.0)
    q_dot_tau_max = np.asarray(q_dot_tau_max, dtype=np.float32) * noise.get(
        "q_dot_tau_max", 1.0
    )
    q_dot_max = np.asarray(q_dot_max, dtype=np.float32) * noise.get("q_dot_max", 1.0)
    kd_min = np.asarray(kd_min, dtype=np.float32) * noise.get("kd_min", 1.0)
    tau_brake_max = np.asarray(tau_brake_max, dtype=np.float32) * noise.get(
        "tau_brake_max", 1.0
    )
    tau_q_dot_max = np.asarray(tau_q_dot_max, dtype=np.float32) * noise.get(
        "tau_q_dot_max", 1.0
    )
    passive_active_ratio = np.asarray(
        passive_active_ratio, dtype=np.float32
    ) * noise.get("passive_active_ratio", 1.0)

    q = np.asarray(q, dtype=np.float32)
    q_dot = np.asarray(q_dot, dtype=np.float32)
    q_dot_dot = np.asarray(q_dot_dot, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)

    error = a - q
    real_kp = np.where(q_dot_dot * error < 0, kp * passive_active_ratio, kp)
    tau_m = real_kp * error - (kd_min + kd) * q_dot
    abs_q_dot = np.abs(q_dot)

    # Linear taper between (q_dot_tau_max, tau_max) and (q_dot_max, tau_q_dot_max).
    slope = (tau_q_dot_max - tau_max) / (q_dot_max - q_dot_tau_max)
    taper_limit = tau_max + slope * (abs_q_dot - q_dot_tau_max)

    tau_acc_limit = np.where(abs_q_dot <= q_dot_tau_max, tau_max, taper_limit)
    tau_m_clamped = np.where(
        np.logical_and(abs_q_dot > q_dot_max, q_dot * a > 0),
        np.where(
            q_dot > 0,
            np.ones_like(tau_m) * -tau_brake_max,
            np.ones_like(tau_m) * tau_brake_max,
        ),
        np.where(
            q_dot > 0,
            np.clip(tau_m, -tau_brake_max, tau_acc_limit),
            np.clip(tau_m, -tau_acc_limit, tau_brake_max),
        ),
    )
    return tau_m_clamped
