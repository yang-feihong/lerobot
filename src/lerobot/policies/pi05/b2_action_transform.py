#!/usr/bin/env python

"""B2 action representation helpers for PI0.5.

The dataset stores body-frame planar twist commands.  PI0.5 instead learns a
50-step trajectory in the frame attached to the robot at the start of each
inference.  The conversion uses the exact SE(2) exponential/log maps for a
piecewise-constant body twist, so translation is rotated as yaw accumulates.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch import Tensor

from lerobot.utils.constants import ACTION

DATASET_ACTION_NAMES = (
    "b2_vx",
    "b2_vy",
    "b2_omega_z",
    "arm_teleop_inactive",
    "arm_reset",
    "height_invariant_ee_rot6d_col0_x",
    "height_invariant_ee_rot6d_col0_y",
    "height_invariant_ee_rot6d_col0_z",
    "height_invariant_ee_rot6d_col1_x",
    "height_invariant_ee_rot6d_col1_y",
    "height_invariant_ee_rot6d_col1_z",
    "height_invariant_ee_x",
    "height_invariant_ee_y",
    "height_invariant_ee_z",
    "gripper_target",
    "task_complete",
)
DATASET_ACTION_DIM = len(DATASET_ACTION_NAMES)
DATASET_B2_TWIST_SLICE = slice(0, 3)
B2_TRAJECTORY_NAMES = ("b2_local_x", "b2_local_y", "b2_local_yaw")
B2_GLOBAL_POSE_STATE_NAMES = ("b2_position_x", "b2_position_y", "b2_yaw")
ARM_TELEOP_INACTIVE_NAME = "arm_teleop_inactive"
TASK_COMPLETE_NAME = "task_complete"


def action_dataset_indices(
    *,
    predict_arm_teleop_inactive: bool = True,
    predict_arm_reset: bool = True,
    predict_ee_pose: bool = True,
    predict_gripper: bool = True,
    include_task_complete: bool = True,
) -> tuple[int, ...]:
    indices: list[int] = [0, 1, 2]
    if predict_arm_teleop_inactive:
        indices.append(3)
    if predict_arm_reset:
        indices.append(4)
    if predict_ee_pose:
        indices.extend(range(5, 14))
    if predict_gripper:
        indices.append(14)
    if include_task_complete:
        indices.append(15)
    return tuple(indices)


def action_schema_kwargs(config: Any) -> dict[str, bool | str]:
    """Extract the persisted action-schema switches from a PI0.5 config."""
    return {
        "representation": config.b2_action_representation,
        "predict_arm_teleop_inactive": config.action_predict_arm_teleop_inactive,
        "predict_arm_reset": config.action_predict_arm_reset,
        "predict_ee_pose": config.action_predict_ee_pose,
        "predict_gripper": config.action_predict_gripper,
        "include_task_complete": config.action_predict_task_complete,
    }


def action_sample_offsets(chunk_size: int, dataset_fps: float, control_frequency_hz: float) -> list[int]:
    """Map model action slots to nearest native dataset-frame offsets."""
    if chunk_size < 1 or dataset_fps <= 0 or control_frequency_hz <= 0:
        raise ValueError("chunk_size and both frequencies must be positive")
    return [round(index * dataset_fps / control_frequency_hz) for index in range(chunk_size)]


def action_label_multiplicity(
    episode_length: int, offsets: list[int], *, num_start_frames: int | None = None
) -> Tensor:
    """Count how often each native action appears as a valid chunk label."""
    multiplicity = torch.zeros(episode_length, dtype=torch.int64)
    if num_start_frames is None:
        num_start_frames = episode_length
    if not 0 <= num_start_frames <= episode_length:
        raise ValueError(f"num_start_frames={num_start_frames} is outside [0, {episode_length}]")
    starts = torch.arange(num_start_frames, dtype=torch.int64)
    for offset in offsets:
        targets = starts + offset
        targets = targets[targets < episode_length]
        multiplicity.scatter_add_(0, targets, torch.ones_like(targets))
    return multiplicity


def _sinc_and_cosc(angle: Tensor) -> tuple[Tensor, Tensor]:
    """Return sin(a)/a and (1-cos(a))/a with stable values around zero."""
    sinc = torch.sinc(angle / torch.pi)
    abs_angle = angle.abs()
    angle2 = angle.square()
    cosc_series = angle * (0.5 - angle2 / 24.0 + angle2.square() / 720.0)
    cosc = torch.where(abs_angle < 1e-4, cosc_series, (1.0 - torch.cos(angle)) / angle)
    return sinc, cosc


def integrate_body_twist(twist: Tensor, dt: float, is_pad: Tensor | None = None) -> Tensor:
    """Integrate ``[..., T, (vx, vy, omega)]`` into a local SE(2) trajectory.

    Every returned pose is expressed in the body frame at the start of the
    chunk. Yaw is accumulated on the real line and is deliberately never
    wrapped to ``[-pi, pi]``.
    """
    if twist.ndim < 2 or twist.shape[-1] != 3:
        raise ValueError(f"Expected twist shape (..., T, 3), got {tuple(twist.shape)}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    work = twist
    if is_pad is not None:
        pad = torch.as_tensor(is_pad, dtype=torch.bool, device=twist.device)
        if pad.shape != twist.shape[:-1]:
            raise ValueError(f"is_pad shape {tuple(pad.shape)} != twist time shape {tuple(twist.shape[:-1])}")
        work = torch.where(pad.unsqueeze(-1), torch.zeros_like(twist), twist)

    vx, vy, omega = work.unbind(dim=-1)
    delta_yaw = omega * dt
    sinc, cosc = _sinc_and_cosc(delta_yaw)
    # Exact body-frame displacement over one piecewise-constant twist interval.
    delta_body_x = dt * (sinc * vx - cosc * vy)
    delta_body_y = dt * (cosc * vx + sinc * vy)

    yaw = torch.cumsum(delta_yaw, dim=-1)
    yaw_before = yaw - delta_yaw
    cos_yaw = torch.cos(yaw_before)
    sin_yaw = torch.sin(yaw_before)
    delta_x = cos_yaw * delta_body_x - sin_yaw * delta_body_y
    delta_y = sin_yaw * delta_body_x + cos_yaw * delta_body_y
    x = torch.cumsum(delta_x, dim=-1)
    y = torch.cumsum(delta_y, dim=-1)
    return torch.stack((x, y, yaw), dim=-1)


def global_pose_to_local_trajectory(global_pose: Tensor, is_pad: Tensor | None = None) -> Tensor:
    """Express T future global SE(2) poses in the current pose's body frame.

    ``global_pose`` is ``(..., T + 1, 3)``: the current pose followed by
    future pose targets. Incremental yaw differences are wrapped before they
    are accumulated, so crossing +/-pi does not create a discontinuity.
    """
    if global_pose.ndim < 2 or global_pose.shape[-1] != 3 or global_pose.shape[-2] < 2:
        raise ValueError(f"Expected global pose shape (..., T+1, 3), got {tuple(global_pose.shape)}")

    current = global_pose[..., :1, :]
    future = global_pose[..., 1:, :]
    delta_world = future[..., :2] - current[..., :2]
    yaw0 = current[..., 0, 2]
    cos_yaw = torch.cos(yaw0).unsqueeze(-1)
    sin_yaw = torch.sin(yaw0).unsqueeze(-1)
    local_x = cos_yaw * delta_world[..., 0] + sin_yaw * delta_world[..., 1]
    local_y = -sin_yaw * delta_world[..., 0] + cos_yaw * delta_world[..., 1]

    yaw_steps = global_pose[..., 1:, 2] - global_pose[..., :-1, 2]
    yaw_steps = torch.atan2(torch.sin(yaw_steps), torch.cos(yaw_steps))
    local_yaw = torch.cumsum(yaw_steps, dim=-1)
    trajectory = torch.stack((local_x, local_y, local_yaw), dim=-1)

    if is_pad is not None:
        pad = torch.as_tensor(is_pad, dtype=torch.bool, device=trajectory.device)
        if pad.shape != trajectory.shape[:-1]:
            raise ValueError(
                f"is_pad shape {tuple(pad.shape)} != trajectory shape {tuple(trajectory.shape[:-1])}"
            )
        valid = ~pad
        time_indices = torch.arange(trajectory.shape[-2], device=trajectory.device)
        time_indices = time_indices.view(*([1] * (valid.ndim - 1)), -1).expand_as(valid)
        last_valid = torch.where(valid, time_indices, -1).cummax(dim=-1).values.clamp_min(0)
        trajectory = trajectory.gather(-2, last_valid.unsqueeze(-1).expand(*last_valid.shape, 3))
    return trajectory


def differentiate_local_trajectory(trajectory: Tensor, dt: float) -> Tensor:
    """Invert :func:`integrate_body_twist` without wrapping accumulated yaw."""
    if trajectory.ndim < 2 or trajectory.shape[-1] != 3:
        raise ValueError(f"Expected trajectory shape (..., T, 3), got {tuple(trajectory.shape)}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    origin = torch.zeros_like(trajectory[..., :1, :])
    previous = torch.cat((origin, trajectory[..., :-1, :]), dim=-2)
    delta = trajectory - previous
    yaw_before = previous[..., 2]
    cos_yaw = torch.cos(yaw_before)
    sin_yaw = torch.sin(yaw_before)
    delta_body_x = cos_yaw * delta[..., 0] + sin_yaw * delta[..., 1]
    delta_body_y = -sin_yaw * delta[..., 0] + cos_yaw * delta[..., 1]
    delta_yaw = delta[..., 2]

    sinc, cosc = _sinc_and_cosc(delta_yaw)
    determinant = (sinc.square() + cosc.square()).clamp_min(1e-8)
    vx = (sinc * delta_body_x + cosc * delta_body_y) / (dt * determinant)
    vy = (-cosc * delta_body_x + sinc * delta_body_y) / (dt * determinant)
    omega = delta_yaw / dt
    return torch.stack((vx, vy, omega), dim=-1)


def encode_b2_action_chunk(
    action: Tensor,
    *,
    dt: float,
    is_pad: Tensor | None = None,
    global_pose: Tensor | None = None,
    representation: str = "local_trajectory",
    predict_arm_teleop_inactive: bool = True,
    predict_arm_reset: bool = True,
    predict_ee_pose: bool = True,
    predict_gripper: bool = True,
    include_task_complete: bool = True,
) -> Tensor:
    """Convert the raw 16D action into the model action.

    The three B2 twist dimensions are replaced by a local trajectory. The
    explicit ``arm_teleop_inactive`` and ``task_complete`` dataset labels are
    retained; completion is never inferred from episode padding.
    """
    if action.ndim < 2 or action.shape[-1] != DATASET_ACTION_DIM:
        raise ValueError(
            f"Expected raw dataset action shape (..., T, {DATASET_ACTION_DIM}), got {tuple(action.shape)}"
        )

    pad = None
    if is_pad is not None:
        pad = torch.as_tensor(is_pad, dtype=torch.bool, device=action.device)
        if pad.shape != action.shape[:-1]:
            raise ValueError(
                f"is_pad shape {tuple(pad.shape)} != action time shape {tuple(action.shape[:-1])}"
            )

    indices = action_dataset_indices(
        predict_arm_teleop_inactive=predict_arm_teleop_inactive,
        predict_arm_reset=predict_arm_reset,
        predict_ee_pose=predict_ee_pose,
        predict_gripper=predict_gripper,
        include_task_complete=include_task_complete,
    )
    result = action[..., indices].clone()
    if representation == "local_trajectory":
        result[..., :3] = (
            global_pose_to_local_trajectory(global_pose, pad)
            if global_pose is not None
            else integrate_body_twist(action[..., DATASET_B2_TWIST_SLICE], dt, pad)
        )
    elif representation != "velocity":
        raise ValueError(f"Unknown B2 action representation: {representation!r}")
    return result


def decode_b2_action_chunk(
    action: Tensor,
    *,
    dt: float,
    representation: str = "local_trajectory",
) -> Tensor:
    """Replace the compact predicted trajectory with executable body twist."""
    if action.ndim < 2:
        raise ValueError(f"Expected an action chunk, got {tuple(action.shape)}")
    result = action.clone()
    if representation == "local_trajectory":
        result[..., :3] = differentiate_local_trajectory(action[..., :3], dt)
    elif representation != "velocity":
        raise ValueError(f"Unknown B2 action representation: {representation!r}")
    return result


def integrate_b2_execution_chunk(
    action: Tensor,
    *,
    dt: float,
    is_pad: Tensor | None = None,
) -> Tensor:
    """Convert a compact executable B2 twist chunk back to trajectory space."""
    if action.ndim < 2:
        raise ValueError(f"Expected a compact executable action chunk, got {tuple(action.shape)}")
    result = action.clone()
    result[..., :3] = integrate_body_twist(action[..., :3], dt, is_pad)
    return result


def b2_execution_action_names(
    dataset_names: list[str] | None,
    *,
    representation: str = "velocity",
    predict_arm_teleop_inactive: bool = True,
    predict_arm_reset: bool = True,
    predict_ee_pose: bool = True,
    predict_gripper: bool = True,
    include_task_complete: bool = True,
) -> list[str] | None:
    del representation
    if dataset_names is None:
        return None
    if len(dataset_names) != DATASET_ACTION_DIM:
        raise ValueError(f"B2 compact action expects the 16D B2+Z1 schema, got {len(dataset_names)} names")
    indices = action_dataset_indices(
        predict_arm_teleop_inactive=predict_arm_teleop_inactive,
        predict_arm_reset=predict_arm_reset,
        predict_ee_pose=predict_ee_pose,
        predict_gripper=predict_gripper,
        include_task_complete=include_task_complete,
    )
    return [dataset_names[i] for i in indices]


def b2_trajectory_action_names(dataset_names: list[str] | None, **kwargs: Any) -> list[str] | None:
    names = b2_execution_action_names(dataset_names, **kwargs)
    if names is None:
        return None
    names[:3] = list(B2_TRAJECTORY_NAMES)
    return names


def make_b2_trajectory_stats(
    dataset_stats: dict[str, dict[str, Any]] | None,
    *,
    dt: float,
    chunk_size: int,
    representation: str = "local_trajectory",
    predict_arm_teleop_inactive: bool = True,
    predict_arm_reset: bool = True,
    predict_ee_pose: bool = True,
    predict_gripper: bool = True,
    include_task_complete: bool = True,
    state_indices: tuple[int, ...] | None = None,
) -> dict[str, dict[str, Any]] | None:
    """Build normalization stats for the compact trajectory representation.

    Dataset statistics remain untouched on disk.  The trajectory bounds use the
    robust per-step velocity quantiles over the full chunk duration; x/y share
    one metric scale so rotation never changes their relative units.
    """
    if dataset_stats is None:
        return None
    stats = deepcopy(dataset_stats)
    if ACTION not in stats:
        raise ValueError("Dataset statistics do not contain action")
    action_stats = stats[ACTION]
    if "q01" not in action_stats or "q99" not in action_stats:
        raise ValueError("B2 trajectory normalization requires action q01/q99 statistics")

    q01 = torch.as_tensor(action_stats["q01"], dtype=torch.float32).reshape(-1)
    q99 = torch.as_tensor(action_stats["q99"], dtype=torch.float32).reshape(-1)
    if q01.numel() != 16 or q99.numel() != 16:
        raise ValueError(f"B2 local trajectory expects 16D dataset action stats, got {q01.numel()}")
    velocity_extent = torch.maximum(q01[DATASET_B2_TWIST_SLICE].abs(), q99[DATASET_B2_TWIST_SLICE].abs())
    duration = dt * chunk_size
    xy_scale = max(float(torch.linalg.vector_norm(velocity_extent[:2]).item() * duration), 1e-3)
    yaw_scale = max(float(velocity_extent[2].item() * duration), 1e-3)
    action_indices = action_dataset_indices(
        predict_arm_teleop_inactive=predict_arm_teleop_inactive,
        predict_arm_reset=predict_arm_reset,
        predict_ee_pose=predict_ee_pose,
        predict_gripper=predict_gripper,
        include_task_complete=include_task_complete,
    )
    gripper_min = float(torch.as_tensor(action_stats.get("min", q01)).reshape(-1)[14])
    gripper_max = float(torch.as_tensor(action_stats.get("max", q99)).reshape(-1)[14])
    if gripper_max <= gripper_min:
        raise ValueError("gripper_target must contain two distinct classes in dataset statistics")

    def converted(name: str, value: Any) -> Any:
        tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1).clone()
        if name == "count":
            return value.clone() if isinstance(value, Tensor) else deepcopy(value)
        if tensor.numel() != 16:
            raise ValueError(f"Action stat {name!r} has {tensor.numel()} dims, expected 16")
        if representation == "local_trajectory" and name in {"q01", "min", "q10"}:
            tensor[DATASET_B2_TWIST_SLICE] = torch.tensor([-xy_scale, -xy_scale, -yaw_scale])
        elif representation == "local_trajectory" and name in {"q99", "max", "q90"}:
            tensor[DATASET_B2_TWIST_SLICE] = torch.tensor([xy_scale, xy_scale, yaw_scale])
        elif representation == "local_trajectory" and name == "std":
            tensor[DATASET_B2_TWIST_SLICE] = torch.tensor([xy_scale / 2, xy_scale / 2, yaw_scale / 2])
        elif representation == "local_trajectory":
            tensor[DATASET_B2_TWIST_SLICE] = 0.0
        tensor = tensor[list(action_indices)]
        selected_names = [DATASET_ACTION_NAMES[index] for index in action_indices]
        # Quantile normalization must remain well-defined even when a boolean is
        # extremely imbalanced and both empirical quantiles collapse to one class.
        positive_bool_names = {ARM_TELEOP_INACTIVE_NAME, "arm_reset", TASK_COMPLETE_NAME}
        for index, selected_name in enumerate(selected_names):
            if selected_name not in positive_bool_names:
                continue
            if name in {"q01", "min", "q10"}:
                tensor[index] = 0.0
            elif name in {"q99", "max", "q90"}:
                tensor[index] = 1.0
            elif name in {"mean", "std"}:
                tensor[index] = 0.5
        if "gripper_target" in selected_names:
            index = selected_names.index("gripper_target")
            if name in {"q01", "min", "q10"}:
                tensor[index] = gripper_min
            elif name in {"q99", "max", "q90"}:
                tensor[index] = gripper_max
            elif name == "mean":
                tensor[index] = (gripper_min + gripper_max) / 2
            elif name == "std":
                tensor[index] = (gripper_max - gripper_min) / 2
        if isinstance(value, Tensor):
            return tensor.to(device=value.device, dtype=value.dtype)
        return tensor.numpy()

    stats[ACTION] = {name: converted(name, value) for name, value in action_stats.items()}
    if state_indices is not None and "observation.state" in stats:
        state_stats = stats["observation.state"]

        def select_state(value: Any) -> Any:
            tensor = torch.as_tensor(value).reshape(-1)
            if tensor.numel() == 1:
                return deepcopy(value)
            selected = tensor[list(state_indices)]
            if isinstance(value, Tensor):
                return selected.to(device=value.device, dtype=value.dtype)
            return selected.numpy()

        stats["observation.state"] = {name: select_state(value) for name, value in state_stats.items()}
    return stats
