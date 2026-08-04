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

DATASET_ACTION_DIM = 16
DATASET_B2_TWIST_SLICE = slice(1, 4)
B2_TRAJECTORY_NAMES = ("b2_local_x", "b2_local_y", "b2_local_yaw")
B2_GLOBAL_POSE_STATE_NAMES = ("b2_position_x", "b2_position_y", "b2_yaw")
TASK_COMPLETE_NAME = "task_complete"


def action_dataset_indices(
    *,
    predict_b2_active: bool = False,
    predict_arm_active: bool = False,
    predict_arm_reset: bool = True,
    predict_ee_pose: bool = True,
    predict_gripper: bool = True,
) -> tuple[int, ...]:
    indices: list[int] = []
    if predict_b2_active:
        indices.append(0)
    indices.extend(range(1, 4))
    if predict_arm_active:
        indices.append(4)
    if predict_arm_reset:
        indices.append(5)
    if predict_ee_pose:
        indices.extend(range(6, 15))
    if predict_gripper:
        indices.append(15)
    return tuple(indices)


def action_schema_kwargs(config: Any) -> dict[str, bool | str]:
    """Extract the persisted action-schema switches from a PI0.5 config."""
    return {
        "representation": config.b2_action_representation,
        "predict_b2_active": config.action_predict_b2_active,
        "predict_arm_active": config.action_predict_arm_active,
        "predict_arm_reset": config.action_predict_arm_reset,
        "predict_ee_pose": config.action_predict_ee_pose,
        "predict_gripper": config.action_predict_gripper,
        "append_completion": config.action_predict_task_complete,
    }


def action_sample_offsets(chunk_size: int, dataset_fps: float, control_frequency_hz: float) -> list[int]:
    """Map model action slots to nearest native dataset-frame offsets."""
    if chunk_size < 1 or dataset_fps <= 0 or control_frequency_hz <= 0:
        raise ValueError("chunk_size and both frequencies must be positive")
    return [round(index * dataset_fps / control_frequency_hz) for index in range(chunk_size)]


def action_label_multiplicity(episode_length: int, offsets: list[int]) -> Tensor:
    """Count how often each native action appears as a valid chunk label."""
    multiplicity = torch.zeros(episode_length, dtype=torch.int64)
    starts = torch.arange(episode_length, dtype=torch.int64)
    for offset in offsets:
        targets = starts + offset
        targets = targets[targets < episode_length]
        multiplicity.scatter_add_(0, targets, torch.ones_like(targets))
    return multiplicity


def task_complete_class_counts(
    episode_lengths: list[int], chunk_size: int, offsets: list[int] | None = None
) -> tuple[int, int]:
    """Count known positive and negative completion labels produced by action chunks.

    Every frame is a possible chunk start. ``task_complete`` becomes true at
    the final valid action and remains true in right-padding. For a completely
    unpadded chunk its last label is unknown (there is no next action in the
    chunk), matching the validity mask used by PI0.5's gate-aware loss.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    offsets = list(range(chunk_size)) if offsets is None else offsets
    if len(offsets) != chunk_size:
        raise ValueError(f"Expected {chunk_size} action offsets, got {len(offsets)}")

    positive = 0
    negative = 0
    for length in episode_lengths:
        if length < 1:
            raise ValueError(f"Episode lengths must be >= 1, got {length}")
        starts = torch.arange(length, dtype=torch.int64).unsqueeze(1)
        targets = starts + torch.tensor(offsets, dtype=torch.int64).unsqueeze(0)
        pad = targets >= length
        complete = pad.clone()
        if chunk_size > 1:
            complete[:, :-1] = pad[:, 1:]
        complete[:, -1] = pad[:, -1]
        known = torch.ones_like(complete)
        known[:, -1] = pad[:, -1]
        positive += int((complete & known).sum())
        negative += int((~complete & known).sum())

    return positive, negative


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


def completion_from_padding(is_pad: Tensor) -> Tensor:
    """Mark the last valid action and all post-episode padding as complete."""
    pad = torch.as_tensor(is_pad, dtype=torch.bool)
    if pad.ndim < 1:
        raise ValueError(f"Expected padding mask with a time dimension, got {tuple(pad.shape)}")
    complete = pad.clone()
    if pad.shape[-1] > 1:
        complete[..., :-1] = pad[..., 1:]
    # When the final slot is valid we do not know whether the episode ends at
    # the following (unloaded) step, so it must remain false.
    complete[..., -1] = pad[..., -1]
    return complete


def encode_b2_action_chunk(
    action: Tensor,
    *,
    dt: float,
    is_pad: Tensor | None = None,
    global_pose: Tensor | None = None,
    representation: str = "local_trajectory",
    predict_b2_active: bool = False,
    predict_arm_active: bool = False,
    predict_arm_reset: bool = True,
    predict_ee_pose: bool = True,
    predict_gripper: bool = True,
    append_completion: bool = True,
) -> Tensor:
    """Convert the raw 16D dataset action into the compact model action.

    ``b2_active`` and ``arm_active`` are deliberately dropped. The three B2
    twist dimensions are replaced by a local trajectory and ``task_complete``
    is optionally appended, producing 15 model dimensions in the normal path.
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
        predict_b2_active=predict_b2_active,
        predict_arm_active=predict_arm_active,
        predict_arm_reset=predict_arm_reset,
        predict_ee_pose=predict_ee_pose,
        predict_gripper=predict_gripper,
    )
    result = action[..., indices].clone()
    b2_start = int(predict_b2_active)
    if representation == "local_trajectory":
        result[..., b2_start : b2_start + 3] = (
            global_pose_to_local_trajectory(global_pose, pad)
            if global_pose is not None
            else integrate_body_twist(action[..., DATASET_B2_TWIST_SLICE], dt, pad)
        )
    elif representation != "velocity":
        raise ValueError(f"Unknown B2 action representation: {representation!r}")
    if not append_completion:
        return result

    if pad is None:
        complete = torch.zeros(action.shape[:-1], dtype=action.dtype, device=action.device)
    else:
        complete = completion_from_padding(pad).to(device=action.device, dtype=action.dtype)
    return torch.cat((result, complete.unsqueeze(-1)), dim=-1)


def decode_b2_action_chunk(
    action: Tensor,
    *,
    dt: float,
    representation: str = "local_trajectory",
    predict_b2_active: bool = False,
    has_completion: bool = True,
) -> Tensor:
    """Replace the compact predicted trajectory with executable body twist."""
    if action.ndim < 2:
        raise ValueError(f"Expected an action chunk, got {tuple(action.shape)}")
    result = action.clone()
    b2_start = int(predict_b2_active)
    if representation == "local_trajectory":
        result[..., b2_start : b2_start + 3] = differentiate_local_trajectory(
            action[..., b2_start : b2_start + 3], dt
        )
    elif representation != "velocity":
        raise ValueError(f"Unknown B2 action representation: {representation!r}")
    return result


def integrate_b2_execution_chunk(
    action: Tensor,
    *,
    dt: float,
    predict_b2_active: bool = False,
    is_pad: Tensor | None = None,
) -> Tensor:
    """Convert a compact executable B2 twist chunk back to trajectory space."""
    if action.ndim < 2:
        raise ValueError(f"Expected a compact executable action chunk, got {tuple(action.shape)}")
    result = action.clone()
    b2_start = int(predict_b2_active)
    result[..., b2_start : b2_start + 3] = integrate_body_twist(
        action[..., b2_start : b2_start + 3], dt, is_pad
    )
    return result


def b2_execution_action_names(
    dataset_names: list[str] | None,
    *,
    representation: str = "velocity",
    predict_b2_active: bool = False,
    predict_arm_active: bool = False,
    predict_arm_reset: bool = True,
    predict_ee_pose: bool = True,
    predict_gripper: bool = True,
    append_completion: bool = True,
) -> list[str] | None:
    del representation
    if dataset_names is None:
        return None
    if len(dataset_names) != DATASET_ACTION_DIM:
        raise ValueError(f"B2 compact action expects the 16D B2+Z1 schema, got {len(dataset_names)} names")
    indices = action_dataset_indices(
        predict_b2_active=predict_b2_active,
        predict_arm_active=predict_arm_active,
        predict_arm_reset=predict_arm_reset,
        predict_ee_pose=predict_ee_pose,
        predict_gripper=predict_gripper,
    )
    names = [dataset_names[i] for i in indices]
    if append_completion:
        names.append(TASK_COMPLETE_NAME)
    return names


def b2_trajectory_action_names(dataset_names: list[str] | None, **kwargs: Any) -> list[str] | None:
    names = b2_execution_action_names(dataset_names, **kwargs)
    if names is None:
        return None
    b2_start = int(bool(kwargs.get("predict_b2_active", False)))
    names[b2_start : b2_start + 3] = list(B2_TRAJECTORY_NAMES)
    return names


def make_b2_trajectory_stats(
    dataset_stats: dict[str, dict[str, Any]] | None,
    *,
    dt: float,
    chunk_size: int,
    representation: str = "local_trajectory",
    predict_b2_active: bool = False,
    predict_arm_active: bool = False,
    predict_arm_reset: bool = True,
    predict_ee_pose: bool = True,
    predict_gripper: bool = True,
    append_completion: bool = True,
    state_indices: tuple[int, ...] | None = None,
) -> dict[str, dict[str, Any]] | None:
    """Build normalization stats for the compact 15D trajectory representation.

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
        predict_b2_active=predict_b2_active,
        predict_arm_active=predict_arm_active,
        predict_arm_reset=predict_arm_reset,
        predict_ee_pose=predict_ee_pose,
        predict_gripper=predict_gripper,
    )

    def converted(name: str, value: Any) -> Any:
        tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1).clone()
        if name == "count":
            return value.clone() if isinstance(value, Tensor) else deepcopy(value)
        if tensor.numel() != 16:
            raise ValueError(f"Action stat {name!r} has {tensor.numel()} dims, expected 16")
        if representation == "local_trajectory" and name in {"q01", "min", "q10"}:
            tensor[DATASET_B2_TWIST_SLICE] = torch.tensor([-xy_scale, -xy_scale, -yaw_scale])
            completion_value = 0.0
        elif representation == "local_trajectory" and name in {"q99", "max", "q90"}:
            tensor[DATASET_B2_TWIST_SLICE] = torch.tensor([xy_scale, xy_scale, yaw_scale])
            completion_value = 1.0
        elif representation == "local_trajectory" and name == "std":
            tensor[DATASET_B2_TWIST_SLICE] = torch.tensor([xy_scale / 2, xy_scale / 2, yaw_scale / 2])
            completion_value = 0.5
        elif representation == "local_trajectory":
            tensor[DATASET_B2_TWIST_SLICE] = 0.0
            completion_value = 0.5 if name == "mean" else 0.0
        else:
            if name in {"q99", "max", "q90"}:
                completion_value = 1.0
            elif name in {"mean", "std"}:
                completion_value = 0.5
            else:
                completion_value = 0.0
        tensor = tensor[list(action_indices)]
        if append_completion:
            tensor = torch.cat((tensor, tensor.new_tensor([completion_value])))
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
