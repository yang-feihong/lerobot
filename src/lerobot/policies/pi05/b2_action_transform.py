#!/usr/bin/env python

"""PI0.5 transforms for B2 velocity/SE(2) pose delta and Z1 EE delta targets."""

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
CONTROL_EE_ACTION_NAMES = tuple(f"control_{name}" for name in DATASET_ACTION_NAMES[5:14])
CONTROL_EXTENDED_DATASET_ACTION_NAMES = (*DATASET_ACTION_NAMES, *CONTROL_EE_ACTION_NAMES)
CONTROL_EXTENDED_DATASET_ACTION_DIM = len(CONTROL_EXTENDED_DATASET_ACTION_NAMES)
HEIGHT_INVARIANT_EE_STATE_NAMES = tuple(
    name.replace("height_invariant_ee_", "height_invariant_ee_state_")
    for name in DATASET_ACTION_NAMES[5:14]
)
RAW_EE_STATE_SLICE = slice(49, 58)
DATASET_B2_TWIST_SLICE = slice(0, 3)
B2_POSE_DELTA_NAMES = ("b2_delta_x", "b2_delta_y", "b2_delta_yaw")
ARM_TELEOP_INACTIVE_NAME = "arm_teleop_inactive"
TASK_COMPLETE_NAME = "task_complete"
EE_DELTA_VALID_KEY = f"{ACTION}_ee_delta_is_valid"
EE_DELTA_ROTVEC_NAMES = (
    "height_invariant_ee_delta_rotvec_x",
    "height_invariant_ee_delta_rotvec_y",
    "height_invariant_ee_delta_rotvec_z",
)


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
        "z1_representation": config.z1_action_representation,
        "ee_delta_rotation_representation": config.ee_delta_rotation_representation,
        "predict_arm_teleop_inactive": config.action_predict_arm_teleop_inactive,
        "predict_arm_reset": config.action_predict_arm_reset,
        "predict_ee_pose": config.action_predict_ee_pose,
        "predict_gripper": config.action_predict_gripper,
        "include_task_complete": config.action_predict_task_complete,
    }


def select_dataset_action_supervision(
    action: Tensor,
    *,
    source: str,
) -> Tensor:
    """Select B2 velocity and Z1 control-target channels from the stored action."""
    if action.ndim < 2 or action.shape[-1] != CONTROL_EXTENDED_DATASET_ACTION_DIM:
        raise ValueError(
            f"Control-action supervision requires {CONTROL_EXTENDED_DATASET_ACTION_DIM}D action, "
            f"got {tuple(action.shape)}"
        )
    if source != "control_action":
        raise ValueError(f"Unsupported supervision source {source!r}; expected 'control_action'")
    result = action[..., :DATASET_ACTION_DIM].clone()
    result[..., 5:14] = action[..., 16:25]
    return result


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


def ee_delta_transition_validity(
    action: Tensor,
    is_pad: Tensor | None = None,
    *,
    supervision_mode: str = "active_only",
) -> Tensor:
    """Return the valid t→t+1 EE-delta transitions for one supervision policy.

    ``active_only`` is the historical gate-aware definition. ``all`` is used
    only with a dataset whose inactive EE targets have already been rewritten
    into a meaningful continuous trajectory.
    """
    if action.ndim < 2 or action.shape[-1] != DATASET_ACTION_DIM or action.shape[-2] < 2:
        raise ValueError(
            f"Expected raw action shape (..., T+1, {DATASET_ACTION_DIM}), got {tuple(action.shape)}"
        )
    if supervision_mode == "active_only":
        source = action[..., :-1, :]
        target = action[..., 1:, :]
        source_active = (source[..., 3] < 0.5) & (source[..., 4] < 0.5)
        target_active = (target[..., 3] < 0.5) & (target[..., 4] < 0.5)
        valid = source_active & target_active
    elif supervision_mode == "all":
        valid = torch.ones(action.shape[:-2] + (action.shape[-2] - 1,), dtype=torch.bool, device=action.device)
    else:
        raise ValueError(
            f"Unknown EE-delta supervision mode {supervision_mode!r}; expected 'active_only' or 'all'"
        )
    if is_pad is not None:
        pad = torch.as_tensor(is_pad, dtype=torch.bool, device=action.device)
        if pad.shape != action.shape[:-1]:
            raise ValueError(f"is_pad shape {tuple(pad.shape)} != action shape {tuple(action.shape[:-1])}")
        valid = valid & ~pad[..., :-1] & ~pad[..., 1:]
    return valid


def ee_delta_sample_validity(
    action: Tensor,
    is_pad: Tensor | None,
    *,
    representation: str,
    supervision_mode: str,
) -> Tensor:
    if representation == "ee_delta":
        return ee_delta_transition_validity(action, is_pad, supervision_mode=supervision_mode)
    if representation != "ee_state_delta":
        raise ValueError(f"Unknown EE delta representation: {representation!r}")
    if supervision_mode == "active_only":
        valid = (action[..., 3] < 0.5) & (action[..., 4] < 0.5)
    elif supervision_mode == "all":
        valid = torch.ones(action.shape[:-1], dtype=torch.bool, device=action.device)
    else:
        raise ValueError(f"Unknown EE-delta supervision mode {supervision_mode!r}")
    if is_pad is not None:
        pad = torch.as_tensor(is_pad, dtype=torch.bool, device=action.device)
        if pad.shape != action.shape[:-1]:
            raise ValueError(f"is_pad shape {tuple(pad.shape)} != action shape {tuple(action.shape[:-1])}")
        valid = valid & ~pad
    return valid


def _sinc_and_cosc(angle: Tensor) -> tuple[Tensor, Tensor]:
    """Return sin(a)/a and (1-cos(a))/a with stable values around zero."""
    sinc = torch.sinc(angle / torch.pi)
    abs_angle = angle.abs()
    angle2 = angle.square()
    cosc_series = angle * (0.5 - angle2 / 24.0 + angle2.square() / 720.0)
    cosc = torch.where(abs_angle < 1e-4, cosc_series, (1.0 - torch.cos(angle)) / angle)
    return sinc, cosc


def integrate_body_twist(twist: Tensor, dt: float, is_pad: Tensor | None = None) -> Tensor:
    """Convert body twist into one local SE(2) increment per control interval."""
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

    return torch.stack((delta_body_x, delta_body_y, delta_yaw), dim=-1)


def integrate_body_twist_to_pose_delta(
    twist: Tensor, dt: float, is_pad: Tensor | None = None
) -> Tensor:
    """Integrate body-frame velocity commands into poses relative to the chunk start."""
    increments = integrate_body_twist(twist, dt, is_pad)
    position = torch.zeros(increments.shape[:-2] + (2,), dtype=twist.dtype, device=twist.device)
    yaw = torch.zeros(increments.shape[:-2], dtype=twist.dtype, device=twist.device)
    outputs = []
    for step in range(increments.shape[-2]):
        dx, dy, dyaw = increments[..., step, :].unbind(dim=-1)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        position = position + torch.stack(
            (cos_yaw * dx - sin_yaw * dy, sin_yaw * dx + cos_yaw * dy), dim=-1
        )
        yaw = yaw + dyaw
        outputs.append(torch.cat((position, yaw.unsqueeze(-1)), dim=-1))
    return torch.stack(outputs, dim=-2)


def _rot6d_to_matrix(rot6d: Tensor) -> Tensor:
    first = torch.nn.functional.normalize(rot6d[..., :3], dim=-1)
    second_raw = rot6d[..., 3:6]
    second = torch.nn.functional.normalize(
        second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first, dim=-1
    )
    third = torch.linalg.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def _matrix_to_rot6d(matrix: Tensor) -> Tensor:
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def _skew(vector: Tensor) -> Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(vector.shape[:-1] + (3, 3))


def _rotvec_to_matrix(rotvec: Tensor) -> Tensor:
    if rotvec.shape[-1] != 3:
        raise ValueError(f"Rotation vectors must have width 3, got {tuple(rotvec.shape)}")
    theta2 = rotvec.square().sum(dim=-1, keepdim=True)
    theta = theta2.sqrt()
    small = theta2 < 1e-8
    sinc = torch.where(
        small,
        1.0 - theta2 / 6.0 + theta2.square() / 120.0,
        torch.sin(theta) / theta.clamp_min(1e-12),
    )
    cosc = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2.square() / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-12),
    )
    skew = _skew(rotvec)
    identity = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).expand(rotvec.shape[:-1] + (3, 3))
    return identity + sinc.unsqueeze(-1) * skew + cosc.unsqueeze(-1) * (skew @ skew)


def _matrix_to_rotvec(matrix: Tensor) -> Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Rotation matrices must end in (3, 3), got {tuple(matrix.shape)}")
    vee = (
        torch.stack(
            (
                matrix[..., 2, 1] - matrix[..., 1, 2],
                matrix[..., 0, 2] - matrix[..., 2, 0],
                matrix[..., 1, 0] - matrix[..., 0, 1],
            ),
            dim=-1,
        )
        * 0.5
    )
    sin_theta = torch.linalg.vector_norm(vee, dim=-1)
    cos_theta = ((matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.atan2(sin_theta, cos_theta)
    scale = torch.where(
        sin_theta > 1e-6,
        theta / sin_theta,
        1.0 + theta.square() / 6.0,
    )
    rotvec = vee * scale.unsqueeze(-1)

    # Adjacent 50 Hz targets should never approach pi, but keep the conversion
    # well-defined for validation and malformed inputs instead of returning zero.
    near_pi = cos_theta < -0.9999
    if bool(near_pi.any()):
        diagonal = matrix.diagonal(dim1=-2, dim2=-1)
        axis = torch.sqrt(((diagonal + 1.0) * 0.5).clamp_min(0.0))
        axis = torch.copysign(
            axis,
            torch.stack((vee[..., 0], vee[..., 1], vee[..., 2]), dim=-1) + 1e-12,
        )
        axis = torch.nn.functional.normalize(axis, dim=-1)
        rotvec = torch.where(near_pi.unsqueeze(-1), axis * theta.unsqueeze(-1), rotvec)
    return rotvec


def absolute_ee_pose_to_delta(
    targets: Tensor,
    reference: Tensor,
    *,
    rotation_representation: str = "rot6d",
) -> Tensor:
    """Encode absolute height-invariant EE targets as parent-frame step deltas."""
    if targets.shape[-1] != 9 or reference.shape[-1] != 9:
        raise ValueError("EE poses must use rot6d(6) + xyz(3)")
    if reference.shape != targets.shape[:-2] + (9,):
        raise ValueError(
            f"EE reference shape {tuple(reference.shape)} is incompatible with {tuple(targets.shape)}"
        )
    previous = torch.cat((reference.unsqueeze(-2), targets[..., :-1, :]), dim=-2)
    target_rotation = _rot6d_to_matrix(targets[..., :6])
    previous_rotation = _rot6d_to_matrix(previous[..., :6])
    delta_rotation = target_rotation @ previous_rotation.transpose(-1, -2)
    delta_position = targets[..., 6:9] - previous[..., 6:9]
    if rotation_representation == "rotvec":
        encoded_rotation = _matrix_to_rotvec(delta_rotation)
    elif rotation_representation == "rot6d":
        encoded_rotation = _matrix_to_rot6d(delta_rotation)
    else:
        raise ValueError(f"Unknown EE delta rotation representation: {rotation_representation!r}")
    return torch.cat((encoded_rotation, delta_position), dim=-1)


def absolute_ee_pose_to_reference_delta(
    targets: Tensor,
    reference: Tensor,
    *,
    rotation_representation: str = "rot6d",
) -> Tensor:
    """Encode every future EE target relative to one inference-time EE state."""
    if targets.shape[-1] != 9 or reference.shape != targets.shape[:-2] + (9,):
        raise ValueError(
            f"EE reference shape {tuple(reference.shape)} is incompatible with {tuple(targets.shape)}"
        )
    target_rotation = _rot6d_to_matrix(targets[..., :6])
    reference_rotation = _rot6d_to_matrix(reference[..., :6]).unsqueeze(-3)
    delta_rotation = target_rotation @ reference_rotation.transpose(-1, -2)
    delta_position = targets[..., 6:9] - reference[..., 6:9].unsqueeze(-2)
    if rotation_representation == "rotvec":
        encoded_rotation = _matrix_to_rotvec(delta_rotation)
    elif rotation_representation == "rot6d":
        encoded_rotation = _matrix_to_rot6d(delta_rotation)
    else:
        raise ValueError(f"Unknown EE delta rotation representation: {rotation_representation!r}")
    return torch.cat((encoded_rotation, delta_position), dim=-1)


def ee_pose_delta_to_absolute(
    deltas: Tensor,
    reference: Tensor,
    *,
    rotation_representation: str = "rot6d",
) -> Tensor:
    """Decode parent-frame EE increments into an absolute target sequence."""
    expected_delta_dim = 6 if rotation_representation == "rotvec" else 9
    if deltas.shape[-1] != expected_delta_dim or reference.shape[-1] != 9:
        raise ValueError(
            f"EE deltas must use {rotation_representation} + xyz ({expected_delta_dim}D) and "
            "the reference must use rot6d + xyz (9D)"
        )
    rotation = _rot6d_to_matrix(reference[..., :6])
    position = reference[..., 6:9]
    outputs = []
    for step in range(deltas.shape[-2]):
        if rotation_representation == "rotvec":
            delta_rotation = _rotvec_to_matrix(deltas[..., step, :3])
            delta_position = deltas[..., step, 3:6]
        elif rotation_representation == "rot6d":
            delta_rotation = _rot6d_to_matrix(deltas[..., step, :6])
            delta_position = deltas[..., step, 6:9]
        else:
            raise ValueError(f"Unknown EE delta rotation representation: {rotation_representation!r}")
        rotation = delta_rotation @ rotation
        position = position + delta_position
        outputs.append(torch.cat((_matrix_to_rot6d(rotation), position), dim=-1))
    return torch.stack(outputs, dim=-2)


def ee_reference_delta_to_absolute(
    deltas: Tensor,
    reference: Tensor,
    *,
    rotation_representation: str = "rot6d",
) -> Tensor:
    """Decode independently anchored EE deltas using one fixed reference pose."""
    expected_delta_dim = 6 if rotation_representation == "rotvec" else 9
    if deltas.shape[-1] != expected_delta_dim or reference.shape != deltas.shape[:-2] + (9,):
        raise ValueError(
            f"EE deltas must be {expected_delta_dim}D and reference shape must match the batch, "
            f"got deltas={tuple(deltas.shape)}, reference={tuple(reference.shape)}"
        )
    if rotation_representation == "rotvec":
        delta_rotation = _rotvec_to_matrix(deltas[..., :3])
        delta_position = deltas[..., 3:6]
    elif rotation_representation == "rot6d":
        delta_rotation = _rot6d_to_matrix(deltas[..., :6])
        delta_position = deltas[..., 6:9]
    else:
        raise ValueError(f"Unknown EE delta rotation representation: {rotation_representation!r}")
    reference_rotation = _rot6d_to_matrix(reference[..., :6]).unsqueeze(-3)
    rotation = delta_rotation @ reference_rotation
    position = reference[..., 6:9].unsqueeze(-2) + delta_position
    return torch.cat((_matrix_to_rot6d(rotation), position), dim=-1)


def se2_increment_to_body_twist(trajectory: Tensor, dt: float) -> Tensor:
    """Convert per-step local SE(2) increments to body twist."""
    if trajectory.ndim < 2 or trajectory.shape[-1] != 3:
        raise ValueError(f"Expected trajectory shape (..., T, 3), got {tuple(trajectory.shape)}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    delta_body_x = trajectory[..., 0]
    delta_body_y = trajectory[..., 1]
    delta_yaw = trajectory[..., 2]

    sinc, cosc = _sinc_and_cosc(delta_yaw)
    determinant = (sinc.square() + cosc.square()).clamp_min(1e-8)
    vx = (sinc * delta_body_x + cosc * delta_body_y) / (dt * determinant)
    vy = (-cosc * delta_body_x + sinc * delta_body_y) / (dt * determinant)
    omega = delta_yaw / dt
    return torch.stack((vx, vy, omega), dim=-1)


def differentiate_pose_delta_trajectory(trajectory: Tensor, dt: float) -> Tensor:
    """Differentiate chunk-start-relative SE(2) poses into body-frame velocity commands."""
    if trajectory.ndim < 2 or trajectory.shape[-1] != 3:
        raise ValueError(f"Expected trajectory shape (..., T, 3), got {tuple(trajectory.shape)}")
    origin = torch.zeros_like(trajectory[..., :1, :])
    previous = torch.cat((origin, trajectory[..., :-1, :]), dim=-2)
    delta_xy = trajectory[..., :2] - previous[..., :2]
    previous_yaw = previous[..., 2]
    cos_yaw = torch.cos(previous_yaw)
    sin_yaw = torch.sin(previous_yaw)
    local_x = cos_yaw * delta_xy[..., 0] + sin_yaw * delta_xy[..., 1]
    local_y = -sin_yaw * delta_xy[..., 0] + cos_yaw * delta_xy[..., 1]
    local_yaw = trajectory[..., 2] - previous_yaw
    local_yaw = torch.atan2(torch.sin(local_yaw), torch.cos(local_yaw))
    increments = torch.stack((local_x, local_y, local_yaw), dim=-1)
    return se2_increment_to_body_twist(increments, dt)


def encode_b2_action_chunk(
    action: Tensor,
    *,
    dt: float,
    is_pad: Tensor | None = None,
    ee_state_anchor: Tensor | None = None,
    representation: str = "velocity",
    z1_representation: str = "ee_delta",
    ee_delta_rotation_representation: str = "rot6d",
    predict_arm_teleop_inactive: bool = True,
    predict_arm_reset: bool = True,
    predict_ee_pose: bool = True,
    predict_gripper: bool = True,
    include_task_complete: bool = True,
) -> Tensor:
    """Convert raw velocity/EE-target controls into the configured model action."""
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

    if z1_representation == "ee_delta":
        if action.shape[-2] < 2:
            raise ValueError("EE delta representation requires current plus future target actions")
        source_action = action[..., :-1, :]
        next_action = action[..., 1:, :]
        if pad is not None:
            pad = pad[..., :-1] | pad[..., 1:]
    elif z1_representation == "ee_state_delta":
        next_action = None
        source_action = action
    else:
        raise ValueError(f"Unknown Z1 action representation: {z1_representation!r}")

    indices = action_dataset_indices(
        predict_arm_teleop_inactive=predict_arm_teleop_inactive,
        predict_arm_reset=predict_arm_reset,
        predict_ee_pose=predict_ee_pose,
        predict_gripper=predict_gripper,
        include_task_complete=include_task_complete,
    )
    result = source_action[..., indices].clone()
    if representation == "pose_delta":
        result[..., :3] = integrate_body_twist_to_pose_delta(
            source_action[..., DATASET_B2_TWIST_SLICE], dt, pad
        )
    elif representation != "velocity":
        raise ValueError(f"Unknown B2 action representation: {representation!r}")
    if z1_representation in {"ee_delta", "ee_state_delta"} and predict_ee_pose:
        selected_names = [DATASET_ACTION_NAMES[index] for index in indices]
        ee_output_indices = [
            index for index, name in enumerate(selected_names) if name.startswith("height_invariant_ee_")
        ]
        if z1_representation == "ee_delta":
            ee_delta = absolute_ee_pose_to_delta(
                next_action[..., 5:14],
                source_action[..., 0, 5:14],
                rotation_representation=ee_delta_rotation_representation,
            )
        else:
            if ee_state_anchor is None:
                raise ValueError("ee_state_delta requires the inference-time height-invariant EE state")
            ee_delta = absolute_ee_pose_to_reference_delta(
                source_action[..., 5:14],
                ee_state_anchor,
                rotation_representation=ee_delta_rotation_representation,
            )
        if ee_delta_rotation_representation == "rotvec":
            first, last = ee_output_indices[0], ee_output_indices[-1] + 1
            result = torch.cat((result[..., :first], ee_delta, result[..., last:]), dim=-1)
        else:
            result[..., ee_output_indices] = ee_delta
    return result


def decode_b2_action_chunk(
    action: Tensor,
    *,
    dt: float,
    representation: str = "velocity",
) -> Tensor:
    """Replace the compact predicted trajectory with executable body twist."""
    if action.ndim < 2:
        raise ValueError(f"Expected an action chunk, got {tuple(action.shape)}")
    result = action.clone()
    if representation == "pose_delta":
        result[..., :3] = differentiate_pose_delta_trajectory(action[..., :3], dt)
    elif representation != "velocity":
        raise ValueError(f"Unknown B2 action representation: {representation!r}")
    return result


def b2_execution_action_names(
    dataset_names: list[str] | None,
    *,
    representation: str = "velocity",
    z1_representation: str = "ee_delta",
    ee_delta_rotation_representation: str = "rot6d",
    predict_arm_teleop_inactive: bool = True,
    predict_arm_reset: bool = True,
    predict_ee_pose: bool = True,
    predict_gripper: bool = True,
    include_task_complete: bool = True,
) -> list[str] | None:
    del representation
    if dataset_names is None:
        return None
    if len(dataset_names) == CONTROL_EXTENDED_DATASET_ACTION_DIM:
        extended_names = tuple(dataset_names)
        if extended_names != CONTROL_EXTENDED_DATASET_ACTION_NAMES:
            raise ValueError(
                "B2 compact action received an unknown 25D extended dataset schema; "
                "expected the control-EE storage layout"
            )
        dataset_names = dataset_names[:DATASET_ACTION_DIM]
    elif len(dataset_names) != DATASET_ACTION_DIM:
        raise ValueError(
            "B2 compact action expects the 16D B2+Z1 schema or a known 25D extended "
            f"storage schema, got {len(dataset_names)} names"
        )
    indices = action_dataset_indices(
        predict_arm_teleop_inactive=predict_arm_teleop_inactive,
        predict_arm_reset=predict_arm_reset,
        predict_ee_pose=predict_ee_pose,
        predict_gripper=predict_gripper,
        include_task_complete=include_task_complete,
    )
    names = [dataset_names[i] for i in indices]
    if z1_representation in {"ee_delta", "ee_state_delta"}:
        ee_indices = [i for i, name in enumerate(names) if name.startswith("height_invariant_ee_")]
        if ee_delta_rotation_representation == "rotvec" and ee_indices:
            first, last = ee_indices[0], ee_indices[-1] + 1
            names = (
                names[:first]
                + list(EE_DELTA_ROTVEC_NAMES)
                + [
                    "height_invariant_ee_delta_x",
                    "height_invariant_ee_delta_y",
                    "height_invariant_ee_delta_z",
                ]
                + names[last:]
            )
        elif ee_delta_rotation_representation == "rot6d":
            names = [name.replace("height_invariant_ee_", "height_invariant_ee_delta_") for name in names]
        else:
            raise ValueError(
                f"Unknown EE delta rotation representation: {ee_delta_rotation_representation!r}"
            )
    else:
        raise ValueError(f"Unknown Z1 action representation: {z1_representation!r}")
    return names


def b2_pose_delta_action_names(dataset_names: list[str] | None, **kwargs: Any) -> list[str] | None:
    names = b2_execution_action_names(dataset_names, **kwargs)
    if names is None:
        return None
    names[:3] = list(B2_POSE_DELTA_NAMES)
    return names


def make_pi05_action_stats(
    dataset_stats: dict[str, dict[str, Any]] | None,
    *,
    transformed_action_stats: dict[str, Any] | None = None,
    dt: float,
    chunk_size: int,
    representation: str = "velocity",
    z1_representation: str = "ee_delta",
    ee_delta_rotation_representation: str = "rot6d",
    predict_arm_teleop_inactive: bool = True,
    predict_arm_reset: bool = True,
    predict_ee_pose: bool = True,
    predict_gripper: bool = True,
    include_task_complete: bool = True,
    state_indices: tuple[int, ...] | None = None,
) -> dict[str, dict[str, Any]] | None:
    """Install exact statistics measured after the configured temporal transform."""
    del dt, chunk_size
    if dataset_stats is None:
        return None
    if representation not in {"velocity", "pose_delta"}:
        raise ValueError(f"Unknown B2 action representation: {representation!r}")
    if z1_representation not in {"ee_delta", "ee_state_delta"}:
        raise ValueError(f"Unknown Z1 action representation: {z1_representation!r}")
    if transformed_action_stats is None:
        raise ValueError(
            "Configured action normalization requires statistics measured after the temporal transform"
        )
    stats = deepcopy(dataset_stats)
    if ACTION not in stats:
        raise ValueError("Dataset statistics do not contain action")
    expected_names = b2_pose_delta_action_names(
        list(DATASET_ACTION_NAMES),
        representation=representation,
        z1_representation=z1_representation,
        ee_delta_rotation_representation=ee_delta_rotation_representation,
        predict_arm_teleop_inactive=predict_arm_teleop_inactive,
        predict_arm_reset=predict_arm_reset,
        predict_ee_pose=predict_ee_pose,
        predict_gripper=predict_gripper,
        include_task_complete=include_task_complete,
    )
    if representation == "velocity":
        expected_names = b2_execution_action_names(
            list(DATASET_ACTION_NAMES),
            representation=representation,
            z1_representation=z1_representation,
            ee_delta_rotation_representation=ee_delta_rotation_representation,
            predict_arm_teleop_inactive=predict_arm_teleop_inactive,
            predict_arm_reset=predict_arm_reset,
            predict_ee_pose=predict_ee_pose,
            predict_gripper=predict_gripper,
            include_task_complete=include_task_complete,
        )
    q01_exact = torch.as_tensor(transformed_action_stats["q01"]).reshape(-1)
    if expected_names is None or q01_exact.numel() != len(expected_names):
        raise ValueError("Exact transformed-action statistics do not match the configured model action schema")
    stats[ACTION] = deepcopy(transformed_action_stats)
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
