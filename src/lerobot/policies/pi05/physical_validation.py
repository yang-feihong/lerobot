"""Comparable physical-space validation metrics for PI0.5 action chunks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from lerobot.policies.pi05.b2_action_transform import (
    _rot6d_to_matrix,
    ee_pose_delta_to_absolute,
    ee_reference_delta_to_absolute,
    integrate_body_twist_to_pose_delta,
)

_BOOL_ACTION_NAMES = (
    "arm_teleop_inactive",
    "arm_reset",
    "task_complete",
    "task_blocked",
)


def physical_metric_names(action_names: Sequence[str]) -> tuple[str, ...]:
    """Return the physical metrics supported by one resolved action schema."""
    names = set(action_names)
    metrics = [
        "b2_x_mae_m",
        "b2_y_mae_m",
        "b2_yaw_mae_rad",
        "b2_xy_ade_m",
        "b2_xy_fde_m",
        "b2_yaw_fde_rad",
    ]
    if any(name.startswith("height_invariant_ee_") for name in names):
        metrics.extend(
            (
                "ee_translation_ade_m",
                "ee_translation_fde_m",
                "ee_rotation_ade_rad",
                "ee_rotation_fde_rad",
            )
        )
    if "gripper_target" in names:
        metrics.append("gripper_mae_rad")
    metrics.extend(f"{name}_accuracy" for name in _BOOL_ACTION_NAMES if name in names)
    return tuple(metrics)


def _masked_sum_count(values: Tensor, valid: Tensor) -> tuple[float, int]:
    selected = values[valid]
    return float(selected.sum().detach().cpu()), int(selected.numel())


def _last_valid(values: Tensor, valid: Tensor) -> tuple[float, int]:
    lengths = valid.to(dtype=torch.long).sum(dim=1)
    rows = torch.nonzero(lengths > 0, as_tuple=False).flatten()
    if not len(rows):
        return 0.0, 0
    selected = values[rows, lengths[rows] - 1]
    return float(selected.sum().detach().cpu()), int(selected.numel())


def _ee_trajectory(
    actions: Tensor,
    *,
    representation: str,
    rotation_representation: str,
) -> Tensor:
    reference = actions.new_zeros(actions.shape[0], 9)
    reference[:, 0] = 1.0
    reference[:, 4] = 1.0
    if representation == "ee_delta":
        return ee_pose_delta_to_absolute(
            actions,
            reference,
            rotation_representation=rotation_representation,
        )
    if representation == "ee_state_delta":
        return ee_reference_delta_to_absolute(
            actions,
            reference,
            rotation_representation=rotation_representation,
        )
    raise ValueError(f"Unsupported Z1 physical validation representation: {representation!r}")


def _rotation_geodesic(expert_pose: Tensor, predicted_pose: Tensor) -> Tensor:
    expert_rotation = _rot6d_to_matrix(expert_pose[..., :6])
    predicted_rotation = _rot6d_to_matrix(predicted_pose[..., :6])
    relative = predicted_rotation @ expert_rotation.transpose(-1, -2)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cosine)


def physical_action_metric_sums(
    expert: Tensor,
    predicted: Tensor,
    valid: Tensor,
    *,
    action_names: Sequence[str],
    b2_representation: str,
    z1_representation: str,
    ee_rotation_representation: str,
    action_dt_seconds: float,
    ee_valid: Tensor | None = None,
) -> dict[str, tuple[float, int]]:
    """Return additive metric numerators and denominators in physical units.

    Inputs must already be unnormalized into the checkpoint's physical action
    representation.  The additive result can be reduced exactly across batches
    and distributed ranks before division.
    """
    if expert.shape != predicted.shape or expert.ndim != 3:
        raise ValueError(
            "Physical validation expects matching [batch, horizon, action] chunks, got "
            f"expert={tuple(expert.shape)} predicted={tuple(predicted.shape)}"
        )
    if valid.shape != expert.shape[:2]:
        raise ValueError(f"valid shape {tuple(valid.shape)} does not match {tuple(expert.shape[:2])}")
    if expert.shape[-1] != len(action_names):
        raise ValueError("Action names do not match physical action chunk width")

    name_to_dim = {name: index for index, name in enumerate(action_names)}
    b2_names = (
        ("b2_delta_x", "b2_delta_y", "b2_delta_yaw")
        if b2_representation == "pose_delta"
        else ("b2_vx", "b2_vy", "b2_omega_z")
    )
    b2_indices = [name_to_dim[name] for name in b2_names]
    expert_b2 = expert[..., b2_indices]
    predicted_b2 = predicted[..., b2_indices]
    if b2_representation == "velocity":
        expert_b2 = integrate_body_twist_to_pose_delta(expert_b2, action_dt_seconds, ~valid)
        predicted_b2 = integrate_body_twist_to_pose_delta(predicted_b2, action_dt_seconds, ~valid)
    elif b2_representation != "pose_delta":
        raise ValueError(f"Unsupported B2 physical validation representation: {b2_representation!r}")

    b2_error = predicted_b2 - expert_b2
    b2_error[..., 2] = torch.atan2(torch.sin(b2_error[..., 2]), torch.cos(b2_error[..., 2]))
    b2_abs = b2_error.abs()
    b2_xy = torch.linalg.vector_norm(b2_error[..., :2], dim=-1)
    result = {
        "b2_x_mae_m": _masked_sum_count(b2_abs[..., 0], valid),
        "b2_y_mae_m": _masked_sum_count(b2_abs[..., 1], valid),
        "b2_yaw_mae_rad": _masked_sum_count(b2_abs[..., 2], valid),
        "b2_xy_ade_m": _masked_sum_count(b2_xy, valid),
        "b2_xy_fde_m": _last_valid(b2_xy, valid),
        "b2_yaw_fde_rad": _last_valid(b2_abs[..., 2], valid),
    }

    ee_indices = [
        index for index, name in enumerate(action_names) if name.startswith("height_invariant_ee_")
    ]
    if ee_indices:
        expected_width = 6 if ee_rotation_representation == "rotvec" else 9
        if ee_indices != list(range(ee_indices[0], ee_indices[0] + expected_width)):
            raise ValueError(f"EE action block is not contiguous: {ee_indices}")
        effective_ee_valid = valid if ee_valid is None else valid & ee_valid.to(valid.device).bool()
        for gate_name in ("arm_teleop_inactive", "arm_reset"):
            if gate_name in name_to_dim:
                effective_ee_valid = effective_ee_valid & (
                    expert[..., name_to_dim[gate_name]] <= 0.5
                )
        expert_ee = _ee_trajectory(
            expert[..., ee_indices],
            representation=z1_representation,
            rotation_representation=ee_rotation_representation,
        )
        predicted_ee = _ee_trajectory(
            predicted[..., ee_indices],
            representation=z1_representation,
            rotation_representation=ee_rotation_representation,
        )
        translation_error = torch.linalg.vector_norm(
            predicted_ee[..., 6:9] - expert_ee[..., 6:9], dim=-1
        )
        rotation_error = _rotation_geodesic(expert_ee, predicted_ee)
        result.update(
            {
                "ee_translation_ade_m": _masked_sum_count(translation_error, effective_ee_valid),
                "ee_translation_fde_m": _last_valid(translation_error, effective_ee_valid),
                "ee_rotation_ade_rad": _masked_sum_count(rotation_error, effective_ee_valid),
                "ee_rotation_fde_rad": _last_valid(rotation_error, effective_ee_valid),
            }
        )

    if "gripper_target" in name_to_dim:
        gripper_error = (
            predicted[..., name_to_dim["gripper_target"]]
            - expert[..., name_to_dim["gripper_target"]]
        ).abs()
        result["gripper_mae_rad"] = _masked_sum_count(gripper_error, valid)

    for name in _BOOL_ACTION_NAMES:
        if name not in name_to_dim:
            continue
        index = name_to_dim[name]
        correct = (predicted[..., index] > 0.5) == (expert[..., index] > 0.5)
        result[f"{name}_accuracy"] = _masked_sum_count(correct.to(torch.float32), valid)
    return result
