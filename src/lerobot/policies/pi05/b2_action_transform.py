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

B2_ACTIVE_INDEX = 0
B2_TWIST_SLICE = slice(1, 4)
B2_TRAJECTORY_NAMES = ("b2_local_x", "b2_local_y", "b2_local_yaw")
TASK_COMPLETE_NAME = "task_complete"


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
    append_completion: bool = True,
) -> Tensor:
    """Replace B2 twist with local trajectory and optionally append completion."""
    if action.ndim < 2 or action.shape[-1] < 4:
        raise ValueError(f"Expected action shape (..., T, A>=4), got {tuple(action.shape)}")
    if append_completion and action.shape[-1] == 17:
        raise ValueError("Action already has 17 dimensions; refusing to append task_complete twice")

    pad = None
    if is_pad is not None:
        pad = torch.as_tensor(is_pad, dtype=torch.bool, device=action.device)
        if pad.shape != action.shape[:-1]:
            raise ValueError(f"is_pad shape {tuple(pad.shape)} != action time shape {tuple(action.shape[:-1])}")

    result = action.clone()
    result[..., B2_TWIST_SLICE] = integrate_body_twist(action[..., B2_TWIST_SLICE], dt, pad)
    if not append_completion:
        return result

    if pad is None:
        complete = torch.zeros(action.shape[:-1], dtype=action.dtype, device=action.device)
    else:
        complete = completion_from_padding(pad).to(device=action.device, dtype=action.dtype)
    return torch.cat((result, complete.unsqueeze(-1)), dim=-1)


def decode_b2_action_chunk(action: Tensor, *, dt: float, has_completion: bool = True) -> Tensor:
    """Replace predicted local trajectory with executable body-frame twist."""
    expected_min_dim = 17 if has_completion else 16
    if action.ndim < 2 or action.shape[-1] < expected_min_dim:
        raise ValueError(
            f"Expected trajectory action shape (..., T, A>={expected_min_dim}), got {tuple(action.shape)}"
        )
    result = action.clone()
    result[..., B2_TWIST_SLICE] = differentiate_local_trajectory(
        action[..., B2_TWIST_SLICE], dt
    )
    return result


def b2_trajectory_action_names(dataset_names: list[str] | None) -> list[str] | None:
    if dataset_names is None:
        return None
    if len(dataset_names) != 16:
        raise ValueError(f"B2 local trajectory expects the 16D B2+Z1 schema, got {len(dataset_names)} names")
    names = list(dataset_names)
    names[B2_TWIST_SLICE] = list(B2_TRAJECTORY_NAMES)
    names.append(TASK_COMPLETE_NAME)
    return names


def make_b2_trajectory_stats(
    dataset_stats: dict[str, dict[str, Any]] | None,
    *,
    dt: float,
    chunk_size: int,
) -> dict[str, dict[str, Any]] | None:
    """Build normalization stats for the derived 17D trajectory representation.

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
    velocity_extent = torch.maximum(q01[1:4].abs(), q99[1:4].abs())
    duration = dt * chunk_size
    xy_scale = max(float(torch.linalg.vector_norm(velocity_extent[:2]).item() * duration), 1e-3)
    yaw_scale = max(float(velocity_extent[2].item() * duration), 1e-3)

    def converted(name: str, value: Any) -> Any:
        tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1).clone()
        if name == "count":
            return value.clone() if isinstance(value, Tensor) else deepcopy(value)
        if tensor.numel() != 16:
            raise ValueError(f"Action stat {name!r} has {tensor.numel()} dims, expected 16")
        if name in {"q01", "min", "q10"}:
            tensor[1:4] = torch.tensor([-xy_scale, -xy_scale, -yaw_scale])
            completion_value = 0.0
        elif name in {"q99", "max", "q90"}:
            tensor[1:4] = torch.tensor([xy_scale, xy_scale, yaw_scale])
            completion_value = 1.0
        elif name == "std":
            tensor[1:4] = torch.tensor([xy_scale / 2, xy_scale / 2, yaw_scale / 2])
            completion_value = 0.5
        else:
            tensor[1:4] = 0.0
            completion_value = 0.5 if name == "mean" else 0.0
        tensor = torch.cat((tensor, tensor.new_tensor([completion_value])))
        if isinstance(value, Tensor):
            return tensor.to(device=value.device, dtype=value.dtype)
        return tensor.numpy()

    stats[ACTION] = {name: converted(name, value) for name, value in action_stats.items()}
    return stats
