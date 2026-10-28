#!/usr/bin/env python
"""Source-agnostic HTTP/NPZ PI0.5 inference: complete inputs to action chunks.

Supports direct request/response and latest-only asynchronous submissions.
Source acquisition, history selection and execution clocks belong to callers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
import traceback
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from threading import Condition, Event, Lock, Thread
from urllib.parse import parse_qs, urlparse
from zipfile import BadZipFile

import cv2
import numpy as np
import torch

from lerobot.configs import FeatureType, NormalizationMode, PreTrainedConfig, RTCAttentionSchedule
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.pi05.b2_action_transform import (
    EE_DELTA_ROTVEC_NAMES,
    _matrix_to_rot6d,
    _rotvec_to_matrix,
    absolute_ee_pose_to_reference_delta,
    action_dataset_indices,
    ee_reference_delta_to_absolute,
)
from lerobot.policies.pi05.processor_pi05 import OBS_ACTION_HISTORY
from lerobot.policies.rtc import RTCConfig
from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from lerobot.types import TransitionKey
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.utils import init_logging

LOG = logging.getLogger("pi05_vla_server")
PROTOCOL_VERSION = 7
B2_POSE_DELTA_CONTROL_MODES = ("differentiate", "se2_feedback")
EXECUTION_ACTION_PREFIX_NAMES = (
    "b2_vx",
    "b2_vy",
    "b2_omega_z",
    "b2_active",
    "arm_active",
    "arm_reset",
)
EE_POSE_ACTION_NAMES = (
    "height_invariant_ee_rot6d_col0_x",
    "height_invariant_ee_rot6d_col0_y",
    "height_invariant_ee_rot6d_col0_z",
    "height_invariant_ee_rot6d_col1_x",
    "height_invariant_ee_rot6d_col1_y",
    "height_invariant_ee_rot6d_col1_z",
    "height_invariant_ee_x",
    "height_invariant_ee_y",
    "height_invariant_ee_z",
)
EE_DELTA_ACTION_NAMES = tuple(
    name.replace("height_invariant_ee_", "height_invariant_ee_delta_") for name in EE_POSE_ACTION_NAMES
)
EE_STATE_NAMES = tuple(
    name.replace("height_invariant_ee_", "height_invariant_ee_state_") for name in EE_POSE_ACTION_NAMES
)
EXECUTION_ACTION_SUFFIX_NAMES = (
    "gripper_target",
    "task_complete",
    "task_blocked",
)
B2_EXECUTION_VELOCITY_NAMES = ("b2_vx", "b2_vy", "b2_omega_z")
B2_OBSERVED_VELOCITY_NAMES = ("b2_body_vx", "b2_body_vy", "b2_body_wz")
B2_GLOBAL_POSE_NAMES = ("b2_position_x", "b2_position_y", "b2_yaw")


def _rtc_reanchor_group_is_well_conditioned(
    normalizer: NormalizerProcessorStep | UnnormalizerProcessorStep,
    action_names: tuple[str, ...],
    group_names: tuple[str, ...],
) -> bool:
    """Whether every action in a coupled pose group has a usable normalization scale."""
    if not hasattr(normalizer, "_tensor_stats"):
        return True
    stats = normalizer._tensor_stats.get("action")  # noqa: SLF001
    if not stats:
        return True
    mode = normalizer.norm_map.get(FeatureType.ACTION, NormalizationMode.IDENTITY)
    if mode == NormalizationMode.IDENTITY:
        return True
    if mode == NormalizationMode.QUANTILES:
        lower, upper = stats.get("q01"), stats.get("q99")
    elif mode == NormalizationMode.QUANTILE10:
        lower, upper = stats.get("q10"), stats.get("q90")
    elif mode == NormalizationMode.MIN_MAX:
        lower, upper = stats.get("min"), stats.get("max")
    elif mode == NormalizationMode.MEAN_STD:
        upper = stats.get("std")
        lower = None if upper is None else torch.zeros_like(upper)
    else:
        raise ValueError(f"Unsupported action normalization mode for RTC: {mode}")
    if lower is None or upper is None:
        raise ValueError(f"Missing action normalization statistics for RTC mode {mode}")
    indices = torch.as_tensor(
        [action_names.index(name) for name in group_names],
        dtype=torch.long,
        device=lower.device,
    )
    scales = torch.abs((upper - lower).index_select(0, indices))
    return bool(torch.all(torch.isfinite(scales) & (scales > normalizer.eps)).item())


def _reanchor_b2_pose_delta(
    trajectory: torch.Tensor,
    old_world_anchor: torch.Tensor,
    new_world_anchor: torch.Tensor,
) -> torch.Tensor:
    """Express old-anchor-relative SE(2) poses in the new inference frame."""
    if trajectory.ndim != 2 or trajectory.shape[-1] != 3:
        raise ValueError(f"Expected B2 pose-delta chunk (T, 3), got {tuple(trajectory.shape)}")
    if old_world_anchor.shape != (3,) or new_world_anchor.shape != (3,):
        raise ValueError("B2 world anchors must both have shape (3,)")
    old_yaw = old_world_anchor[2]
    old_cos, old_sin = torch.cos(old_yaw), torch.sin(old_yaw)
    world_x = old_world_anchor[0] + old_cos * trajectory[:, 0] - old_sin * trajectory[:, 1]
    world_y = old_world_anchor[1] + old_sin * trajectory[:, 0] + old_cos * trajectory[:, 1]
    dx = world_x - new_world_anchor[0]
    dy = world_y - new_world_anchor[1]
    new_yaw = new_world_anchor[2]
    new_cos, new_sin = torch.cos(new_yaw), torch.sin(new_yaw)
    relative_yaw = old_yaw + trajectory[:, 2] - new_yaw
    return torch.stack(
        (
            new_cos * dx + new_sin * dy,
            -new_sin * dx + new_cos * dy,
            torch.atan2(torch.sin(relative_yaw), torch.cos(relative_yaw)),
        ),
        dim=-1,
    )


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}")


CHUNK_SCHEDULING_MODES = ("RTC", "SYNC")


def _rtc_enabled_for_chunk_scheduling(mode: str) -> bool:
    if mode not in CHUNK_SCHEDULING_MODES:
        raise ValueError(f"Unknown chunk scheduling mode: {mode!r}")
    return mode == "RTC"


def _deployment_rtc_config(config: PreTrainedConfig, args: argparse.Namespace) -> RTCConfig:
    """Derive deployment conditioning from checkpoint training metadata."""
    training_config = config.training_rtc_config
    mode = "training" if training_config is not None and training_config.enabled else "inference"
    return RTCConfig(
        enabled=_rtc_enabled_for_chunk_scheduling(args.chunk_scheduling_mode),
        mode=mode,
        prefix_attention_schedule=RTCAttentionSchedule(args.rtc_schedule.upper()),
        max_guidance_weight=args.rtc_max_guidance_weight,
        execution_horizon=args.rtc_execution_horizon,
    )


def execution_action_names(z1_action_representation: str) -> tuple[str, ...]:
    if z1_action_representation in {"ee_delta", "ee_state_delta"}:
        ee_names = EE_DELTA_ACTION_NAMES
    else:
        raise ValueError(f"Unknown Z1 action representation: {z1_action_representation!r}")
    return EXECUTION_ACTION_PREFIX_NAMES + ee_names + EXECUTION_ACTION_SUFFIX_NAMES


def _to_execution_actions(
    actions: torch.Tensor,
    names: tuple[str, ...],
    z1_action_representation: str,
) -> torch.Tensor:
    """Map named checkpoint output into the explicitly represented low-level protocol."""
    if actions.ndim != 2 or actions.shape[-1] != len(names) or len(set(names)) != len(names):
        raise ValueError(f"Invalid named postprocessed actions: shape={tuple(actions.shape)}, names={names}")
    columns = {name: actions[:, index] for index, name in enumerate(names)}
    output_names = execution_action_names(z1_action_representation)
    if z1_action_representation in {"ee_delta", "ee_state_delta"} and set(EE_DELTA_ROTVEC_NAMES).issubset(
        columns
    ):
        rotvec = torch.stack([columns[name] for name in EE_DELTA_ROTVEC_NAMES], dim=-1)
        rot6d = _matrix_to_rot6d(_rotvec_to_matrix(rotvec))
        for index, name in enumerate(EE_DELTA_ACTION_NAMES[:6]):
            columns[name] = rot6d[:, index]
    required = set(output_names) - {
        "b2_active",
        "arm_active",
        "arm_reset",
        "task_complete",
        "task_blocked",
    }
    missing = sorted(required - columns.keys())
    if missing:
        raise ValueError(f"Checkpoint postprocessor is missing executable actions: {missing}")
    if "arm_active" in columns:
        raise ValueError("Schema-v2 checkpoint must not output legacy arm_active")
    if "arm_teleop_inactive" in columns:
        columns["arm_active"] = 1.0 - columns["arm_teleop_inactive"]
    else:
        columns["arm_active"] = torch.ones_like(actions[:, 0])
    columns.setdefault("b2_active", torch.ones_like(actions[:, 0]))
    columns.setdefault("arm_reset", torch.zeros_like(actions[:, 0]))
    columns.setdefault("task_complete", torch.zeros_like(actions[:, 0]))
    columns.setdefault("task_blocked", torch.zeros_like(actions[:, 0]))
    return torch.stack([columns[name] for name in output_names], dim=-1)


def _decode_discrete_actions(
    normalized_actions: torch.Tensor,
    postprocessed_actions: torch.Tensor,
    names: tuple[str, ...],
    *,
    mode: str,
    gripper_target_representation: str,
    gripper_negative_value: float,
    gripper_nonnegative_value: float,
    arm_mode_encoding: str | None = None,
) -> torch.Tensor:
    """Apply the checkpoint's normalized-domain discrete protocol to physical actions."""
    if normalized_actions.shape != postprocessed_actions.shape:
        raise ValueError(
            "Normalized and postprocessed action shapes differ: "
            f"{tuple(normalized_actions.shape)} != {tuple(postprocessed_actions.shape)}"
        )
    if normalized_actions.ndim != 2 or normalized_actions.shape[-1] != len(names):
        raise ValueError(
            f"Invalid named action chunk: shape={tuple(normalized_actions.shape)}, names={names}"
        )
    if mode not in {"continuous_flow", "structured_temporal"}:
        raise ValueError(f"Unsupported discrete action training mode: {mode!r}")
    if arm_mode_encoding not in {None, "paired_flow_channels"}:
        raise ValueError(f"Unsupported arm mode encoding: {arm_mode_encoding!r}")
    decoded = postprocessed_actions.clone()
    indices = {name: index for index, name in enumerate(names)}
    arm_mode_names = ("arm_teleop_inactive", "arm_reset")
    if arm_mode_encoding == "paired_flow_channels":
        if mode != "continuous_flow":
            raise ValueError("paired_flow_channels requires continuous_flow training")
        if not set(arm_mode_names).issubset(indices):
            raise ValueError(f"paired_flow_channels requires both arm mode channels: {names}")
        arm_mode_indices = [indices[name] for name in arm_mode_names]
        normalized_arm_modes = normalized_actions[:, arm_mode_indices]
        prototypes = normalized_arm_modes.new_tensor(((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)))
        state_indices = torch.argmin(
            torch.sum((normalized_arm_modes[:, None, :] - prototypes[None, :, :]) ** 2, dim=-1),
            dim=-1,
        )
        physical_states = decoded.new_tensor(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
        decoded[:, arm_mode_indices] = physical_states[state_indices]
    else:
        for name in arm_mode_names:
            if name in indices:
                index = indices[name]
                decoded[:, index] = (normalized_actions[:, index] > 0).to(decoded.dtype)
    if gripper_target_representation not in {"binary_position", "continuous_position"}:
        raise ValueError(f"Unsupported gripper target representation: {gripper_target_representation!r}")
    if "gripper_target" in indices and gripper_target_representation == "binary_position":
        index = indices["gripper_target"]
        negative_class = normalized_actions[:, index] < 0
        decoded[:, index] = torch.where(
            negative_class,
            decoded.new_tensor(gripper_negative_value),
            decoded.new_tensor(gripper_nonnegative_value),
        )
    if {"task_complete", "task_blocked"}.issubset(indices):
        status_indices = [indices["task_complete"], indices["task_blocked"]]
        normalized_status = normalized_actions[:, status_indices]
        prototypes = normalized_status.new_tensor(((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)))
        state_indices = torch.argmin(
            torch.sum((normalized_status[:, None, :] - prototypes[None, :, :]) ** 2, dim=-1),
            dim=-1,
        )
        terminal = state_indices != 0
        if bool(terminal.any()):
            first = int(torch.nonzero(terminal, as_tuple=False)[0, 0])
            state_indices[first:] = state_indices[first]
        physical_states = decoded.new_tensor(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
        decoded[:, status_indices] = physical_states[state_indices]
    elif "task_complete" in indices:
        index = indices["task_complete"]
        complete = torch.cummax(normalized_actions[:, index] > 0, dim=0).values
        decoded[:, index] = complete.to(decoded.dtype)
    elif "task_blocked" in indices:
        index = indices["task_blocked"]
        blocked = torch.cummax(normalized_actions[:, index] > 0, dim=0).values
        decoded[:, index] = blocked.to(decoded.dtype)
    if mode == "structured_temporal" and {"arm_teleop_inactive", "arm_reset"}.issubset(indices):
        inactive = decoded[:, indices["arm_teleop_inactive"]] > 0.5
        reset = decoded[:, indices["arm_reset"]] > 0.5
        if bool((inactive & reset).any()):
            raise ValueError("Structured arm mode decoded an impossible inactive+reset state")
    return decoded


def _smooth_b2_execution_velocity(
    actions: torch.Tensor,
    names: tuple[str, ...],
    initial_velocity: torch.Tensor,
    *,
    dt: float,
    time_constant_s: float,
    transition_start_step: int = 0,
    prefix_velocity: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a causal first-order low-pass filter to executable B2 velocity only."""
    if actions.ndim != 2 or actions.shape[-1] != len(names):
        raise ValueError(f"Invalid named execution actions: shape={tuple(actions.shape)}, names={names}")
    if len(set(names)) != len(names) or not set(B2_EXECUTION_VELOCITY_NAMES).issubset(names):
        raise ValueError(f"Execution actions do not uniquely contain {B2_EXECUTION_VELOCITY_NAMES}: {names}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if time_constant_s < 0:
        raise ValueError(f"time_constant_s must be non-negative, got {time_constant_s}")
    if transition_start_step < 0:
        raise ValueError(f"transition_start_step must be non-negative, got {transition_start_step}")
    initial_velocity = torch.as_tensor(initial_velocity, dtype=actions.dtype, device=actions.device).reshape(
        -1
    )
    if initial_velocity.shape != (len(B2_EXECUTION_VELOCITY_NAMES),):
        raise ValueError(
            f"initial_velocity must have shape ({len(B2_EXECUTION_VELOCITY_NAMES)},), "
            f"got {tuple(initial_velocity.shape)}"
        )

    velocity_indices = [names.index(name) for name in B2_EXECUTION_VELOCITY_NAMES]
    if prefix_velocity is not None:
        prefix_velocity = torch.as_tensor(prefix_velocity, dtype=actions.dtype, device=actions.device)
        if prefix_velocity.ndim != 2 or prefix_velocity.shape[-1] != len(velocity_indices):
            raise ValueError(
                f"prefix_velocity must have shape (steps, {len(velocity_indices)}), "
                f"got {tuple(prefix_velocity.shape)}"
            )

    smoothed = actions.clone()
    if time_constant_s == 0 or len(actions) == 0:
        return smoothed
    alpha = -math.expm1(-dt / time_constant_s)
    filtered_velocity = initial_velocity
    for step in range(len(smoothed)):
        if step < transition_start_step:
            if prefix_velocity is not None and step < len(prefix_velocity):
                filtered_velocity = prefix_velocity[step]
            smoothed[step, velocity_indices] = filtered_velocity
            continue
        target_velocity = actions[step, velocity_indices]
        filtered_velocity = filtered_velocity + alpha * (target_velocity - filtered_velocity)
        smoothed[step, velocity_indices] = filtered_velocity
    return smoothed


def _b2_velocity_smoothing_transition_step(
    inference_delay_steps: int,
    previous_velocity_prefix: torch.Tensor | None,
) -> int:
    if inference_delay_steps < 0:
        raise ValueError(f"inference_delay_steps must be non-negative, got {inference_delay_steps}")
    return inference_delay_steps if previous_velocity_prefix is not None else 0


def _apply_completion_stop(actions: torch.Tensor, names: tuple[str, ...]) -> torch.Tensor:
    """Make the completion row and its suffix non-actuating while retaining the stop signal."""
    if "task_complete" not in names:
        return actions
    stopped = actions.clone()
    complete_index = names.index("task_complete")
    complete = torch.cummax(stopped[:, complete_index] > 0.5, dim=0).values
    stopped[:, complete_index] = complete.to(stopped.dtype)
    for name in (*B2_EXECUTION_VELOCITY_NAMES, "b2_active", "arm_active", "arm_reset"):
        if name in names:
            stopped[complete, names.index(name)] = 0.0
    return stopped


@dataclass(frozen=True)
class CheckpointContract:
    """Versioned checkpoint semantics validated before policy construction."""

    model_action_representation: str
    z1_action_representation: str
    ee_delta_rotation_representation: str
    metadata_version: int
    state_dim: int
    camera_keys: tuple[str, ...]
    control_frequency_hz: float
    metadata_path: Path
    source_state_names: tuple[str, ...]
    selected_state_names: tuple[str, ...]
    source_action_names: tuple[str, ...]
    discrete_action_training_mode: str
    arm_mode_encoding: str | None
    ee_target_dataset_semantics: str
    ee_delta_supervision_mode: str
    gripper_target_representation: str
    gripper_negative_value: float
    gripper_nonnegative_value: float


def _feature_shape(feature: object) -> tuple[int, ...]:
    shape = feature.get("shape") if isinstance(feature, dict) else getattr(feature, "shape", None)
    if shape is None:
        raise ValueError(f"Feature has no shape: {feature!r}")
    return tuple(int(value) for value in shape)


def _camera_bindings(camera_keys: tuple[str, ...]) -> tuple[str, str]:
    wrist = tuple(key for key in camera_keys if "wrist" in key.lower())
    base = tuple(key for key in camera_keys if key.rsplit(".", 1)[-1].lower().startswith("base"))
    if len(base) != 1 or len(wrist) != 1:
        raise ValueError(f"Cannot bind checkpoint cameras to base/wrist packets: {camera_keys}")
    return base[0], wrist[0]


def _resolve_checkpoint_action_representations(
    metadata_version: int,
    action_metadata: dict[str, object],
    raw_config: dict[str, object],
) -> tuple[str, str]:
    b2_representation = str(action_metadata["representation"])
    del raw_config
    if metadata_version != 10:
        raise ValueError(f"Deployment metadata version {metadata_version} uses a retired contract")
    z1_representation = str(action_metadata["z1_representation"])
    if b2_representation not in {"pose_delta", "velocity"}:
        raise ValueError(f"Unsupported B2 deployment action representation: {b2_representation!r}")
    if z1_representation not in {"ee_delta", "ee_state_delta"}:
        raise ValueError(f"Unsupported Z1 deployment action representation: {z1_representation!r}")
    return b2_representation, z1_representation


def _load_checkpoint_contract(
    policy_path: Path,
    config: PreTrainedConfig,
    low_level_hz: float,
) -> CheckpointContract:
    """Validate the current checkpoint-local deployment contract."""
    raw_config = json.loads((policy_path / "config.json").read_text(encoding="utf-8"))
    if not raw_config.get("io_schema_resolved", False):
        raise ValueError(
            "This server only supports the versioned PI0.5 I/O schema; "
            "legacy checkpoints must use the temporary compatibility runner"
        )
    metadata_path = policy_path / "pi05_deployment_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Checkpoint is missing deployment metadata: {metadata_path}")
    saved = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_version = saved.get("version")
    if saved.get("format") != "lerobot.pi05.deployment" or metadata_version != 10:
        raise ValueError("Checkpoint deployment metadata must use lerobot.pi05.deployment version 10")
    if not config.io_schema_resolved:
        raise ValueError("Checkpoint contains unresolved PI0.5 I/O metadata")
    config.restore_camera_metadata(saved)

    action = saved["action"]
    observation = saved["observation"]
    timing = saved["timing"]
    b2_action_representation, z1_action_representation = _resolve_checkpoint_action_representations(
        int(metadata_version), action, raw_config
    )
    expected = config.deployment_metadata()
    if "reset_preset_selector" not in action and not config.action_predict_reset_preset:
        expected["action"].pop("reset_preset_selector")
    if "arm_mode_encoding" not in action:
        expected["action"].pop("arm_mode_encoding", None)
    saved_predict = action.get("predict", {})
    expected_predict = expected["action"].get("predict", {})
    if (
        isinstance(saved_predict, dict)
        and "task_blocked" not in saved_predict
        and expected_predict.get("task_blocked") is False
    ):
        expected_predict.pop("task_blocked", None)
        expected["action"].get("boolean_decoding", {}).get("true_side", {}).pop("task_blocked", None)
        expected["action"].get("boolean_decoding", {}).get("output_values", {}).pop("task_blocked", None)
        expected["action"].pop("task_blocked_semantics", None)
    if saved != expected:
        raise ValueError("pi05_deployment_metadata.json disagrees with checkpoint config.json")
    discrete_mode = str(action.get("discrete_training_mode", "continuous_flow"))
    if discrete_mode not in {"continuous_flow", "structured_temporal"}:
        raise ValueError(f"Unsupported discrete action training mode: {discrete_mode!r}")
    config_mode = str(raw_config.get("discrete_action_training_mode", "continuous_flow"))
    if config_mode != discrete_mode:
        raise ValueError("Discrete action training mode disagrees between metadata and config.json")
    arm_mode_metadata = action.get("arm_mode_encoding")
    arm_mode_encoding = None
    if arm_mode_metadata is not None:
        expected_arm_mode_metadata = {
            "representation": "paired_flow_channels",
            "channels": ["arm_teleop_inactive", "arm_reset"],
            "states": {
                "teleop": {"arm_teleop_inactive": 0, "arm_reset": 0},
                "inactive": {"arm_teleop_inactive": 1, "arm_reset": 0},
                "reset": {"arm_teleop_inactive": 0, "arm_reset": 1},
            },
            "invalid": {"arm_teleop_inactive": 1, "arm_reset": 1},
            "training": "flow_matching_with_class_balanced_channel_loss",
        }
        if arm_mode_metadata != expected_arm_mode_metadata:
            raise ValueError(f"Unsupported arm mode encoding: {arm_mode_metadata!r}")
        arm_mode_encoding = "paired_flow_channels"
    gripper_target_representation = str(action.get("gripper_target_representation", "binary_position"))
    if gripper_target_representation != config.gripper_target_representation:
        raise ValueError("Gripper target representation disagrees between metadata and config.json")
    decoding = action.get("boolean_decoding", {})
    output_values = decoding.get("output_values", decoding.get("physical_values", {}))
    gripper_values = output_values.get("gripper_target", {})
    if raw_config["b2_action_representation"] != b2_action_representation:
        raise ValueError("B2 action representation disagrees between metadata and config.json")
    source_action_names = tuple(str(name) for name in action["source_names"])
    if "arm_teleop_inactive" not in source_action_names or "arm_active" in source_action_names:
        raise ValueError(
            "Schema-v2 action source must contain arm_teleop_inactive and must not contain legacy arm_active"
        )
    source_state_names = tuple(str(name) for name in observation["state"]["source_names"])
    state_names = tuple(str(name) for name in observation["state"]["selected_names"])
    if not source_state_names or len(set(source_state_names)) != len(source_state_names):
        raise ValueError(f"Checkpoint has invalid source state names: {source_state_names}")
    if not set(state_names).issubset(source_state_names):
        raise ValueError("Checkpoint selected state fields are not a subset of its source state fields")
    if z1_action_representation == "ee_state_delta":
        missing_anchor_state = tuple(name for name in EE_STATE_NAMES if name not in source_state_names)
        if missing_anchor_state:
            raise ValueError(
                f"ee_state_delta checkpoint is missing source EE anchor fields: {missing_anchor_state}"
            )
    state_dim = _feature_shape(raw_config["input_features"][OBS_STATE])[-1]
    if state_dim != len(state_names):
        raise ValueError(
            f"Checkpoint state feature shape {state_dim} disagrees with metadata names {len(state_names)}"
        )
    camera_keys = tuple(str(key) for key in observation["camera_keys_in_model_order"])
    if len(camera_keys) != 2 or len(set(camera_keys)) != 2:
        raise ValueError(f"Checkpoint must define two distinct cameras, got {camera_keys}")
    control_hz = float(timing["control_frequency_hz"])
    if not math.isclose(control_hz, low_level_hz, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"Checkpoint control frequency {control_hz} Hz does not match low level {low_level_hz} Hz"
        )
    return CheckpointContract(
        model_action_representation=b2_action_representation,
        z1_action_representation=str(z1_action_representation),
        ee_delta_rotation_representation=str(action.get("ee_delta_rotation_representation") or "rot6d"),
        metadata_version=int(metadata_version),
        state_dim=state_dim,
        camera_keys=camera_keys,
        control_frequency_hz=control_hz,
        metadata_path=metadata_path,
        source_state_names=source_state_names,
        selected_state_names=state_names,
        source_action_names=source_action_names,
        discrete_action_training_mode=discrete_mode,
        arm_mode_encoding=arm_mode_encoding,
        ee_target_dataset_semantics=str(action.get("ee_target_dataset_semantics", "legacy_raw")),
        ee_delta_supervision_mode=str(action.get("ee_delta_supervision_mode", "active_only")),
        gripper_target_representation=gripper_target_representation,
        gripper_negative_value=float(
            gripper_values.get("normalized_negative", gripper_values.get("true", -1.0471976))
        ),
        gripper_nonnegative_value=float(
            gripper_values.get("normalized_nonnegative", gripper_values.get("false", 0.0))
        ),
    )


def _resolve_policy_path(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    if (path / "config.json").exists():
        return path
    if (path / "pretrained_model" / "config.json").exists():
        return path / "pretrained_model"
    checkpoint_root = path / "checkpoints"
    if checkpoint_root.exists():
        steps = sorted(
            (item for item in checkpoint_root.iterdir() if item.is_dir() and item.name.isdigit()),
            key=lambda item: int(item.name),
        )
        if steps and (steps[-1] / "pretrained_model" / "config.json").exists():
            return steps[-1] / "pretrained_model"
    raise FileNotFoundError(f"Cannot resolve a policy checkpoint from {path}")


def _checkpoint_hot_swap_signature(policy_path: Path) -> str:
    """Return the inference-compatibility signature for PEFT weight replacement.

    Checkpoints trained from the same inference architecture may legitimately
    contain different training priors or newer explicit copies of old default
    values.  Those values must not force a full model reconstruction.  The
    pre/post processors are rebuilt on every hot swap, so their training-only
    masking options are likewise outside the compatibility contract.
    """
    documents = {}
    for name in (
        "config.json",
        "adapter_config.json",
        "pi05_deployment_metadata.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        path = policy_path / name
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint is missing hot-swap metadata: {path}")
        documents[name] = json.loads(path.read_text(encoding="utf-8"))
    # Use a structural allowlist.  A denylist of training fields is inherently
    # unsafe here: every newly recorded loss/sampling key would spuriously make
    # an otherwise compatible checkpoint rebuild the multi-billion-parameter
    # base model.  Unknown config and deployment-metadata keys are therefore
    # non-structural by default.
    structural_config_fields = {
        "type",
        "dtype",
        "input_features",
        "output_features",
        "normalization_mapping",
        "chunk_size",
        "n_action_steps",
        "n_obs_steps",
        "max_state_dim",
        "max_action_dim",
        "image_resolution",
        "empty_cameras",
        "paligemma_variant",
        "action_expert_variant",
        "tokenizer_max_length",
        "max_period",
        "min_period",
        "use_peft",
        "state_action_encoding",
        "io_schema_resolved",
        "action_feature_names",
        "dataset_action_feature_names",
        "dataset_camera_keys",
        "dataset_state_feature_names",
        "resolved_state_feature_names",
        "state_feature_indices",
        "action_dt_seconds",
        "control_frequency_hz",
        "dataset_frequency_hz",
        "b2_action_representation",
        "b2_global_pose_state_indices",
        "z1_action_representation",
        "ee_delta_rotation_representation",
        "ee_state_anchor_indices",
        "ee_supervision_source",
        "ee_target_dataset_semantics",
        "gripper_target_representation",
        "relative_exclude_joints",
        "use_relative_actions",
        "action_gripper_negative_value",
        "action_gripper_nonnegative_value",
        "action_gripper_target_true_side",
        "action_predict_arm_reset",
        "action_predict_reset_preset",
        "reset_preset_catalog",
        "action_predict_arm_teleop_inactive",
        "action_predict_ee_pose",
        "action_predict_gripper",
        "action_predict_task_blocked",
        "action_predict_task_complete",
        "discrete_action_training_mode",
        "action_history_enabled",
        "state_num_frames",
        "state_history_frame_interval_seconds",
        "state_history_frame_stride",
        "state_use_arm_gripper_feedback",
        "state_use_arm_joint_positions",
        "state_use_arm_joint_velocities",
        "state_use_b2_angular_velocity",
        "state_use_b2_joint_positions",
        "state_use_b2_joint_velocities",
        "state_use_b2_linear_velocity",
        "state_use_b2_trunk_pose",
        "mem_vit_enabled",
        "mem_vit_finetune_mode",
        "mem_vit_num_frames",
        "mem_vit_min_num_frames",
        "mem_vit_max_num_frames",
        "mem_vit_frame_interval_seconds",
        "mem_vit_frame_stride",
        "mem_vit_temporal_every",
        "mem_vit_use_original_for_k1",
        "mem_vit_base_weights_embedded",
        "mem_vit_embedded_tensor_count",
        "mem_vit_artifact_id",
        "mem_vit_sha256",
        "policy_base_artifact_id",
        "policy_base_sha256",
        "policy_tokenizer_artifact_id",
        "policy_tokenizer_sha256",
        "rtc_config",
        "training_rtc_config",
    }
    config = documents["config.json"]
    config["action_predict_reset_preset"] = config.get("action_predict_reset_preset", False)
    if not config["action_predict_reset_preset"]:
        config.pop("reset_preset_catalog", None)
    documents["config.json"] = {key: config[key] for key in sorted(structural_config_fields) if key in config}

    # Adapter metadata is itself structural, except for filesystem/revision and
    # initialization bookkeeping which does not change the saved tensor layout.
    adapter_config = documents["adapter_config.json"]
    for non_structural_adapter_field in (
        "base_model_name_or_path",
        "revision",
        "inference_mode",
        "init_lora_weights",
        "peft_version",
    ):
        adapter_config.pop(non_structural_adapter_field, None)

    # Deployment metadata is descriptive and is validated when the checkpoint
    # contract is loaded.  It must never decide architecture compatibility;
    # the structural config and processor topology below are authoritative.
    documents.pop("pi05_deployment_metadata.json", None)

    for processor_name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        for step in documents[processor_name].get("steps", ()):
            step.get("config", {}).pop("supervise_terminal_static_padding", None)
    canonical = json.dumps(documents, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _decode_jpeg(encoded: np.ndarray, name: str) -> np.ndarray:
    image = cv2.imdecode(np.asarray(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode {name} JPEG")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def decode_observation_packet(payload: bytes, max_state_dim: int = 4096) -> ObservationPacket:
    with np.load(BytesIO(payload), allow_pickle=False) as packet:
        version = int(packet["protocol_version"].item())
        if version != PROTOCOL_VERSION:
            raise ValueError(f"Protocol version {version} != {PROTOCOL_VERSION}")
        state = np.asarray(packet["state"], dtype=np.float32).reshape(-1)
        state_names = tuple(str(name) for name in packet["state_names"].tolist())
        if state.size == 0 or state.size > max_state_dim or not np.isfinite(state).all():
            raise ValueError(f"Invalid state vector shape/content: {state.shape}")
        if len(state_names) != state.size or len(set(state_names)) != len(state_names):
            raise ValueError(f"Invalid named state vector: shape={state.shape}, names={state_names}")
        executed_ee_target = np.asarray(packet["executed_ee_target"], dtype=np.float32).reshape(-1)
        executed_ee_target_names = tuple(str(name) for name in packet["executed_ee_target_names"].tolist())
        if (
            executed_ee_target.shape != (len(EE_POSE_ACTION_NAMES),)
            or executed_ee_target_names != EE_POSE_ACTION_NAMES
            or not np.isfinite(executed_ee_target).all()
        ):
            raise ValueError(
                "Invalid executed EE target feedback: "
                f"shape={executed_ee_target.shape}, names={executed_ee_target_names}"
            )
        actual_ee_state = np.asarray(packet["actual_ee_state"], dtype=np.float32).reshape(-1)
        if actual_ee_state.shape != (len(EE_POSE_ACTION_NAMES),) or not np.isfinite(actual_ee_state).all():
            raise ValueError(f"Invalid actual EE state anchor: shape={actual_ee_state.shape}")
        state_index = {name: index for index, name in enumerate(state_names)}
        if all(name in state_index for name in EE_STATE_NAMES):
            state_anchor = state[[state_index[name] for name in EE_STATE_NAMES]]
            if not np.array_equal(actual_ee_state, state_anchor):
                raise ValueError("Actual EE anchor disagrees with the named observation.state channels")
        if any(key.startswith("history_") for key in packet.files):
            raise ValueError("Raw sensor histories are not model inputs; assemble them in the deployment adapter")
        prepared_state = np.asarray(packet["model_state"], dtype=np.float32)
        prepared_pad = np.asarray(packet["model_state_is_pad"])
        if (prepared_state.ndim != 2 or not np.isfinite(prepared_state).all()
                or prepared_pad.dtype != np.bool_ or prepared_pad.shape != prepared_state.shape[:1]):
            raise ValueError("Invalid prepared state window")
        sim_step = int(packet["sim_step"].item())
        active_sequence = int(packet["active_sequence"].item())
        active_index = int(packet["active_index"].item())
        delay_steps = int(packet["rtc_inference_delay_steps"].item())
        source_rate = float(packet["source_steps_per_wall_second"].item())
        if delay_steps < 0 or not math.isfinite(source_rate) or source_rate <= 0:
            raise ValueError("Invalid adapter-supplied RTC delay/source rate")
        scheduling_mode = str(packet["chunk_scheduling_mode"].item()) if "chunk_scheduling_mode" in packet else None
        if scheduling_mode is not None and scheduling_mode not in CHUNK_SCHEDULING_MODES:
            raise ValueError("Invalid request scheduling mode")
        return ObservationPacket(
            control_epoch=int(packet["control_epoch"].item()),
            sim_step=sim_step,
            active_sequence=active_sequence,
            active_index=active_index,
            task=str(packet["task"].item()),
            state=state,
            state_names=state_names,
            base_rgb=_decode_jpeg(packet["base_jpeg"], "base"),
            wrist_rgb=_decode_jpeg(packet["wrist_jpeg"], "wrist"),
            base_jpeg=np.asarray(packet["base_jpeg"], dtype=np.uint8),
            wrist_jpeg=np.asarray(packet["wrist_jpeg"], dtype=np.uint8),
            capture_started_ns=int(packet["capture_started_ns"].item())
            if "capture_started_ns" in packet
            else -1,
            capture_finished_ns=int(packet["capture_finished_ns"].item())
            if "capture_finished_ns" in packet
            else -1,
            encoded_ns=int(packet["encoded_ns"].item()) if "encoded_ns" in packet else -1,
            executed_ee_target=executed_ee_target,
            executed_ee_target_names=executed_ee_target_names,
            actual_ee_state=actual_ee_state,
            input_schema=str(packet["model_input_schema"].item()),
            prepared_state=prepared_state,
            prepared_state_is_pad=prepared_pad,
            prepared_action_history=np.asarray(packet["model_action_history"], dtype=np.float32)
            if "model_action_history" in packet else None,
            prepared_action_history_is_pad=np.asarray(packet["model_action_history_is_pad"])
            if "model_action_history_is_pad" in packet else None,
            inference_delay_steps=delay_steps,
            source_steps_per_wall_second=source_rate,
            diagnostic_flow_seed=(
                int(packet["diagnostic_flow_seed"].item()) if "diagnostic_flow_seed" in packet else None
            ),
            server_received_ns=time.monotonic_ns(),
            model_image_history=np.asarray(packet["model_image_history"], dtype=np.uint8).tobytes(),
            scheduling_mode=scheduling_mode,
        )


def encode_action_packet(record: ActionRecord) -> bytes:
    output = BytesIO()
    np.savez(
        output,
        model_image_history=np.frombuffer(record.model_image_history or b"", dtype=np.uint8),
        protocol_version=np.asarray(PROTOCOL_VERSION, dtype=np.int64),
        sequence=np.asarray(record.sequence, dtype=np.int64),
        source_step=np.asarray(record.source_step, dtype=np.int64),
        inference_seconds=np.asarray(record.inference_seconds, dtype=np.float64),
        action_names=np.asarray(record.action_names, dtype=np.str_),
        reset_preset_catalog=np.asarray(json.dumps(record.reset_preset_catalog), dtype=np.str_),
        model_action_names=np.asarray(record.model_action_names, dtype=np.str_),
        control_epoch=np.asarray(record.control_epoch, dtype=np.int64),
        b2_execution_mode=np.asarray(record.b2_execution_mode, dtype=np.str_),
        z1_action_representation=np.asarray(record.z1_action_representation, dtype=np.str_),
        b2_anchor_kind=np.asarray(record.b2_anchor_kind, dtype=np.str_),
        b2_anchor=record.b2_anchor,
        b2_reference=(
            record.b2_reference.numpy().astype(np.float32, copy=False)
            if record.b2_reference is not None
            else np.empty((0, 3), dtype=np.float32)
        ),
        ee_anchor_kind=np.asarray(record.ee_anchor_kind, dtype=np.str_),
        ee_anchor=record.ee_anchor,
        original_actions=record.original.numpy().astype(np.float32, copy=False),
        unsmoothed_execution_actions=record.unsmoothed.numpy().astype(np.float32, copy=False),
        processed_actions=record.processed.numpy().astype(np.float32, copy=False),
        inference_started_ns=np.asarray(record.inference_started_ns, dtype=np.int64),
        inference_finished_ns=np.asarray(record.inference_finished_ns, dtype=np.int64),
        observation_received_ns=np.asarray(record.observation_received_ns, dtype=np.int64),
        preprocess_seconds=np.asarray(record.preprocess_seconds, dtype=np.float64),
        predict_seconds=np.asarray(record.predict_seconds, dtype=np.float64),
        postprocess_seconds=np.asarray(record.postprocess_seconds, dtype=np.float64),
        inference_delay_steps=np.asarray(record.inference_delay_steps, dtype=np.int64),
        rtc_mode=np.asarray(record.rtc_mode, dtype=np.str_),
        rtc_conditioning_delay_steps=np.asarray(record.rtc_conditioning_delay_steps, dtype=np.int64),
        velocity_smoothing_transition_step=np.asarray(
            record.velocity_smoothing_transition_step, dtype=np.int64
        ),
        sim_steps_per_wall_second=np.asarray(record.sim_steps_per_wall_second, dtype=np.float64),
    )
    return output.getvalue()


@dataclass(frozen=True)
class ImagePacket:
    base_jpeg: np.ndarray
    wrist_jpeg: np.ndarray
    base_capture_ns: int
    wrist_capture_ns: int
    capture_ns: int
    encoded_ns: int
    server_received_ns: int
    sim_step: int | None


@dataclass(frozen=True)
class ObservationPacket:
    control_epoch: int
    sim_step: int
    active_sequence: int
    active_index: int
    task: str
    state: np.ndarray
    state_names: tuple[str, ...]
    base_rgb: np.ndarray
    wrist_rgb: np.ndarray
    base_jpeg: np.ndarray
    wrist_jpeg: np.ndarray
    capture_started_ns: int
    capture_finished_ns: int
    encoded_ns: int
    executed_ee_target: np.ndarray
    executed_ee_target_names: tuple[str, ...]
    actual_ee_state: np.ndarray
    input_schema: str
    prepared_state: np.ndarray
    prepared_state_is_pad: np.ndarray
    prepared_action_history: np.ndarray | None
    prepared_action_history_is_pad: np.ndarray | None
    inference_delay_steps: int
    source_steps_per_wall_second: float
    # Optional and intentionally absent from production packets. It lets
    # open-loop counterfactuals reuse identical flow noise, so a modality swap
    # is not confounded by a fresh stochastic sample.
    diagnostic_flow_seed: int | None
    server_received_ns: int
    model_image_history: bytes
    scheduling_mode: str | None = None


@dataclass(frozen=True)
class ActionRecord:
    sequence: int
    source_step: int
    inference_seconds: float
    action_names: tuple[str, ...]
    model_action_names: tuple[str, ...]
    control_epoch: int
    b2_execution_mode: str
    z1_action_representation: str
    b2_anchor_kind: str
    b2_anchor: np.ndarray
    b2_reference: torch.Tensor | None
    ee_anchor_kind: str
    ee_anchor: np.ndarray
    original: torch.Tensor
    unsmoothed: torch.Tensor
    processed: torch.Tensor
    inference_started_ns: int
    inference_finished_ns: int
    preprocess_seconds: float
    predict_seconds: float
    postprocess_seconds: float
    observation_received_ns: int
    inference_delay_steps: int
    velocity_smoothing_transition_step: int
    sim_steps_per_wall_second: float
    rtc_mode: str = "inference"
    rtc_conditioning_delay_steps: int = 0
    model_image_history: bytes | None = None
    reset_preset_catalog: list[dict] | None = None


class _RecordedImageBatch(dict):
    """Keep recording data outside the dictionary passed to the model processors."""

    model_image_history: bytes


def _decode_model_image_history(payload, *, source_step, count, base_key, wrist_key):
    with np.load(BytesIO(payload), allow_pickle=False) as data:
        if int(data["format_version"]) != 1 or int(data["source_step"]) != source_step:
            raise ValueError("Recorded model image history version/source_step mismatch")
        if str(data["base_camera_key"].item()) != base_key or str(data["wrist_camera_key"].item()) != wrist_key:
            raise ValueError("Recorded model image history physical camera roles mismatch")
        pad = np.asarray(data["is_pad"])
        if pad.dtype != np.bool_ or pad.ndim != 1 or len(pad) != count:
            raise ValueError("Recorded model image history frame count/padding mismatch")
        for field in ("capture_ns", "base_capture_ns", "wrist_capture_ns", "encoded_ns", "server_received_ns", "sim_step"):
            if data[field].shape != (count,):
                raise ValueError(f"Recorded model image history invalid {field}")
        records = []
        for index in range(count):
            images = {}
            for role in ("base", "wrist"):
                array = data[f"{role}_jpeg_{index}"]
                if array.dtype != np.uint8 or array.ndim != 1 or not array.size:
                    raise ValueError("Recorded model image history invalid JPEG bytes")
                images[f"{role}_jpeg"] = array.copy()
            values = {field: int(data[field][index]) for field in ("capture_ns", "base_capture_ns", "wrist_capture_ns", "encoded_ns", "server_received_ns", "sim_step")}
            values["sim_step"] = None if values["sim_step"] < 0 else values["sim_step"]
            records.append(ImagePacket(**images, **values))
    return records, torch.from_numpy(pad.copy()).unsqueeze(0)


class RolloutRecorder:
    """Write replayable VLA inputs/outputs and structured latency events."""

    def __init__(self, root: str):
        self.root = Path(root) if root else None
        self._lock = Lock()
        if self.root is not None:
            (self.root / "vla" / "observations").mkdir(parents=True, exist_ok=True)
            (self.root / "vla" / "actions").mkdir(parents=True, exist_ok=True)
            self.events_path = self.root / "vla" / "events.jsonl"

    def event(self, kind: str, **values: object) -> None:
        if self.root is None:
            return
        record = {
            "event": kind,
            "wall_time_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            **values,
        }
        with self._lock, self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    def observation(self, packet: ObservationPacket) -> None:
        if self.root is None:
            return
        path = self.root / "vla" / "observations" / f"{packet.sim_step:06d}.npz"
        with self._lock:
            np.savez_compressed(
                path,
                protocol_version=np.asarray(PROTOCOL_VERSION),
                control_epoch=np.asarray(packet.control_epoch),
                sim_step=np.asarray(packet.sim_step),
                active_sequence=np.asarray(packet.active_sequence),
                active_index=np.asarray(packet.active_index),
                task=np.asarray(packet.task),
                state=packet.state,
                state_names=np.asarray(packet.state_names, dtype=np.str_),
                base_jpeg=packet.base_jpeg,
                wrist_jpeg=packet.wrist_jpeg,
                capture_started_ns=np.asarray(packet.capture_started_ns),
                capture_finished_ns=np.asarray(packet.capture_finished_ns),
                encoded_ns=np.asarray(packet.encoded_ns),
                executed_ee_target=packet.executed_ee_target,
                executed_ee_target_names=np.asarray(packet.executed_ee_target_names, dtype=np.str_),
                actual_ee_state=packet.actual_ee_state,
                model_input_schema=np.asarray(packet.input_schema),
                model_state=packet.prepared_state,
                model_state_is_pad=packet.prepared_state_is_pad,
                model_action_history=packet.prepared_action_history if packet.prepared_action_history is not None else np.empty((0, 0), dtype=np.float32),
                model_action_history_is_pad=packet.prepared_action_history_is_pad if packet.prepared_action_history_is_pad is not None else np.empty(0, dtype=np.bool_),
                model_image_history=np.frombuffer(packet.model_image_history, dtype=np.uint8),
                server_received_ns=np.asarray(packet.server_received_ns),
            )
            prompt_path = self.root / "prompt.txt"
            if not prompt_path.exists():
                prompt_path.write_text(packet.task + "\n", encoding="utf-8")
        self.event(
            "observation_received",
            sim_step=packet.sim_step,
            active_sequence=packet.active_sequence,
            active_index=packet.active_index,
            control_epoch=packet.control_epoch,
            prepared_state_frames=len(packet.prepared_state),
            transport_ms=(packet.server_received_ns - packet.encoded_ns) / 1e6
            if packet.encoded_ns >= 0
            else None,
            base_jpeg_sha256=hashlib.sha256(packet.base_jpeg).hexdigest(),
            wrist_jpeg_sha256=hashlib.sha256(packet.wrist_jpeg).hexdigest(),
        )

    def action(self, record: ActionRecord) -> None:
        if self.root is None:
            return
        path = self.root / "vla" / "actions" / f"{record.sequence:06d}.npz"
        with self._lock:
            np.savez_compressed(
                path,
                model_image_history=np.frombuffer(record.model_image_history or b"", dtype=np.uint8),
                sequence=np.asarray(record.sequence),
                source_step=np.asarray(record.source_step),
                action_names=np.asarray(record.action_names),
                reset_preset_catalog=np.asarray(json.dumps(record.reset_preset_catalog)),
                model_action_names=np.asarray(record.model_action_names),
                control_epoch=np.asarray(record.control_epoch),
                b2_execution_mode=np.asarray(record.b2_execution_mode),
                z1_action_representation=np.asarray(record.z1_action_representation),
                b2_anchor_kind=np.asarray(record.b2_anchor_kind),
                b2_anchor=record.b2_anchor,
                b2_reference=(
                    record.b2_reference.numpy()
                    if record.b2_reference is not None
                    else np.empty((0, 3), dtype=np.float32)
                ),
                ee_anchor_kind=np.asarray(record.ee_anchor_kind),
                ee_anchor=record.ee_anchor,
                original_actions=record.original.numpy(),
                unsmoothed_execution_actions=record.unsmoothed.numpy(),
                processed_actions=record.processed.numpy(),
                inference_seconds=np.asarray(record.inference_seconds),
                inference_started_ns=np.asarray(record.inference_started_ns),
                inference_finished_ns=np.asarray(record.inference_finished_ns),
                inference_delay_steps=np.asarray(record.inference_delay_steps),
                rtc_mode=np.asarray(record.rtc_mode),
                rtc_conditioning_delay_steps=np.asarray(record.rtc_conditioning_delay_steps),
                velocity_smoothing_transition_step=np.asarray(record.velocity_smoothing_transition_step),
                sim_steps_per_wall_second=np.asarray(record.sim_steps_per_wall_second),
            )
        self.event(
            "inference_finished",
            sequence=record.sequence,
            source_step=record.source_step,
            queue_ms=(record.inference_started_ns - record.observation_received_ns) / 1e6,
            preprocess_ms=record.preprocess_seconds * 1000.0,
            predict_ms=record.predict_seconds * 1000.0,
            postprocess_ms=record.postprocess_seconds * 1000.0,
            inference_ms=record.inference_seconds * 1000.0,
            inference_delay_steps=record.inference_delay_steps,
            rtc_mode=record.rtc_mode,
            rtc_conditioning_delay_steps=record.rtc_conditioning_delay_steps,
            b2_anchor_kind=record.b2_anchor_kind,
            b2_anchor=record.b2_anchor.tolist(),
            ee_anchor_kind=record.ee_anchor_kind,
            ee_anchor=record.ee_anchor.tolist(),
            velocity_smoothing_transition_step=record.velocity_smoothing_transition_step,
            sim_steps_per_wall_second=record.sim_steps_per_wall_second,
        )


class InferenceBusyError(RuntimeError):
    pass


class AsyncRTCPolicy:
    """Asynchronous inference worker with selectable RTC or synchronous scheduling."""

    def __init__(self, args: argparse.Namespace):
        self.chunk_scheduling_mode = str(args.chunk_scheduling_mode)
        self.recorder = RolloutRecorder(args.rollout_dir)
        self.recorder.event(
            "vla_loading_started",
            policy_path=args.policy_path,
            chunk_scheduling_mode=self.chunk_scheduling_mode,
        )
        policy_path = _resolve_policy_path(args.policy_path)
        self.policy_path = policy_path
        self._hot_swap_signature = _checkpoint_hot_swap_signature(policy_path)
        config = PreTrainedConfig.from_pretrained(policy_path)
        config.pretrained_path = policy_path
        config.device = args.device
        self.device = torch.device(args.device)
        contract = _load_checkpoint_contract(policy_path, config, float(args.low_level_hz))
        if args.num_inference_steps is not None:
            config.num_inference_steps = args.num_inference_steps
        if args.compile_model is not None:
            config.compile_model = args.compile_model
        if args.compile_mode is not None:
            config.compile_mode = args.compile_mode
        self.rtc_config = _deployment_rtc_config(config, args)
        config.rtc_config = self.rtc_config
        training_config = config.training_rtc_config
        self.training_rtc_max_delay = (
            training_config.simulated_delay - 1 if self.rtc_config.mode == "training" else None
        )

        LOG.info("Loading checkpoint-only policy=%s", policy_path)
        self.policy = make_policy(config)
        self.policy.eval()
        if hasattr(self.policy, "active_adapters") and hasattr(self.policy, "set_adapter"):
            active_adapters = list(self.policy.active_adapters)
            if len(active_adapters) != 1:
                raise ValueError(f"Expected one active PEFT adapter, got {active_adapters}")
            self._active_adapter_name = str(active_adapters[0])
            self.policy.set_adapter(self._active_adapter_name, inference_mode=True)
        else:
            raise TypeError("PI0.5 deployment requires a PEFT policy with adapter switching support")
        if hasattr(self.policy, "gradient_checkpointing_disable"):
            self.policy.gradient_checkpointing_disable()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(policy_path),
            pretrained_revision=getattr(config, "pretrained_revision", None),
        )
        action_unnormalizers = [
            step for step in self.postprocessor.steps if isinstance(step, UnnormalizerProcessorStep)
        ]
        if len(action_unnormalizers) != 1:
            raise ValueError(f"Expected one action unnormalizer, got {len(action_unnormalizers)}")
        self._action_unnormalizer = action_unnormalizers[0]
        self._action_normalizer = NormalizerProcessorStep(
            features=self._action_unnormalizer.features,
            norm_map=self._action_unnormalizer.norm_map,
            stats=self._action_unnormalizer.stats,
            device=self.device,
        )

        self.policy.config.rtc_config = self.rtc_config
        self.policy.init_rtc_processor()
        self.low_level_hz = float(args.low_level_hz)
        self.stop_on_model_task_complete = bool(args.stop_on_model_task_complete)
        self.b2_velocity_smoothing_time_constant_s = float(args.b2_velocity_smoothing_time_constant_s)
        self.camera_keys = contract.camera_keys
        if tuple(self.policy.config.image_features) != self.camera_keys:
            raise ValueError("Loaded model camera slots disagree with checkpoint metadata")
        self.base_camera_key, self.wrist_camera_key = _camera_bindings(self.camera_keys)
        self.camera_shapes = {key: _feature_shape(config.input_features[key]) for key in self.camera_keys}
        if any(len(shape) != 3 for shape in self.camera_shapes.values()):
            raise ValueError(f"Checkpoint camera features must be CHW: {self.camera_shapes}")

        names = getattr(self.policy.config, "action_feature_names", None)
        if getattr(self.policy.config, "io_schema_resolved", False):
            from lerobot.policies.pi05.b2_action_transform import (
                action_schema_kwargs,
                b2_execution_action_names,
            )

            names = b2_execution_action_names(
                list(contract.source_action_names),
                **action_schema_kwargs(self.policy.config),
            )
        if names is None:
            raise ValueError("Checkpoint does not define named actions")
        self.postprocessed_action_names = tuple(str(name) for name in names)
        self.model_action_names = tuple(
            str(name) for name in (self.policy.config.action_feature_names or self.postprocessed_action_names)
        )
        self.discrete_action_training_mode = contract.discrete_action_training_mode
        self.arm_mode_encoding = contract.arm_mode_encoding
        self.gripper_target_representation = contract.gripper_target_representation
        self.gripper_negative_value = contract.gripper_negative_value
        self.gripper_nonnegative_value = contract.gripper_nonnegative_value
        self.z1_action_representation = contract.z1_action_representation
        self.b2_action_representation = contract.model_action_representation
        self.b2_pose_delta_control_mode = args.b2_pose_delta_control_mode
        self.b2_execution_mode = (
            "se2_feedback"
            if self.b2_action_representation == "pose_delta"
            and self.b2_pose_delta_control_mode == "se2_feedback"
            else "velocity_chunk"
        )
        self.action_names = execution_action_names(self.z1_action_representation)
        self.reset_preset_catalog = config.reset_preset_catalog if config.action_predict_reset_preset else None
        if self.reset_preset_catalog is not None:
            self.action_names += ("arm_reset_preset_id",)
        self.source_state_names = contract.source_state_names
        self.selected_state_names = contract.selected_state_names
        self.source_action_names = contract.source_action_names
        self.mem_enabled = bool(config.mem_vit_enabled)
        self.mem_image_num_frames = int(config.mem_vit_num_frames) if self.mem_enabled else 1
        self.mem_image_interval_ns = max(1, round(float(config.mem_vit_frame_interval_seconds or 0.0) * 1e9))
        self.continuous_state_enabled = config.state_action_encoding == "continuous"
        self.state_num_frames = int(config.state_num_frames) if self.continuous_state_enabled else 1
        self.state_history_stride = max(
            1, round(float(config.state_history_frame_interval_seconds) * self.low_level_hz)
        )
        self.action_history_enabled = bool(config.action_history_enabled)
        self.action_history_indices = action_dataset_indices(
            **{
                key: value
                for key, value in action_schema_kwargs(config).items()
                if key
                not in {
                    "representation",
                    "z1_representation",
                    "ee_delta_rotation_representation",
                    "include_task_blocked",
                }
            }
        )
        self._mailbox_condition = Condition()
        self._model_lock = Lock()
        self._request_gate = Lock()
        self._mailbox: ObservationPacket | None = None
        self._mailbox_version = 0
        self._last_inferred_version = 0
        self._records_lock = Lock()
        self._records: dict[int, ActionRecord] = {}
        self._latest_record: ActionRecord | None = None
        self._next_sequence = 1
        self._control_epoch: int | None = None
        self._task_complete_latched = False
        self._stop = Event()
        self._warmup_requested = Event()
        self._warmup_finished = Event()
        self._warmup_error: BaseException | None = None
        self._worker = Thread(target=self._run, name="pi05-rtc-worker", daemon=True)
        self._last_error: str | None = None
        self._last_error_traceback: str | None = None
        self._inference_count = 0
        self._warmup_inferences = int(args.warmup_inferences)
        b2_rtc_reanchor, ee_rtc_reanchor, _, _ = self._rtc_reanchor_status()
        if self.b2_action_representation == "pose_delta" and not b2_rtc_reanchor:
            LOG.warning(
                "B2 RTC prefix re-anchoring is disabled because its coupled SE(2) "
                "normalization group has a degenerate scale"
            )
        if self.z1_action_representation == "ee_state_delta" and not ee_rtc_reanchor:
            LOG.warning(
                "Z1 RTC prefix re-anchoring is disabled because its coupled SE(3) "
                "normalization group has a degenerate scale"
            )
        self.recorder.event(
            "vla_loading_finished",
            action_names=self.action_names,
            model_action_names=self.model_action_names,
            postprocessed_action_names=self.postprocessed_action_names,
            camera_keys=self.camera_keys,
            base_camera_key=self.base_camera_key,
            wrist_camera_key=self.wrist_camera_key,
            checkpoint_metadata_version=contract.metadata_version,
            model_action_representation=contract.model_action_representation,
            z1_model_action_representation=contract.z1_action_representation,
            ee_delta_rotation_representation=contract.ee_delta_rotation_representation,
            b2_execution_action_representation="velocity",
            b2_pose_delta_control_mode=self.b2_pose_delta_control_mode,
            b2_execution_mode=self.b2_execution_mode,
            b2_rtc_prefix_reanchor_enabled=b2_rtc_reanchor,
            z1_execution_action_representation=contract.z1_action_representation,
            ee_rtc_prefix_reanchor_enabled=ee_rtc_reanchor,
            execution_action_protocol="rtc_action_packet_v6_optional_high_level_se2_feedback",
            rtc_mode=self.rtc_config.mode,
            training_rtc_max_delay=self.training_rtc_max_delay,
            training_rtc_delay_overflow="error" if self.rtc_config.mode == "training" else None,
            stop_on_model_task_complete=self.stop_on_model_task_complete,
            b2_velocity_smoothing="causal_first_order_low_pass",
            b2_velocity_smoothing_time_constant_s=self.b2_velocity_smoothing_time_constant_s,
            arm_gate_conversion=(
                "one_minus_arm_teleop_inactive"
                if "arm_teleop_inactive" in self.postprocessed_action_names
                else "not_present_low_level_always_ee"
            ),
            discrete_action_training_mode=self.discrete_action_training_mode,
            arm_mode_encoding=self.arm_mode_encoding,
            ee_target_dataset_semantics=contract.ee_target_dataset_semantics,
            ee_delta_supervision_mode=contract.ee_delta_supervision_mode,
            gripper_target_representation=self.gripper_target_representation,
            structured_action_crf_initial_stay_bias=(
                self.policy.config.structured_action_crf_initial_stay_bias
                if self.discrete_action_training_mode == "structured_temporal"
                else None
            ),
            gripper_output_values={
                "normalized_negative": self.gripper_negative_value,
                "normalized_nonnegative": self.gripper_nonnegative_value,
            },
            checkpoint_control_frequency_hz=contract.control_frequency_hz,
            checkpoint_state_dim=contract.state_dim,
            source_state_names=self.source_state_names,
            selected_state_names=self.selected_state_names,
            deployment_metadata_path=str(contract.metadata_path),
            torch_version=torch.__version__,
            cuda_runtime_version=torch.version.cuda,
            device_name=torch.cuda.get_device_name(self.device)
            if self.device.type == "cuda"
            else str(self.device),
            model_dtype=str(config.dtype),
            num_inference_steps=int(config.num_inference_steps),
            compile_model=bool(config.compile_model),
            inference_context="torch.inference_mode",
            mem_image_sampling_hz=(1.0 / config.mem_vit_frame_interval_seconds if self.mem_enabled else None),
            state_action_encoding=config.state_action_encoding,
            state_history_sampling_hz=(
                1.0 / config.state_history_frame_interval_seconds if self.continuous_state_enabled else None
            ),
            state_history_num_frames=self.state_num_frames,
            state_history_stride_steps=self.state_history_stride,
            action_history_enabled=self.action_history_enabled,
            action_history_num_frames=(self.state_num_frames - 1 if self.action_history_enabled else 0),
            warmup_inferences=self._warmup_inferences,
        )

    def _reset_runtime_after_model_switch(self) -> None:
        with self._mailbox_condition:
            self._mailbox = None
            self._last_inferred_version = self._mailbox_version
        with self._records_lock:
            self._records.clear()
            self._latest_record = None
        self._control_epoch = None
        self._task_complete_latched = False
        self._last_error = None
        self._last_error_traceback = None

    def hot_swap_checkpoint(self, value: str | Path) -> dict[str, object]:
        """Replace compatible PEFT weights without reconstructing the base model."""
        from lerobot.policies.base_artifacts import resolve_checkpoint_bases

        started = time.perf_counter()
        policy_path = _resolve_policy_path(value)
        # A hot swap reuses the already-loaded bases, so first prove that the
        # requested adapter declares the exact artifacts available locally.
        resolve_checkpoint_bases(policy_path)
        signature = _checkpoint_hot_swap_signature(policy_path)
        if signature != self._hot_swap_signature:
            return {
                "accepted": False,
                "hot_swap_supported": False,
                "reason": "checkpoint architecture, deployment contract, or processor schema differs",
                "current_model": str(self.policy_path),
                "requested_model": str(policy_path),
            }

        config = PreTrainedConfig.from_pretrained(policy_path)
        config.pretrained_path = policy_path
        config.device = str(self.device)
        _load_checkpoint_contract(policy_path, config, self.low_level_hz)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(policy_path),
            pretrained_revision=getattr(config, "pretrained_revision", None),
        )
        action_unnormalizers = [
            step for step in postprocessor.steps if isinstance(step, UnnormalizerProcessorStep)
        ]
        if len(action_unnormalizers) != 1:
            raise ValueError(f"Expected one action unnormalizer, got {len(action_unnormalizers)}")
        action_unnormalizer = action_unnormalizers[0]
        action_normalizer = NormalizerProcessorStep(
            features=action_unnormalizer.features,
            norm_map=action_unnormalizer.norm_map,
            stats=action_unnormalizer.stats,
            device=self.device,
        )

        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

        from lerobot.utils.reset_presets import validate_selector_checkpoint

        validate_selector_checkpoint(policy_path, config.action_predict_reset_preset)
        adapter_state = load_peft_weights(str(policy_path), device="cpu")
        with self._model_lock:
            load_result = set_peft_model_state_dict(
                self.policy,
                adapter_state,
                adapter_name=self._active_adapter_name,
            )
            if load_result.unexpected_keys:
                raise ValueError(
                    f"Hot-swapped adapter has unexpected keys: {load_result.unexpected_keys[:8]}"
                )
            self.preprocessor = preprocessor
            self.postprocessor = postprocessor
            self._action_unnormalizer = action_unnormalizer
            self._action_normalizer = action_normalizer
            self.policy_path = policy_path
            b2_rtc_reanchor, ee_rtc_reanchor, _, _ = self._rtc_reanchor_status()
            self._reset_runtime_after_model_switch()
        self._request_worker_warmup()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.recorder.event(
            "vla_checkpoint_hot_swapped",
            policy_path=str(policy_path),
            elapsed_ms=elapsed_ms,
            active_adapter=self._active_adapter_name,
            b2_rtc_prefix_reanchor_enabled=b2_rtc_reanchor,
            ee_rtc_prefix_reanchor_enabled=ee_rtc_reanchor,
        )
        LOG.info("Hot-swapped checkpoint=%s in %.3fs", policy_path, elapsed_ms / 1000.0)
        return {
            "accepted": True,
            "hot_swap_supported": True,
            "load_strategy": "hot_swap",
            "current_model": str(policy_path),
            "elapsed_ms": elapsed_ms,
        }

    def start(self) -> None:
        self._warmup_requested.set()
        self._worker.start()
        self._wait_for_worker_warmup()

    def _request_worker_warmup(self) -> None:
        self._warmup_error = None
        self._warmup_finished.clear()
        self._warmup_requested.set()
        with self._mailbox_condition:
            self._mailbox_condition.notify_all()
        self._wait_for_worker_warmup()

    def _wait_for_worker_warmup(self) -> None:
        if not self._warmup_finished.wait(timeout=600.0):
            raise TimeoutError("Timed out waiting for VLA warmup on the inference worker")
        if self._warmup_error is not None:
            raise RuntimeError("VLA warmup failed on the inference worker") from self._warmup_error

    def _warmup_batch(self) -> dict[str, object]:
        batch: dict[str, object] = {
            OBS_STATE: torch.zeros(
                1, self.state_num_frames, len(self.source_state_names), dtype=torch.float32
            ),
            f"{OBS_STATE}_is_pad": torch.zeros(1, self.state_num_frames, dtype=torch.bool),
            "task": ["warm up the VLA deployment model"],
        }
        for key, (channels, height, width) in self.camera_shapes.items():
            image = torch.zeros(1, channels, height, width, dtype=torch.float32)
            if self.mem_enabled:
                image = image.unsqueeze(1).expand(-1, self.mem_image_num_frames, -1, -1, -1).clone()
                image_pad = torch.zeros(1, self.mem_image_num_frames, dtype=torch.bool)
            else:
                image_pad = torch.zeros(1, dtype=torch.bool)
            batch[key] = image
            batch[f"{key}_is_pad"] = image_pad
        if self.action_history_enabled:
            action_frames = self.state_num_frames - 1
            batch[OBS_ACTION_HISTORY] = torch.zeros(
                1, action_frames, len(self.action_history_indices), dtype=torch.float32
            )
            batch[f"{OBS_ACTION_HISTORY}_is_pad"] = torch.ones(1, action_frames, dtype=torch.bool)
        return batch

    def _warmup(self) -> None:
        if self._warmup_inferences == 0:
            return
        started = time.perf_counter()
        rtc_prefix = None
        completed = 0
        for warmup_index in range(self._warmup_inferences):
            batch = self.preprocessor(self._warmup_batch())
            with torch.inference_mode():
                conditioning: dict[str, object] = {}
                if self.rtc_config.enabled:
                    warmup_delay = 0
                    if self.rtc_config.mode == "training" and rtc_prefix is not None:
                        warmup_delay = min(1, self.training_rtc_max_delay)
                    conditioning = {
                        "inference_delay": warmup_delay,
                        "prev_chunk_left_over": rtc_prefix,
                    }
                actions = self.policy.predict_action_chunk(batch, **conditioning)
                if warmup_index == 0 and self.chunk_scheduling_mode == "RTC" and self._warmup_inferences > 1:
                    # Keep the prefix independent from torch.compile's reusable CUDA graph
                    # outputs, then exercise RTC guidance before the service reports ready.
                    rtc_prefix = actions.detach().cpu().clone().to(self.device)
                self.postprocessor(actions).detach().cpu()
                completed += 1
        if self.rtc_config.mode in {"inference", "training"} and not self.rtc_config.enabled:
            self.rtc_config.enabled = True
            try:
                batch = self.preprocessor(self._warmup_batch())
                with torch.inference_mode():
                    actions = self.policy.predict_action_chunk(
                        batch,
                        inference_delay=0,
                        prev_chunk_left_over=None,
                    )
                    self.postprocessor(actions).detach().cpu()
                    prefix = actions.detach().cpu().clone().to(self.device)
                    actions = self.policy.predict_action_chunk(
                        batch,
                        inference_delay=(
                            min(1, self.training_rtc_max_delay) if self.rtc_config.mode == "training" else 0
                        ),
                        prev_chunk_left_over=prefix,
                    )
                    self.postprocessor(actions).detach().cpu()
                    completed += 2
            finally:
                self.rtc_config.enabled = False
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        self.recorder.event(
            "vla_warmup_finished",
            inference_count=completed,
            elapsed_ms=elapsed * 1000.0,
        )
        LOG.info("Completed %d VLA warmup inference(s) in %.3fs", completed, elapsed)

    def _run_worker_warmup(self) -> None:
        try:
            with self._model_lock:
                self._warmup()
            self._warmup_error = None
        except BaseException as exc:
            self._warmup_error = exc
        finally:
            self._warmup_requested.clear()
            self._warmup_finished.set()

    def stop(self) -> None:
        self._stop.set()
        with self._mailbox_condition:
            self._mailbox_condition.notify_all()
        self._worker.join(timeout=5.0)

    def submit(self, packet: ObservationPacket, *, enqueue: bool = True) -> None:
        executed_completion = False
        if self.stop_on_model_task_complete and packet.active_sequence >= 0 and packet.active_index > 0:
            with self._records_lock:
                active_record = self._records.get(packet.active_sequence)
            if active_record is not None and "task_complete" in active_record.action_names:
                complete_index = active_record.action_names.index("task_complete")
                executed_count = min(packet.active_index, len(active_record.processed))
                if bool((active_record.processed[:executed_count, complete_index] > 0.5).any()):
                    executed_completion = True
        with self._mailbox_condition:
            epoch_reset = self._control_epoch != packet.control_epoch
            if epoch_reset:
                self._task_complete_latched = False
                self._mailbox = None
            if executed_completion:
                self._task_complete_latched = True
            self._control_epoch = packet.control_epoch
            if enqueue:
                if self._mailbox is not None and packet.sim_step < self._mailbox.sim_step:
                    return
                self._mailbox = packet
                self._mailbox_version += 1
                self._mailbox_condition.notify()
        self.recorder.observation(packet)

    def _publish_record(self, record: ActionRecord) -> None:
        with self._records_lock:
            self._records[record.sequence] = record
            if self._latest_record is None or record.sequence > self._latest_record.sequence:
                self._latest_record = record
            while len(self._records) > 16:
                self._records.pop(min(self._records))
            self._inference_count += 1
        self._last_error = None
        self._last_error_traceback = None
        self.recorder.action(record)

    def infer_request(self, packet: ObservationPacket) -> ActionRecord:
        # A request/response call must never be replaced by the RTC mailbox.
        # Bound admission instead of building an unbounded HTTP inference queue.
        if not self._request_gate.acquire(blocking=False):
            raise InferenceBusyError("Another request/response inference is in progress")
        try:
            with self._model_lock:
                self.submit(packet, enqueue=False)
                record = self._infer_unlocked(packet)
                self._publish_record(record)
                return record
        finally:
            self._request_gate.release()

    def latest_after(self, sequence: int) -> ActionRecord | None:
        with self._records_lock:
            record = self._latest_record
            return record if record is not None and record.sequence > sequence else None

    def health(self) -> dict[str, object]:
        with self._records_lock:
            latest = self._latest_record
        return {
            "ok": self._worker.is_alive() and self._last_error is None,
            "protocol_version": PROTOCOL_VERSION,
            "policy_path": str(self.policy_path),
            "active_adapter": self._active_adapter_name,
            "worker_alive": self._worker.is_alive(),
            "inference_count": self._inference_count,
            "latest_sequence": latest.sequence if latest else 0,
            "latest_source_step": latest.source_step if latest else -1,
            "chunk_scheduling_mode": self.chunk_scheduling_mode,
            "stop_on_model_task_complete": self.stop_on_model_task_complete,
            "b2_execution_mode": self.b2_execution_mode,
            "rtc_mode": self.rtc_config.mode,
            "training_rtc_max_delay": self.training_rtc_max_delay,
            "last_error": self._last_error,
            "last_error_traceback": self._last_error_traceback,
        }

    def set_chunk_scheduling_mode(self, mode: str) -> dict[str, object]:
        if mode not in CHUNK_SCHEDULING_MODES:
            raise ValueError(
                f"Unknown chunk scheduling mode {mode!r}; expected one of {CHUNK_SCHEDULING_MODES}"
            )
        previous = self.chunk_scheduling_mode
        self.chunk_scheduling_mode = mode
        self.rtc_config.enabled = _rtc_enabled_for_chunk_scheduling(mode)
        LOG.info("Chunk scheduling mode changed: %s -> %s", previous, mode)
        return {
            "accepted": previous != mode,
            "previous_mode": previous,
            "chunk_scheduling_mode": mode,
            "rtc_enabled": self.rtc_config.enabled,
        }

    def set_stop_on_model_task_complete(self, enabled: bool) -> dict[str, object]:
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        previous = self.stop_on_model_task_complete
        self.stop_on_model_task_complete = enabled
        if not enabled:
            self._task_complete_latched = False
        LOG.info(
            "Task-complete termination changed: %s -> %s",
            previous,
            enabled,
        )
        return {
            "accepted": previous != enabled,
            "previous_enabled": previous,
            "stop_on_model_task_complete": enabled,
        }

    @staticmethod
    def _b2_global_pose(packet: ObservationPacket) -> np.ndarray:
        indices = {name: index for index, name in enumerate(packet.state_names)}
        missing = tuple(name for name in B2_GLOBAL_POSE_NAMES if name not in indices)
        if missing:
            raise ValueError(f"Simulator observation is missing B2 global-pose fields: {missing}")
        return packet.state[[indices[name] for name in B2_GLOBAL_POSE_NAMES]].astype(np.float32, copy=True)

    @staticmethod
    def _apply_action_processor(step, action: torch.Tensor) -> torch.Tensor:
        transition = step({TransitionKey.ACTION: action})
        result = transition[TransitionKey.ACTION]
        if not isinstance(result, torch.Tensor):
            raise TypeError(f"Action processor returned {type(result)}")
        return result

    def _rtc_reanchor_status(self) -> tuple[bool, bool, tuple[str, ...], str]:
        b2_names = ("b2_delta_x", "b2_delta_y", "b2_delta_yaw")
        b2_enabled = self.b2_action_representation == "pose_delta" and (
            _rtc_reanchor_group_is_well_conditioned(
                self._action_unnormalizer, self.model_action_names, b2_names
            )
        )
        if set(EE_DELTA_ROTVEC_NAMES).issubset(self.model_action_names):
            rotation_representation = "rotvec"
            ee_names = (*EE_DELTA_ROTVEC_NAMES, *EE_DELTA_ACTION_NAMES[6:9])
        elif set(EE_DELTA_ACTION_NAMES).issubset(self.model_action_names):
            rotation_representation = "rot6d"
            ee_names = EE_DELTA_ACTION_NAMES
        else:
            ee_names = ()
            rotation_representation = ""
        ee_enabled = (
            self.z1_action_representation == "ee_state_delta"
            and bool(ee_names)
            and (
                _rtc_reanchor_group_is_well_conditioned(
                    self._action_unnormalizer, self.model_action_names, ee_names
                )
            )
        )
        return b2_enabled, ee_enabled, ee_names, rotation_representation

    def _reanchor_rtc_prefix(
        self,
        normalized_prefix: torch.Tensor,
        record: ActionRecord,
        packet: ObservationPacket,
    ) -> torch.Tensor:
        physical = self._apply_action_processor(self._action_unnormalizer, normalized_prefix)
        b2_reanchor, ee_reanchor, ee_names, rotation_representation = self._rtc_reanchor_status()
        if b2_reanchor:
            b2_names = ("b2_delta_x", "b2_delta_y", "b2_delta_yaw")
            indices = [self.model_action_names.index(name) for name in b2_names]
            old_anchor = torch.as_tensor(record.b2_anchor, dtype=physical.dtype, device=physical.device)
            new_anchor = torch.as_tensor(
                self._b2_global_pose(packet), dtype=physical.dtype, device=physical.device
            )
            physical[:, indices] = _reanchor_b2_pose_delta(physical[:, indices], old_anchor, new_anchor)

        if self.z1_action_representation == "ee_state_delta" and not ee_names:
            raise ValueError("Model action names do not contain a complete EE-delta representation")
        if ee_reanchor:
            indices = [self.model_action_names.index(name) for name in ee_names]
            old_anchor = torch.as_tensor(record.ee_anchor, dtype=physical.dtype, device=physical.device)
            new_anchor = torch.as_tensor(packet.actual_ee_state, dtype=physical.dtype, device=physical.device)
            absolute_targets = ee_reference_delta_to_absolute(
                physical[:, indices].unsqueeze(0),
                old_anchor.unsqueeze(0),
                rotation_representation=rotation_representation,
            )
            physical[:, indices] = absolute_ee_pose_to_reference_delta(
                absolute_targets,
                new_anchor.unsqueeze(0),
                rotation_representation=rotation_representation,
            ).squeeze(0)

        normalized = self._apply_action_processor(self._action_normalizer, physical)
        if not torch.isfinite(normalized).all():
            raise RuntimeError("RTC prefix re-anchoring produced non-finite normalized actions")
        return normalized

    def _previous_prefix(self, packet: ObservationPacket) -> torch.Tensor | None:
        if self.chunk_scheduling_mode == "SYNC":
            return None
        if packet.active_sequence < 0:
            return None
        with self._records_lock:
            record = self._records.get(packet.active_sequence)
        if record is None:
            LOG.warning("RTC prefix sequence %d is no longer available", packet.active_sequence)
            return None
        index = max(0, min(packet.active_index, len(record.original)))
        prefix = record.original[index:].clone().to(self.device)
        if prefix.numel() == 0:
            return None
        if self.b2_action_representation == "pose_delta" or self.z1_action_representation == "ee_state_delta":
            prefix = self._reanchor_rtc_prefix(prefix, record, packet)
            b2_reanchored, ee_reanchored, _, _ = self._rtc_reanchor_status()
            self.recorder.event(
                "rtc_prefix_reanchored",
                active_sequence=record.sequence,
                active_index=index,
                source_step=packet.sim_step,
                b2_reanchored=b2_reanchored,
                b2_reanchor_skipped_degenerate_normalization=(
                    self.b2_action_representation == "pose_delta" and not b2_reanchored
                ),
                b2_old_anchor=record.b2_anchor.tolist(),
                b2_new_anchor=self._b2_global_pose(packet).tolist(),
                ee_reanchored=ee_reanchored,
                ee_reanchor_skipped_degenerate_normalization=(
                    self.z1_action_representation == "ee_state_delta" and not ee_reanchored
                ),
                ee_old_anchor=record.ee_anchor.tolist(),
                ee_new_anchor=packet.actual_ee_state.tolist(),
            )
        if self.rtc_config.mode == "training":
            # Hard conditioning must only freeze actions that are really queued.
            # The inference-RTC execution horizon and zero padding do not apply.
            return prefix
        horizon = self.rtc_config.execution_horizon
        if len(prefix) >= horizon:
            return prefix[:horizon]
        padded = torch.zeros((horizon, prefix.shape[-1]), dtype=prefix.dtype, device=prefix.device)
        padded[: len(prefix)] = prefix
        return padded

    def _rtc_conditioning(self, packet: ObservationPacket, inference_delay_steps: int) -> dict[str, object]:
        if self.chunk_scheduling_mode == "SYNC":
            return {}
        prefix = self._previous_prefix(packet)
        delay = inference_delay_steps
        if self.rtc_config.mode == "training":
            # First chunks and drained queues have no actions to freeze.
            delay = inference_delay_steps if prefix is not None else 0
            if delay > self.training_rtc_max_delay:
                raise ValueError(
                    f"Training-RTC estimated inference delay {delay} exceeds the checkpoint's "
                    f"trained maximum {self.training_rtc_max_delay}; fine-tune with a larger "
                    "training_rtc_config.simulated_delay"
                )
            if prefix is not None and delay > len(prefix):
                raise ValueError(
                    f"Training-RTC estimated inference delay {delay} exceeds the available "
                    f"queued prefix ({len(prefix)} actions)"
                )
        return {"inference_delay": delay, "prev_chunk_left_over": prefix}

    def input_spec(self) -> dict[str, object]:
        spec = {
            "protocol_version": PROTOCOL_VERSION, "model_input_format": 1,
            "source_state_names": list(self.source_state_names),
            "selected_state_names": list(self.selected_state_names),
            "base_camera_key": self.base_camera_key, "wrist_camera_key": self.wrist_camera_key,
            "image_num_frames": self.mem_image_num_frames,
            "image_interval_ns": self.mem_image_interval_ns,
            "state_num_frames": self.state_num_frames,
            "state_stride_steps": self.state_history_stride,
            "control_frequency_hz": self.low_level_hz,
            "action_history_enabled": self.action_history_enabled,
            "action_history_names": [self.source_action_names[index] for index in self.action_history_indices],
        }
        spec["schema_id"] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        return spec

    def validate_input(self, packet: ObservationPacket) -> None:
        spec = self.input_spec()
        if packet.input_schema != spec["schema_id"]:
            raise ValueError("Prepared input contract differs from the loaded model")
        if packet.prepared_state.shape != (self.state_num_frames, len(self.source_state_names)):
            raise ValueError("Prepared state window does not match model shape")
        if self.action_history_enabled:
            actions, pad = packet.prepared_action_history, packet.prepared_action_history_is_pad
            expected = (self.state_num_frames - 1, len(self.action_history_indices))
            if (actions is None or actions.shape != expected or not np.isfinite(actions).all()
                    or pad is None or pad.dtype != np.bool_ or pad.shape != expected[:1]):
                raise ValueError("Prepared action history does not match model shape")
        _decode_model_image_history(packet.model_image_history, source_step=packet.sim_step,
            count=self.mem_image_num_frames, base_key=self.base_camera_key, wrist_key=self.wrist_camera_key)

    def _make_batch(self, packet: ObservationPacket) -> dict[str, object]:
        self.validate_input(packet)
        batch = _RecordedImageBatch({
            OBS_STATE: torch.from_numpy(packet.prepared_state.copy()).unsqueeze(0),
            f"{OBS_STATE}_is_pad": torch.from_numpy(packet.prepared_state_is_pad.copy()).unsqueeze(0),
            "task": [packet.task],
        })
        image_records, image_pad = _decode_model_image_history(
            packet.model_image_history, source_step=packet.sim_step,
            count=self.mem_image_num_frames, base_key=self.base_camera_key, wrist_key=self.wrist_camera_key,
        )
        batch.model_image_history = packet.model_image_history
        for key, role in ((self.base_camera_key, "base_jpeg"), (self.wrist_camera_key, "wrist_jpeg")):
            image_window = torch.stack([
                torch.from_numpy(np.ascontiguousarray(_decode_jpeg(getattr(record, role), key)))
                .permute(2, 0, 1).to(torch.float32).div_(255.0) for record in image_records
            ]).unsqueeze(0)
            batch[key] = image_window if self.mem_enabled else image_window[:, -1]
            batch[f"{key}_is_pad"] = image_pad if self.mem_enabled else image_pad[:, -1]
        if self.action_history_enabled:
            actions, pad = packet.prepared_action_history, packet.prepared_action_history_is_pad
            batch[OBS_ACTION_HISTORY] = torch.from_numpy(actions.copy()).unsqueeze(0)
            batch[f"{OBS_ACTION_HISTORY}_is_pad"] = torch.from_numpy(pad.copy()).unsqueeze(0)
        return batch

    @staticmethod
    def _observed_b2_velocity(packet: ObservationPacket) -> torch.Tensor:
        state_indices = {name: index for index, name in enumerate(packet.state_names)}
        missing = tuple(name for name in B2_OBSERVED_VELOCITY_NAMES if name not in state_indices)
        if missing:
            raise ValueError(f"Simulator observation is missing B2 velocity fields: {missing}")
        return torch.from_numpy(
            packet.state[[state_indices[name] for name in B2_OBSERVED_VELOCITY_NAMES]]
        ).to(torch.float32)

    def _b2_velocity_filter_context(
        self, packet: ObservationPacket
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.chunk_scheduling_mode == "SYNC":
            return torch.zeros(3, dtype=torch.float32), None
        if packet.active_sequence >= 0 and packet.active_index > 0:
            with self._records_lock:
                active_record = self._records.get(packet.active_sequence)
            if active_record is not None and len(active_record.processed) > 0:
                previous_index = min(packet.active_index, len(active_record.processed)) - 1
                velocity_indices = [self.action_names.index(name) for name in B2_EXECUTION_VELOCITY_NAMES]
                anchor = active_record.processed[previous_index, velocity_indices].clone()
                prefix_start = min(packet.active_index, len(active_record.processed))
                prefix = active_record.processed[prefix_start:, velocity_indices].clone()
                return anchor, prefix
        return self._observed_b2_velocity(packet), None

    def _infer(self, packet: ObservationPacket) -> ActionRecord:
        with self._model_lock:
            return self._infer_unlocked(packet)

    def _infer_unlocked(self, packet: ObservationPacket) -> ActionRecord:
        # Adopt request context inside the inference lock, never from a global
        # management switch racing an already-running prediction.
        if packet.scheduling_mode is not None and packet.scheduling_mode != self.chunk_scheduling_mode:
            self.set_chunk_scheduling_mode(packet.scheduling_mode)
        started_ns = time.monotonic_ns()
        started = time.perf_counter()
        raw_batch = self._make_batch(packet)
        model_image_history = getattr(raw_batch, "model_image_history", None)
        batch = self.preprocessor(raw_batch)
        preprocessed = time.perf_counter()
        inference_delay_steps = 0 if self.chunk_scheduling_mode == "SYNC" else packet.inference_delay_steps
        sim_rate = packet.source_steps_per_wall_second
        conditioning = self._rtc_conditioning(packet, inference_delay_steps)
        diagnostic_flow_seed = getattr(packet, "diagnostic_flow_seed", None)
        if diagnostic_flow_seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(diagnostic_flow_seed)
            conditioning["noise"] = torch.randn(
                (
                    1,
                    int(self.policy.config.chunk_size),
                    int(self.policy.config.max_action_dim),
                ),
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            ).to(self.device)
        # Both RTC modes operate without autograd or tensor version counters.
        with torch.inference_mode():
            if self.reset_preset_catalog is not None:
                actions, preset_ids = self.policy.predict_action_chunk(batch, return_reset_presets=True, **conditioning)
            else:
                actions = self.policy.predict_action_chunk(batch, **conditioning)
            original = actions.squeeze(0).detach().cpu().clone()
            b2_reference = None
            if self.b2_execution_mode == "se2_feedback":
                physical_model_actions = (
                    self._apply_action_processor(self._action_unnormalizer, actions.squeeze(0))
                    .detach()
                    .cpu()
                    .to(torch.float32)
                )
                reference_indices = [
                    self.model_action_names.index(name)
                    for name in ("b2_delta_x", "b2_delta_y", "b2_delta_yaw")
                ]
                b2_reference = physical_model_actions[:, reference_indices].clone()
            predicted = time.perf_counter()
            postprocessed = self.postprocessor(actions).squeeze(0).detach().cpu().to(torch.float32)
            postprocessed = _decode_discrete_actions(
                original,
                postprocessed,
                self.postprocessed_action_names,
                mode=self.discrete_action_training_mode,
                arm_mode_encoding=self.arm_mode_encoding,
                gripper_target_representation=self.gripper_target_representation,
                gripper_negative_value=self.gripper_negative_value,
                gripper_nonnegative_value=self.gripper_nonnegative_value,
            )
            unsmoothed = _to_execution_actions(
                postprocessed,
                self.postprocessed_action_names,
                self.z1_action_representation,
            )
            if self.reset_preset_catalog is not None:
                unsmoothed = torch.cat((unsmoothed, preset_ids.squeeze(0).detach().cpu().float()[:, None]), dim=-1)
            if self.stop_on_model_task_complete:
                if self._task_complete_latched:
                    unsmoothed[:, self.action_names.index("task_complete")] = 1.0
                unsmoothed = _apply_completion_stop(unsmoothed, self.action_names)
            velocity_anchor, previous_velocity_prefix = self._b2_velocity_filter_context(packet)
            smoothing_transition_step = _b2_velocity_smoothing_transition_step(
                inference_delay_steps,
                previous_velocity_prefix,
            )
            processed = _smooth_b2_execution_velocity(
                unsmoothed,
                self.action_names,
                velocity_anchor,
                dt=1.0 / self.low_level_hz,
                time_constant_s=self.b2_velocity_smoothing_time_constant_s,
                transition_start_step=smoothing_transition_step,
                prefix_velocity=previous_velocity_prefix,
            )
        finished = time.perf_counter()
        finished_ns = time.monotonic_ns()
        elapsed = finished - started
        if processed.ndim != 2 or processed.shape[-1] != len(self.action_names):
            raise ValueError(
                f"Processed action shape {tuple(processed.shape)} does not match names {self.action_names}"
            )
        record = ActionRecord(
            reset_preset_catalog=self.reset_preset_catalog,
            sequence=self._next_sequence,
            source_step=packet.sim_step,
            inference_seconds=elapsed,
            action_names=self.action_names,
            model_action_names=self.model_action_names,
            control_epoch=packet.control_epoch,
            b2_execution_mode=self.b2_execution_mode,
            z1_action_representation=self.z1_action_representation,
            b2_anchor_kind="actual_world_pose",
            b2_anchor=self._b2_global_pose(packet),
            b2_reference=b2_reference,
            ee_anchor_kind=(
                "actual_ee_state"
                if self.z1_action_representation == "ee_state_delta"
                else "executed_ee_target"
            ),
            ee_anchor=(
                packet.actual_ee_state.copy()
                if self.z1_action_representation == "ee_state_delta"
                else packet.executed_ee_target.copy()
            ),
            original=original,
            unsmoothed=unsmoothed,
            processed=processed,
            inference_started_ns=started_ns,
            inference_finished_ns=finished_ns,
            preprocess_seconds=preprocessed - started,
            predict_seconds=predicted - preprocessed,
            postprocess_seconds=finished - predicted,
            observation_received_ns=packet.server_received_ns,
            inference_delay_steps=inference_delay_steps,
            rtc_mode=self.rtc_config.mode,
            rtc_conditioning_delay_steps=int(conditioning.get("inference_delay", 0)),
            velocity_smoothing_transition_step=smoothing_transition_step,
            sim_steps_per_wall_second=sim_rate,
            model_image_history=model_image_history,
        )
        self._next_sequence += 1
        return record

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._warmup_requested.is_set():
                self._run_worker_warmup()
                if self._warmup_error is not None:
                    return
                continue
            with self._mailbox_condition:
                self._mailbox_condition.wait_for(
                    lambda: (
                        self._stop.is_set()
                        or self._warmup_requested.is_set()
                        or (
                            self._mailbox is not None and self._mailbox_version != self._last_inferred_version
                        )
                    ),
                    timeout=0.25,
                )
                if self._stop.is_set():
                    return
                if self._warmup_requested.is_set():
                    continue
                packet = self._mailbox
                version = self._mailbox_version
            if packet is None or version == self._last_inferred_version:
                continue
            try:
                record = self._infer(packet)
                self._publish_record(record)
                self._last_inferred_version = version
                LOG.info(
                    "chunk=%d obs_step=%d latency=%.3fs delay=%d actions=%s",
                    record.sequence,
                    record.source_step,
                    record.inference_seconds,
                    record.inference_delay_steps,
                    tuple(record.processed.shape),
                )
            except Exception as exc:  # keep the server observable, but stop retry storms
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._last_error_traceback = traceback.format_exc()
                self._last_inferred_version = version
                self.recorder.event(
                    "inference_failed",
                    sim_step=packet.sim_step,
                    error=self._last_error,
                    traceback=self._last_error_traceback,
                )
                LOG.error(
                    "Inference failed for sim_step=%d\n%s",
                    packet.sim_step,
                    self._last_error_traceback,
                )
                if self._stop.wait(0.5):
                    return


class VLARequestHandler(BaseHTTPRequestHandler):
    server_version = "Pi05VLA/1"

    @property
    def engine(self) -> AsyncRTCPolicy:
        return self.server.engine  # type: ignore[attr-defined,no-any-return]

    @property
    def max_payload_bytes(self) -> int:
        return self.server.max_payload_bytes  # type: ignore[attr-defined,no-any-return]

    def log_message(self, fmt: str, *args: object) -> None:
        LOG.debug("http %s - %s", self.address_string(), fmt % args)

    def _write_json(self, status: int, value: dict[str, object]) -> None:
        payload = json.dumps(value, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _error(self, status: int, code: str, message: str) -> None:
        self._write_json(status, {"error": {"code": code, "message": message}})

    def _write_actions(self, record: ActionRecord) -> None:
        payload = encode_action_packet(record)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-npz")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/v1/model":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 16384:
                    raise ValueError(f"Invalid Content-Length: {length}")
                request_value = json.loads(self.rfile.read(length))
                result = self.engine.hot_swap_checkpoint(str(request_value["path"]))
                status = HTTPStatus.OK if result["accepted"] else HTTPStatus.CONFLICT
                self._write_json(status, result)
            except Exception as exc:
                LOG.exception("Checkpoint hot swap failed")
                self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return
        if path == "/v1/task-complete-termination":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 4096:
                    raise ValueError(f"Invalid Content-Length: {length}")
                request_value = json.loads(self.rfile.read(length))
                if not isinstance(request_value.get("enabled"), bool):
                    raise ValueError("enabled must be a boolean")
                result = self.engine.set_stop_on_model_task_complete(request_value["enabled"])
                self._write_json(HTTPStatus.OK, result)
            except Exception as exc:
                LOG.warning("Rejected task-complete termination setting: %s", exc)
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        if path not in {"/v1/observations", "/v1/infer"}:
            self._error(HTTPStatus.NOT_FOUND, "not_found", "Unknown endpoint")
            return
        if self.headers.get_content_type() != "application/x-npz":
            self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "unsupported_media_type", "Use application/x-npz")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > self.max_payload_bytes:
                self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "payload_too_large", "Request exceeds max_payload_bytes")
                return
            if length <= 0:
                raise ValueError(f"Invalid Content-Length: {length}")
            packet = decode_observation_packet(self.rfile.read(length))
            if packet.input_schema != self.engine.input_spec()["schema_id"]:
                self._error(HTTPStatus.CONFLICT, "schema_mismatch", "Refresh /v1/input-spec and reassemble the input")
                return
            self.engine.validate_input(packet)
            if path == "/v1/infer":
                self._write_actions(self.engine.infer_request(packet))
            else:
                self.engine.submit(packet)
                self._write_json(HTTPStatus.ACCEPTED, {"accepted_step": packet.sim_step, "control_epoch": packet.control_epoch})
        except InferenceBusyError as exc:
            self._error(HTTPStatus.TOO_MANY_REQUESTS, "inference_busy", str(exc))
        except (ValueError, KeyError, TypeError, OSError, EOFError, BadZipFile) as exc:
            LOG.warning("Rejected observation: %s", exc)
            self._error(HTTPStatus.BAD_REQUEST, "invalid_input", str(exc))
        except Exception:
            LOG.exception("Inference request failed")
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "inference_failed", "See model worker logs")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/v1/input-spec":
            self._write_json(HTTPStatus.OK, self.engine.input_spec())
            return
        if parsed.path == "/v1/schema":
            self._write_json(HTTPStatus.OK, {
                "api_version": 1, "protocol_version": PROTOCOL_VERSION,
                "content_type": "application/x-npz", "max_payload_bytes": self.max_payload_bytes,
                "input": self.engine.input_spec(),
                "output": {"chunk_size": self.engine.policy.config.chunk_size,
                           "action_names": list(self.engine.action_names),
                           "model_action_names": list(self.engine.model_action_names),
                           "b2_execution_mode": self.engine.b2_execution_mode,
                           "z1_action_representation": self.engine.z1_action_representation,
                           "ee_pose_names": list(EE_POSE_ACTION_NAMES)},
                "endpoints": {"infer": "/v1/infer", "submit": "/v1/observations", "poll": "/v1/actions"},
                "concurrency": "one execution session per worker port; one in-flight /v1/infer; busy returns 429",
            })
            return
        if parsed.path == "/healthz":
            health = self.engine.health()
            self._write_json(HTTPStatus.OK if health["ok"] else HTTPStatus.SERVICE_UNAVAILABLE, health)
            return
        if parsed.path != "/v1/actions":
            self._error(HTTPStatus.NOT_FOUND, "not_found", "Unknown endpoint")
            return
        try:
            after = int(parse_qs(parsed.query).get("after", ["0"])[0])
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "invalid_cursor", "after must be an integer")
            return
        record = self.engine.latest_after(after)
        if record is None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        self._write_actions(record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument(
        "--compile-model",
        type=_parse_bool,
        default=None,
        help="Override the checkpoint torch.compile setting for inference.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
        default=None,
        help="torch.compile mode; only used when compilation is enabled.",
    )
    parser.add_argument("--warmup-inferences", type=int, default=1)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--low-level-hz", type=float, default=50.0)
    parser.add_argument(
        "--chunk-scheduling-mode",
        choices=CHUNK_SCHEDULING_MODES,
        default="RTC",
    )
    parser.add_argument("--stop-on-model-task-complete", type=_parse_bool, required=True)
    parser.add_argument("--b2-velocity-smoothing-time-constant-s", type=float, required=True)
    parser.add_argument(
        "--b2-pose-delta-control-mode",
        choices=B2_POSE_DELTA_CONTROL_MODES,
        required=True,
    )
    parser.add_argument("--rtc-execution-horizon", type=int, default=25)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=10.0)
    parser.add_argument(
        "--rtc-schedule", choices=[item.value for item in RTCAttentionSchedule], default="EXP"
    )
    parser.add_argument("--max-payload-mb", type=float, default=12.0)
    parser.add_argument("--rollout-dir", default="")
    args = parser.parse_args()
    if args.low_level_hz <= 0:
        parser.error("low-level-hz must be positive")
    if args.b2_velocity_smoothing_time_constant_s < 0:
        parser.error("b2-velocity-smoothing-time-constant-s must be non-negative")
    if args.rtc_execution_horizon <= 0:
        parser.error("RTC horizon must be positive")
    if args.num_inference_steps is not None and args.num_inference_steps <= 0:
        parser.error("num-inference-steps must be positive")
    if args.warmup_inferences < 0:
        parser.error("warmup-inferences must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    init_logging()
    engine = AsyncRTCPolicy(args)
    server = ThreadingHTTPServer((args.host, args.port), VLARequestHandler)
    server.engine = engine  # type: ignore[attr-defined]
    server.max_payload_bytes = int(args.max_payload_mb * 1024 * 1024)  # type: ignore[attr-defined]
    engine.start()
    LOG.info("Listening on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.stop()


if __name__ == "__main__":
    main()
