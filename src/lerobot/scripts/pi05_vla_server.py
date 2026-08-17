#!/usr/bin/env python
"""Serve asynchronous PI0.5 RTC chunks to the visual-wholebody simulator.

The simulator is the clock authority.  Observations carry a monotonically
increasing 50 Hz simulation step and the currently executing chunk/index.
Inference always consumes the newest observation and never blocks simulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from threading import Condition, Event, Lock, Thread
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import torch

from lerobot.configs import PreTrainedConfig, RTCAttentionSchedule
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.pi05.b2_action_transform import (
    EE_DELTA_ROTVEC_NAMES,
    _matrix_to_rot6d,
    _rotvec_to_matrix,
    action_dataset_indices,
)
from lerobot.policies.pi05.processor_pi05 import OBS_ACTION_HISTORY
from lerobot.policies.rtc import RTCConfig
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.utils import init_logging

LOG = logging.getLogger("pi05_vla_server")
PROTOCOL_VERSION = 4
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
EXECUTION_ACTION_SUFFIX_NAMES = (
    "gripper_target",
    "task_complete",
)
B2_EXECUTION_VELOCITY_NAMES = ("b2_vx", "b2_vy", "b2_omega_z")
B2_OBSERVED_VELOCITY_NAMES = ("b2_body_vx", "b2_body_vy", "b2_body_wz")


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}")


def execution_action_names(z1_action_representation: str) -> tuple[str, ...]:
    if z1_action_representation == "ee_pose":
        ee_names = EE_POSE_ACTION_NAMES
    elif z1_action_representation == "ee_delta":
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
    if z1_action_representation == "ee_delta" and set(EE_DELTA_ROTVEC_NAMES).issubset(columns):
        rotvec = torch.stack([columns[name] for name in EE_DELTA_ROTVEC_NAMES], dim=-1)
        rot6d = _matrix_to_rot6d(_rotvec_to_matrix(rotvec))
        for index, name in enumerate(EE_DELTA_ACTION_NAMES[:6]):
            columns[name] = rot6d[:, index]
    required = set(output_names) - {"b2_active", "arm_active", "task_complete"}
    missing = sorted(required - columns.keys())
    if missing:
        raise ValueError(f"Checkpoint postprocessor is missing executable actions: {missing}")
    if "arm_active" in columns or "arm_teleop_inactive" not in columns:
        raise ValueError(
            "Schema-v2 checkpoint must output arm_teleop_inactive and must not output legacy arm_active"
        )
    columns["arm_active"] = 1.0 - columns["arm_teleop_inactive"]
    columns.setdefault("b2_active", torch.ones_like(actions[:, 0]))
    columns.setdefault("task_complete", torch.zeros_like(actions[:, 0]))
    return torch.stack([columns[name] for name in output_names], dim=-1)


def _decode_discrete_actions(
    normalized_actions: torch.Tensor,
    postprocessed_actions: torch.Tensor,
    names: tuple[str, ...],
    *,
    mode: str,
    gripper_negative_value: float,
    gripper_nonnegative_value: float,
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
    decoded = postprocessed_actions.clone()
    indices = {name: index for index, name in enumerate(names)}
    for name in ("arm_teleop_inactive", "arm_reset"):
        if name in indices:
            index = indices[name]
            decoded[:, index] = (normalized_actions[:, index] > 0).to(decoded.dtype)
    if "gripper_target" in indices:
        index = indices["gripper_target"]
        negative_class = normalized_actions[:, index] < 0
        decoded[:, index] = torch.where(
            negative_class,
            decoded.new_tensor(gripper_negative_value),
            decoded.new_tensor(gripper_nonnegative_value),
        )
    if "task_complete" in indices:
        index = indices["task_complete"]
        complete = torch.cummax(normalized_actions[:, index] > 0, dim=0).values
        decoded[:, index] = complete.to(decoded.dtype)
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
    gripper_negative_value: float
    gripper_nonnegative_value: float


@dataclass(frozen=True)
class MemoryObservation:
    """One exact low-level telemetry sample from the 50 Hz simulator clock."""

    sim_step: int
    state: np.ndarray
    state_names: tuple[str, ...]
    active_sequence: int
    active_index: int
    executed_ee_target: np.ndarray


def _feature_shape(feature: object) -> tuple[int, ...]:
    shape = feature.get("shape") if isinstance(feature, dict) else getattr(feature, "shape", None)
    if shape is None:
        raise ValueError(f"Feature has no shape: {feature!r}")
    return tuple(int(value) for value in shape)


def _camera_bindings(camera_keys: tuple[str, ...]) -> tuple[str, str]:
    wrist = tuple(key for key in camera_keys if "wrist" in key.lower())
    base = tuple(key for key in camera_keys if "wrist" not in key.lower())
    if len(base) != 1 or len(wrist) != 1:
        raise ValueError(f"Cannot bind checkpoint cameras to base/wrist packets: {camera_keys}")
    return base[0], wrist[0]


def _resolve_checkpoint_action_representations(
    metadata_version: int,
    action_metadata: dict[str, object],
    raw_config: dict[str, object],
) -> tuple[str, str]:
    b2_representation = str(action_metadata["representation"])
    if metadata_version >= 4:
        z1_representation = str(action_metadata["z1_representation"])
    else:
        if b2_representation != "velocity":
            raise ValueError(
                "Deployment metadata versions 2 and 3 are supported only for B2 velocity; "
                "their local_trajectory labels use the retired cumulative-pose semantics"
            )
        if "z1_representation" in action_metadata or "z1_action_representation" in raw_config:
            raise ValueError(
                f"Malformed deployment metadata version {metadata_version}: Z1 representation must be absent"
            )
        z1_representation = "ee_pose"
    if b2_representation not in {"local_trajectory", "velocity"}:
        raise ValueError(f"Unsupported B2 deployment action representation: {b2_representation!r}")
    if z1_representation not in {"ee_pose", "ee_delta"}:
        raise ValueError(f"Unsupported Z1 deployment action representation: {z1_representation!r}")
    return b2_representation, z1_representation


def _load_checkpoint_contract(
    policy_path: Path,
    config: PreTrainedConfig,
    low_level_hz: float,
) -> CheckpointContract:
    """Validate the checkpoint-local contract and its explicit version migration."""
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
    if saved.get("format") != "lerobot.pi05.deployment" or metadata_version not in {2, 3, 4, 5, 6}:
        raise ValueError(
            "Checkpoint deployment metadata must use lerobot.pi05.deployment version 2 through 6"
        )
    if not config.io_schema_resolved:
        raise ValueError("Checkpoint contains unresolved PI0.5 I/O metadata")

    action = saved["action"]
    observation = saved["observation"]
    timing = saved["timing"]
    b2_action_representation, z1_action_representation = _resolve_checkpoint_action_representations(
        int(metadata_version), action, raw_config
    )
    if metadata_version == 6:
        expected = config.deployment_metadata()
        if saved != expected:
            raise ValueError("pi05_deployment_metadata.json disagrees with checkpoint config.json")
    elif metadata_version == 5:
        config.ee_delta_rotation_representation = "rot6d"
        expected = config.deployment_metadata()
        expected["version"] = 5
        expected["action"].pop("ee_delta_rotation_representation", None)
        if saved != expected:
            raise ValueError("pi05_deployment_metadata.json disagrees with checkpoint config.json")
    else:
        config.z1_action_representation = z1_action_representation
    discrete_mode = str(action.get("discrete_training_mode", "continuous_flow"))
    if discrete_mode not in {"continuous_flow", "structured_temporal"}:
        raise ValueError(f"Unsupported discrete action training mode: {discrete_mode!r}")
    config_mode = str(raw_config.get("discrete_action_training_mode", "continuous_flow"))
    if config_mode != discrete_mode:
        raise ValueError("Discrete action training mode disagrees between metadata and config.json")
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
        history_sim_steps = np.asarray(packet["history_sim_steps"], dtype=np.int64).reshape(-1)
        history_states = np.asarray(packet["history_states"], dtype=np.float32)
        history_active_sequences = np.asarray(packet["history_active_sequences"], dtype=np.int64).reshape(-1)
        history_active_indices = np.asarray(packet["history_active_indices"], dtype=np.int64).reshape(-1)
        history_executed_ee_targets = np.asarray(packet["history_executed_ee_targets"], dtype=np.float32)
        history_count = len(history_sim_steps)
        expected_shapes = (
            history_states.shape == (history_count, state.size)
            and history_active_sequences.shape == (history_count,)
            and history_active_indices.shape == (history_count,)
            and history_executed_ee_targets.shape == (history_count, len(EE_POSE_ACTION_NAMES))
        )
        if history_count == 0 or not expected_shapes:
            raise ValueError(
                "Invalid 50 Hz telemetry history shapes: "
                f"steps={history_sim_steps.shape}, states={history_states.shape}, "
                f"sequences={history_active_sequences.shape}, indices={history_active_indices.shape}, "
                f"ee={history_executed_ee_targets.shape}"
            )
        if np.any(np.diff(history_sim_steps) != 1):
            raise ValueError("50 Hz telemetry history must contain consecutive simulation steps")
        if (
            not np.isfinite(history_states).all()
            or not np.isfinite(history_executed_ee_targets).all()
            or np.any(history_active_indices < 0)
        ):
            raise ValueError("50 Hz telemetry history contains invalid values")
        sim_step = int(packet["sim_step"].item())
        active_sequence = int(packet["active_sequence"].item())
        active_index = int(packet["active_index"].item())
        if (
            history_sim_steps[-1] != sim_step
            or history_active_sequences[-1] != active_sequence
            or history_active_indices[-1] != active_index
            or not np.array_equal(history_states[-1], state)
            or not np.array_equal(history_executed_ee_targets[-1], executed_ee_target)
        ):
            raise ValueError("Current observation does not match the final 50 Hz telemetry sample")
        telemetry_history = tuple(
            MemoryObservation(
                sim_step=int(history_sim_steps[index]),
                state=history_states[index].copy(),
                state_names=state_names,
                active_sequence=int(history_active_sequences[index]),
                active_index=int(history_active_indices[index]),
                executed_ee_target=history_executed_ee_targets[index].copy(),
            )
            for index in range(history_count)
        )
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
            telemetry_history=telemetry_history,
            server_received_ns=time.monotonic_ns(),
        )


def encode_action_packet(record: ActionRecord) -> bytes:
    output = BytesIO()
    np.savez(
        output,
        protocol_version=np.asarray(PROTOCOL_VERSION, dtype=np.int64),
        sequence=np.asarray(record.sequence, dtype=np.int64),
        source_step=np.asarray(record.source_step, dtype=np.int64),
        inference_seconds=np.asarray(record.inference_seconds, dtype=np.float64),
        action_names=np.asarray(record.action_names, dtype=np.str_),
        z1_action_representation=np.asarray(record.z1_action_representation, dtype=np.str_),
        processed_actions=record.processed.numpy().astype(np.float32, copy=False),
    )
    return output.getvalue()


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
    telemetry_history: tuple[MemoryObservation, ...]
    server_received_ns: int


@dataclass(frozen=True)
class ActionRecord:
    sequence: int
    source_step: int
    inference_seconds: float
    action_names: tuple[str, ...]
    z1_action_representation: str
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
                history_sim_steps=np.asarray(
                    [record.sim_step for record in packet.telemetry_history], dtype=np.int64
                ),
                history_states=np.stack([record.state for record in packet.telemetry_history]),
                history_active_sequences=np.asarray(
                    [record.active_sequence for record in packet.telemetry_history], dtype=np.int64
                ),
                history_active_indices=np.asarray(
                    [record.active_index for record in packet.telemetry_history], dtype=np.int64
                ),
                history_executed_ee_targets=np.stack(
                    [record.executed_ee_target for record in packet.telemetry_history]
                ),
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
            telemetry_history_first_step=packet.telemetry_history[0].sim_step,
            telemetry_history_count=len(packet.telemetry_history),
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
                sequence=np.asarray(record.sequence),
                source_step=np.asarray(record.source_step),
                action_names=np.asarray(record.action_names),
                z1_action_representation=np.asarray(record.z1_action_representation),
                original_actions=record.original.numpy(),
                unsmoothed_execution_actions=record.unsmoothed.numpy(),
                processed_actions=record.processed.numpy(),
                inference_seconds=np.asarray(record.inference_seconds),
                inference_started_ns=np.asarray(record.inference_started_ns),
                inference_finished_ns=np.asarray(record.inference_finished_ns),
                inference_delay_steps=np.asarray(record.inference_delay_steps),
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
            velocity_smoothing_transition_step=record.velocity_smoothing_transition_step,
            sim_steps_per_wall_second=record.sim_steps_per_wall_second,
        )


class AsyncRTCPolicy:
    """Latest-only observation mailbox plus a single asynchronous GPU worker."""

    def __init__(self, args: argparse.Namespace):
        self.recorder = RolloutRecorder(args.rollout_dir)
        self.recorder.event("vla_loading_started", policy_path=args.policy_path)
        policy_path = _resolve_policy_path(args.policy_path)
        config = PreTrainedConfig.from_pretrained(policy_path)
        config.pretrained_path = policy_path
        config.device = args.device
        contract = _load_checkpoint_contract(policy_path, config, float(args.low_level_hz))
        if args.num_inference_steps is not None:
            config.num_inference_steps = args.num_inference_steps

        LOG.info("Loading checkpoint-only policy=%s", policy_path)
        self.policy = make_policy(config)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(policy_path),
            pretrained_revision=getattr(config, "pretrained_revision", None),
        )

        rtc_config = RTCConfig(
            enabled=True,
            prefix_attention_schedule=RTCAttentionSchedule(args.rtc_schedule.upper()),
            max_guidance_weight=args.rtc_max_guidance_weight,
            execution_horizon=args.rtc_execution_horizon,
        )
        self.policy.config.rtc_config = rtc_config
        self.policy.init_rtc_processor()
        self.rtc_config = rtc_config
        self.device = torch.device(args.device)
        self.low_level_hz = float(args.low_level_hz)
        self.stop_on_model_task_complete = bool(args.stop_on_model_task_complete)
        self.b2_velocity_smoothing_time_constant_s = float(args.b2_velocity_smoothing_time_constant_s)
        self.period_s = 1.0 / float(args.high_level_hz)
        self.camera_keys = contract.camera_keys
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
        self.discrete_action_training_mode = contract.discrete_action_training_mode
        self.gripper_negative_value = contract.gripper_negative_value
        self.gripper_nonnegative_value = contract.gripper_nonnegative_value
        self.z1_action_representation = contract.z1_action_representation
        self.action_names = execution_action_names(self.z1_action_representation)
        self.source_state_names = contract.source_state_names
        self.selected_state_names = contract.selected_state_names
        self.source_action_names = contract.source_action_names
        self.mem_enabled = bool(config.mem_vit_enabled)
        self.mem_image_num_frames = int(config.mem_vit_num_frames) if self.mem_enabled else 1
        self.mem_image_stride = max(
            1, round(float(config.mem_vit_frame_interval_seconds or 0.0) * self.low_level_hz)
        )
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
                if key not in {"representation", "z1_representation"}
            }
        )
        max_history_steps = max(
            (self.mem_image_num_frames - 1) * self.mem_image_stride,
            (self.state_num_frames - 1) * self.state_history_stride,
        )
        self._observation_history: deque[MemoryObservation] = deque(maxlen=max_history_steps + 2)
        self._image_history: deque[ObservationPacket] = deque(maxlen=256)

        self._mailbox_condition = Condition()
        self._mailbox: ObservationPacket | None = None
        self._mailbox_version = 0
        self._last_inferred_version = 0
        self._records_lock = Lock()
        self._records: dict[int, ActionRecord] = {}
        self._latest_record: ActionRecord | None = None
        self._next_sequence = 1
        self._latencies: deque[float] = deque(maxlen=args.rtc_latency_window)
        self._sim_rates: deque[float] = deque(maxlen=args.rtc_latency_window)
        self._previous_observation_clock: tuple[int, int] | None = None
        self._control_epoch: int | None = None
        self._task_complete_latched = False
        self._stop = Event()
        self._worker = Thread(target=self._run, name="pi05-rtc-worker", daemon=True)
        self._last_error: str | None = None
        self._inference_count = 0
        self._warmup_inferences = int(args.warmup_inferences)
        self.recorder.event(
            "vla_loading_finished",
            action_names=self.action_names,
            postprocessed_action_names=self.postprocessed_action_names,
            camera_keys=self.camera_keys,
            base_camera_key=self.base_camera_key,
            wrist_camera_key=self.wrist_camera_key,
            checkpoint_metadata_version=contract.metadata_version,
            model_action_representation=contract.model_action_representation,
            z1_model_action_representation=contract.z1_action_representation,
            ee_delta_rotation_representation=contract.ee_delta_rotation_representation,
            b2_execution_action_representation="velocity",
            z1_execution_action_representation=contract.z1_action_representation,
            execution_action_protocol="rtc_action_packet_v4_50hz_history",
            stop_on_model_task_complete=self.stop_on_model_task_complete,
            b2_velocity_smoothing="causal_first_order_low_pass",
            b2_velocity_smoothing_time_constant_s=self.b2_velocity_smoothing_time_constant_s,
            arm_gate_conversion="one_minus_arm_teleop_inactive",
            discrete_action_training_mode=self.discrete_action_training_mode,
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

    def start(self) -> None:
        self._warmup()
        self._worker.start()

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
        for _ in range(self._warmup_inferences):
            batch = self.preprocessor(self._warmup_batch())
            with torch.inference_mode():
                actions = self.policy.predict_action_chunk(
                    batch,
                    inference_delay=0,
                    prev_chunk_left_over=None,
                )
                self.postprocessor(actions).detach().cpu()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        self.recorder.event(
            "vla_warmup_finished",
            inference_count=self._warmup_inferences,
            elapsed_ms=elapsed * 1000.0,
        )
        LOG.info("Completed %d VLA warmup inference(s) in %.3fs", self._warmup_inferences, elapsed)

    def stop(self) -> None:
        self._stop.set()
        with self._mailbox_condition:
            self._mailbox_condition.notify_all()
        self._worker.join(timeout=5.0)

    def submit(self, packet: ObservationPacket) -> None:
        if self.stop_on_model_task_complete and packet.active_sequence >= 0 and packet.active_index > 0:
            with self._records_lock:
                active_record = self._records.get(packet.active_sequence)
            if active_record is not None and "task_complete" in active_record.action_names:
                complete_index = active_record.action_names.index("task_complete")
                executed_count = min(packet.active_index, len(active_record.processed))
                if bool((active_record.processed[:executed_count, complete_index] > 0.5).any()):
                    self._task_complete_latched = True
        with self._mailbox_condition:
            epoch_reset = self._control_epoch is not None and packet.control_epoch != self._control_epoch
            if epoch_reset:
                self._task_complete_latched = False
                self._mailbox = None
                self._observation_history.clear()
                self._image_history.clear()
                self._previous_observation_clock = None
            self._control_epoch = packet.control_epoch
            if self._mailbox is not None and packet.sim_step < self._mailbox.sim_step:
                return
            if self._previous_observation_clock is not None:
                previous_step, previous_ns = self._previous_observation_clock
                step_delta = packet.sim_step - previous_step
                wall_delta_s = (packet.server_received_ns - previous_ns) / 1e9
                if step_delta > 0 and wall_delta_s > 0:
                    self._sim_rates.append(step_delta / wall_delta_s)
            self._previous_observation_clock = (packet.sim_step, packet.server_received_ns)
            last_history_step = self._observation_history[-1].sim_step if self._observation_history else -1
            new_records = tuple(
                record for record in packet.telemetry_history if record.sim_step > last_history_step
            )
            if new_records and last_history_step >= 0 and new_records[0].sim_step != last_history_step + 1:
                raise ValueError(
                    "Gap in simulator 50 Hz telemetry history: "
                    f"last={last_history_step}, next={new_records[0].sim_step}"
                )
            self._observation_history.extend(new_records)
            if not self._observation_history or self._observation_history[-1].sim_step != packet.sim_step:
                raise ValueError(f"Telemetry history did not reach current step {packet.sim_step}")
            if self._image_history and packet.sim_step <= self._image_history[-1].sim_step:
                self._image_history.clear()
            self._image_history.append(packet)
            self._mailbox = packet
            self._mailbox_version += 1
            self._mailbox_condition.notify()
        self.recorder.observation(packet)

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
            "worker_alive": self._worker.is_alive(),
            "inference_count": self._inference_count,
            "latest_sequence": latest.sequence if latest else 0,
            "latest_source_step": latest.source_step if latest else -1,
            "last_error": self._last_error,
        }

    def _previous_prefix(self, packet: ObservationPacket) -> torch.Tensor | None:
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
        horizon = self.rtc_config.execution_horizon
        if len(prefix) >= horizon:
            return prefix[:horizon]
        padded = torch.zeros((horizon, prefix.shape[-1]), dtype=prefix.dtype, device=prefix.device)
        padded[: len(prefix)] = prefix
        return padded

    def _estimated_delay(self) -> tuple[int, float]:
        with self._mailbox_condition:
            sim_rate = max(self._sim_rates) if self._sim_rates else self.low_level_hz
        if not self._latencies:
            return 0, sim_rate
        return int(math.ceil(max(self._latencies) * sim_rate)), sim_rate

    @staticmethod
    def _sample_memory_records(
        history: tuple[MemoryObservation, ...], current_step: int, *, count: int, stride: int
    ) -> tuple[list[MemoryObservation], torch.Tensor]:
        if not history:
            raise ValueError("Observation memory is empty")
        records: list[MemoryObservation] = []
        is_pad: list[bool] = []
        first_step = history[0].sim_step
        for slot in range(count):
            target_step = current_step - (count - 1 - slot) * stride
            if target_step < first_step:
                records.append(history[0])
                is_pad.append(True)
                continue
            record = next(
                (candidate for candidate in reversed(history) if candidate.sim_step == target_step),
                None,
            )
            if record is None:
                raise ValueError(f"Missing exact 50 Hz telemetry sample for sim_step={target_step}")
            records.append(record)
            is_pad.append(False)
        return records, torch.tensor(is_pad, dtype=torch.bool).unsqueeze(0)

    @staticmethod
    def _sample_image_records(
        history: tuple[ObservationPacket, ...], current_step: int, *, count: int, stride: int
    ) -> tuple[list[ObservationPacket], torch.Tensor]:
        if not history:
            raise ValueError("Image observation memory is empty")
        records: list[ObservationPacket] = []
        is_pad: list[bool] = []
        first_step = history[0].sim_step
        for slot in range(count):
            target_step = current_step - (count - 1 - slot) * stride
            if target_step < first_step:
                records.append(history[0])
                is_pad.append(True)
                continue
            record = next(
                (candidate for candidate in reversed(history) if candidate.sim_step <= target_step),
                history[0],
            )
            records.append(record)
            is_pad.append(False)
        return records, torch.tensor(is_pad, dtype=torch.bool).unsqueeze(0)

    def _source_state(self, record: MemoryObservation) -> torch.Tensor:
        indices = {name: index for index, name in enumerate(record.state_names)}
        missing = tuple(name for name in self.source_state_names if name not in indices)
        if missing:
            raise ValueError(f"Simulator observation is missing checkpoint state fields: {missing}")
        return torch.from_numpy(record.state[[indices[name] for name in self.source_state_names]])

    def _executed_source_action(self, record: MemoryObservation) -> torch.Tensor | None:
        if record.active_sequence < 0:
            return None
        with self._records_lock:
            action_record = self._records.get(record.active_sequence)
        if action_record is None or record.active_index <= 0 or len(action_record.processed) == 0:
            return None
        executed_index = min(record.active_index, len(action_record.processed)) - 1
        executed = action_record.processed[executed_index]
        columns = {name: executed[index] for index, name in enumerate(action_record.action_names)}
        columns["arm_teleop_inactive"] = 1.0 - columns["arm_active"]
        columns.update(zip(EE_POSE_ACTION_NAMES, torch.from_numpy(record.executed_ee_target), strict=True))
        try:
            source = torch.stack([columns[name] for name in self.source_action_names])
        except KeyError as exc:
            raise ValueError(f"Cannot reconstruct historical source action {exc.args[0]!r}") from exc
        return source[list(self.action_history_indices)]

    def _make_batch(self, packet: ObservationPacket) -> dict[str, object]:
        with self._mailbox_condition:
            history = tuple(self._observation_history)
            image_history = tuple(self._image_history)

        state_records, state_pad = self._sample_memory_records(
            history,
            packet.sim_step,
            count=self.state_num_frames,
            stride=self.state_history_stride,
        )
        state = torch.stack([self._source_state(record) for record in state_records]).unsqueeze(0)
        batch: dict[str, object] = {
            OBS_STATE: state,
            f"{OBS_STATE}_is_pad": state_pad,
        }

        image_records, image_pad = self._sample_image_records(
            image_history,
            packet.sim_step,
            count=self.mem_image_num_frames,
            stride=self.mem_image_stride,
        )
        encoded_images = {
            self.base_camera_key: [record.base_jpeg for record in image_records],
            self.wrist_camera_key: [record.wrist_jpeg for record in image_records],
        }
        for key, encoded_window in encoded_images.items():
            image_window = torch.stack(
                [
                    torch.from_numpy(np.ascontiguousarray(_decode_jpeg(encoded, key)))
                    .permute(2, 0, 1)
                    .to(dtype=torch.float32)
                    .div_(255.0)
                    for encoded in encoded_window
                ]
            ).unsqueeze(0)
            batch[key] = image_window if self.mem_enabled else image_window[:, -1]
            batch[f"{key}_is_pad"] = image_pad if self.mem_enabled else image_pad[:, -1]

        if self.action_history_enabled:
            action_count = self.state_num_frames - 1
            action_records, action_pad = self._sample_memory_records(
                history,
                packet.sim_step - self.state_history_stride,
                count=action_count,
                stride=self.state_history_stride,
            )
            action_dim = len(self.action_history_indices)
            action_values = []
            for index, record in enumerate(action_records):
                action = self._executed_source_action(record)
                if action is None:
                    action = torch.zeros(action_dim, dtype=torch.float32)
                    action_pad[0, index] = True
                action_values.append(action)
            batch[OBS_ACTION_HISTORY] = torch.stack(action_values).unsqueeze(0)
            batch[f"{OBS_ACTION_HISTORY}_is_pad"] = action_pad

        batch["task"] = [packet.task]
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
        started_ns = time.monotonic_ns()
        started = time.perf_counter()
        batch = self.preprocessor(self._make_batch(packet))
        preprocessed = time.perf_counter()
        prefix = self._previous_prefix(packet)
        inference_delay_steps, sim_rate = self._estimated_delay()
        # RTC prefix guidance is a closed-form forward correction, so this
        # online path never needs autograd or tensor version counters.
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(
                batch,
                inference_delay=inference_delay_steps,
                prev_chunk_left_over=prefix,
            )
            original = actions.squeeze(0).detach().cpu().clone()
            predicted = time.perf_counter()
            postprocessed = self.postprocessor(actions).squeeze(0).detach().cpu().to(torch.float32)
            postprocessed = _decode_discrete_actions(
                original,
                postprocessed,
                self.postprocessed_action_names,
                mode=self.discrete_action_training_mode,
                gripper_negative_value=self.gripper_negative_value,
                gripper_nonnegative_value=self.gripper_nonnegative_value,
            )
            unsmoothed = _to_execution_actions(
                postprocessed,
                self.postprocessed_action_names,
                self.z1_action_representation,
            )
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
        self._latencies.append(elapsed)
        record = ActionRecord(
            sequence=self._next_sequence,
            source_step=packet.sim_step,
            inference_seconds=elapsed,
            action_names=self.action_names,
            z1_action_representation=self.z1_action_representation,
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
            velocity_smoothing_transition_step=smoothing_transition_step,
            sim_steps_per_wall_second=sim_rate,
        )
        self._next_sequence += 1
        return record

    def _run(self) -> None:
        next_start = 0.0
        while not self._stop.is_set():
            with self._mailbox_condition:
                self._mailbox_condition.wait_for(
                    lambda: (
                        self._stop.is_set()
                        or (
                            self._mailbox is not None and self._mailbox_version != self._last_inferred_version
                        )
                    ),
                    timeout=0.25,
                )
                if self._stop.is_set():
                    return
                packet = self._mailbox
                version = self._mailbox_version
            if packet is None or version == self._last_inferred_version:
                continue
            remaining = next_start - time.perf_counter()
            if remaining > 0 and self._stop.wait(remaining):
                return
            # A newer observation may have arrived while enforcing the 4 Hz
            # start-rate cap. Refresh once so an older mailbox item is never
            # inferred merely because it was current before the wait.
            with self._mailbox_condition:
                if self._mailbox_version != version:
                    packet = self._mailbox
                    version = self._mailbox_version
            if packet is None or version == self._last_inferred_version:
                continue
            inference_started = time.perf_counter()
            try:
                record = self._infer(packet)
                with self._records_lock:
                    self._records[record.sequence] = record
                    self._latest_record = record
                    while len(self._records) > 16:
                        self._records.pop(min(self._records))
                self._last_inferred_version = version
                self._inference_count += 1
                self._last_error = None
                self.recorder.action(record)
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
                self._last_inferred_version = version
                LOG.exception("Inference failed for sim_step=%d", packet.sim_step)
                if self._stop.wait(0.5):
                    return
            # Cap inference starts at high_level_hz. If inference itself is
            # slower than the period, immediately consume the newest mailbox.
            next_start = inference_started + self.period_s


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
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/v1/observations":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > self.max_payload_bytes:
                raise ValueError(f"Invalid Content-Length: {length}")
            packet = decode_observation_packet(self.rfile.read(length))
            self.engine.submit(packet)
            self._write_json(HTTPStatus.ACCEPTED, {"accepted_step": packet.sim_step})
        except Exception as exc:
            LOG.warning("Rejected observation: %s", exc)
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/healthz":
            health = self.engine.health()
            self._write_json(HTTPStatus.OK if health["ok"] else HTTPStatus.SERVICE_UNAVAILABLE, health)
            return
        if parsed.path != "/v1/actions":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            after = int(parse_qs(parsed.query).get("after", ["0"])[0])
        except ValueError:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "after must be an integer"})
            return
        record = self.engine.latest_after(after)
        if record is None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        payload = encode_action_packet(record)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-npz")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--warmup-inferences", type=int, default=1)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--high-level-hz", type=float, default=4.0)
    parser.add_argument("--low-level-hz", type=float, default=50.0)
    parser.add_argument("--stop-on-model-task-complete", type=_parse_bool, required=True)
    parser.add_argument("--b2-velocity-smoothing-time-constant-s", type=float, required=True)
    parser.add_argument("--rtc-execution-horizon", type=int, default=13)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=10.0)
    parser.add_argument(
        "--rtc-schedule", choices=[item.value for item in RTCAttentionSchedule], default="EXP"
    )
    parser.add_argument("--rtc-latency-window", type=int, default=10)
    parser.add_argument("--max-payload-mb", type=float, default=12.0)
    parser.add_argument("--rollout-dir", default="")
    args = parser.parse_args()
    if args.high_level_hz <= 0 or args.low_level_hz <= 0:
        parser.error("both frequencies must be positive")
    if args.b2_velocity_smoothing_time_constant_s < 0:
        parser.error("b2-velocity-smoothing-time-constant-s must be non-negative")
    if args.rtc_execution_horizon <= 0 or args.rtc_latency_window <= 0:
        parser.error("RTC horizon/window must be positive")
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
