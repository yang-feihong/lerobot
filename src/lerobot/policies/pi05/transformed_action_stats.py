#!/usr/bin/env python

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.utils.constants import ACTION, OBS_STATE

from .b2_action_transform import (
    ARM_TELEOP_INACTIVE_NAME,
    TASK_COMPLETE_NAME,
    action_schema_kwargs,
    b2_execution_action_names,
    b2_pose_delta_action_names,
    ee_delta_sample_validity,
    ee_delta_transition_validity,
    encode_b2_action_chunk,
    select_dataset_action_supervision,
)

PI05_TRANSFORMED_ACTION_STATS_NAME = "pi05_transformed_action_stats.json"
EE_VALID_TRANSITION_COUNT_KEYS = (
    "ee_active_non_reset_both_endpoints",
    "ee_all_transitions",
)


def transformed_action_stats_ee_valid_count(payload: dict[str, Any]) -> int:
    counts = payload["counts"]
    present = [key for key in EE_VALID_TRANSITION_COUNT_KEYS if key in counts]
    if len(present) != 1:
        raise ValueError(f"Expected exactly one EE-valid transition count, got {present}")
    return int(counts[present[0]])


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_transformed_action_stats(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def load_transformed_action_stats(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "lerobot.pi05.transformed_action_stats" or payload.get("version") != 1:
        raise ValueError(f"Unsupported PI0.5 transformed-action statistics: {path}")
    return payload


def validate_transformed_action_stats(payload: dict[str, Any], dataset, config) -> None:
    dataset_record = payload["dataset"]
    expected_dataset = {
        "num_frames": int(dataset.num_frames),
        "num_episodes": int(dataset.num_episodes),
        "fps": float(dataset.meta.fps),
    }
    for key, expected in expected_dataset.items():
        if dataset_record.get(key) != expected:
            raise ValueError(
                f"Transformed-action statistics dataset mismatch for {key}: "
                f"saved={dataset_record.get(key)!r}, current={expected!r}"
            )
    saved_schema = payload["schema"]
    for key, expected in action_schema_kwargs(config).items():
        if saved_schema.get(key) != expected:
            raise ValueError(
                f"Transformed-action statistics schema mismatch for {key}: "
                f"saved={saved_schema.get(key)!r}, current={expected!r}"
            )
    for key in (
        "ee_target_dataset_semantics",
        "ee_delta_supervision_mode",
        "gripper_target_representation",
        "ee_supervision_source",
    ):
        expected = getattr(config, key)
        if saved_schema.get(key) != expected:
            raise ValueError(
                f"Transformed-action statistics schema mismatch for {key}: "
                f"saved={saved_schema.get(key)!r}, current={expected!r}"
            )
    if float(saved_schema["control_frequency_hz"]) != float(config.control_frequency_hz):
        raise ValueError("Transformed-action statistics control frequency does not match the policy")
    if list(saved_schema["action_names"]) != list(config.action_feature_names):
        raise ValueError("Transformed-action statistics action names do not match the policy")


def assert_transformed_action_stats_equal(saved: dict[str, Any], measured: dict[str, Any]) -> None:
    if saved["schema"] != measured["schema"] or saved["counts"] != _jsonable(measured["counts"]):
        raise ValueError("Checkpoint transformed-action statistics provenance does not match the dataset")
    saved_stats = saved["stats"][ACTION]
    measured_stats = measured["stats"][ACTION]
    if saved_stats.keys() != measured_stats.keys():
        raise ValueError("Checkpoint transformed-action statistics fields do not match the measured fields")
    for name in saved_stats:
        if not np.array_equal(np.asarray(saved_stats[name]), np.asarray(measured_stats[name])):
            raise ValueError(
                f"Checkpoint transformed-action statistic {name!r} does not match a fresh episode traversal"
            )


def _episode_arrays(dataset):
    columns = [ACTION, "episode_index", "frame_index"]
    use_global_pose = bool(dataset.meta.features.get(OBS_STATE))
    if use_global_pose:
        columns.append(OBS_STATE)
    table = dataset.hf_dataset.select_columns(columns).with_format("numpy")

    current_episode: int | None = None
    action_parts: list[np.ndarray] = []
    state_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []
    for batch in table.iter(batch_size=65_536):
        episode_indices = np.asarray(batch["episode_index"], dtype=np.int64)
        actions = np.asarray(batch[ACTION], dtype=np.float32)
        frames = np.asarray(batch["frame_index"], dtype=np.int64)
        states = np.asarray(batch[OBS_STATE], dtype=np.float32) if OBS_STATE in batch else None
        start = 0
        while start < len(episode_indices):
            episode = int(episode_indices[start])
            end = start + 1
            while end < len(episode_indices) and int(episode_indices[end]) == episode:
                end += 1
            if current_episode is not None and episode != current_episode:
                yield (
                    current_episode,
                    np.concatenate(frame_parts),
                    np.concatenate(action_parts),
                    (np.concatenate(state_parts) if state_parts else None),
                )
                action_parts = []
                state_parts = []
                frame_parts = []
            current_episode = episode
            action_parts.append(actions[start:end])
            frame_parts.append(frames[start:end])
            if states is not None:
                state_parts.append(states[start:end])
            start = end
    if current_episode is not None:
        yield (
            current_episode,
            np.concatenate(frame_parts),
            np.concatenate(action_parts),
            np.concatenate(state_parts) if state_parts else None,
        )


def compute_transformed_action_stats(dataset, config) -> dict[str, Any]:
    """Traverse continuous episodes and measure statistics after the configured action transform."""
    if not config.io_schema_resolved:
        raise ValueError("Transformed B2+Z1 statistics require a resolved PI0.5 I/O schema")
    if config.action_dt_seconds is None or config.control_frequency_hz is None:
        raise ValueError("PI0.5 action timing must be resolved before transformed statistics are computed")
    dataset_fps = float(dataset.meta.fps)
    stride_float = dataset_fps / float(config.control_frequency_hz)
    stride = round(stride_float)
    if stride < 1 or abs(stride - stride_float) > 1e-6:
        raise ValueError(
            "Exact transformed statistics require the model interval to be an integer number of dataset frames: "
            f"dataset_fps={dataset_fps}, control_frequency_hz={config.control_frequency_hz}"
        )

    schema = action_schema_kwargs(config)
    name_fn = (
        b2_pose_delta_action_names
        if config.b2_action_representation == "pose_delta"
        else b2_execution_action_names
    )
    action_names = name_fn(list(config.dataset_action_feature_names[:16]), **schema)
    if action_names is None:
        raise ValueError("Resolved PI0.5 schema has no model action names")
    values_by_dimension: list[list[np.ndarray]] = [[] for _ in action_names]
    counts_by_dimension = np.zeros(len(action_names), dtype=np.int64)
    total_transitions = 0
    ee_valid_transitions = 0
    ee_indices = [i for i, name in enumerate(action_names) if name.startswith("height_invariant_ee_")]
    def accumulate(transformed: torch.Tensor, ee_valid: torch.Tensor) -> None:
        nonlocal total_transitions, ee_valid_transitions
        transformed_np = transformed.reshape(-1, transformed.shape[-1]).numpy()
        ee_valid_np = ee_valid.reshape(-1).numpy()
        total_transitions += len(transformed_np)
        ee_valid_transitions += int(ee_valid_np.sum())
        for index in range(len(action_names)):
            selected = transformed_np[ee_valid_np, index] if index in ee_indices else transformed_np[:, index]
            if selected.size:
                values_by_dimension[index].append(selected.astype(np.float32, copy=False))
                counts_by_dimension[index] += selected.size

    for episode_index, frame_indices, actions_np, states_np in _episode_arrays(dataset):
        if len(actions_np) <= stride:
            continue
        if frame_indices[0] != 0 or not np.all(np.diff(frame_indices) == 1):
            raise ValueError(
                f"Episode {episode_index} is not a continuous frame sequence: "
                f"first={frame_indices[0]}, last={frame_indices[-1]}, rows={len(frame_indices)}"
            )
        needs_full_chunk = (
            config.b2_action_representation == "pose_delta"
            or config.z1_action_representation == "ee_state_delta"
        )
        if needs_full_chunk:
            action_count = config.chunk_size + int(config.z1_action_representation == "ee_delta")
            last_offset = (action_count - 1) * stride
            num_starts = len(actions_np) - last_offset
            if num_starts <= 0:
                continue
            offsets = np.arange(action_count, dtype=np.int64) * stride
            for batch_start in range(0, num_starts, 2048):
                starts = np.arange(batch_start, min(batch_start + 2048, num_starts), dtype=np.int64)
                raw = torch.from_numpy(actions_np[starts[:, None] + offsets[None, :]])
                raw = select_dataset_action_supervision(raw, source=config.ee_supervision_source)
                ee_state_anchor = None
                if config.z1_action_representation == "ee_state_delta":
                    if states_np is None or not config.ee_state_anchor_indices:
                        raise ValueError("ee_state_delta statistics require resolved EE state indices")
                    ee_state_anchor = torch.from_numpy(
                        states_np[starts][:, config.ee_state_anchor_indices]
                    )
                transformed = encode_b2_action_chunk(
                    raw,
                    dt=float(config.action_dt_seconds),
                    ee_state_anchor=ee_state_anchor,
                    **schema,
                )
                valid = ee_delta_sample_validity(
                    raw,
                    None,
                    representation=config.z1_action_representation,
                    supervision_mode=config.ee_delta_supervision_mode,
                )
                accumulate(transformed, valid)
            continue
        source = torch.from_numpy(actions_np[:-stride])
        target = torch.from_numpy(actions_np[stride:])
        pairs = torch.stack((source, target), dim=1)
        pairs = select_dataset_action_supervision(
            pairs,
            source=config.ee_supervision_source,
        )
        transformed = encode_b2_action_chunk(
            pairs,
            dt=float(config.action_dt_seconds),
            **schema,
        ).squeeze(1)
        if transformed.shape[-1] != len(action_names):
            raise ValueError(
                f"Transformed action width {transformed.shape[-1]} does not match names {len(action_names)}"
            )
        ee_valid = ee_delta_transition_validity(
            pairs,
            supervision_mode=config.ee_delta_supervision_mode,
        ).squeeze(1)
        accumulate(transformed, ee_valid)

    if total_transitions == 0 or any(count == 0 for count in counts_by_dimension):
        raise ValueError(
            f"Insufficient continuous transitions for transformed statistics: total={total_transitions}, "
            f"counts={counts_by_dimension.tolist()}"
        )

    arrays = [np.concatenate(parts) for parts in values_by_dimension]
    stats: dict[str, Any] = {
        "min": np.asarray([array.min() for array in arrays], dtype=np.float32),
        "max": np.asarray([array.max() for array in arrays], dtype=np.float32),
        "mean": np.asarray([array.mean(dtype=np.float64) for array in arrays], dtype=np.float32),
        "std": np.asarray([array.std(dtype=np.float64) for array in arrays], dtype=np.float32),
        "count": np.asarray([total_transitions], dtype=np.int64),
    }
    for quantile, name in ((0.01, "q01"), (0.10, "q10"), (0.50, "q50"), (0.90, "q90"), (0.99, "q99")):
        stats[name] = np.asarray([np.quantile(array, quantile) for array in arrays], dtype=np.float32)

    categorical_names = {ARM_TELEOP_INACTIVE_NAME, "arm_reset", TASK_COMPLETE_NAME}
    for index, name in enumerate(action_names):
        if name in categorical_names:
            for key in ("min", "q01", "q10"):
                stats[key][index] = 0.0
            for key in ("max", "q90", "q99"):
                stats[key][index] = 1.0
            stats["mean"][index] = 0.5
            stats["std"][index] = 0.5
        elif name == "gripper_target" and config.gripper_target_representation == "binary_position":
            low = float(stats["min"][index])
            high = float(stats["max"][index])
            if not low < high:
                raise ValueError("gripper_target must contain both physical classes")
            for key in ("min", "q01", "q10"):
                stats[key][index] = low
            for key in ("max", "q90", "q99"):
                stats[key][index] = high
            stats["mean"][index] = (low + high) * 0.5
            stats["std"][index] = (high - low) * 0.5

    return {
        "format": "lerobot.pi05.transformed_action_stats",
        "version": 1,
        "dataset": {
            "root": str(dataset.root),
            "repo_id": str(dataset.repo_id),
            "num_frames": int(dataset.num_frames),
            "num_episodes": int(dataset.num_episodes),
            "fps": dataset_fps,
        },
        "schema": {
            **schema,
            "ee_target_dataset_semantics": config.ee_target_dataset_semantics,
            "ee_delta_supervision_mode": config.ee_delta_supervision_mode,
            "gripper_target_representation": config.gripper_target_representation,
            "ee_supervision_source": config.ee_supervision_source,
            "control_frequency_hz": float(config.control_frequency_hz),
            "stride_dataset_frames": stride,
            "action_names": action_names,
        },
        "counts": {
            "all_transitions": total_transitions,
            (
                "ee_active_non_reset_both_endpoints"
                if config.ee_delta_supervision_mode == "active_only"
                else "ee_all_transitions"
            ): ee_valid_transitions,
            "per_dimension": counts_by_dimension,
        },
        "stats": {ACTION: stats},
    }
