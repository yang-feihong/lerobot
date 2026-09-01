#!/usr/bin/env python
"""Open-loop action evaluation for VLA policies on a LeRobotDataset.

This evaluates a trained policy on recorded dataset observations without sending
actions to a robot or simulator. For every sampled frame it predicts an action
chunk and compares it with the physical supervision reconstructed through the
checkpoint's training preprocessor and postprocessor.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import logging
import math
import multiprocessing
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.pi05.b2_action_transform import (
    absolute_ee_pose_to_reference_delta,
    action_schema_kwargs,
    b2_execution_action_names,
    b2_pose_delta_action_names,
    ee_reference_delta_to_absolute,
    integrate_body_twist_to_pose_delta,
    make_pi05_action_stats,
)
from lerobot.policies.pi05.manipulation_metrics import (
    aggregate_manipulation_metrics,
    compute_manipulation_onset_metrics,
)
from lerobot.policies.pi05.transformed_action_stats import (
    PI05_TRANSFORMED_ACTION_STATS_NAME,
    load_transformed_action_stats,
)
from lerobot.processor.normalize_processor import UnnormalizerProcessorStep
from lerobot.scripts.lerobot_train import apply_task_variants_to_batch, load_task_variants
from lerobot.scripts.pi05_vla_server import _load_checkpoint_contract
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.utils import init_logging

_B2_WORLD_TRAJECTORY_NAMES = ["b2_world_x", "b2_world_y", "b2_world_yaw"]


def _unnormalize_model_action(normalized: torch.Tensor, postprocessor) -> torch.Tensor:
    """Undo checkpoint normalization without applying deployment representation transforms."""
    if not isinstance(normalized, torch.Tensor):
        raise ValueError("Expected a tensor in the model action space")
    transition = postprocessor.to_transition(normalized)
    found_unnormalizer = False
    for processor_step in postprocessor.steps:
        transition = processor_step(transition)
        if isinstance(processor_step, UnnormalizerProcessorStep):
            found_unnormalizer = True
            break
    if not found_unnormalizer:
        raise ValueError("Checkpoint postprocessor has no action unnormalizer")
    model_action = postprocessor.to_output(transition)
    if not isinstance(model_action, torch.Tensor):
        raise ValueError("Checkpoint action unnormalizer did not return a tensor")
    return _as_action_chunk(model_action.detach().cpu().to(torch.float32))


def _supervision_contract(policy_cfg: PreTrainedConfig, io_schema_enabled: bool) -> dict[str, Any]:
    if not io_schema_enabled:
        return {
            "construction": "checkpoint_preprocessor_then_action_unnormalizer",
            "action_source": "dataset_action",
        }
    metadata = policy_cfg.deployment_metadata()
    action = metadata["action"]
    return {
        "construction": "checkpoint_preprocessor_then_action_unnormalizer",
        "b2_source": action["trajectory_source"],
        "b2_representation": action["representation"],
        "ee_source": action["ee_supervision_source"],
        "ee_dataset_semantics": action["ee_target_dataset_semantics"],
        "ee_representation": action["z1_representation"],
        "ee_delta_reference": action["ee_delta_reference"],
        "gripper_source": "dataset_action.gripper_target",
        "gripper_representation": action["gripper_target_representation"],
    }


def _current_state_slice(
    raw_state: torch.Tensor,
    pose_indices: list[int],
    *,
    history_length: int,
) -> torch.Tensor:
    """Read current physical state fields before policy state-feature selection.

    A non-MEM batch has shape ``[B, D]``. A temporal-state batch can have
    shape ``[B, T, D]``, with the current state at the end of the history prefix.
    The pose is an evaluation/visualization input even when the policy itself
    does not consume it (notably for a velocity checkpoint).
    """
    if raw_state.ndim == 2:
        return raw_state[:, pose_indices]
    if raw_state.ndim == 3:
        if history_length < 1 or history_length > raw_state.shape[1]:
            raise ValueError(
                f"Invalid history_length={history_length} for observation.state shape "
                f"{tuple(raw_state.shape)}"
            )
        return raw_state[:, history_length - 1, pose_indices]
    raise ValueError(
        f"Expected observation.state with shape [B, D] or [B, T, D], got {tuple(raw_state.shape)}"
    )


def _episode_b2_command_anchors(
    dataset: LeRobotDataset,
    episode_index: int,
    frame_index: np.ndarray,
    *,
    dataset_dt_seconds: float,
) -> np.ndarray:
    """Return each plotted frame's command-integrated pose from the episode start."""
    episode = dataset.meta.episodes[episode_index]
    absolute_indices = list(range(int(episode["dataset_from_index"]), int(episode["dataset_to_index"])))
    relative_index = dataset.absolute_to_relative_idx
    rows = (
        [relative_index[index] for index in absolute_indices]
        if relative_index is not None
        else absolute_indices
    )
    raw_action = dataset.hf_dataset[rows][ACTION]
    if isinstance(raw_action, list):
        raw_action = torch.stack([torch.as_tensor(value) for value in raw_action])
    else:
        raw_action = torch.as_tensor(raw_action)
    if raw_action.ndim != 2 or raw_action.shape[1] < 3:
        raise ValueError(
            f"Expected episode action matrix with at least 3 columns, got {tuple(raw_action.shape)}"
        )

    pose_after_interval = integrate_body_twist_to_pose_delta(
        raw_action[:, :3].to(dtype=torch.float32).unsqueeze(0),
        dataset_dt_seconds,
    )[0]
    anchors = torch.cat((torch.zeros_like(pose_after_interval[:1]), pose_after_interval[:-1]), dim=0)
    selected = torch.as_tensor(frame_index, dtype=torch.long)
    if selected.numel() and (selected.min() < 0 or selected.max() >= len(anchors)):
        raise IndexError(
            f"Episode {episode_index} plot frame indices are outside [0, {len(anchors)}): "
            f"min={int(selected.min())}, max={int(selected.max())}"
        )
    return anchors[selected].numpy()


def _reanchor_episode_plot_actions(
    actions: np.ndarray,
    action_names: list[str],
    *,
    b2_representation: str,
    z1_representation: str,
    ee_rotation_representation: str,
    b2_episode_anchors: np.ndarray | None,
    current_ee_anchor: np.ndarray | None,
) -> np.ndarray:
    """Express anchored model actions relative to the episode's first inference anchor."""
    original = np.asarray(actions)
    if original.ndim not in (2, 3):
        raise ValueError(f"Expected [N, A] or [N, H, A] plot actions, got {original.shape}")
    values = torch.from_numpy(original.astype(np.float32, copy=False))
    squeeze_horizon = values.ndim == 2
    if squeeze_horizon:
        values = values.unsqueeze(1)
    result = values.clone()
    inference_count = result.shape[0]

    if b2_representation == "pose_delta":
        if b2_episode_anchors is None or np.asarray(b2_episode_anchors).shape != (inference_count, 3):
            raise ValueError(
                "B2 pose-delta episode plots require one command-integrated anchor per inference"
            )
        anchors = torch.as_tensor(b2_episode_anchors, dtype=result.dtype)
        local = result[..., :3]
        anchor_yaw = anchors[:, None, 2]
        cos_yaw = torch.cos(anchor_yaw)
        sin_yaw = torch.sin(anchor_yaw)
        target_x = anchors[:, None, 0] + cos_yaw * local[..., 0] - sin_yaw * local[..., 1]
        target_y = anchors[:, None, 1] + sin_yaw * local[..., 0] + cos_yaw * local[..., 1]
        target_yaw = anchors[:, None, 2] + local[..., 2]

        first_anchor = anchors[0]
        world_x = target_x - first_anchor[0]
        world_y = target_y - first_anchor[1]
        first_cos = torch.cos(first_anchor[2])
        first_sin = torch.sin(first_anchor[2])
        result[..., 0] = first_cos * world_x + first_sin * world_y
        result[..., 1] = -first_sin * world_x + first_cos * world_y
        yaw_delta = target_yaw - first_anchor[2]
        result[..., 2] = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta))
    elif b2_representation != "velocity":
        raise ValueError(f"Unknown B2 representation: {b2_representation!r}")

    if z1_representation == "ee_state_delta":
        expected_ee_dim = 6 if ee_rotation_representation == "rotvec" else 9
        ee_indices = [
            index for index, name in enumerate(action_names) if name.startswith("height_invariant_ee_")
        ]
        if not ee_indices:
            output = result[:, 0] if squeeze_horizon else result
            return output.numpy().astype(original.dtype, copy=False)
        if len(ee_indices) != expected_ee_dim or ee_indices != list(
            range(ee_indices[0], ee_indices[0] + expected_ee_dim)
        ):
            raise ValueError(
                "Cannot identify one contiguous EE-delta block in model actions: "
                f"indices={ee_indices}, expected_dim={expected_ee_dim}"
            )
        if current_ee_anchor is None or np.asarray(current_ee_anchor).shape != (inference_count, 9):
            raise ValueError("EE state-delta episode plots require one measured EE anchor per inference")
        anchors = torch.as_tensor(current_ee_anchor, dtype=result.dtype)
        ee_slice = slice(ee_indices[0], ee_indices[-1] + 1)
        absolute_targets = ee_reference_delta_to_absolute(
            result[..., ee_slice],
            anchors,
            rotation_representation=ee_rotation_representation,
        )
        first_anchor = anchors[0].expand(inference_count, -1)
        result[..., ee_slice] = absolute_ee_pose_to_reference_delta(
            absolute_targets,
            first_anchor,
            rotation_representation=ee_rotation_representation,
        )
    elif z1_representation != "ee_delta":
        raise ValueError(f"Unknown Z1 representation: {z1_representation!r}")

    output = result[:, 0] if squeeze_horizon else result
    return output.numpy().astype(original.dtype, copy=False)


def _episode_anchor_plot_names(
    action_names: list[str],
    *,
    b2_representation: str,
    z1_representation: str,
) -> list[str]:
    names = list(action_names)
    if b2_representation == "pose_delta":
        names[:3] = ["b2_episode_anchor_x", "b2_episode_anchor_y", "b2_episode_anchor_yaw"]
    if z1_representation == "ee_state_delta":
        names = [
            name.replace("height_invariant_ee_", "episode_anchor_ee_")
            if name.startswith("height_invariant_ee_")
            else name
            for name in names
        ]
    return names


def _resolve_policy_path(path: str | Path) -> Path:
    """Accept a run dir, checkpoint dir, or pretrained_model dir."""
    path = Path(path).expanduser().resolve()
    if (path / "config.json").exists():
        return path
    if (path / "pretrained_model" / "config.json").exists():
        return path / "pretrained_model"
    ckpt_root = path / "checkpoints"
    if ckpt_root.exists():
        step_dirs = sorted(
            [p for p in ckpt_root.iterdir() if p.is_dir() and p.name.isdigit()],
            key=lambda p: int(p.name),
        )
        if not step_dirs:
            raise FileNotFoundError(f"No numeric checkpoint directories found under {ckpt_root}")
        latest = step_dirs[-1] / "pretrained_model"
        if not (latest / "config.json").exists():
            raise FileNotFoundError(f"Latest checkpoint has no pretrained_model/config.json: {latest}")
        return latest
    raise FileNotFoundError(
        "policy path must be a pretrained_model dir, checkpoint step dir, or run dir with checkpoints/: "
        f"{path}"
    )


def _parse_episodes(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    episodes: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            episodes.extend(range(int(start), int(end) + 1))
        else:
            episodes.append(int(part))
    return sorted(set(episodes))


def _resolve_image_source(
    *,
    policy_path: Path,
    dataset_root: Path,
    requested_source: str,
    sim_image_manifest: str | None,
    sim_image_root: str | None,
    mixed_sim_probability: float | None,
    image_source_seed: int | None,
) -> dict[str, Any]:
    """Resolve the same paired-image source used to train the checkpoint."""
    if requested_source == "checkpoint":
        train_config_path = policy_path / "train_config.json"
        if not train_config_path.is_file():
            return {"image_source": "real"}
        train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
        dataset_config = train_config.get("dataset", {})
        source = str(dataset_config.get("image_source", "real"))
        manifest_value = dataset_config.get("sim_image_manifest")
        root_value = dataset_config.get("sim_image_root")
        probability = float(dataset_config.get("mixed_sim_probability", 0.5))
        seed = int(dataset_config.get("image_source_seed", 0))
    else:
        source = requested_source
        manifest_value = sim_image_manifest
        root_value = sim_image_root
        probability = 0.5 if mixed_sim_probability is None else mixed_sim_probability
        seed = 0 if image_source_seed is None else image_source_seed

    if source == "real":
        return {"image_source": "real"}
    if source not in {"sim", "mixed"}:
        raise ValueError(f"Unsupported open-loop image source: {source!r}")
    if manifest_value is None or root_value is None:
        raise ValueError(f"image_source={source!r} requires simulator manifest and root")

    manifest = Path(manifest_value).expanduser()
    root = Path(root_value).expanduser()
    if requested_source == "checkpoint" and not manifest.is_file():
        # Training paths are container-local. Rebase the paired dataset beside
        # the explicitly supplied evaluation dataset without changing its name.
        manifest = dataset_root.parent / manifest.parent.name / manifest.name
        root = dataset_root.parent
    if not manifest.is_file():
        raise FileNotFoundError(f"Paired simulator image manifest does not exist: {manifest}")
    if not root.is_dir():
        raise FileNotFoundError(f"Paired simulator image root does not exist: {root}")
    return {
        "image_source": source,
        "sim_image_manifest": manifest,
        "sim_image_root": root,
        "mixed_sim_probability": probability,
        "image_source_seed": seed,
    }


def _split_episodes(meta: LeRobotDatasetMetadata, split: str, eval_split: float) -> list[int]:
    all_episodes = list(range(meta.total_episodes))
    if split == "all" or eval_split == 0.0:
        return all_episodes

    episode_tasks = meta.episodes["tasks"]
    task_to_episodes: dict[str, list[int]] = {}
    for ep_idx in all_episodes:
        task_key = episode_tasks[ep_idx][0] if episode_tasks[ep_idx] else ""
        task_to_episodes.setdefault(task_key, []).append(ep_idx)

    selected: list[int] = []
    for eps in task_to_episodes.values():
        n_eval = math.ceil(len(eps) * eval_split)
        if split == "eval":
            selected.extend(eps[len(eps) - n_eval :])
        elif split == "train":
            selected.extend(eps[: len(eps) - n_eval])
        else:
            raise ValueError(f"Unknown split={split!r}; expected train/eval/all")
    return selected


def _limit_episodes(episodes: list[int], max_episodes: int) -> list[int]:
    if max_episodes <= 0 or len(episodes) <= max_episodes:
        return episodes
    return episodes[:max_episodes]


def _default_action_names(meta: LeRobotDatasetMetadata, action_dim: int) -> list[str]:
    names = meta.features.get(ACTION, {}).get("names")
    if names is None:
        return [f"action_{i}" for i in range(action_dim)]
    return [str(name) for name in names]


def _openloop_model_action_names(
    dataset_names: list[str],
    *,
    representation: str,
    **schema_kwargs: Any,
) -> list[str]:
    if representation == "velocity":
        names = b2_execution_action_names(
            dataset_names,
            representation=representation,
            **schema_kwargs,
        )
    elif representation == "pose_delta":
        names = b2_pose_delta_action_names(
            dataset_names,
            representation=representation,
            **schema_kwargs,
        )
    else:
        raise ValueError(f"Unknown B2 action representation: {representation!r}")
    assert names is not None
    return names


def _to_numpy_1d(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _write_metrics_csv(
    path: Path,
    rows: list[dict[str, Any]],
    action_names: list[str],
    discrete_action_names: frozenset[str] = frozenset(
        {"arm_teleop_inactive", "arm_reset", "gripper_target", "task_complete"}
    ),
) -> None:
    fieldnames = ["scope", "episode_index", "num_frames", "mae_mean", "rmse_mean"]
    for name in action_names:
        fieldnames.extend([f"mae/{name}", f"rmse/{name}"])
        if name in discrete_action_names:
            fieldnames.extend(
                [
                    f"discrete_accuracy/{name}",
                    f"discrete_precision/{name}",
                    f"discrete_recall/{name}",
                    f"discrete_f1/{name}",
                    f"discrete_supervision_true_frac/{name}",
                    f"discrete_pred_true_frac/{name}",
                ]
            )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_predictions_csv(path: Path, rows: list[dict[str, Any]], action_names: list[str]) -> None:
    fieldnames = ["episode_index", "frame_index", "timestamp", "task"]
    for name in action_names:
        fieldnames.extend([f"supervision/{name}", f"pred/{name}", f"error/{name}"])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _compute_metrics(
    episode_index: int | str,
    expert: np.ndarray,
    pred: np.ndarray,
    action_names: list[str],
    discrete_action_names: frozenset[str] = frozenset(
        {"arm_teleop_inactive", "arm_reset", "gripper_target", "task_complete"}
    ),
) -> dict[str, Any]:
    err = pred - expert
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(np.square(err), axis=0))
    row: dict[str, Any] = {
        "scope": "all" if episode_index == "all" else "episode",
        "episode_index": episode_index,
        "num_frames": int(expert.shape[0]),
        "mae_mean": float(np.mean(mae)),
        "rmse_mean": float(np.mean(rmse)),
    }
    for i, name in enumerate(action_names):
        row[f"mae/{name}"] = float(mae[i])
        row[f"rmse/{name}"] = float(rmse[i])
        if name not in discrete_action_names:
            continue
        expert_true = expert[:, i] < 0 if name == "gripper_target" else expert[:, i] > 0
        pred_true = pred[:, i] < 0 if name == "gripper_target" else pred[:, i] > 0
        true_positive = int(np.count_nonzero(expert_true & pred_true))
        predicted_positive = int(np.count_nonzero(pred_true))
        expected_positive = int(np.count_nonzero(expert_true))
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / expected_positive if expected_positive else 0.0
        row[f"discrete_accuracy/{name}"] = float(np.mean(expert_true == pred_true))
        row[f"discrete_precision/{name}"] = precision
        row[f"discrete_recall/{name}"] = recall
        row[f"discrete_f1/{name}"] = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        row[f"discrete_supervision_true_frac/{name}"] = float(np.mean(expert_true))
        row[f"discrete_pred_true_frac/{name}"] = float(np.mean(pred_true))
    return row


def _normalize_action_array(actions: np.ndarray, meta: LeRobotDatasetMetadata) -> np.ndarray:
    stats = meta.stats[ACTION]
    q01 = np.asarray(stats["q01"], dtype=np.float32).reshape(-1)
    q99 = np.asarray(stats["q99"], dtype=np.float32).reshape(-1)
    denom = q99 - q01
    denom = np.where(denom == 0, 1e-8, denom)
    return 2.0 * (actions - q01) / denom - 1.0


def _normalize_execution_action_array(
    actions: np.ndarray,
    meta: LeRobotDatasetMetadata,
    execution_stats: dict[str, Any] | None = None,
) -> np.ndarray:
    """Normalize the compact action against corresponding raw-dataset stats."""
    if execution_stats is not None:
        return _normalize_with_stats(actions, execution_stats)
    return _normalize_action_array(actions, meta)


def _normalize_with_stats(actions: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    q01 = np.asarray(stats["q01"], dtype=np.float32).reshape(-1)
    q99 = np.asarray(stats["q99"], dtype=np.float32).reshape(-1)
    denom = np.where(q99 == q01, 1e-8, q99 - q01)
    return 2.0 * (actions - q01) / denom - 1.0


def _combined_base_plot_arrays(
    execution: np.ndarray,
    trajectory: np.ndarray,
    b2_start: int = 0,
) -> np.ndarray:
    """Show velocity and integrated position together while retaining all other actions."""
    return np.concatenate(
        (
            execution[..., : b2_start + 3],
            trajectory[..., b2_start : b2_start + 3],
            execution[..., b2_start + 3 :],
        ),
        axis=-1,
    )


def _combined_base_plot_names(execution_names: list[str], b2_start: int = 0) -> list[str]:
    return execution_names[: b2_start + 3] + _B2_WORLD_TRAJECTORY_NAMES + execution_names[b2_start + 3 :]


def _world_trajectory_plot_limits(
    expert_world_chunks: np.ndarray,
    pred_world_chunks: np.ndarray,
) -> list[tuple[float, float]]:
    """Use one physical scale for world x/y and a separate stable scale for yaw."""
    xy = np.concatenate((expert_world_chunks[..., :2].reshape(-1), pred_world_chunks[..., :2].reshape(-1)))
    yaw = np.concatenate((expert_world_chunks[..., 2].reshape(-1), pred_world_chunks[..., 2].reshape(-1)))
    xy_limits = _padded_limits(float(np.nanmin(xy)), float(np.nanmax(xy)))
    yaw_limits = _padded_limits(float(np.nanmin(yaw)), float(np.nanmax(yaw)))
    return [xy_limits, xy_limits, yaw_limits]


def _padded_limits(low: float, high: float, *, padding_fraction: float = 0.08) -> tuple[float, float]:
    """Return stable plot limits instead of allowing per-subplot autoscaling."""
    span = high - low
    if not np.isfinite(span) or span <= 0:
        center = low if np.isfinite(low) else 0.0
        span = max(abs(center) * 0.2, 1.0)
        low = center - span / 2
        high = center + span / 2
    padding = span * padding_fraction
    return float(low - padding), float(high + padding)


def _action_plot_y_limits(meta: LeRobotDatasetMetadata, action_names: list[str]) -> list[tuple[float, float]]:
    """Build semantic, run-independent y-axis limits for every action dimension.

    Dimensions with the same physical representation share one scale. In
    particular, this prevents a nearly constant rotation component around 1.0
    from being autoscaled to a misleading 0.998..1.000 plot.
    """
    stats = meta.stats[ACTION]
    action_min = np.asarray(stats["min"], dtype=np.float32).reshape(-1)
    action_max = np.asarray(stats["max"], dtype=np.float32).reshape(-1)
    name_to_index = {name: i for i, name in enumerate(action_names)}

    limits = [_padded_limits(float(action_min[i]), float(action_max[i])) for i in range(len(action_names))]

    def set_group(names: list[str], group_limits: tuple[float, float]) -> None:
        for name in names:
            index = name_to_index.get(name)
            if index is not None:
                limits[index] = group_limits

    set_group(["arm_teleop_inactive", "arm_reset", "task_complete"], (-0.1, 1.1))

    velocity_names = ["b2_vx", "b2_vy", "b2_omega_z"]
    velocity_indices = [name_to_index[name] for name in velocity_names if name in name_to_index]
    if velocity_indices:
        velocity_extent = max(
            0.1,
            max(float(abs(action_min[i])) for i in velocity_indices),
            max(float(abs(action_max[i])) for i in velocity_indices),
        )
        set_group(velocity_names, (-1.1 * velocity_extent, 1.1 * velocity_extent))

    rotation_names = [name for name in action_names if "_rot6d_" in name]
    set_group(rotation_names, (-1.1, 1.1))

    ee_position_names = ["height_invariant_ee_x", "height_invariant_ee_y", "height_invariant_ee_z"]
    ee_position_indices = [name_to_index[name] for name in ee_position_names if name in name_to_index]
    if ee_position_indices:
        ee_limits = _padded_limits(
            min(float(action_min[i]) for i in ee_position_indices),
            max(float(action_max[i]) for i in ee_position_indices),
        )
        set_group(ee_position_names, ee_limits)

    gripper_index = name_to_index.get("gripper_target")
    if gripper_index is not None:
        limits[gripper_index] = _padded_limits(
            min(float(action_min[gripper_index]), 0.0),
            max(float(action_max[gripper_index]), 0.0),
        )

    return limits


def _plot_episode(
    output_path: Path,
    episode_index: int,
    frame_index: np.ndarray,
    expert: np.ndarray,
    pred: np.ndarray,
    action_names: list[str],
    y_limits: list[tuple[float, float]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    action_dim = expert.shape[1]
    ncols = min(5, action_dim)
    nrows = math.ceil(action_dim / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 2.4 * nrows), squeeze=False)
    x = frame_index

    for i, ax in enumerate(axes.flat):
        if i >= action_dim:
            ax.axis("off")
            continue
        ax.plot(x, expert[:, i], label="supervision", linewidth=1.3, marker="o", markersize=5.0)
        ax.plot(x, pred[:, i], label="pred", linewidth=1.1, marker="x", markersize=5.0, alpha=0.85)
        ax.set_ylim(*y_limits[i])
        if len(x) == 1:
            # A single-point smoke run otherwise renders like an empty line plot.
            ax.set_xlim(float(x[0]) - 1.0, float(x[0]) + 1.0)
        ax.set_title(action_names[i], fontsize=9)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(loc="best", fontsize=8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.995, 0.995), fontsize=9)
    fig.suptitle(f"Open-loop action comparison: episode {episode_index} ({len(frame_index)} frames)")
    fig.supxlabel("frame_index")
    fig.tight_layout(rect=(0, 0, 0.98, 0.97))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_episode_rolling_chunks(
    output_path: Path,
    episode_index: int,
    start_frame_index: np.ndarray,
    expert_chunks: np.ndarray,
    pred_chunks: np.ndarray,
    action_names: list[str],
    y_limits: list[tuple[float, float]],
    valid_lengths: np.ndarray,
) -> None:
    """Plot every evaluated 50-step chunk as a rolling future trajectory.

    Row semantics in data terms:
      frame 0 predicts/actions 0..49,
      frame 1 predicts/actions 1..50,
      frame 2 predicts/actions 2..51,
      ...

    The figure overlays those rolling chunks per action dimension. This makes it
    clear whether the policy's whole short-horizon plan matches the supervision, not
    just the first action of each chunk.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    action_dim = expert_chunks.shape[2]
    horizon = expert_chunks.shape[1]
    ncols = min(5, action_dim)
    nrows = math.ceil(action_dim / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 2.5 * nrows), squeeze=False)

    for dim, ax in enumerate(axes.flat):
        if dim >= action_dim:
            ax.axis("off")
            continue
        for row_idx, start in enumerate(start_frame_index):
            valid_length = int(valid_lengths[row_idx])
            x = int(start) + np.arange(valid_length)
            ax.plot(
                x,
                expert_chunks[row_idx, :valid_length, dim],
                color="tab:blue",
                alpha=0.18,
                linewidth=0.9,
                label="supervision chunks" if row_idx == 0 else None,
            )
            ax.plot(
                x,
                pred_chunks[row_idx, :valid_length, dim],
                color="tab:orange",
                alpha=0.22,
                linewidth=0.9,
                label="pred chunks" if row_idx == 0 else None,
            )
        ax.set_ylim(*y_limits[dim])
        ax.set_title(action_names[dim], fontsize=9)
        ax.grid(True, alpha=0.25)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.995, 0.995), fontsize=9)
    fig.suptitle(
        f"Rolling open-loop action chunks: episode {episode_index} "
        f"({len(start_frame_index)} starts × max horizon {horizon}; clipped at episode end)"
    )
    fig.supxlabel("absolute frame_index")
    fig.tight_layout(rect=(0, 0, 0.98, 0.97))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_single_chunk(
    output_path: Path,
    episode_index: int,
    start_frame_index: int,
    expert_chunk: np.ndarray,
    pred_chunk: np.ndarray,
    action_names: list[str],
    y_limits: list[tuple[float, float]],
) -> None:
    """Plot one policy inference: a single 50-step predicted action chunk."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    action_dim = expert_chunk.shape[1]
    horizon = expert_chunk.shape[0]
    ncols = min(5, action_dim)
    nrows = math.ceil(action_dim / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 2.5 * nrows), squeeze=False)
    x = int(start_frame_index) + np.arange(horizon)

    for dim, ax in enumerate(axes.flat):
        if dim >= action_dim:
            ax.axis("off")
            continue
        ax.plot(x, expert_chunk[:, dim], label="supervision", color="tab:blue", linewidth=1.4)
        ax.plot(x, pred_chunk[:, dim], label="pred", color="tab:orange", linewidth=1.2)
        ax.set_ylim(*y_limits[dim])
        ax.set_title(action_names[dim], fontsize=9)
        ax.grid(True, alpha=0.25)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.995, 0.995), fontsize=9)
    fig.suptitle(
        f"Single inference action chunk: episode {episode_index}, "
        f"start frame {start_frame_index}, horizon {horizon}"
    )
    fig.supxlabel("absolute frame_index")
    fig.tight_layout(rect=(0, 0, 0.98, 0.97))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_single_chunk_job(
    job: tuple[
        Path,
        int,
        int,
        np.ndarray,
        np.ndarray,
        list[str],
        list[tuple[float, float]],
    ],
) -> None:
    """Process-pool entry point for one independent action-chunk plot."""
    _plot_single_chunk(*job)


def _batch_to_device_and_float_images(batch: dict[str, Any], camera_keys: list[str]) -> dict[str, Any]:
    for cam_key in camera_keys:
        if (
            cam_key in batch
            and isinstance(batch[cam_key], torch.Tensor)
            and batch[cam_key].dtype == torch.uint8
        ):
            batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
    return batch


def _as_action_chunk(action: torch.Tensor) -> torch.Tensor:
    if action.ndim == 3:
        return action
    if action.ndim == 2:
        return action[:, None, :]
    raise ValueError(f"Expected action shape (B, A) or (B, T, A), got {tuple(action.shape)}")


def _first_action(action: torch.Tensor) -> torch.Tensor:
    """Return the single action aligned with the current observation.

    Some policies request action chunks from LeRobotDataset, yielding
    ``(batch, horizon, action_dim)`` targets. Open-loop comparison uses the same
    convention as deployment: compare the first predicted action with the first
    supervision action in that chunk.
    """
    return _as_action_chunk(action)[:, 0]


def _valid_chunk_lengths(
    meta: LeRobotDatasetMetadata,
    episode_index: int,
    start_frame_index: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Return the non-padded future length for each chunk start."""
    episode_length = int(meta.episodes[episode_index]["length"])
    remaining = episode_length - start_frame_index
    return np.clip(remaining, 0, horizon).astype(np.int64)


def _flatten_valid_chunks(chunks: np.ndarray, valid_lengths: np.ndarray) -> np.ndarray:
    """Flatten chunks while excluding samples padded beyond the episode end."""
    valid_parts = [
        chunk[: int(length)] for chunk, length in zip(chunks, valid_lengths, strict=True) if length
    ]
    if not valid_parts:
        return np.empty((0, chunks.shape[-1]), dtype=chunks.dtype)
    return np.concatenate(valid_parts, axis=0)


def _plot_limits_covering_arrays(
    base_limits: list[tuple[float, float]],
    *arrays: np.ndarray,
) -> list[tuple[float, float]]:
    """Expand configured limits when a display-only representation has a wider range."""
    if not arrays:
        return list(base_limits)
    flattened = [np.asarray(array).reshape(-1, np.asarray(array).shape[-1]) for array in arrays]
    values = np.concatenate(flattened, axis=0)
    if values.shape[1] != len(base_limits):
        raise ValueError(f"Plot values have {values.shape[1]} dims but limits have {len(base_limits)}")
    limits = []
    for index, (base_low, base_high) in enumerate(base_limits):
        finite = values[:, index][np.isfinite(values[:, index])]
        if not len(finite):
            limits.append((base_low, base_high))
            continue
        data_low, data_high = _padded_limits(float(np.min(finite)), float(np.max(finite)))
        limits.append((min(base_low, data_low), max(base_high, data_high)))
    return limits


def _manipulation_onset_windows(
    dataset: LeRobotDataset,
    *,
    pre_frames: int,
    post_frames: int,
) -> tuple[set[int], set[tuple[int, int]]]:
    table = dataset.hf_dataset.select_columns([ACTION, "episode_index", "frame_index"]).with_format("numpy")
    required_episode_frames: set[tuple[int, int]] = set()
    index_by_episode_frame: dict[tuple[int, int], int] = {}
    offset = 0
    previous_episode = None
    previous_active = False
    for batch in table.iter(batch_size=65_536):
        actions = np.asarray(batch[ACTION])
        episode_indices = np.asarray(batch["episode_index"], dtype=np.int64)
        frame_indices = np.asarray(batch["frame_index"], dtype=np.int64)
        active = (actions[:, 3] < 0.5) & (actions[:, 4] < 0.5)
        for local_index, (episode, frame, is_active) in enumerate(
            zip(episode_indices, frame_indices, active, strict=True)
        ):
            episode = int(episode)
            frame = int(frame)
            index_by_episode_frame[(episode, frame)] = local_index + offset
            if episode != previous_episode:
                previous_active = False
            if is_active and not previous_active:
                for required_frame in range(max(0, frame - pre_frames), frame + post_frames + 1):
                    required_episode_frames.add((episode, required_frame))
            previous_episode = episode
            previous_active = bool(is_active)
        offset += len(actions)
    required_episode_frames.intersection_update(index_by_episode_frame)
    required_indices = {index_by_episode_frame[key] for key in required_episode_frames}
    return required_indices, required_episode_frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-path", required=True, help="Run dir, checkpoint dir, or pretrained_model dir."
    )
    parser.add_argument("--dataset-repo-id", default="local/b2_z1_vla")
    parser.add_argument("--dataset-root", default="/data/b2_z1_vla_lerobot")
    parser.add_argument(
        "--image-source",
        choices=["checkpoint", "real", "sim", "mixed"],
        default="checkpoint",
        help="Image channel used for evaluation. 'checkpoint' reproduces the training config.",
    )
    parser.add_argument("--sim-image-manifest", default=None)
    parser.add_argument("--sim-image-root", default=None)
    parser.add_argument("--mixed-sim-probability", type=float, default=None)
    parser.add_argument("--image-source-seed", type=int, default=None)
    parser.add_argument(
        "--dataset-group",
        default=None,
        help="Metric group such as staff1/staff2. By default it is inferred from --dataset-root.",
    )
    parser.add_argument(
        "--onset-window-seconds",
        type=float,
        default=1.0,
        help="Always evaluate this many seconds before and after every ground-truth manipulation onset.",
    )
    parser.add_argument(
        "--include-onset-windows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Augment strided samples with ground-truth manipulation-onset windows.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", choices=["train", "eval", "all"], default="eval")
    parser.add_argument("--eval-split", type=float, default=0.1)
    parser.add_argument("--episodes", default=None, help="Comma/range list, e.g. 1,5,9-12. Overrides split.")
    parser.add_argument("--max-episodes", type=int, default=8, help="0 means no limit.")
    parser.add_argument(
        "--max-frames-per-episode",
        type=int,
        default=None,
        help=(
            "Maximum frames per episode. 0 means no limit. By default, explicitly selected "
            "--episodes are evaluated in full; split-based evaluation is limited to 300 frames."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Run inference once every N source frames, resetting the stride at each episode.",
    )
    parser.add_argument("--task-variants-path", default=None)
    parser.add_argument("--task-variant", choices=["dataset", "first", "random"], default="first")
    parser.add_argument(
        "--task-override",
        default=None,
        help="Use one exact instruction for every evaluated frame after task-variant selection.",
    )
    parser.add_argument(
        "--max-chunk-plots-per-episode",
        type=int,
        default=None,
        help=(
            "Maximum number of per-inference 50-step chunk plots to save per episode. 0 means no "
            "limit. By default, every inference is plotted for explicitly selected --episodes; "
            "split-based evaluation saves at most 20 plots per episode."
        ),
    )
    parser.add_argument(
        "--chunk-plot-stride",
        type=int,
        default=1,
        help=(
            "Save one independent action-chunk plot every N evaluated frames. Metrics and the "
            "episode-level plots still use every evaluated frame."
        ),
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument(
        "--plot-workers",
        type=int,
        default=1,
        help="Processes used for independent chunk plots. Use more than 1 for large full episodes.",
    )
    args = parser.parse_args()

    if args.plot_workers < 1:
        raise ValueError(f"plot_workers must be at least 1, got {args.plot_workers}")
    if args.frame_stride < 1:
        raise ValueError(f"frame_stride must be at least 1, got {args.frame_stride}")
    if args.chunk_plot_stride < 1:
        raise ValueError(f"chunk_plot_stride must be at least 1, got {args.chunk_plot_stride}")

    init_logging()

    policy_path = _resolve_policy_path(args.policy_path)
    if args.output_dir is None:
        output_dir = policy_path.parent.parent / "openloop_eval" / f"{args.split}_latest"
    else:
        output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_group = args.dataset_group
    if dataset_group is None:
        root_name = Path(args.dataset_root).name.lower()
        dataset_group = "staff1" if "staff1" in root_name else "staff2" if "staff2" in root_name else "mixed"
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    chunk_plot_dir = output_dir / "rolling_chunk_plots"
    chunk_plot_dir.mkdir(parents=True, exist_ok=True)
    single_chunk_plot_dir = output_dir / "single_chunk_plots"
    single_chunk_plot_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Policy path: %s", policy_path)
    logging.info("Output dir: %s", output_dir)

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = args.device
    checkpoint_contract = None
    if getattr(policy_cfg, "io_schema_resolved", False):
        checkpoint_contract = _load_checkpoint_contract(
            policy_path,
            policy_cfg,
            float(policy_cfg.control_frequency_hz),
        )
    discrete_metric_names = {"arm_teleop_inactive", "arm_reset", "task_complete"}
    if checkpoint_contract is None or checkpoint_contract.gripper_target_representation == "binary_position":
        discrete_metric_names.add("gripper_target")
    discrete_metric_names = frozenset(discrete_metric_names)

    def compute_metrics(scope, expert, pred, action_names):
        return _compute_metrics(
            scope,
            expert,
            pred,
            action_names,
            discrete_action_names=discrete_metric_names,
        )

    meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    io_schema_enabled = bool(getattr(policy_cfg, "io_schema_resolved", False))
    supervision_contract = _supervision_contract(policy_cfg, io_schema_enabled)
    logging.info("Supervision contract: %s", json.dumps(supervision_contract, sort_keys=True))
    action_dt = getattr(policy_cfg, "action_dt_seconds", None)
    if io_schema_enabled:
        control_frequency_hz = getattr(policy_cfg, "control_frequency_hz", None)
        if control_frequency_hz is None:
            raise ValueError("B2+Z1 checkpoint is missing control_frequency_hz")
        expected_dt = 1.0 / float(control_frequency_hz)
        if action_dt is None:
            raise ValueError("B2+Z1 checkpoint is missing action_dt_seconds")
        if abs(float(action_dt) - expected_dt) > 1e-9:
            raise ValueError(
                f"Action checkpoint dt={action_dt} does not match control_frequency_hz={control_frequency_hz}"
            )
    delta_timestamps = resolve_delta_timestamps(policy_cfg, meta)

    episodes = _parse_episodes(args.episodes)
    explicit_episode_selection = episodes is not None
    if args.max_frames_per_episode is None:
        args.max_frames_per_episode = 0 if explicit_episode_selection else 300
    if args.max_chunk_plots_per_episode is None:
        args.max_chunk_plots_per_episode = 0 if explicit_episode_selection else 20
    if episodes is None:
        episodes = _split_episodes(meta, args.split, args.eval_split)
    episodes = _limit_episodes(episodes, args.max_episodes)
    if not episodes:
        raise ValueError("No episodes selected for open-loop evaluation.")
    logging.info("Selected %d episode(s): %s", len(episodes), episodes)
    logging.info(
        "Per-episode limits: frames=%s, frame_stride=%d, chunk_plots=%s, chunk_plot_stride=%d",
        "all" if args.max_frames_per_episode == 0 else args.max_frames_per_episode,
        args.frame_stride,
        "all" if args.max_chunk_plots_per_episode == 0 else args.max_chunk_plots_per_episode,
        args.chunk_plot_stride,
    )

    image_source_kwargs = _resolve_image_source(
        policy_path=policy_path,
        dataset_root=Path(args.dataset_root).expanduser(),
        requested_source=args.image_source,
        sim_image_manifest=args.sim_image_manifest,
        sim_image_root=args.sim_image_root,
        mixed_sim_probability=args.mixed_sim_probability,
        image_source_seed=args.image_source_seed,
    )
    logging.info(
        "Evaluation image source: %s",
        {key: str(value) if isinstance(value, Path) else value for key, value in image_source_kwargs.items()},
    )
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=None,
        video_backend=args.video_backend,
        return_uint8=True,
        **image_source_kwargs,
    )
    meta = dataset.meta

    dataset_action_dim = int(meta.features[ACTION]["shape"][0])
    dataset_action_names = _default_action_names(meta, dataset_action_dim)
    execution_action_names = list(dataset_action_names)
    trajectory_action_names = list(dataset_action_names)
    trajectory_stats = None
    execution_stats = None
    checkpoint_transformed_action_stats = None
    schema_kwargs = action_schema_kwargs(policy_cfg) if io_schema_enabled else {}
    if io_schema_enabled:
        if dataset_action_dim != 25:
            raise ValueError(f"B2 evaluation expects the 25D control-action schema, got {dataset_action_dim}")
        canonical_action_names = dataset_action_names[:16]
        model_action_names = _openloop_model_action_names(canonical_action_names, **schema_kwargs)
        execution_action_names = list(model_action_names)
        trajectory_action_names = list(model_action_names)
        if policy_cfg.z1_action_representation in {"ee_delta", "ee_state_delta"}:
            transformed_stats_path = policy_path / PI05_TRANSFORMED_ACTION_STATS_NAME
            if not transformed_stats_path.is_file():
                raise FileNotFoundError(
                    "EE-delta checkpoint is missing exact transformed-action statistics: "
                    f"{transformed_stats_path}"
                )
            checkpoint_transformed_action_stats = load_transformed_action_stats(transformed_stats_path)[
                "stats"
            ][ACTION]
        transformed_stats = make_pi05_action_stats(
            meta.stats,
            transformed_action_stats=checkpoint_transformed_action_stats,
            dt=float(action_dt),
            chunk_size=int(policy_cfg.chunk_size),
            **schema_kwargs,
        )
        assert transformed_stats is not None
        model_stats = transformed_stats[ACTION]
        trajectory_stats = deepcopy(model_stats)
        execution_stats = deepcopy(model_stats)
        trajectory_q01 = np.asarray(trajectory_stats["q01"], dtype=np.float32)
        trajectory_q99 = np.asarray(trajectory_stats["q99"], dtype=np.float32)
        trajectory_limits = [
            _padded_limits(float(trajectory_q01[i]), float(trajectory_q99[i]))
            for i in range(len(trajectory_action_names))
        ]
        plot_action_names = trajectory_action_names
        plot_y_limits = trajectory_limits
    else:
        plot_action_names = execution_action_names
        plot_y_limits = _action_plot_y_limits(meta, execution_action_names)
    sampled_indices = [
        index
        for index, frame_index in enumerate(dataset.hf_dataset["frame_index"])
        if int(frame_index) % args.frame_stride == 0
    ]
    if args.max_frames_per_episode > 0 and not args.include_onset_windows:
        limited_indices: list[int] = []
        limited_counts: dict[int, int] = defaultdict(int)
        episode_index_column = dataset.hf_dataset["episode_index"]
        for index in sampled_indices:
            episode_index = int(episode_index_column[index])
            if limited_counts[episode_index] >= args.max_frames_per_episode:
                continue
            limited_indices.append(index)
            limited_counts[episode_index] += 1
        sampled_indices = limited_indices
    required_onset_frames: set[tuple[int, int]] = set()
    if (
        args.include_onset_windows
        and io_schema_enabled
        and policy_cfg.z1_action_representation in {"ee_delta", "ee_state_delta"}
    ):
        onset_window_frames = int(round(args.onset_window_seconds * float(meta.fps)))
        required_onset_indices, required_onset_frames = _manipulation_onset_windows(
            dataset,
            pre_frames=onset_window_frames,
            post_frames=onset_window_frames,
        )
        sampled_indices = sorted(set(sampled_indices) | required_onset_indices)
    logging.info(
        "Inference sampling: %d/%d source frames (stride=%d)",
        len(sampled_indices),
        len(dataset),
        args.frame_stride,
    )
    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    dataloader = DataLoader(
        dataset,
        sampler=sampled_indices,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
        collate_fn=collate_fn,
    )

    task_variants = load_task_variants(dataset.root, args.task_variants_path)
    policy = make_policy(policy_cfg, ds_meta=dataset.meta)
    policy.eval()
    episode_plot_action_names = _episode_anchor_plot_names(
        plot_action_names,
        b2_representation=policy_cfg.b2_action_representation,
        z1_representation=policy_cfg.z1_action_representation,
    )
    state_history_length = (
        policy_cfg.state_num_frames if policy_cfg.state_action_encoding == "continuous" else 1
    )
    dataset_state_names = meta.features[OBS_STATE].get("names")
    if dataset_state_names is None:
        raise ValueError("Open-loop anchor visualization requires named observation.state dimensions")
    ee_plot_anchor_indices = None
    if policy_cfg.z1_action_representation == "ee_state_delta":
        if policy_cfg.ee_state_anchor_indices is None:
            raise ValueError("EE state-delta checkpoint did not resolve its measured-state anchor indices")
        ee_plot_anchor_indices = list(policy_cfg.ee_state_anchor_indices)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_path),
        pretrained_revision=getattr(policy_cfg, "pretrained_revision", None),
        # The dataset may have grown after this checkpoint was trained. Keep the
        # normalization state serialized with the checkpoint instead of silently
        # replacing it with statistics from the current dataset directory.
        dataset_stats=None,
        preprocessor_overrides={
            "device_processor": {"device": torch.device(args.device).type},
        },
    )

    per_episode: dict[int, dict[str, list[Any]]] = defaultdict(
        lambda: {
            "frame_index": [],
            "timestamp": [],
            "expert": [],
            "pred": [],
            "expert_chunk": [],
            "pred_chunk": [],
            "expert_trajectory": [],
            "pred_trajectory": [],
            "expert_trajectory_chunk": [],
            "pred_trajectory_chunk": [],
            "current_ee_anchor": [],
            "task": [],
        }
    )
    counts: dict[int, int] = defaultdict(int)
    remaining_required_onset_frames = set(required_onset_frames)
    prediction_rows: list[dict[str, Any]] = []

    with torch.inference_mode():
        for step, batch in enumerate(tqdm(dataloader, desc="Open-loop eval", unit="batch")):
            if batch is None:
                continue

            episode_indices = batch["episode_index"].detach().cpu().view(-1).tolist()
            frame_indices_in_batch = batch["frame_index"].detach().cpu().view(-1).tolist()
            keep = []
            for i, (ep_idx, frame_idx) in enumerate(
                zip(episode_indices, frame_indices_in_batch, strict=True)
            ):
                ep_idx = int(ep_idx)
                required_onset_frame = (ep_idx, int(frame_idx)) in required_onset_frames
                if (
                    args.max_frames_per_episode > 0
                    and counts[ep_idx] >= args.max_frames_per_episode
                    and not required_onset_frame
                ):
                    continue
                keep.append(i)
                counts[ep_idx] += 1
                remaining_required_onset_frames.discard((ep_idx, int(frame_idx)))
            if not keep:
                continue
            if len(keep) != len(episode_indices):
                keep_t = torch.tensor(keep, dtype=torch.long)
                filtered: dict[str, Any] = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor) and value.shape[:1] == (len(episode_indices),):
                        filtered[key] = value.index_select(0, keep_t)
                    elif isinstance(value, list) and len(value) == len(episode_indices):
                        filtered[key] = [value[i] for i in keep]
                    else:
                        filtered[key] = value
                batch = filtered

            if args.task_variant != "dataset":
                apply_task_variants_to_batch(
                    batch,
                    task_variants,
                    step=step,
                    seed=args.seed,
                    randomize=args.task_variant == "random",
                )
            if args.task_override is not None:
                batch["task"] = [args.task_override] * len(keep)

            batch = _batch_to_device_and_float_images(batch, dataset.meta.camera_keys)
            raw_state = batch.get(OBS_STATE)
            if not isinstance(raw_state, torch.Tensor):
                raise ValueError("Open-loop evaluation batch has no tensor observation.state")
            current_ee_anchor = (
                _current_state_slice(
                    raw_state,
                    ee_plot_anchor_indices,
                    history_length=state_history_length,
                )
                if ee_plot_anchor_indices is not None
                else None
            )
            processed = preprocessor(batch)
            normalized_supervision_chunk = processed.get(ACTION)
            if not isinstance(normalized_supervision_chunk, torch.Tensor):
                raise ValueError("Checkpoint preprocessor did not produce tensor action supervision")
            supervision_model_chunk = _unnormalize_model_action(
                normalized_supervision_chunk,
                postprocessor,
            )
            normalized_pred_chunk = policy.predict_action_chunk(processed)
            pred_model_chunk = _unnormalize_model_action(normalized_pred_chunk, postprocessor)
            if supervision_model_chunk.shape != pred_model_chunk.shape:
                raise ValueError(
                    "Training supervision and model prediction have different shapes: "
                    f"supervision={tuple(supervision_model_chunk.shape)}, "
                    f"prediction={tuple(pred_model_chunk.shape)}"
                )
            expert_chunk = supervision_model_chunk
            pred_chunk = pred_model_chunk
            expert_trajectory_chunk = expert_chunk
            pred_trajectory_chunk = pred_chunk
            expert_action = _first_action(expert_chunk)
            pred_action = _first_action(pred_chunk)
            expert_trajectory_action = _first_action(expert_trajectory_chunk)
            pred_trajectory_action = _first_action(pred_trajectory_chunk)

            ep_np = _to_numpy_1d(batch["episode_index"]).astype(np.int64)
            frame_np = _to_numpy_1d(batch["frame_index"]).astype(np.int64)
            ts_np = _to_numpy_1d(batch["timestamp"])
            tasks = batch.get("task")
            if isinstance(tasks, str):
                task_list = [tasks] * len(ep_np)
            elif isinstance(tasks, list):
                task_list = [str(t) for t in tasks]
            else:
                task_list = [""] * len(ep_np)

            expert_np = expert_action.numpy()
            pred_np = pred_action.numpy()
            expert_trajectory_np = expert_trajectory_action.numpy()
            pred_trajectory_np = pred_trajectory_action.numpy()
            current_ee_anchor_np = current_ee_anchor.numpy() if current_ee_anchor is not None else None
            for i, ep_idx in enumerate(ep_np.tolist()):
                ep_store = per_episode[int(ep_idx)]
                ep_store["frame_index"].append(int(frame_np[i]))
                ep_store["timestamp"].append(float(ts_np[i]))
                ep_store["expert"].append(expert_np[i])
                ep_store["pred"].append(pred_np[i])
                ep_store["expert_chunk"].append(expert_chunk[i].numpy())
                ep_store["pred_chunk"].append(pred_chunk[i].numpy())
                ep_store["expert_trajectory"].append(expert_trajectory_np[i])
                ep_store["pred_trajectory"].append(pred_trajectory_np[i])
                ep_store["expert_trajectory_chunk"].append(expert_trajectory_chunk[i].numpy())
                ep_store["pred_trajectory_chunk"].append(pred_trajectory_chunk[i].numpy())
                if current_ee_anchor_np is not None:
                    ep_store["current_ee_anchor"].append(current_ee_anchor_np[i])
                ep_store["task"].append(task_list[i])

                row: dict[str, Any] = {
                    "episode_index": int(ep_idx),
                    "frame_index": int(frame_np[i]),
                    "timestamp": float(ts_np[i]),
                    "task": task_list[i],
                }
                err = pred_np[i] - expert_np[i]
                for j, name in enumerate(execution_action_names):
                    row[f"supervision/{name}"] = float(expert_np[i, j])
                    row[f"pred/{name}"] = float(pred_np[i, j])
                    row[f"error/{name}"] = float(err[j])
                prediction_rows.append(row)

            if (
                args.max_frames_per_episode > 0
                and all(counts[int(ep_idx)] >= args.max_frames_per_episode for ep_idx in episodes)
                and not remaining_required_onset_frames
            ):
                break

    if not per_episode:
        raise RuntimeError("No frames were evaluated. Check selected episodes and frame limits.")

    metrics_rows: list[dict[str, Any]] = []
    chunk_metrics_rows: list[dict[str, Any]] = []
    normalized_metrics_rows: list[dict[str, Any]] = []
    normalized_chunk_metrics_rows: list[dict[str, Any]] = []
    trajectory_metrics_rows: list[dict[str, Any]] = []
    normalized_trajectory_metrics_rows: list[dict[str, Any]] = []
    trajectory_chunk_metrics_rows: list[dict[str, Any]] = []
    normalized_trajectory_chunk_metrics_rows: list[dict[str, Any]] = []
    manipulation_metric_rows: list[dict[str, Any]] = []
    all_expert, all_pred = [], []
    all_expert_chunks, all_pred_chunks = [], []
    all_expert_trajectories, all_pred_trajectories = [], []
    all_expert_trajectory_chunks, all_pred_trajectory_chunks = [], []
    npz_payload: dict[str, np.ndarray] = {}
    single_chunk_plot_jobs: list[
        tuple[
            Path,
            int,
            int,
            np.ndarray,
            np.ndarray,
            list[str],
            list[tuple[float, float]],
        ]
    ] = []
    episode_plot_y_limits_by_episode: dict[str, dict[str, list[float]]] = {}
    for ep_idx in sorted(per_episode):
        item = per_episode[ep_idx]
        frame_index = np.asarray(item["frame_index"], dtype=np.int64)
        expert = np.stack(item["expert"]).astype(np.float32)
        pred = np.stack(item["pred"]).astype(np.float32)
        expert_chunks = np.stack(item["expert_chunk"]).astype(np.float32)
        pred_chunks = np.stack(item["pred_chunk"]).astype(np.float32)
        expert_trajectory = np.stack(item["expert_trajectory"]).astype(np.float32)
        pred_trajectory = np.stack(item["pred_trajectory"]).astype(np.float32)
        expert_trajectory_chunks = np.stack(item["expert_trajectory_chunk"]).astype(np.float32)
        pred_trajectory_chunks = np.stack(item["pred_trajectory_chunk"]).astype(np.float32)
        if (
            args.include_onset_windows
            and io_schema_enabled
            and policy_cfg.z1_action_representation in {"ee_delta", "ee_state_delta"}
        ):
            onset_window_mask = np.asarray(
                [(int(ep_idx), int(frame)) in required_onset_frames for frame in frame_index],
                dtype=bool,
            )
            if not onset_window_mask.any():
                raise RuntimeError(f"Episode {ep_idx} has no evaluated manipulation-onset window")
            onset_metrics = compute_manipulation_onset_metrics(
                frame_indices=frame_index[onset_window_mask],
                expert_chunks=expert_chunks[onset_window_mask],
                predicted_chunks=pred_chunks[onset_window_mask],
                action_names=execution_action_names,
                control_frequency_hz=float(policy_cfg.control_frequency_hz),
                dataset_frequency_hz=float(meta.fps),
                dataset_group=dataset_group,
            )
            onset_metrics["episode_index"] = int(ep_idx)
            manipulation_metric_rows.append(onset_metrics)
        b2_episode_anchors = (
            _episode_b2_command_anchors(
                dataset,
                ep_idx,
                frame_index,
                dataset_dt_seconds=1.0 / float(meta.fps),
            )
            if policy_cfg.b2_action_representation == "pose_delta"
            else None
        )
        current_ee_anchor = (
            np.stack(item["current_ee_anchor"]).astype(np.float32) if item["current_ee_anchor"] else None
        )
        valid_chunk_lengths = _valid_chunk_lengths(
            meta,
            ep_idx,
            frame_index,
            expert_chunks.shape[1],
        )
        valid_expert_chunks = _flatten_valid_chunks(expert_chunks, valid_chunk_lengths)
        valid_pred_chunks = _flatten_valid_chunks(pred_chunks, valid_chunk_lengths)
        valid_expert_trajectory_chunks = _flatten_valid_chunks(expert_trajectory_chunks, valid_chunk_lengths)
        valid_pred_trajectory_chunks = _flatten_valid_chunks(pred_trajectory_chunks, valid_chunk_lengths)
        all_expert.append(expert)
        all_pred.append(pred)
        all_expert_chunks.append(valid_expert_chunks)
        all_pred_chunks.append(valid_pred_chunks)
        all_expert_trajectories.append(expert_trajectory)
        all_pred_trajectories.append(pred_trajectory)
        all_expert_trajectory_chunks.append(valid_expert_trajectory_chunks)
        all_pred_trajectory_chunks.append(valid_pred_trajectory_chunks)
        metrics_rows.append(compute_metrics(ep_idx, expert, pred, execution_action_names))
        normalized_metrics_rows.append(
            compute_metrics(
                ep_idx,
                _normalize_execution_action_array(expert, meta, execution_stats),
                _normalize_execution_action_array(pred, meta, execution_stats),
                execution_action_names,
            )
        )
        chunk_metrics_rows.append(
            compute_metrics(
                ep_idx,
                valid_expert_chunks,
                valid_pred_chunks,
                execution_action_names,
            )
        )
        normalized_chunk_metrics_rows.append(
            compute_metrics(
                ep_idx,
                _normalize_execution_action_array(valid_expert_chunks, meta, execution_stats),
                _normalize_execution_action_array(valid_pred_chunks, meta, execution_stats),
                execution_action_names,
            )
        )
        trajectory_metrics_rows.append(
            compute_metrics(
                ep_idx,
                expert_trajectory,
                pred_trajectory,
                trajectory_action_names,
            )
        )
        trajectory_chunk_metrics_rows.append(
            compute_metrics(
                ep_idx,
                valid_expert_trajectory_chunks,
                valid_pred_trajectory_chunks,
                trajectory_action_names,
            )
        )
        if trajectory_stats is not None:
            normalized_trajectory_metrics_rows.append(
                compute_metrics(
                    ep_idx,
                    _normalize_with_stats(expert_trajectory, trajectory_stats),
                    _normalize_with_stats(pred_trajectory, trajectory_stats),
                    trajectory_action_names,
                )
            )
            normalized_trajectory_chunk_metrics_rows.append(
                compute_metrics(
                    ep_idx,
                    _normalize_with_stats(valid_expert_trajectory_chunks, trajectory_stats),
                    _normalize_with_stats(valid_pred_trajectory_chunks, trajectory_stats),
                    trajectory_action_names,
                )
            )
        else:
            normalized_trajectory_metrics_rows.append(normalized_metrics_rows[-1].copy())
            normalized_trajectory_chunk_metrics_rows.append(normalized_chunk_metrics_rows[-1].copy())

        reanchor_kwargs = {
            "b2_representation": policy_cfg.b2_action_representation,
            "z1_representation": policy_cfg.z1_action_representation,
            "ee_rotation_representation": policy_cfg.ee_delta_rotation_representation,
            "b2_episode_anchors": b2_episode_anchors,
            "current_ee_anchor": current_ee_anchor,
        }
        plot_expert = _reanchor_episode_plot_actions(
            expert,
            plot_action_names,
            **reanchor_kwargs,
        )
        plot_pred = _reanchor_episode_plot_actions(
            pred,
            plot_action_names,
            **reanchor_kwargs,
        )
        plot_expert_chunks = _reanchor_episode_plot_actions(
            expert_chunks,
            plot_action_names,
            **reanchor_kwargs,
        )
        plot_pred_chunks = _reanchor_episode_plot_actions(
            pred_chunks,
            plot_action_names,
            **reanchor_kwargs,
        )
        episode_plot_y_limits = _plot_limits_covering_arrays(
            plot_y_limits,
            plot_expert,
            plot_pred,
            _flatten_valid_chunks(plot_expert_chunks, valid_chunk_lengths),
            _flatten_valid_chunks(plot_pred_chunks, valid_chunk_lengths),
        )
        episode_plot_y_limits_by_episode[str(ep_idx)] = {
            name: [float(low), float(high)]
            for name, (low, high) in zip(
                episode_plot_action_names,
                episode_plot_y_limits,
                strict=True,
            )
        }
        _plot_episode(
            plot_dir / f"episode_{ep_idx:06d}.png",
            ep_idx,
            frame_index,
            plot_expert,
            plot_pred,
            episode_plot_action_names,
            episode_plot_y_limits,
        )
        _plot_episode_rolling_chunks(
            chunk_plot_dir / f"episode_{ep_idx:06d}_rolling_chunks.png",
            ep_idx,
            frame_index,
            plot_expert_chunks,
            plot_pred_chunks,
            episode_plot_action_names,
            episode_plot_y_limits,
            valid_chunk_lengths,
        )
        ep_single_dir = single_chunk_plot_dir / f"episode_{ep_idx:06d}"
        ep_single_dir.mkdir(parents=True, exist_ok=True)
        single_plot_indices = np.arange(0, len(frame_index), args.chunk_plot_stride)
        if args.max_chunk_plots_per_episode > 0:
            single_plot_indices = single_plot_indices[: args.max_chunk_plots_per_episode]
        for i in single_plot_indices:
            single_chunk_plot_jobs.append(
                (
                    ep_single_dir / f"chunk_start_{int(frame_index[i]):06d}.png",
                    ep_idx,
                    int(frame_index[i]),
                    expert_chunks[i, : valid_chunk_lengths[i]],
                    pred_chunks[i, : valid_chunk_lengths[i]],
                    plot_action_names,
                    plot_y_limits,
                )
            )
        npz_payload[f"episode_{ep_idx:06d}_frame_index"] = frame_index
        npz_payload[f"episode_{ep_idx:06d}_supervision"] = expert
        npz_payload[f"episode_{ep_idx:06d}_pred"] = pred
        npz_payload[f"episode_{ep_idx:06d}_supervision_chunk"] = expert_chunks
        npz_payload[f"episode_{ep_idx:06d}_pred_chunk"] = pred_chunks
        npz_payload[f"episode_{ep_idx:06d}_supervision_trajectory"] = expert_trajectory
        npz_payload[f"episode_{ep_idx:06d}_pred_trajectory"] = pred_trajectory
        npz_payload[f"episode_{ep_idx:06d}_supervision_trajectory_chunk"] = expert_trajectory_chunks
        npz_payload[f"episode_{ep_idx:06d}_pred_trajectory_chunk"] = pred_trajectory_chunks
        npz_payload[f"episode_{ep_idx:06d}_valid_chunk_length"] = valid_chunk_lengths
        npz_payload[f"episode_{ep_idx:06d}_episode_anchor_plot_supervision"] = plot_expert
        npz_payload[f"episode_{ep_idx:06d}_episode_anchor_plot_pred"] = plot_pred
        npz_payload[f"episode_{ep_idx:06d}_episode_anchor_plot_supervision_chunk"] = plot_expert_chunks
        npz_payload[f"episode_{ep_idx:06d}_episode_anchor_plot_pred_chunk"] = plot_pred_chunks
        if b2_episode_anchors is not None:
            npz_payload[f"episode_{ep_idx:06d}_b2_episode_command_anchor"] = b2_episode_anchors
        if current_ee_anchor is not None:
            npz_payload[f"episode_{ep_idx:06d}_current_ee_anchor"] = current_ee_anchor

    logging.info(
        "Generating %d independent action-chunk plot(s) with %d worker(s)",
        len(single_chunk_plot_jobs),
        args.plot_workers,
    )
    if args.plot_workers == 1:
        for job in tqdm(single_chunk_plot_jobs, desc="Chunk plots", unit="plot"):
            _plot_single_chunk_job(job)
    else:
        # Spawn avoids forking a process after CUDA has been initialized by policy inference.
        mp_context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.plot_workers,
            mp_context=mp_context,
        ) as executor:
            results = executor.map(_plot_single_chunk_job, single_chunk_plot_jobs, chunksize=1)
            for _ in tqdm(results, total=len(single_chunk_plot_jobs), desc="Chunk plots", unit="plot"):
                pass

    all_expert_arr = np.concatenate(all_expert, axis=0)
    all_pred_arr = np.concatenate(all_pred, axis=0)
    all_expert_chunk_arr = np.concatenate(all_expert_chunks, axis=0)
    all_pred_chunk_arr = np.concatenate(all_pred_chunks, axis=0)
    all_expert_trajectory_arr = np.concatenate(all_expert_trajectories, axis=0)
    all_pred_trajectory_arr = np.concatenate(all_pred_trajectories, axis=0)
    all_expert_trajectory_chunk_arr = np.concatenate(all_expert_trajectory_chunks, axis=0)
    all_pred_trajectory_chunk_arr = np.concatenate(all_pred_trajectory_chunks, axis=0)
    metrics_rows.insert(0, compute_metrics("all", all_expert_arr, all_pred_arr, execution_action_names))
    normalized_metrics_rows.insert(
        0,
        compute_metrics(
            "all",
            _normalize_execution_action_array(all_expert_arr, meta, execution_stats),
            _normalize_execution_action_array(all_pred_arr, meta, execution_stats),
            execution_action_names,
        ),
    )
    chunk_metrics_rows.insert(
        0,
        compute_metrics("all", all_expert_chunk_arr, all_pred_chunk_arr, execution_action_names),
    )
    normalized_chunk_metrics_rows.insert(
        0,
        compute_metrics(
            "all",
            _normalize_execution_action_array(all_expert_chunk_arr, meta, execution_stats),
            _normalize_execution_action_array(all_pred_chunk_arr, meta, execution_stats),
            execution_action_names,
        ),
    )

    trajectory_metrics_rows.insert(
        0,
        compute_metrics(
            "all",
            all_expert_trajectory_arr,
            all_pred_trajectory_arr,
            trajectory_action_names,
        ),
    )
    trajectory_chunk_metrics_rows.insert(
        0,
        compute_metrics(
            "all",
            all_expert_trajectory_chunk_arr,
            all_pred_trajectory_chunk_arr,
            trajectory_action_names,
        ),
    )
    if trajectory_stats is not None:
        normalized_trajectory_metrics_rows.insert(
            0,
            compute_metrics(
                "all",
                _normalize_with_stats(all_expert_trajectory_arr, trajectory_stats),
                _normalize_with_stats(all_pred_trajectory_arr, trajectory_stats),
                trajectory_action_names,
            ),
        )
        normalized_trajectory_chunk_metrics_rows.insert(
            0,
            compute_metrics(
                "all",
                _normalize_with_stats(all_expert_trajectory_chunk_arr, trajectory_stats),
                _normalize_with_stats(all_pred_trajectory_chunk_arr, trajectory_stats),
                trajectory_action_names,
            ),
        )
    else:
        normalized_trajectory_metrics_rows.insert(0, normalized_metrics_rows[0].copy())
        normalized_trajectory_chunk_metrics_rows.insert(0, normalized_chunk_metrics_rows[0].copy())

    _write_metrics_csv(
        output_dir / "metrics.csv", metrics_rows, execution_action_names, discrete_metric_names
    )
    _write_metrics_csv(
        output_dir / "normalized_metrics.csv",
        normalized_metrics_rows,
        execution_action_names,
        discrete_metric_names,
    )
    _write_metrics_csv(
        output_dir / "chunk_metrics.csv", chunk_metrics_rows, execution_action_names, discrete_metric_names
    )
    _write_metrics_csv(
        output_dir / "normalized_chunk_metrics.csv",
        normalized_chunk_metrics_rows,
        execution_action_names,
        discrete_metric_names,
    )
    _write_metrics_csv(
        output_dir / "trajectory_metrics.csv",
        trajectory_metrics_rows,
        trajectory_action_names,
        discrete_metric_names,
    )
    _write_metrics_csv(
        output_dir / "normalized_trajectory_metrics.csv",
        normalized_trajectory_metrics_rows,
        trajectory_action_names,
        discrete_metric_names,
    )
    _write_metrics_csv(
        output_dir / "trajectory_chunk_metrics.csv",
        trajectory_chunk_metrics_rows,
        trajectory_action_names,
        discrete_metric_names,
    )
    _write_metrics_csv(
        output_dir / "normalized_trajectory_chunk_metrics.csv",
        normalized_trajectory_chunk_metrics_rows,
        trajectory_action_names,
        discrete_metric_names,
    )
    _write_predictions_csv(output_dir / "predictions.csv", prediction_rows, execution_action_names)
    np.savez_compressed(output_dir / "predictions.npz", **npz_payload)

    manipulation_metrics = None
    if manipulation_metric_rows:
        manipulation_metrics = aggregate_manipulation_metrics(manipulation_metric_rows, dataset_group)
        (output_dir / "manipulation_onset_metrics.json").write_text(
            json.dumps(
                {"overall": manipulation_metrics, "episodes": manipulation_metric_rows},
                indent=2,
            ),
            encoding="utf-8",
        )

    summary = {
        "policy_path": str(policy_path),
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": args.dataset_root,
        "evaluation_dataset_frequency_hz": float(meta.fps),
        "model_control_frequency_hz": getattr(policy_cfg, "control_frequency_hz", None),
        "split": args.split,
        "eval_split": args.eval_split,
        "episodes": sorted(per_episode),
        "num_frames": int(all_expert_arr.shape[0]),
        "deployment_metadata": policy_cfg.deployment_metadata() if io_schema_enabled else None,
        "supervision_contract": supervision_contract,
        "action_dt_seconds": action_dt,
        "execution_action_names": execution_action_names,
        "trajectory_action_names": trajectory_action_names,
        "plot_action_names": episode_plot_action_names,
        "single_chunk_plot_action_names": plot_action_names,
        "plot_anchor_semantics": {
            "episode_and_rolling_chunk_plots": {
                "b2_pose_delta": "episode_first_frame_command_integrated_se2_anchor",
                "z1_ee_state_delta": "episode_first_frame_measured_ee_anchor",
            },
            "single_chunk_plots": "each_inference_own_training_anchor",
            "metrics_and_csv": "checkpoint_training_representation",
        },
        "plot_y_limits_by_episode": episode_plot_y_limits_by_episode,
        "metrics": metrics_rows[0],
        "normalized_metrics": normalized_metrics_rows[0],
        "chunk_metrics": chunk_metrics_rows[0],
        "normalized_chunk_metrics": normalized_chunk_metrics_rows[0],
        "trajectory_metrics": trajectory_metrics_rows[0],
        "normalized_trajectory_metrics": normalized_trajectory_metrics_rows[0],
        "trajectory_chunk_metrics": trajectory_chunk_metrics_rows[0],
        "normalized_trajectory_chunk_metrics": normalized_trajectory_chunk_metrics_rows[0],
        "manipulation_onset_metrics": manipulation_metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logging.info("Open-loop eval finished.")
    logging.info("Frames: %d", all_expert_arr.shape[0])
    logging.info("MAE mean: %.6f", metrics_rows[0]["mae_mean"])
    logging.info("RMSE mean: %.6f", metrics_rows[0]["rmse_mean"])
    logging.info("Chunk MAE mean: %.6f", chunk_metrics_rows[0]["mae_mean"])
    logging.info("Chunk RMSE mean: %.6f", chunk_metrics_rows[0]["rmse_mean"])
    logging.info("Normalized chunk MAE mean: %.6f", normalized_chunk_metrics_rows[0]["mae_mean"])
    logging.info("Normalized chunk RMSE mean: %.6f", normalized_chunk_metrics_rows[0]["rmse_mean"])
    if io_schema_enabled:
        logging.info("Trajectory chunk MAE mean: %.6f", trajectory_chunk_metrics_rows[0]["mae_mean"])
        logging.info(
            "Normalized trajectory chunk MAE mean: %.6f",
            normalized_trajectory_chunk_metrics_rows[0]["mae_mean"],
        )
    logging.info("Metrics: %s", output_dir / "metrics.csv")
    logging.info("Plots: %s", plot_dir)


if __name__ == "__main__":
    main()
