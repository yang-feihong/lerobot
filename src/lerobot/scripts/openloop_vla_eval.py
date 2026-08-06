#!/usr/bin/env python
"""Open-loop action evaluation for VLA policies on a LeRobotDataset.

This evaluates a trained policy on recorded dataset observations without sending
actions to a robot or simulator. For every sampled frame it predicts an action
chunk, takes the first action, unnormalizes it with the policy postprocessor, and
compares it with the dataset action at the same frame.
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
    B2_TRAJECTORY_NAMES,
    action_dataset_indices,
    action_schema_kwargs,
    b2_execution_action_names,
    b2_trajectory_action_names,
    decode_b2_action_chunk,
    encode_b2_action_chunk,
    integrate_b2_execution_chunk,
    make_b2_trajectory_stats,
)
from lerobot.scripts.lerobot_train import apply_task_variants_to_batch, load_task_variants
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.utils import init_logging


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


def _to_numpy_1d(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]], action_names: list[str]) -> None:
    fieldnames = ["scope", "episode_index", "num_frames", "mae_mean", "rmse_mean"]
    for name in action_names:
        fieldnames.extend([f"mae/{name}", f"rmse/{name}"])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_predictions_csv(path: Path, rows: list[dict[str, Any]], action_names: list[str]) -> None:
    fieldnames = ["episode_index", "frame_index", "timestamp", "task"]
    for name in action_names:
        fieldnames.extend([f"expert/{name}", f"pred/{name}", f"error/{name}"])
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
    return execution_names[: b2_start + 3] + list(B2_TRAJECTORY_NAMES) + execution_names[b2_start + 3 :]


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
        ax.plot(x, expert[:, i], label="expert", linewidth=1.3, marker="o", markersize=5.0)
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
    clear whether the policy's whole short-horizon plan matches the expert, not
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
                label="expert chunks" if row_idx == 0 else None,
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
        ax.plot(x, expert_chunk[:, dim], label="expert", color="tab:blue", linewidth=1.4)
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
    expert action in that chunk.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-path", required=True, help="Run dir, checkpoint dir, or pretrained_model dir."
    )
    parser.add_argument("--dataset-repo-id", default="local/b2_z1_vla")
    parser.add_argument("--dataset-root", default="/data/b2_z1_vla_lerobot")
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
    parser.add_argument("--task-variants-path", default=None)
    parser.add_argument("--task-variant", choices=["dataset", "first", "random"], default="first")
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

    init_logging()

    policy_path = _resolve_policy_path(args.policy_path)
    if args.output_dir is None:
        output_dir = policy_path.parent.parent / "openloop_eval" / f"{args.split}_latest"
    else:
        output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
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
    deployment_metadata_path = policy_path / "pi05_deployment_metadata.json"
    if getattr(policy_cfg, "io_schema_resolved", False):
        if not deployment_metadata_path.exists():
            raise FileNotFoundError(
                f"Checkpoint is missing required deployment metadata: {deployment_metadata_path}"
            )
        saved_deployment_metadata = json.loads(deployment_metadata_path.read_text(encoding="utf-8"))
        if saved_deployment_metadata != policy_cfg.deployment_metadata():
            raise ValueError("pi05_deployment_metadata.json disagrees with the checkpoint policy config")

    meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    io_schema_enabled = bool(getattr(policy_cfg, "io_schema_resolved", False))
    b2_trajectory_dt = getattr(policy_cfg, "b2_local_trajectory_dt", None)
    if io_schema_enabled:
        control_frequency_hz = getattr(policy_cfg, "control_frequency_hz", None)
        if control_frequency_hz is None:
            raise ValueError("B2+Z1 checkpoint is missing control_frequency_hz")
        expected_dt = 1.0 / float(control_frequency_hz)
        if b2_trajectory_dt is None:
            raise ValueError("B2+Z1 checkpoint is missing b2_local_trajectory_dt")
        if abs(float(b2_trajectory_dt) - expected_dt) > 1e-9:
            raise ValueError(
                f"Trajectory checkpoint dt={b2_trajectory_dt} does not match "
                f"control_frequency_hz={control_frequency_hz}"
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
        "Per-episode limits: frames=%s, chunk_plots=%s",
        "all" if args.max_frames_per_episode == 0 else args.max_frames_per_episode,
        "all" if args.max_chunk_plots_per_episode == 0 else args.max_chunk_plots_per_episode,
    )

    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=None,
        video_backend=args.video_backend,
        return_uint8=True,
    )

    dataset_action_dim = int(meta.features[ACTION]["shape"][0])
    dataset_action_names = _default_action_names(meta, dataset_action_dim)
    execution_action_names = list(dataset_action_names)
    trajectory_action_names = list(dataset_action_names)
    trajectory_stats = None
    execution_stats = None
    schema_kwargs = action_schema_kwargs(policy_cfg) if io_schema_enabled else {}
    if io_schema_enabled:
        if dataset_action_dim != 16:
            raise ValueError(
                f"B2 local trajectory evaluation expects a 16D dataset action, got {dataset_action_dim}"
            )
        execution_action_names = b2_execution_action_names(dataset_action_names, **schema_kwargs)
        trajectory_action_names = b2_trajectory_action_names(dataset_action_names, **schema_kwargs)
        assert execution_action_names is not None
        assert trajectory_action_names is not None
        trajectory_schema = {**schema_kwargs, "representation": "local_trajectory"}
        transformed_stats = make_b2_trajectory_stats(
            meta.stats,
            dt=float(b2_trajectory_dt),
            chunk_size=int(policy_cfg.chunk_size),
            **trajectory_schema,
        )
        assert transformed_stats is not None
        trajectory_stats = transformed_stats[ACTION]
        execution_schema = {**schema_kwargs, "representation": "velocity"}
        execution_transformed_stats = make_b2_trajectory_stats(
            meta.stats,
            dt=float(b2_trajectory_dt),
            chunk_size=int(policy_cfg.chunk_size),
            **execution_schema,
        )
        assert execution_transformed_stats is not None
        execution_stats = execution_transformed_stats[ACTION]

        raw_limits = _action_plot_y_limits(meta, dataset_action_names)
        selected_indices = action_dataset_indices(
            predict_arm_teleop_inactive=policy_cfg.action_predict_arm_teleop_inactive,
            predict_arm_reset=policy_cfg.action_predict_arm_reset,
            predict_ee_pose=policy_cfg.action_predict_ee_pose,
            predict_gripper=policy_cfg.action_predict_gripper,
            include_task_complete=policy_cfg.action_predict_task_complete,
        )
        execution_limits = [raw_limits[i] for i in selected_indices]
        trajectory_q01 = np.asarray(trajectory_stats["q01"], dtype=np.float32)
        trajectory_q99 = np.asarray(trajectory_stats["q99"], dtype=np.float32)
        b2_start = 0
        trajectory_limits = [
            _padded_limits(float(trajectory_q01[i]), float(trajectory_q99[i]))
            for i in range(b2_start, b2_start + 3)
        ]
        plot_action_names = _combined_base_plot_names(execution_action_names, b2_start)
        plot_y_limits = (
            execution_limits[: b2_start + 3] + trajectory_limits + execution_limits[b2_start + 3 :]
        )
    else:
        plot_action_names = execution_action_names
        plot_y_limits = _action_plot_y_limits(meta, execution_action_names)
    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    dataloader = DataLoader(
        dataset,
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

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_path),
        pretrained_revision=getattr(policy_cfg, "pretrained_revision", None),
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": torch.device(args.device).type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
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
            "task": [],
        }
    )
    counts: dict[int, int] = defaultdict(int)
    prediction_rows: list[dict[str, Any]] = []

    with torch.inference_mode():
        for step, batch in enumerate(tqdm(dataloader, desc="Open-loop eval", unit="batch")):
            if batch is None:
                continue

            dataset_expert_chunk = _as_action_chunk(batch[ACTION].detach().cpu().to(torch.float32))
            episode_indices = batch["episode_index"].detach().cpu().view(-1).tolist()
            keep = []
            for i, ep_idx in enumerate(episode_indices):
                ep_idx = int(ep_idx)
                if args.max_frames_per_episode > 0 and counts[ep_idx] >= args.max_frames_per_episode:
                    continue
                keep.append(i)
                counts[ep_idx] += 1
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
                dataset_expert_chunk = dataset_expert_chunk.index_select(0, keep_t)

            if args.task_variant != "dataset":
                apply_task_variants_to_batch(
                    batch,
                    task_variants,
                    step=step,
                    seed=args.seed,
                    randomize=args.task_variant == "random",
                )

            batch = _batch_to_device_and_float_images(batch, dataset.meta.camera_keys)
            processed = preprocessor(batch)
            pred_execution_chunk = (
                postprocessor(policy.predict_action_chunk(processed)).detach().cpu().to(torch.float32)
            )
            if io_schema_enabled:
                action_is_pad = batch.get(f"{ACTION}_is_pad")
                if action_is_pad is not None:
                    action_is_pad = action_is_pad.detach().cpu().to(torch.bool)
                global_pose = None
                global_pose_indices = getattr(policy_cfg, "b2_global_pose_state_indices", None)
                if global_pose_indices is not None:
                    raw_state = batch[OBS_STATE].detach().cpu().to(torch.float32)
                    history_length = policy_cfg.mem_vit_num_frames if policy_cfg.mem_vit_enabled else 1
                    current_pose = raw_state[
                        ..., history_length - 1 : history_length, list(global_pose_indices)
                    ]
                    future_pose = raw_state[..., history_length:, list(global_pose_indices)]
                    global_pose = torch.cat((current_pose, future_pose), dim=-2)
                expert_trajectory_chunk = encode_b2_action_chunk(
                    dataset_expert_chunk,
                    dt=float(b2_trajectory_dt),
                    is_pad=action_is_pad,
                    global_pose=global_pose,
                    **trajectory_schema,
                )
                expert_chunk = decode_b2_action_chunk(
                    expert_trajectory_chunk,
                    dt=float(b2_trajectory_dt),
                    representation="local_trajectory",
                )
                pred_chunk = pred_execution_chunk
                pred_trajectory_chunk = integrate_b2_execution_chunk(
                    pred_execution_chunk,
                    dt=float(b2_trajectory_dt),
                )
            else:
                expert_chunk = dataset_expert_chunk
                pred_chunk = pred_execution_chunk
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
                ep_store["task"].append(task_list[i])

                row: dict[str, Any] = {
                    "episode_index": int(ep_idx),
                    "frame_index": int(frame_np[i]),
                    "timestamp": float(ts_np[i]),
                    "task": task_list[i],
                }
                err = pred_np[i] - expert_np[i]
                for j, name in enumerate(execution_action_names):
                    row[f"expert/{name}"] = float(expert_np[i, j])
                    row[f"pred/{name}"] = float(pred_np[i, j])
                    row[f"error/{name}"] = float(err[j])
                prediction_rows.append(row)

            if args.max_frames_per_episode > 0 and all(
                counts[int(ep_idx)] >= args.max_frames_per_episode for ep_idx in episodes
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
        metrics_rows.append(_compute_metrics(ep_idx, expert, pred, execution_action_names))
        normalized_metrics_rows.append(
            _compute_metrics(
                ep_idx,
                _normalize_execution_action_array(expert, meta, execution_stats),
                _normalize_execution_action_array(pred, meta, execution_stats),
                execution_action_names,
            )
        )
        chunk_metrics_rows.append(
            _compute_metrics(
                ep_idx,
                valid_expert_chunks,
                valid_pred_chunks,
                execution_action_names,
            )
        )
        normalized_chunk_metrics_rows.append(
            _compute_metrics(
                ep_idx,
                _normalize_execution_action_array(valid_expert_chunks, meta, execution_stats),
                _normalize_execution_action_array(valid_pred_chunks, meta, execution_stats),
                execution_action_names,
            )
        )
        trajectory_metrics_rows.append(
            _compute_metrics(
                ep_idx,
                expert_trajectory,
                pred_trajectory,
                trajectory_action_names,
            )
        )
        trajectory_chunk_metrics_rows.append(
            _compute_metrics(
                ep_idx,
                valid_expert_trajectory_chunks,
                valid_pred_trajectory_chunks,
                trajectory_action_names,
            )
        )
        if trajectory_stats is not None:
            normalized_trajectory_metrics_rows.append(
                _compute_metrics(
                    ep_idx,
                    _normalize_with_stats(expert_trajectory, trajectory_stats),
                    _normalize_with_stats(pred_trajectory, trajectory_stats),
                    trajectory_action_names,
                )
            )
            normalized_trajectory_chunk_metrics_rows.append(
                _compute_metrics(
                    ep_idx,
                    _normalize_with_stats(valid_expert_trajectory_chunks, trajectory_stats),
                    _normalize_with_stats(valid_pred_trajectory_chunks, trajectory_stats),
                    trajectory_action_names,
                )
            )
        else:
            normalized_trajectory_metrics_rows.append(normalized_metrics_rows[-1].copy())
            normalized_trajectory_chunk_metrics_rows.append(normalized_chunk_metrics_rows[-1].copy())

        plot_expert = (
            _combined_base_plot_arrays(expert, expert_trajectory, b2_start) if io_schema_enabled else expert
        )
        plot_pred = _combined_base_plot_arrays(pred, pred_trajectory, b2_start) if io_schema_enabled else pred
        plot_expert_chunks = (
            _combined_base_plot_arrays(expert_chunks, expert_trajectory_chunks, b2_start)
            if io_schema_enabled
            else expert_chunks
        )
        plot_pred_chunks = (
            _combined_base_plot_arrays(pred_chunks, pred_trajectory_chunks, b2_start)
            if io_schema_enabled
            else pred_chunks
        )
        _plot_episode(
            plot_dir / f"episode_{ep_idx:06d}.png",
            ep_idx,
            frame_index,
            plot_expert,
            plot_pred,
            plot_action_names,
            plot_y_limits,
        )
        _plot_episode_rolling_chunks(
            chunk_plot_dir / f"episode_{ep_idx:06d}_rolling_chunks.png",
            ep_idx,
            frame_index,
            plot_expert_chunks,
            plot_pred_chunks,
            plot_action_names,
            plot_y_limits,
            valid_chunk_lengths,
        )
        ep_single_dir = single_chunk_plot_dir / f"episode_{ep_idx:06d}"
        ep_single_dir.mkdir(parents=True, exist_ok=True)
        n_single_plots = len(frame_index)
        if args.max_chunk_plots_per_episode > 0:
            n_single_plots = min(n_single_plots, args.max_chunk_plots_per_episode)
        for i in range(n_single_plots):
            single_chunk_plot_jobs.append(
                (
                    ep_single_dir / f"chunk_start_{int(frame_index[i]):06d}.png",
                    ep_idx,
                    int(frame_index[i]),
                    plot_expert_chunks[i, : valid_chunk_lengths[i]],
                    plot_pred_chunks[i, : valid_chunk_lengths[i]],
                    plot_action_names,
                    plot_y_limits,
                )
            )
        npz_payload[f"episode_{ep_idx:06d}_frame_index"] = frame_index
        npz_payload[f"episode_{ep_idx:06d}_expert"] = expert
        npz_payload[f"episode_{ep_idx:06d}_pred"] = pred
        npz_payload[f"episode_{ep_idx:06d}_expert_chunk"] = expert_chunks
        npz_payload[f"episode_{ep_idx:06d}_pred_chunk"] = pred_chunks
        npz_payload[f"episode_{ep_idx:06d}_expert_trajectory"] = expert_trajectory
        npz_payload[f"episode_{ep_idx:06d}_pred_trajectory"] = pred_trajectory
        npz_payload[f"episode_{ep_idx:06d}_expert_trajectory_chunk"] = expert_trajectory_chunks
        npz_payload[f"episode_{ep_idx:06d}_pred_trajectory_chunk"] = pred_trajectory_chunks
        npz_payload[f"episode_{ep_idx:06d}_valid_chunk_length"] = valid_chunk_lengths

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
    metrics_rows.insert(0, _compute_metrics("all", all_expert_arr, all_pred_arr, execution_action_names))
    normalized_metrics_rows.insert(
        0,
        _compute_metrics(
            "all",
            _normalize_execution_action_array(all_expert_arr, meta, execution_stats),
            _normalize_execution_action_array(all_pred_arr, meta, execution_stats),
            execution_action_names,
        ),
    )
    chunk_metrics_rows.insert(
        0,
        _compute_metrics("all", all_expert_chunk_arr, all_pred_chunk_arr, execution_action_names),
    )
    normalized_chunk_metrics_rows.insert(
        0,
        _compute_metrics(
            "all",
            _normalize_execution_action_array(all_expert_chunk_arr, meta, execution_stats),
            _normalize_execution_action_array(all_pred_chunk_arr, meta, execution_stats),
            execution_action_names,
        ),
    )

    trajectory_metrics_rows.insert(
        0,
        _compute_metrics(
            "all",
            all_expert_trajectory_arr,
            all_pred_trajectory_arr,
            trajectory_action_names,
        ),
    )
    trajectory_chunk_metrics_rows.insert(
        0,
        _compute_metrics(
            "all",
            all_expert_trajectory_chunk_arr,
            all_pred_trajectory_chunk_arr,
            trajectory_action_names,
        ),
    )
    if trajectory_stats is not None:
        normalized_trajectory_metrics_rows.insert(
            0,
            _compute_metrics(
                "all",
                _normalize_with_stats(all_expert_trajectory_arr, trajectory_stats),
                _normalize_with_stats(all_pred_trajectory_arr, trajectory_stats),
                trajectory_action_names,
            ),
        )
        normalized_trajectory_chunk_metrics_rows.insert(
            0,
            _compute_metrics(
                "all",
                _normalize_with_stats(all_expert_trajectory_chunk_arr, trajectory_stats),
                _normalize_with_stats(all_pred_trajectory_chunk_arr, trajectory_stats),
                trajectory_action_names,
            ),
        )
    else:
        normalized_trajectory_metrics_rows.insert(0, normalized_metrics_rows[0].copy())
        normalized_trajectory_chunk_metrics_rows.insert(0, normalized_chunk_metrics_rows[0].copy())

    _write_metrics_csv(output_dir / "metrics.csv", metrics_rows, execution_action_names)
    _write_metrics_csv(output_dir / "normalized_metrics.csv", normalized_metrics_rows, execution_action_names)
    _write_metrics_csv(output_dir / "chunk_metrics.csv", chunk_metrics_rows, execution_action_names)
    _write_metrics_csv(
        output_dir / "normalized_chunk_metrics.csv",
        normalized_chunk_metrics_rows,
        execution_action_names,
    )
    _write_metrics_csv(
        output_dir / "trajectory_metrics.csv", trajectory_metrics_rows, trajectory_action_names
    )
    _write_metrics_csv(
        output_dir / "normalized_trajectory_metrics.csv",
        normalized_trajectory_metrics_rows,
        trajectory_action_names,
    )
    _write_metrics_csv(
        output_dir / "trajectory_chunk_metrics.csv",
        trajectory_chunk_metrics_rows,
        trajectory_action_names,
    )
    _write_metrics_csv(
        output_dir / "normalized_trajectory_chunk_metrics.csv",
        normalized_trajectory_chunk_metrics_rows,
        trajectory_action_names,
    )
    _write_predictions_csv(output_dir / "predictions.csv", prediction_rows, execution_action_names)
    np.savez_compressed(output_dir / "predictions.npz", **npz_payload)

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
        "b2_local_trajectory_dt": b2_trajectory_dt,
        "execution_action_names": execution_action_names,
        "trajectory_action_names": trajectory_action_names,
        "plot_action_names": plot_action_names,
        "plot_y_limits": {
            name: [float(low), float(high)]
            for name, (low, high) in zip(plot_action_names, plot_y_limits, strict=True)
        },
        "metrics": metrics_rows[0],
        "normalized_metrics": normalized_metrics_rows[0],
        "chunk_metrics": chunk_metrics_rows[0],
        "normalized_chunk_metrics": normalized_chunk_metrics_rows[0],
        "trajectory_metrics": trajectory_metrics_rows[0],
        "normalized_trajectory_metrics": normalized_trajectory_metrics_rows[0],
        "trajectory_chunk_metrics": trajectory_chunk_metrics_rows[0],
        "normalized_trajectory_chunk_metrics": normalized_trajectory_chunk_metrics_rows[0],
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
