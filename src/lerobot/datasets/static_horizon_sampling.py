#!/usr/bin/env python

"""Classify fully static action horizons for PI0.5 sampling."""

from dataclasses import dataclass

import numpy as np

from lerobot.utils.constants import ACTION


@dataclass(frozen=True)
class StaticHorizonPool:
    interior_frame_indices: np.ndarray
    terminal_frame_indices: np.ndarray
    total_frames: int


def static_horizon_masks(
    actions: np.ndarray,
    *,
    horizon_frames: int,
    b2_indices: tuple[int, int, int],
    inactive_index: int,
    reset_index: int,
    gripper_index: int,
    b2_tolerance: float,
    gripper_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return interior/terminal masks for starts whose complete horizon is static.

    Terminal starts may have fewer than ``horizon_frames`` real frames: the
    final legal hold command is continued by right padding. Interior starts
    must contain the complete horizon before later motion resumes.
    """
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2:
        raise ValueError(f"actions must have shape (frames, dims), got {actions.shape}")
    if horizon_frames < 1:
        raise ValueError(f"horizon_frames must be >= 1, got {horizon_frames}")
    length = len(actions)
    interior = np.zeros(length, dtype=bool)
    terminal = np.zeros(length, dtype=bool)
    if length == 0:
        return interior, terminal

    frame_static = np.all(np.abs(actions[:, list(b2_indices)]) <= b2_tolerance, axis=1)
    frame_static &= actions[:, inactive_index] >= 0.5
    frame_static &= actions[:, reset_index] < 0.5

    transition_static = np.ones(length, dtype=bool)
    if length > 1:
        gripper_ok = np.abs(np.diff(actions[:, gripper_index])) <= gripper_tolerance
        # Inactive masks the EE channels in the PI0.5 loss and means that the
        # controller holds its joints. Stored interpolated EE values therefore
        # do not turn an inactive wait into arm motion.
        transition_static[:-1] = gripper_ok

    # A start belongs to the terminal pool exactly when every remaining real
    # command is a legal hold and every remaining transition is stationary.
    terminal[:] = np.logical_and.accumulate((frame_static & transition_static)[::-1])[::-1]

    if length >= horizon_frames:
        invalid_frame_prefix = np.concatenate(([0], np.cumsum(~frame_static)))
        invalid_transition_prefix = np.concatenate(([0], np.cumsum(~transition_static)))
        starts = np.arange(length - horizon_frames + 1)
        ends = starts + horizon_frames
        full_static = (invalid_frame_prefix[ends] - invalid_frame_prefix[starts] == 0) & (
            invalid_transition_prefix[ends - 1] - invalid_transition_prefix[starts] == 0
        )
        interior[starts] = full_static & ~terminal[starts]
    return interior, terminal


def build_static_horizon_pool(
    dataset,
    *,
    horizon_frames: int,
    b2_tolerance: float,
    gripper_tolerance: float,
) -> StaticHorizonPool:
    action_names = list(dataset.meta.features.get(ACTION, {}).get("names") or [])

    def require(name: str) -> int:
        if name not in action_names:
            raise ValueError(f"Static-horizon sampling requires action field {name!r}")
        return action_names.index(name)

    b2_indices = tuple(require(name) for name in ("b2_vx", "b2_vy", "b2_omega_z"))
    inactive_index = require("arm_teleop_inactive")
    reset_index = require("arm_reset")
    gripper_index = require("gripper_target")
    columns = dataset.hf_dataset.select_columns(["index", "episode_index", ACTION]).with_format("numpy")
    interior_parts: list[np.ndarray] = []
    terminal_parts: list[np.ndarray] = []
    total_frames = 0
    current_episode: int | None = None
    index_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []

    def flush_episode() -> None:
        nonlocal total_frames
        if not index_parts:
            return
        indices = np.concatenate(index_parts)
        actions = np.concatenate(action_parts)
        interior, terminal = static_horizon_masks(
            actions,
            horizon_frames=horizon_frames,
            b2_indices=b2_indices,
            inactive_index=inactive_index,
            reset_index=reset_index,
            gripper_index=gripper_index,
            b2_tolerance=b2_tolerance,
            gripper_tolerance=gripper_tolerance,
        )
        interior_parts.append(indices[interior])
        terminal_parts.append(indices[terminal])
        total_frames += len(indices)

    for batch in columns.iter(batch_size=65_536):
        episodes = np.asarray(batch["episode_index"], dtype=np.int64)
        boundaries = np.flatnonzero(np.diff(episodes)) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(episodes)]))
        for start, end in zip(starts, ends, strict=True):
            episode = int(episodes[start])
            if current_episode is not None and episode != current_episode:
                flush_episode()
                index_parts.clear()
                action_parts.clear()
            current_episode = episode
            index_parts.append(np.asarray(batch["index"][start:end], dtype=np.int64))
            action_parts.append(np.asarray(batch[ACTION][start:end], dtype=np.float64))
    flush_episode()
    return StaticHorizonPool(
        interior_frame_indices=(
            np.concatenate(interior_parts) if interior_parts else np.empty(0, dtype=np.int64)
        ),
        terminal_frame_indices=(
            np.concatenate(terminal_parts) if terminal_parts else np.empty(0, dtype=np.int64)
        ),
        total_frames=total_frames,
    )
