#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass

import numpy as np

from lerobot.utils.constants import ACTION, OBS_STATE

HEIGHT_INVARIANT_EE_STATE_KEY = "observation.height_invariant_ee_state"


@dataclass(frozen=True)
class MotionPriorityPool:
    frame_indices: np.ndarray
    total_frames: int
    translation_frames: int
    rotation_frames: int
    gripper_frames: int


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-8):
        raise ValueError("height-invariant EE state contains an invalid zero-length rotation axis")
    return vector / norm


def _rotation_matrices_from_6d(rotation_6d: np.ndarray) -> np.ndarray:
    first = _normalize(rotation_6d[:, :3])
    second_raw = rotation_6d[:, 3:6]
    second = _normalize(second_raw - np.sum(first * second_raw, axis=-1, keepdims=True) * first)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-1)


def motion_priority_masks(
    ee_state: np.ndarray,
    gripper: np.ndarray,
    *,
    horizon_frames: int,
    translation_threshold_m: float,
    rotation_threshold_rad: float,
    gripper_change_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ee_state = np.asarray(ee_state, dtype=np.float64)
    gripper = np.asarray(gripper, dtype=np.float64)
    if ee_state.ndim != 2 or ee_state.shape[1] != 9:
        raise ValueError(f"height-invariant EE state must have shape (N, 9), got {ee_state.shape}")
    if gripper.shape != (len(ee_state),):
        raise ValueError(f"gripper must have shape ({len(ee_state)},), got {gripper.shape}")
    if horizon_frames < 1:
        raise ValueError(f"horizon_frames must be >= 1, got {horizon_frames}")
    if len(ee_state) == 0:
        empty = np.zeros(0, dtype=bool)
        return empty, empty, empty

    rotations = _rotation_matrices_from_6d(ee_state[:, :6])
    translation = np.zeros(len(ee_state), dtype=bool)
    rotation = np.zeros(len(ee_state), dtype=bool)
    gripper_change = np.zeros(len(ee_state), dtype=bool)
    for offset in range(1, min(horizon_frames, len(ee_state) - 1) + 1):
        current = slice(None, -offset)
        future = slice(offset, None)
        translation[:-offset] |= (
            np.linalg.norm(ee_state[future, 6:9] - ee_state[current, 6:9], axis=-1) >= translation_threshold_m
        )
        relative = np.einsum("nji,njk->nik", rotations[current], rotations[future])
        cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
        rotation[:-offset] |= np.arccos(cosine) >= rotation_threshold_rad
        gripper_change[:-offset] |= np.abs(gripper[future] - gripper[current]) >= gripper_change_threshold
    return translation, rotation, gripper_change


def build_motion_priority_pool(
    dataset,
    *,
    horizon_frames: int,
    translation_threshold_m: float,
    rotation_threshold_rad: float,
    gripper_change_threshold: float,
) -> MotionPriorityPool:
    state_names = dataset.meta.features.get(OBS_STATE, {}).get("names") or []
    action_names = dataset.meta.features.get(ACTION, {}).get("names") or []
    control_ee_names = [
        f"control_height_invariant_ee_{suffix}"
        for suffix in (
            "rot6d_col0_x", "rot6d_col0_y", "rot6d_col0_z",
            "rot6d_col1_x", "rot6d_col1_y", "rot6d_col1_z", "x", "y", "z",
        )
    ]
    control_ee_indices = (
        [action_names.index(name) for name in control_ee_names]
        if all(name in action_names for name in control_ee_names)
        else None
    )
    continuous_ee_names = [
        f"continuous_height_invariant_ee_state_{suffix}"
        for suffix in (
            "rot6d_col0_x", "rot6d_col0_y", "rot6d_col0_z",
            "rot6d_col1_x", "rot6d_col1_y", "rot6d_col1_z", "x", "y", "z",
        )
    ]
    embedded_ee_indices = (
        [state_names.index(name) for name in continuous_ee_names]
        if all(name in state_names for name in continuous_ee_names)
        else None
    )
    if (
        control_ee_indices is None
        and embedded_ee_indices is None
        and HEIGHT_INVARIANT_EE_STATE_KEY not in dataset.hf_dataset.column_names
    ):
        raise ValueError("Motion-balanced sampling requires the configured continuous EE supervision trajectory")
    if "gripper_target" not in action_names:
        raise ValueError("Motion-balanced sampling requires a named gripper_target action dimension")
    gripper_dim = action_names.index("gripper_target")
    ee_column = (
        ACTION
        if control_ee_indices is not None
        else OBS_STATE if embedded_ee_indices is not None else HEIGHT_INVARIANT_EE_STATE_KEY
    )
    selected_columns = list(dict.fromkeys(["index", "episode_index", ee_column, ACTION]))
    columns = dataset.hf_dataset.select_columns(selected_columns).with_format("numpy")

    priority_indices: list[np.ndarray] = []
    total_frames = translation_frames = rotation_frames = gripper_frames = 0
    current_episode: int | None = None
    index_parts: list[np.ndarray] = []
    ee_parts: list[np.ndarray] = []
    gripper_parts: list[np.ndarray] = []

    def flush_episode() -> None:
        nonlocal total_frames, translation_frames, rotation_frames, gripper_frames
        if not index_parts:
            return
        indices = np.concatenate(index_parts)
        ee_state = np.concatenate(ee_parts)
        gripper = np.concatenate(gripper_parts)
        translation, rotation, gripper_change = motion_priority_masks(
            ee_state,
            gripper,
            horizon_frames=horizon_frames,
            translation_threshold_m=translation_threshold_m,
            rotation_threshold_rad=rotation_threshold_rad,
            gripper_change_threshold=gripper_change_threshold,
        )
        priority_indices.append(indices[translation | rotation | gripper_change])
        total_frames += len(indices)
        translation_frames += int(translation.sum())
        rotation_frames += int(rotation.sum())
        gripper_frames += int(gripper_change.sum())

    for batch in columns.iter(batch_size=65_536):
        batch_episodes = np.asarray(batch["episode_index"], dtype=np.int64)
        boundaries = np.flatnonzero(np.diff(batch_episodes)) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(batch_episodes)]))
        for start, end in zip(starts, ends, strict=True):
            episode = int(batch_episodes[start])
            if current_episode is not None and episode != current_episode:
                flush_episode()
                index_parts.clear()
                ee_parts.clear()
                gripper_parts.clear()
            current_episode = episode
            index_parts.append(np.asarray(batch["index"][start:end], dtype=np.int64))
            ee_values = np.asarray(batch[ee_column][start:end], dtype=np.float64)
            if control_ee_indices is not None:
                ee_values = ee_values[:, control_ee_indices]
            elif embedded_ee_indices is not None:
                ee_values = ee_values[:, embedded_ee_indices]
            ee_parts.append(ee_values)
            actions = np.asarray(batch[ACTION][start:end], dtype=np.float64)
            gripper_parts.append(actions[:, gripper_dim])
    flush_episode()
    frame_indices = np.concatenate(priority_indices) if priority_indices else np.empty(0, dtype=np.int64)
    return MotionPriorityPool(
        frame_indices=frame_indices,
        total_frames=total_frames,
        translation_frames=translation_frames,
        rotation_frames=rotation_frames,
        gripper_frames=gripper_frames,
    )
