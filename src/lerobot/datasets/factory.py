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
import logging
import math
from pprint import pformat

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.configs.rewards import RewardModelConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, IMAGENET_STATS, OBS_IMAGES, OBS_PREFIX, OBS_STATE, REWARD

from .dataset_metadata import LeRobotDatasetMetadata
from .lerobot_dataset import LeRobotDataset
from .multi_dataset import MultiLeRobotDataset
from .streaming_dataset import StreamingLeRobotDataset


def resolve_delta_timestamps(
    cfg: PreTrainedConfig | RewardModelConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the config.

    Args:
        cfg (PreTrainedConfig | RewardModelConfig): The config to read delta_indices from. Both
            ``PreTrainedConfig`` and concrete ``RewardModelConfig`` subclasses expose the
            ``{observation,action,reward}_delta_indices`` properties used below.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    dataset_fps = float(ds_meta.fps)
    model_fps = float(getattr(cfg, "control_frequency_hz", None) or dataset_fps)
    if model_fps > dataset_fps + 1e-9:
        logging.warning(
            "Policy control frequency %.3f Hz exceeds dataset frequency %.3f Hz. "
            "Training is allowed, but nearest-frame repetition cannot recover missing high-frequency behavior.",
            model_fps,
            dataset_fps,
        )

    def model_steps_to_dataset_timestamps(indices: list[int]) -> list[float]:
        return [round(index * dataset_fps / model_fps) / dataset_fps for index in indices]

    mem_vit_enabled = bool(getattr(cfg, "mem_vit_enabled", False))
    mem_vit_num_frames = int(getattr(cfg, "mem_vit_num_frames", 1))
    requested_mem_interval = getattr(cfg, "mem_vit_frame_interval_seconds", None)
    if requested_mem_interval is not None:
        mem_vit_frame_stride = max(1, round(float(requested_mem_interval) * dataset_fps))
        if not bool(getattr(cfg, "io_schema_resolved", False)):
            cfg.mem_vit_frame_stride = mem_vit_frame_stride
    else:
        mem_vit_frame_stride = int(getattr(cfg, "mem_vit_frame_stride", 1))
    if mem_vit_enabled and mem_vit_num_frames > 1:
        mem_vit_delta_indices = [-i * mem_vit_frame_stride for i in reversed(range(mem_vit_num_frames))]
        image_features = getattr(cfg, "image_features", {})
    else:
        mem_vit_delta_indices = None
        image_features = {}

    global_pose_names = ("b2_position_x", "b2_position_y", "b2_yaw")
    state_names = ds_meta.features.get(OBS_STATE, {}).get("names")
    use_global_pose_trajectory = (
        getattr(cfg, "b2_action_representation", None) == "local_trajectory"
        and isinstance(state_names, list)
        and all(name in state_names for name in global_pose_names)
    )
    if use_global_pose_trajectory:
        resolved_pose_indices = [state_names.index(name) for name in global_pose_names]
        if not bool(getattr(cfg, "io_schema_resolved", False)):
            cfg.b2_global_pose_state_indices = resolved_pose_indices
        state_history_indices = mem_vit_delta_indices if mem_vit_delta_indices is not None else [0]
        # Layout consumed by the PI0.5 processor: history ending at the current
        # state, then one future state for every action in the chunk.
        state_delta_indices = state_history_indices + list(range(1, int(cfg.chunk_size) + 1))
    else:
        if hasattr(cfg, "b2_global_pose_state_indices") and not bool(
            getattr(cfg, "io_schema_resolved", False)
        ):
            cfg.b2_global_pose_state_indices = None
        state_delta_indices = mem_vit_delta_indices

    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = model_steps_to_dataset_timestamps(cfg.action_delta_indices)
        if state_delta_indices is not None and key == OBS_STATE:
            history_length = (
                len(state_history_indices) if use_global_pose_trajectory else len(state_delta_indices)
            )
            history_indices = state_delta_indices[:history_length]
            future_indices = state_delta_indices[history_length:]
            history_timestamps = [index / dataset_fps for index in history_indices]
            future_timestamps = model_steps_to_dataset_timestamps(future_indices)
            delta_timestamps[key] = history_timestamps + future_timestamps
            continue
        if (
            mem_vit_delta_indices is not None
            and key.startswith(OBS_IMAGES)
            and key in ds_meta.camera_keys
            and (not image_features or key in image_features)
        ):
            delta_timestamps[key] = [i / dataset_fps for i in mem_vit_delta_indices]
            continue
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    if isinstance(cfg.dataset.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, ds_meta)
        if not cfg.dataset.streaming:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                return_uint8=True,
                depth_output_unit=cfg.dataset.depth_output_unit,
                tolerance_s=cfg.tolerance_s,
            )
        else:
            dataset = StreamingLeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                max_num_shards=cfg.num_workers,
                tolerance_s=cfg.tolerance_s,
                return_uint8=True,
            )
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
        dataset = MultiLeRobotDataset(
            cfg.dataset.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            if key in dataset.meta.depth_keys:
                continue  # Exclude depth keys from ImageNet stats
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset


def make_train_eval_datasets(
    cfg: TrainPipelineConfig,
) -> tuple[LeRobotDataset | MultiLeRobotDataset, LeRobotDataset | None]:
    """Create train and optional eval datasets by splitting episodes based on eval_split.

    The last ceil(n_episodes * eval_split) episodes per task are held out for evaluation.
    If eval_split == 0.0, returns (full_dataset, None).
    """
    full_dataset = make_dataset(cfg)

    if cfg.dataset.eval_split == 0.0:
        return full_dataset, None

    base_episodes = (
        full_dataset.episodes if full_dataset.episodes is not None else list(range(full_dataset.num_episodes))
    )

    episode_tasks = full_dataset.meta.episodes["tasks"]
    task_to_episodes: dict[str, list[int]] = {}
    for ep_idx in base_episodes:
        task_key = episode_tasks[ep_idx][0] if episode_tasks[ep_idx] else ""
        task_to_episodes.setdefault(task_key, []).append(ep_idx)

    train_episodes, eval_episodes = [], []
    for eps in task_to_episodes.values():
        n_eval = math.ceil(len(eps) * cfg.dataset.eval_split)
        train_episodes.extend(eps[: len(eps) - n_eval])
        eval_episodes.extend(eps[len(eps) - n_eval :])

    if not train_episodes:
        raise ValueError(
            f"eval_split={cfg.dataset.eval_split} leaves 0 training episodes from {len(base_episodes)} total."
        )

    logging.info(
        f"Train/eval split: {len(train_episodes)} train, {len(eval_episodes)} eval "
        f"(eval_split={cfg.dataset.eval_split}, {len(task_to_episodes)} tasks)"
    )

    delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, full_dataset.meta)

    train_image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    train_dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=train_episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=train_image_transforms,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
        tolerance_s=cfg.tolerance_s,
    )

    eval_dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=eval_episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=None,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
        tolerance_s=cfg.tolerance_s,
    )

    if cfg.dataset.use_imagenet_stats:
        for ds in (train_dataset, eval_dataset):
            for key in ds.meta.camera_keys:
                for stats_type, stats in IMAGENET_STATS.items():
                    ds.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return train_dataset, eval_dataset
