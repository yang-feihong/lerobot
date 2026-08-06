#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..rtc.configuration_rtc import RTCConfig

DEFAULT_IMAGE_SIZE = 224
PI05_DEPLOYMENT_METADATA_NAME = "pi05_deployment_metadata.json"


@PreTrainedConfig.register_subclass("pi05")
@dataclass
class PI05Config(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Relative actions: converts absolute actions to relative (relative to state).
    use_relative_actions: bool = False
    # Joint names to exclude from relative (kept absolute). Empty list = all dims relative.
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    # Populated at runtime from dataset metadata by make_policy.
    action_feature_names: list[str] | None = None
    dataset_action_feature_names: list[str] | None = None
    dataset_state_feature_names: list[str] | None = None
    resolved_state_feature_names: list[str] | None = None
    state_feature_indices: list[int] | None = None
    io_schema_resolved: bool = False
    # Dataset/runtime timing and camera contract resolved by make_policy.
    control_frequency_hz: float | None = None
    # Training provenance used for timestamp resampling and resume checks. This
    # is intentionally excluded from pi05_deployment_metadata.json.
    dataset_frequency_hz: float | None = None
    dataset_camera_keys: list[str] | None = None

    # Robot I/O schema. These selections are serialized in every checkpoint.
    # The defaults retain only arm q, gripper feedback, and B2 trunk pose/height.
    state_use_arm_joint_positions: bool = True
    state_use_arm_joint_velocities: bool = False
    state_use_arm_gripper_feedback: bool = True
    state_use_b2_joint_positions: bool = False
    state_use_b2_joint_velocities: bool = False
    state_use_b2_trunk_pose: bool = True
    state_use_b2_linear_velocity: bool = False
    state_use_b2_angular_velocity: bool = False
    b2_action_representation: str = "local_trajectory"  # "velocity" or "local_trajectory"
    action_predict_arm_teleop_inactive: bool = True
    action_predict_arm_reset: bool = True
    action_predict_ee_pose: bool = True
    action_predict_gripper: bool = True
    action_predict_task_complete: bool = True

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Finetuning settings
    freeze_vision_encoder: bool = False  # Freeze only the vision encoder
    train_expert_only: bool = False  # Freeze entire VLM, train only action expert and projections

    # B2+Z1 VLA gate-aware action loss.
    #
    # The raw dataset is 16D. The default switches above keep a 16D model
    # schema; disabling optional groups updates names and indices at
    # runtime while keeping the same flow-matching head:
    #
    #   [0:3]    B2 local x, y, yaw     continuous, always supervised
    #   [3]      arm_teleop_inactive    bool/gate, class-balanced
    #   [4]      arm_reset              bool/gate, class-balanced
    #   [5:14]   EE rot6d + xyz         continuous, trained only when teleop is active and not reset
    #   [14]     gripper_target         bool/gate, class-balanced
    #   [15]     task_complete          explicit bool/gate, class-balanced
    #            Current B2+Z1 data stores gripper as two raw values (0 and a negative
    #            target position), so its active/event class lives on the negative side
    #            after quantile normalization. Override action_gripper_target_true_side
    #            if a future dataset stores gripper differently.
    #
    # "auto" applies this to a resolved B2+Z1 schema; "always"
    # forces it; "off" restores the original unweighted mean.
    action_loss_schema: str = "auto"  # "auto", "always", or "off"
    action_bool_loss_weight: float = 4.0
    action_continuous_loss_weight: float = 1.0
    action_masked_continuous_min_weight: float = 0.0
    action_bool_balance_eps: float = 1e-3
    # Resolved once from the complete train split before policy construction.
    # Every enabled bool uses the same fixed prior on every rank/microbatch.
    action_bool_true_fractions: dict[str, float] = field(default_factory=dict)
    action_gripper_target_true_side: str = "negative"  # "negative" or "positive"
    # Keep a bounded number of chunk starts whose current input is already in
    # the explicit completion tail. This teaches stable stopping without letting
    # arbitrarily long post-task bag tails dominate training and validation.
    task_complete_sample_tail_seconds: float | None = 2.0

    # B2 high-level action representation. The dataset remains 16D and stores
    # [vx, vy, omega_z]. At load time the configured groups are selected, B2
    # motion can stay as velocity or become a chunk-local SE(2) trajectory, and
    # task_complete remains the dataset's explicit completion label. The postprocessor
    # converts local trajectories back into executable body twist when needed.
    # Resolved automatically as 1 / model control frequency by make_policy. It is persisted
    # in checkpoints so inference uses exactly the training discretization.
    b2_local_trajectory_dt: float | None = None
    # Resolved from named dataset state fields. When present, measured global
    # poses replace command integration for local-trajectory supervision.
    b2_global_pose_state_indices: list[int] | None = None

    # MEM-ViT settings. Passing mem_vit_checkpoint also enables MEM-ViT.
    mem_vit_enabled: bool = False
    mem_vit_checkpoint: str | None = None
    mem_vit_num_frames: int = 1
    mem_vit_min_num_frames: int | None = None
    mem_vit_max_num_frames: int | None = None
    mem_vit_frame_stride: int = 1
    # Physical history interval. When set, the loader resolves frame_stride
    # against each dataset's native FPS instead of treating rows as time.
    mem_vit_frame_interval_seconds: float | None = 0.5
    mem_vit_temporal_every: int = 4
    mem_vit_use_original_for_k1: bool = True

    # Optimizer settings: see openpi `AdamW`
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    def __post_init__(self):
        super().__post_init__()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")
        if self.action_loss_schema not in ["auto", "always", "off"]:
            raise ValueError(
                f"Invalid action_loss_schema: {self.action_loss_schema}. Expected 'auto', 'always', or 'off'."
            )
        if self.b2_action_representation not in ["velocity", "local_trajectory"]:
            raise ValueError(
                "b2_action_representation must be 'velocity' or 'local_trajectory', got "
                f"{self.b2_action_representation!r}"
            )
        if not any(self.state_feature_switches().values()):
            raise ValueError("At least one state_use_* switch must be true")
        if self.action_bool_loss_weight <= 0:
            raise ValueError(f"action_bool_loss_weight must be > 0, got {self.action_bool_loss_weight}")
        if self.task_complete_sample_tail_seconds is not None and self.task_complete_sample_tail_seconds < 0:
            raise ValueError("task_complete_sample_tail_seconds must be >= 0 or None")
        invalid_bool_fractions = {
            name: fraction
            for name, fraction in self.action_bool_true_fractions.items()
            if not 0.0 < fraction < 1.0
        }
        if invalid_bool_fractions:
            raise ValueError(
                f"action_bool_true_fractions must be strictly between 0 and 1, got {invalid_bool_fractions}"
            )
        if self.action_continuous_loss_weight <= 0:
            raise ValueError(
                f"action_continuous_loss_weight must be > 0, got {self.action_continuous_loss_weight}"
            )
        if self.action_masked_continuous_min_weight < 0:
            raise ValueError(
                "action_masked_continuous_min_weight must be >= 0, got "
                f"{self.action_masked_continuous_min_weight}"
            )
        if self.action_gripper_target_true_side not in ["negative", "positive"]:
            raise ValueError(
                "Invalid action_gripper_target_true_side: "
                f"{self.action_gripper_target_true_side}. Expected 'negative' or 'positive'."
            )
        if self.b2_local_trajectory_dt is not None and self.b2_local_trajectory_dt <= 0:
            raise ValueError(f"b2_local_trajectory_dt must be positive, got {self.b2_local_trajectory_dt}")
        if self.control_frequency_hz is not None and self.control_frequency_hz <= 0:
            raise ValueError(f"control_frequency_hz must be positive, got {self.control_frequency_hz}")
        if self.dataset_frequency_hz is not None and self.dataset_frequency_hz <= 0:
            raise ValueError(f"dataset_frequency_hz must be positive, got {self.dataset_frequency_hz}")
        if self.mem_vit_checkpoint is not None:
            self.mem_vit_enabled = True
        if self.mem_vit_enabled and self.mem_vit_frame_interval_seconds is None:
            raise ValueError("MEM deployment requires mem_vit_frame_interval_seconds")
        if self.mem_vit_num_frames < 1:
            raise ValueError(f"mem_vit_num_frames must be >= 1, got {self.mem_vit_num_frames}")
        if (self.mem_vit_min_num_frames is None) != (self.mem_vit_max_num_frames is None):
            raise ValueError("mem_vit_min_num_frames and mem_vit_max_num_frames must be set together")
        if self.mem_vit_min_num_frames is not None and self.mem_vit_max_num_frames is not None:
            if self.mem_vit_min_num_frames < 1:
                raise ValueError(f"mem_vit_min_num_frames must be >= 1, got {self.mem_vit_min_num_frames}")
            if self.mem_vit_max_num_frames < self.mem_vit_min_num_frames:
                raise ValueError(
                    "mem_vit_max_num_frames must be >= mem_vit_min_num_frames, got "
                    f"{self.mem_vit_max_num_frames} < {self.mem_vit_min_num_frames}"
                )
            self.mem_vit_num_frames = self.mem_vit_max_num_frames
        if self.mem_vit_frame_stride < 1:
            raise ValueError(f"mem_vit_frame_stride must be >= 1, got {self.mem_vit_frame_stride}")
        if self.mem_vit_frame_interval_seconds is not None and self.mem_vit_frame_interval_seconds <= 0:
            raise ValueError(
                f"mem_vit_frame_interval_seconds must be positive, got {self.mem_vit_frame_interval_seconds}"
            )
        if self.mem_vit_temporal_every < 1:
            raise ValueError(f"mem_vit_temporal_every must be >= 1, got {self.mem_vit_temporal_every}")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features[ACTION] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    def deployment_metadata(self) -> dict:
        """Return the complete, versioned deployment contract for this checkpoint."""
        control_hz = self.control_frequency_hz
        action_dt = None if control_hz is None else 1.0 / control_hz
        history_interval = self.mem_vit_frame_interval_seconds if self.mem_vit_enabled else None
        min_history_frames = (
            self.mem_vit_min_num_frames
            if self.mem_vit_min_num_frames is not None
            else self.mem_vit_num_frames
        )
        max_history_frames = (
            self.mem_vit_max_num_frames
            if self.mem_vit_max_num_frames is not None
            else self.mem_vit_num_frames
        )

        def json_value(value):
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, dict):
                return {str(key): json_value(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [json_value(item) for item in value]
            return value

        rtc = None if self.rtc_config is None else json_value(asdict(self.rtc_config))
        image_features = {
            key: {"shape": list(feature.shape), "type": feature.type.value}
            for key, feature in self.image_features.items()
        }
        return {
            "format": "lerobot.pi05.deployment",
            "version": 1,
            "policy": {
                "type": self.type,
                "paligemma_variant": self.paligemma_variant,
                "action_expert_variant": self.action_expert_variant,
                "dtype": self.dtype,
                "max_state_dim": self.max_state_dim,
                "max_action_dim": self.max_action_dim,
                "n_obs_steps": self.n_obs_steps,
                "empty_cameras": self.empty_cameras,
                "image_resolution": list(self.image_resolution),
                "tokenizer_max_length": self.tokenizer_max_length,
                "num_inference_steps": self.num_inference_steps,
                "rtc": rtc,
            },
            "timing": {
                "control_frequency_hz": control_hz,
                "action_dt_seconds": action_dt,
                "chunk_size_steps": self.chunk_size,
                "chunk_horizon_seconds": None if action_dt is None else self.chunk_size * action_dt,
                "execute_steps_per_inference": self.n_action_steps,
                "replan_interval_seconds": (None if action_dt is None else self.n_action_steps * action_dt),
            },
            "observation": {
                "camera_keys_in_model_order": list(self.image_features),
                "dataset_camera_keys": self.dataset_camera_keys,
                "image_features": image_features,
                "state": {
                    "dataset_names": self.dataset_state_feature_names,
                    "switches": self.state_feature_switches(),
                    "selected_names": self.resolved_state_feature_names,
                    "selected_indices": self.state_feature_indices,
                    "encoding": "continuous_memory_tokens" if self.mem_vit_enabled else "text_tokens",
                },
            },
            "action": {
                "dataset_names": self.dataset_action_feature_names,
                "model_names": self.action_feature_names,
                "representation": self.b2_action_representation,
                "trajectory_source": (
                    "global_pose_transform"
                    if self.b2_global_pose_state_indices is not None
                    else "twist_integration"
                )
                if self.b2_action_representation == "local_trajectory"
                else "body_twist",
                "global_pose_state_names": (
                    ["b2_position_x", "b2_position_y", "b2_yaw"]
                    if self.b2_global_pose_state_indices is not None
                    else None
                ),
                "global_pose_state_indices": self.b2_global_pose_state_indices,
                "local_trajectory_reference": "current_body_frame",
                "local_trajectory_samples": "interval_end_t_plus_1_through_t_plus_chunk_size",
                "predict": {
                    "arm_teleop_inactive": self.action_predict_arm_teleop_inactive,
                    "arm_reset": self.action_predict_arm_reset,
                    "ee_pose": self.action_predict_ee_pose,
                    "gripper": self.action_predict_gripper,
                    "task_complete": self.action_predict_task_complete,
                },
                "boolean_decoding": {
                    "threshold": 0.0,
                    "threshold_domain": "normalized_model_action_before_unnormalization",
                    "postprocessed_threshold": "inverse_normalization_of_zero",
                    "true_side": {
                        "arm_teleop_inactive": "positive",
                        "arm_reset": "positive",
                        "gripper_target": self.action_gripper_target_true_side,
                        "task_complete": "positive",
                    },
                },
                "task_complete_semantics": (
                    "explicit_true_in_the_post_task_tail_until_ros_bag_end"
                    if self.action_predict_task_complete
                    else None
                ),
                "task_complete_deployment_behavior": (
                    "stop_before_executing_later_chunk_elements_at_first_true"
                    if self.action_predict_task_complete
                    else None
                ),
                "local_trajectory_deployment_decode": (
                    "se2_finite_difference_to_body_twist"
                    if self.b2_action_representation == "local_trajectory"
                    else None
                ),
                "relative_actions": self.use_relative_actions,
                "relative_exclude_joints": self.relative_exclude_joints,
            },
            "memory": {
                "enabled": self.mem_vit_enabled,
                "history_length_mode": (
                    "random_uniform_integer" if self.mem_vit_min_num_frames is not None else "fixed"
                ),
                "fixed_num_frames": (
                    self.mem_vit_num_frames if self.mem_vit_min_num_frames is None else None
                ),
                "min_num_frames": min_history_frames,
                "max_num_frames": max_history_frames,
                "frame_interval_seconds": history_interval,
                "deployment_sampling_clock": "physical_time_before_current_observation",
                "min_history_span_seconds": (
                    None if history_interval is None else (min_history_frames - 1) * history_interval
                ),
                "max_history_span_seconds": (
                    None if history_interval is None else (max_history_frames - 1) * history_interval
                ),
                "episode_start_policy": "truncate_to_available_history_with_mask",
                "temporal_attention_every_layers": self.mem_vit_temporal_every,
                "use_original_vit_for_k1": self.mem_vit_use_original_for_k1,
            },
            "normalization": {key: json_value(value) for key, value in self.normalization_mapping.items()},
        }

    def state_feature_switches(self) -> dict[str, bool]:
        return {
            "arm_joint_positions": self.state_use_arm_joint_positions,
            "arm_joint_velocities": self.state_use_arm_joint_velocities,
            "arm_gripper_feedback": self.state_use_arm_gripper_feedback,
            "b2_joint_positions": self.state_use_b2_joint_positions,
            "b2_joint_velocities": self.state_use_b2_joint_velocities,
            "b2_trunk_pose": self.state_use_b2_trunk_pose,
            "b2_linear_velocity": self.state_use_b2_linear_velocity,
            "b2_angular_velocity": self.state_use_b2_angular_velocity,
        }

    def resolve_state_feature_names(self, dataset_names: list[str]) -> list[str]:
        """Resolve boolean state-group switches against named physical dataset dimensions."""

        def group_for(name: str) -> str | None:
            if name.startswith("arm_qd_"):
                return "arm_joint_velocities"
            if name.startswith("arm_q_"):
                return "arm_joint_positions"
            if name == "arm_gripper_feedback":
                return "arm_gripper_feedback"
            if name.startswith("b2_joint_pos_"):
                return "b2_joint_positions"
            if name.startswith("b2_joint_vel_"):
                return "b2_joint_velocities"
            if name in {"b2_trunk_roll", "b2_trunk_pitch", "b2_body_height"}:
                return "b2_trunk_pose"
            if name in {"b2_body_vx", "b2_body_vy", "b2_body_vz"}:
                return "b2_linear_velocity"
            if name in {"b2_body_wx", "b2_body_wy", "b2_body_wz"}:
                return "b2_angular_velocity"
            return None

        grouped_names: dict[str, list[str]] = {group: [] for group in self.state_feature_switches()}
        auxiliary_names = {"b2_position_x", "b2_position_y", "b2_yaw"}
        unknown_names = []
        for name in dataset_names:
            if name in auxiliary_names:
                continue
            group = group_for(name)
            if group is None:
                unknown_names.append(name)
            else:
                grouped_names[group].append(name)
        if unknown_names:
            raise ValueError(f"Unrecognized B2+Z1 state feature names: {unknown_names}")

        switches = self.state_feature_switches()
        missing_groups = [
            group for group, enabled in switches.items() if enabled and not grouped_names[group]
        ]
        if missing_groups:
            raise ValueError(f"Enabled state groups are absent from the dataset: {missing_groups}")
        selected_names = []
        for name in dataset_names:
            if name in auxiliary_names:
                continue
            group = group_for(name)
            assert group is not None
            if switches[group]:
                selected_names.append(name)
        return selected_names

    def _save_pretrained(self, save_directory: Path) -> None:
        super()._save_pretrained(save_directory)
        with (save_directory / PI05_DEPLOYMENT_METADATA_NAME).open("w") as f:
            json.dump(self.deployment_metadata(), f, indent=2)

    @property
    def reward_delta_indices(self) -> None:
        return None
