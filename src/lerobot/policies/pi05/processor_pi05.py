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

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.normalize_processor import hotswap_stats
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    ACTION,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .b2_action_transform import (
    EE_DELTA_VALID_KEY,
    action_dataset_indices,
    action_schema_kwargs,
    decode_b2_action_chunk,
    ee_delta_sample_validity,
    encode_b2_action_chunk,
    make_pi05_action_stats,
    select_dataset_action_supervision,
)
from .configuration_pi05 import PI05Config

OBS_ACTION_HISTORY = "observation.action_history"


def _selected_action_history_stats(
    dataset_stats: dict[str, dict[str, Any]] | None, indices: tuple[int, ...]
) -> dict[str, dict[str, Any]] | None:
    if dataset_stats is None or ACTION not in dataset_stats:
        return None
    selected: dict[str, Any] = {}
    for name, value in dataset_stats[ACTION].items():
        tensor = torch.as_tensor(value)
        if tensor.ndim > 0 and tensor.shape[-1] == 16:
            converted = tensor[..., list(indices)]
            selected[name] = converted if isinstance(value, torch.Tensor) else converted.cpu().numpy()
        else:
            selected[name] = deepcopy(value)
    return {OBS_ACTION_HISTORY: selected}


def _action_history_steps(
    config: PI05Config, dataset_stats: dict[str, dict[str, Any]] | None
) -> tuple[Pi05SplitActionHistoryProcessorStep, NormalizerProcessorStep | None] | tuple[()]:
    needs_reference = config.z1_action_representation == "ee_delta"
    if not config.action_history_enabled and not needs_reference:
        return ()
    indices = action_dataset_indices(
        **{
            key: value
            for key, value in action_schema_kwargs(config).items()
            if key
            not in {
                "representation",
                "z1_representation",
                "ee_delta_rotation_representation",
            }
        }
    )
    history_length = config.state_num_frames - 1 if config.action_history_enabled else 0
    split = Pi05SplitActionHistoryProcessorStep(
        history_length=history_length,
        target_length=config.chunk_size + int(needs_reference),
        action_indices=indices,
    )
    if not config.action_history_enabled:
        return split, None
    action_mode = config.normalization_mapping.get(
        FeatureType.ACTION, config.normalization_mapping.get("ACTION")
    )
    normalizer = NormalizerProcessorStep(
        features={OBS_ACTION_HISTORY: PolicyFeature(type=FeatureType.STATE, shape=(len(indices),))},
        norm_map={FeatureType.STATE: action_mode},
        stats=_selected_action_history_stats(dataset_stats, indices),
        normalize_observation_keys={OBS_ACTION_HISTORY},
    )
    return split, normalizer


@ProcessorStepRegistry.register(name="pi05_split_action_history_processor")
@dataclass
class Pi05SplitActionHistoryProcessorStep(ProcessorStep):
    history_length: int
    target_length: int
    action_indices: tuple[int, ...]

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        if not isinstance(action, torch.Tensor):
            raise ValueError(f"Action history expects a tensor action, got {type(action)}")
        expected = self.history_length + self.target_length
        if action.ndim < 2 or action.shape[-2] != expected:
            raise ValueError(
                f"Expected {expected} action samples ({self.history_length} history + "
                f"{self.target_length} targets), got {tuple(action.shape)}"
            )

        new_transition = transition.copy()
        if self.history_length:
            observations = dict(new_transition.get(TransitionKey.OBSERVATION, {}) or {})
            observations[OBS_ACTION_HISTORY] = action[..., : self.history_length, list(self.action_indices)]
            new_transition[TransitionKey.OBSERVATION] = observations
        new_transition[TransitionKey.ACTION] = action[..., self.history_length :, :]

        complementary = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {})
        pad_key = f"{ACTION}_is_pad"
        if pad_key in complementary:
            combined_pad = complementary[pad_key]
            if self.history_length:
                complementary[f"{OBS_ACTION_HISTORY}_is_pad"] = combined_pad[..., : self.history_length]
            complementary[pad_key] = combined_pad[..., self.history_length :]
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "history_length": self.history_length,
            "target_length": self.target_length,
            "action_indices": list(self.action_indices),
        }


@ProcessorStepRegistry.register(name="pi05_action_representation_processor")
@dataclass
class Pi05ActionRepresentationProcessorStep(ProcessorStep):
    """Convert stored controls to/from the configured model representation."""

    # Deliberately has no implicit fallback: this comes from the configured
    # model control frequency and is persisted for inference.
    dt: float
    inverse: bool = False
    include_task_complete: bool = True
    representation: str = "velocity"
    z1_representation: str = "ee_delta"
    ee_delta_rotation_representation: str = "rot6d"
    ee_delta_supervision_mode: str = "all"
    ee_supervision_source: str = "control_action"
    ee_state_anchor_indices: tuple[int, ...] = ()
    predict_arm_teleop_inactive: bool = True
    predict_arm_reset: bool = True
    predict_ee_pose: bool = True
    predict_gripper: bool = True
    state_indices: tuple[int, ...] = ()
    state_history_length: int = 1
    keep_state_history: bool = False

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        action = transition.get(TransitionKey.ACTION)
        if action is not None and not isinstance(action, torch.Tensor):
            raise ValueError(f"B2 action schema expects a tensor action, got {type(action)}")
        if not self.inverse and action is not None:
            action = select_dataset_action_supervision(
                action,
                source=self.ee_supervision_source,
            )
        ee_state_anchor = None
        if not self.inverse and action is not None and self.z1_representation == "ee_state_delta":
            observations = dict(new_transition.get(TransitionKey.OBSERVATION, {}))
            state = observations.get(OBS_STATE)
            if not isinstance(state, torch.Tensor) or not self.ee_state_anchor_indices:
                raise ValueError("ee_state_delta requires height-invariant EE state channels")
            if state.ndim == action.ndim:
                ee_state_anchor = state[
                    ..., self.state_history_length - 1, list(self.ee_state_anchor_indices)
                ]
            elif state.ndim == action.ndim - 1:
                ee_state_anchor = state[..., list(self.ee_state_anchor_indices)]
            else:
                raise ValueError(
                    f"Cannot extract EE anchor from state={tuple(state.shape)}, action={tuple(action.shape)}"
                )
        if not self.inverse and self.state_indices:
            observations = dict(new_transition.get(TransitionKey.OBSERVATION, {}))
            state = observations.get(OBS_STATE)
            if state is not None:
                observations[OBS_STATE] = state[..., list(self.state_indices)]
                new_transition[TransitionKey.OBSERVATION] = observations

        if action is None:
            return new_transition
        if self.inverse:
            transformed = decode_b2_action_chunk(
                action,
                dt=self.dt,
                representation=self.representation,
            )
        else:
            complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}
            is_pad = complementary.get(f"{ACTION}_is_pad")
            transformed = encode_b2_action_chunk(
                action,
                dt=self.dt,
                is_pad=is_pad,
                ee_state_anchor=ee_state_anchor,
                representation=self.representation,
                z1_representation=self.z1_representation,
                ee_delta_rotation_representation=self.ee_delta_rotation_representation,
                predict_arm_teleop_inactive=self.predict_arm_teleop_inactive,
                predict_arm_reset=self.predict_arm_reset,
                predict_ee_pose=self.predict_ee_pose,
                predict_gripper=self.predict_gripper,
                include_task_complete=self.include_task_complete,
            )
            if self.z1_representation in {"ee_delta", "ee_state_delta"}:
                complementary = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {})
                complementary[EE_DELTA_VALID_KEY] = ee_delta_sample_validity(
                    action,
                    is_pad,
                    representation=self.z1_representation,
                    supervision_mode=self.ee_delta_supervision_mode,
                )
                if is_pad is not None and self.z1_representation == "ee_delta":
                    complementary[f"{ACTION}_is_pad"] = is_pad[..., :-1] | is_pad[..., 1:]
                new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        new_transition[TransitionKey.ACTION] = transformed
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "dt": self.dt,
            "inverse": self.inverse,
            "include_task_complete": self.include_task_complete,
            "representation": self.representation,
            "z1_representation": self.z1_representation,
            "ee_delta_rotation_representation": self.ee_delta_rotation_representation,
            "ee_delta_supervision_mode": self.ee_delta_supervision_mode,
            "ee_supervision_source": self.ee_supervision_source,
            "ee_state_anchor_indices": list(self.ee_state_anchor_indices),
            "predict_arm_teleop_inactive": self.predict_arm_teleop_inactive,
            "predict_arm_reset": self.predict_arm_reset,
            "predict_ee_pose": self.predict_ee_pose,
            "predict_gripper": self.predict_gripper,
            "state_indices": list(self.state_indices),
            "state_history_length": self.state_history_length,
            "keep_state_history": self.keep_state_history,
        }


def reconcile_pi05_action_representation_processors(
    config: PI05Config,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    dataset_stats: dict[str, dict[str, Any]] | None,
    transformed_action_stats: dict[str, Any] | None = None,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Insert or refresh PI0.5 action-representation steps during fine-tuning."""
    if not config.io_schema_resolved:
        return preprocessor, postprocessor
    if config.action_dt_seconds is None:
        raise ValueError("action_dt_seconds must be resolved from the model control frequency")

    raw_dataset_stats = dataset_stats
    transformed_stats = make_pi05_action_stats(
        dataset_stats,
        transformed_action_stats=transformed_action_stats,
        dt=config.action_dt_seconds,
        chunk_size=config.chunk_size,
        state_indices=tuple(config.state_feature_indices or ()),
        **action_schema_kwargs(config),
    )
    if transformed_stats is not None:
        preprocessor = hotswap_stats(preprocessor, transformed_stats)
        postprocessor = hotswap_stats(postprocessor, transformed_stats)

    desired_pre_step = Pi05ActionRepresentationProcessorStep(
        dt=config.action_dt_seconds,
        state_indices=tuple(config.state_feature_indices or ()),
        state_history_length=(config.state_num_frames if config.state_action_encoding == "continuous" else 1),
        keep_state_history=config.state_action_encoding == "continuous",
        representation=config.b2_action_representation,
        z1_representation=config.z1_action_representation,
        ee_delta_rotation_representation=config.ee_delta_rotation_representation,
        ee_delta_supervision_mode=config.ee_delta_supervision_mode,
        ee_supervision_source=config.ee_supervision_source,
        ee_state_anchor_indices=tuple(config.ee_state_anchor_indices or ()),
        predict_arm_teleop_inactive=config.action_predict_arm_teleop_inactive,
        predict_arm_reset=config.action_predict_arm_reset,
        predict_ee_pose=config.action_predict_ee_pose,
        predict_gripper=config.action_predict_gripper,
        include_task_complete=config.action_predict_task_complete,
    )
    steps = [
        desired_pre_step if isinstance(step, Pi05ActionRepresentationProcessorStep) else step
        for step in preprocessor.steps
    ]
    if not any(isinstance(step, Pi05ActionRepresentationProcessorStep) for step in steps):
        normalizer_index = next(
            i for i, step in enumerate(steps) if isinstance(step, NormalizerProcessorStep)
        )
        steps.insert(normalizer_index, desired_pre_step)

    history_steps = _action_history_steps(config, raw_dataset_stats)
    steps = [
        step
        for step in steps
        if not isinstance(step, Pi05SplitActionHistoryProcessorStep)
        and not (
            isinstance(step, NormalizerProcessorStep)
            and step.normalize_observation_keys == {OBS_ACTION_HISTORY}
        )
    ]
    if history_steps:
        split_step, history_normalizer = history_steps
        relative_index = next(
            i for i, step in enumerate(steps) if isinstance(step, RelativeActionsProcessorStep)
        )
        steps.insert(relative_index, split_step)
        if history_normalizer is not None:
            main_normalizer_index = next(
                i for i, step in enumerate(steps) if isinstance(step, NormalizerProcessorStep)
            )
            steps.insert(main_normalizer_index + 1, history_normalizer)
    preprocessor.steps = steps

    desired_post_step = Pi05ActionRepresentationProcessorStep(
        dt=config.action_dt_seconds,
        inverse=True,
        representation=config.b2_action_representation,
        z1_representation=config.z1_action_representation,
        ee_delta_rotation_representation=config.ee_delta_rotation_representation,
        predict_arm_teleop_inactive=config.action_predict_arm_teleop_inactive,
        include_task_complete=config.action_predict_task_complete,
    )
    steps = [
        desired_post_step if isinstance(step, Pi05ActionRepresentationProcessorStep) else step
        for step in postprocessor.steps
    ]
    if not any(isinstance(step, Pi05ActionRepresentationProcessorStep) for step in steps):
        unnormalizer_index = next(
            i for i, step in enumerate(steps) if isinstance(step, UnnormalizerProcessorStep)
        )
        steps.insert(unnormalizer_index + 1, desired_post_step)
    postprocessor.steps = steps
    return preprocessor, postprocessor


@ProcessorStepRegistry.register(name="pi05_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to prepare the state and tokenize the language input.
    """

    max_state_dim: int = 32
    task_key: str = "task"
    continuous_state_memory: bool = False

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            if self.continuous_state_memory:
                full_prompt = f"Task: {cleaned_text};\nAction: "
            else:
                # TODO: check if this necessary
                state_i = deepcopy(state[i])
                if state_i.ndim > 1:
                    # Non-MEM training expects a single state in the text prompt.
                    # If a caller manually supplies a state window, keep the current state.
                    state_i = state_i[-1]

                # State should already be normalized to [-1, 1] by the NormalizerProcessorStep that runs before this step
                # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
                state_np = state_i.cpu().numpy()
                discretized_state = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
                state_str = " ".join(map(str, discretized_state))
                full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            full_prompts.append(full_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        # Normalize state to [-1, 1] range if needed (assuming it's already normalized by normalizer processor step!!)
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.
        """
        return features


def make_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    transformed_action_stats: dict[str, Any] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the PI0 policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Appending a newline character to the task description for tokenizer compatibility.
    5. Tokenizing the text prompt using the PaliGemma tokenizer.
    6. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the PI0 policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    raw_dataset_stats = dataset_stats
    if config.io_schema_resolved:
        if config.action_dt_seconds is None:
            raise ValueError("action_dt_seconds must be resolved from the model control frequency")
        dataset_stats = make_pi05_action_stats(
            dataset_stats,
            transformed_action_stats=transformed_action_stats,
            dt=config.action_dt_seconds,
            chunk_size=config.chunk_size,
            state_indices=tuple(config.state_feature_indices or ()),
            **action_schema_kwargs(config),
        )

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=config.relative_exclude_joints,
        action_names=config.action_feature_names,
    )

    # OpenPI order: raw → relative → normalize → model → unnormalize → absolute
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
        relative_step,
        # NOTE: NormalizerProcessorStep MUST come before Pi05PrepareStateTokenizerProcessorStep
        # because the tokenizer step expects normalized state in [-1, 1] range for discretization
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        Pi05PrepareStateTokenizerProcessorStep(
            max_state_dim=config.max_state_dim,
            continuous_state_memory=config.state_action_encoding == "continuous",
        ),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
    ]

    if config.io_schema_resolved:
        normalizer_index = next(
            i for i, step in enumerate(input_steps) if isinstance(step, NormalizerProcessorStep)
        )
        input_steps.insert(
            normalizer_index,
            Pi05ActionRepresentationProcessorStep(
                dt=config.action_dt_seconds,
                state_indices=tuple(config.state_feature_indices or ()),
                state_history_length=(
                    config.state_num_frames if config.state_action_encoding == "continuous" else 1
                ),
                keep_state_history=config.state_action_encoding == "continuous",
                representation=config.b2_action_representation,
                z1_representation=config.z1_action_representation,
                ee_delta_rotation_representation=config.ee_delta_rotation_representation,
                ee_delta_supervision_mode=config.ee_delta_supervision_mode,
                ee_supervision_source=config.ee_supervision_source,
                ee_state_anchor_indices=tuple(config.ee_state_anchor_indices or ()),
                predict_arm_teleop_inactive=config.action_predict_arm_teleop_inactive,
                predict_arm_reset=config.action_predict_arm_reset,
                predict_ee_pose=config.action_predict_ee_pose,
                predict_gripper=config.action_predict_gripper,
                include_task_complete=config.action_predict_task_complete,
            ),
        )

    history_steps = _action_history_steps(config, raw_dataset_stats)
    if history_steps:
        split_step, history_normalizer = history_steps
        relative_index = next(
            i for i, step in enumerate(input_steps) if isinstance(step, RelativeActionsProcessorStep)
        )
        input_steps.insert(relative_index, split_step)
        main_normalizer_index = next(
            i
            for i, step in enumerate(input_steps)
            if isinstance(step, NormalizerProcessorStep)
            and step.normalize_observation_keys != {OBS_ACTION_HISTORY}
        )
        if history_normalizer is not None:
            input_steps.insert(main_normalizer_index + 1, history_normalizer)

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        DeviceProcessorStep(device="cpu"),
    ]
    if config.io_schema_resolved:
        output_steps.insert(
            1,
            Pi05ActionRepresentationProcessorStep(
                dt=config.action_dt_seconds,
                inverse=True,
                representation=config.b2_action_representation,
                z1_representation=config.z1_action_representation,
                ee_delta_rotation_representation=config.ee_delta_rotation_representation,
                predict_arm_teleop_inactive=config.action_predict_arm_teleop_inactive,
                include_task_complete=config.action_predict_task_complete,
            ),
        )

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
