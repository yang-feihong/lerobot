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

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
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
    decode_b2_action_chunk,
    encode_b2_action_chunk,
    make_b2_trajectory_stats,
)
from .configuration_pi05 import PI05Config


@ProcessorStepRegistry.register(name="pi05_b2_local_trajectory_processor")
@dataclass
class Pi05B2LocalTrajectoryProcessorStep(ProcessorStep):
    """Convert between dataset B2 twist and the model's local trajectory."""

    dt: float = 0.1
    inverse: bool = False
    append_task_complete: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        if not isinstance(action, torch.Tensor):
            raise ValueError(f"B2 local trajectory expects a tensor action, got {type(action)}")

        new_transition = transition.copy()
        if self.inverse:
            transformed = decode_b2_action_chunk(
                action,
                dt=self.dt,
                has_completion=self.append_task_complete,
            )
        else:
            complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}
            is_pad = complementary.get(f"{ACTION}_is_pad")
            transformed = encode_b2_action_chunk(
                action,
                dt=self.dt,
                is_pad=is_pad,
                append_completion=self.append_task_complete,
            )
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
            "append_task_complete": self.append_task_complete,
        }


def reconcile_pi05_b2_trajectory_processors(
    config: PI05Config,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    dataset_stats: dict[str, dict[str, Any]] | None,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Insert/refresh B2 trajectory steps when fine-tuning a pretrained PI0.5."""
    if not config.b2_local_trajectory_enabled:
        return preprocessor, postprocessor
    if config.b2_local_trajectory_dt is None:
        raise ValueError("b2_local_trajectory_dt must be resolved from the dataset fps")

    transformed_stats = make_b2_trajectory_stats(
        dataset_stats,
        dt=config.b2_local_trajectory_dt,
        chunk_size=config.chunk_size,
    )
    if transformed_stats is not None:
        preprocessor = hotswap_stats(preprocessor, transformed_stats)
        postprocessor = hotswap_stats(postprocessor, transformed_stats)

    if not any(isinstance(step, Pi05B2LocalTrajectoryProcessorStep) for step in preprocessor.steps):
        steps = list(preprocessor.steps)
        normalizer_index = next(
            i for i, step in enumerate(steps) if isinstance(step, NormalizerProcessorStep)
        )
        steps.insert(
            normalizer_index,
            Pi05B2LocalTrajectoryProcessorStep(dt=config.b2_local_trajectory_dt),
        )
        preprocessor.steps = steps

    if not any(isinstance(step, Pi05B2LocalTrajectoryProcessorStep) for step in postprocessor.steps):
        steps = list(postprocessor.steps)
        unnormalizer_index = next(
            i for i, step in enumerate(steps) if isinstance(step, UnnormalizerProcessorStep)
        )
        steps.insert(
            unnormalizer_index + 1,
            Pi05B2LocalTrajectoryProcessorStep(dt=config.b2_local_trajectory_dt, inverse=True),
        )
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

    if config.b2_local_trajectory_enabled:
        if config.b2_local_trajectory_dt is None:
            raise ValueError("b2_local_trajectory_dt must be resolved from the dataset fps")
        dataset_stats = make_b2_trajectory_stats(
            dataset_stats,
            dt=config.b2_local_trajectory_dt,
            chunk_size=config.chunk_size,
        )

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
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
            continuous_state_memory=config.mem_vit_enabled,
        ),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
    ]

    if config.b2_local_trajectory_enabled:
        normalizer_index = next(
            i for i, step in enumerate(input_steps) if isinstance(step, NormalizerProcessorStep)
        )
        input_steps.insert(
            normalizer_index,
            Pi05B2LocalTrajectoryProcessorStep(dt=config.b2_local_trajectory_dt),
        )

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        DeviceProcessorStep(device="cpu"),
    ]
    if config.b2_local_trajectory_enabled:
        output_steps.insert(
            1,
            Pi05B2LocalTrajectoryProcessorStep(dt=config.b2_local_trajectory_dt, inverse=True),
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
