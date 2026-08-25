from types import SimpleNamespace

import pytest
import torch

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.pi05.processor_pi05 import (
    OBS_ACTION_HISTORY,
    Pi05SplitActionHistoryProcessorStep,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE


def test_mem_modalities_use_independent_50hz_sampling_clocks():
    image_key = "observation.images.base"
    state_names = ["b2_position_x", "b2_position_y", "b2_yaw"]
    config = SimpleNamespace(
        control_frequency_hz=50,
        mem_vit_enabled=True,
        mem_vit_num_frames=6,
        mem_vit_frame_interval_seconds=0.5,
        mem_vit_frame_stride=1,
        state_action_encoding="continuous",
        state_num_frames=13,
        state_history_frame_interval_seconds=0.04,
        state_history_frame_stride=1,
        action_history_enabled=True,
        io_schema_resolved=False,
        b2_action_representation="pose_delta",
        z1_action_representation="ee_delta",
        b2_global_pose_state_indices=None,
        chunk_size=50,
        action_delta_indices=list(range(50)),
        reward_delta_indices=None,
        observation_delta_indices=None,
        image_features={image_key: object()},
    )
    metadata = SimpleNamespace(
        fps=50,
        features={ACTION: {}, OBS_STATE: {"names": state_names}, image_key: {}},
        camera_keys=[image_key],
    )

    timestamps = resolve_delta_timestamps(config, metadata)

    image_history = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0]
    state_history = [-0.48 + 0.04 * index for index in range(13)]
    assert timestamps[image_key] == pytest.approx(image_history)
    assert timestamps[OBS_STATE][:13] == pytest.approx(state_history)
    assert timestamps[OBS_STATE] == pytest.approx(state_history)
    assert timestamps[ACTION][:12] == pytest.approx(state_history[:-1])
    assert timestamps[ACTION][12:] == pytest.approx([index / 50 for index in range(51)])
    assert config.mem_vit_frame_stride == 25
    assert config.state_history_frame_stride == 2


def test_action_history_split_keeps_target_and_padding_independent():
    step = Pi05SplitActionHistoryProcessorStep(
        history_length=12,
        target_length=50,
        action_indices=(0, 2, 4),
    )
    combined = torch.arange(2 * 62 * 16, dtype=torch.float32).reshape(2, 62, 16)
    combined_pad = torch.zeros(2, 62, dtype=torch.bool)
    combined_pad[0, :5] = True
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: torch.zeros(2, 13, 4)},
        TransitionKey.ACTION: combined,
        TransitionKey.COMPLEMENTARY_DATA: {f"{ACTION}_is_pad": combined_pad},
    }

    result = step(transition)

    assert torch.equal(result[TransitionKey.OBSERVATION][OBS_ACTION_HISTORY], combined[:, :12, (0, 2, 4)])
    assert torch.equal(result[TransitionKey.ACTION], combined[:, 12:])
    complementary = result[TransitionKey.COMPLEMENTARY_DATA]
    assert torch.equal(complementary[f"{OBS_ACTION_HISTORY}_is_pad"], combined_pad[:, :12])
    assert torch.equal(complementary[f"{ACTION}_is_pad"], combined_pad[:, 12:])


def test_continuous_state_and_action_history_do_not_require_mem_vit():
    image_key = "observation.images.base"
    config = SimpleNamespace(
        control_frequency_hz=50,
        mem_vit_enabled=False,
        mem_vit_num_frames=6,
        mem_vit_frame_interval_seconds=0.5,
        mem_vit_frame_stride=1,
        state_action_encoding="continuous",
        state_num_frames=13,
        state_history_frame_interval_seconds=0.04,
        state_history_frame_stride=1,
        action_history_enabled=True,
        io_schema_resolved=False,
        b2_action_representation="velocity",
        z1_action_representation="ee_delta",
        b2_global_pose_state_indices=None,
        chunk_size=50,
        action_delta_indices=list(range(50)),
        reward_delta_indices=None,
        observation_delta_indices=None,
        image_features={image_key: object()},
    )
    metadata = SimpleNamespace(
        fps=50,
        features={ACTION: {}, OBS_STATE: {"names": ["state"]}, image_key: {}},
        camera_keys=[image_key],
    )

    timestamps = resolve_delta_timestamps(config, metadata)

    assert image_key not in timestamps
    assert timestamps[OBS_STATE] == pytest.approx([-0.48 + 0.04 * index for index in range(13)])
    assert timestamps[ACTION][:12] == pytest.approx(timestamps[OBS_STATE][:-1])
    assert timestamps[ACTION][12:] == pytest.approx([index / 50 for index in range(51)])


def test_ee_delta_loads_current_plus_one_extra_future_target():
    config = SimpleNamespace(
        control_frequency_hz=50,
        mem_vit_enabled=False,
        mem_vit_num_frames=1,
        mem_vit_frame_interval_seconds=0.5,
        mem_vit_frame_stride=1,
        state_action_encoding="text",
        state_num_frames=1,
        state_history_frame_interval_seconds=0.04,
        state_history_frame_stride=1,
        action_history_enabled=False,
        io_schema_resolved=False,
        b2_action_representation="velocity",
        z1_action_representation="ee_delta",
        b2_global_pose_state_indices=None,
        chunk_size=50,
        action_delta_indices=list(range(50)),
        reward_delta_indices=None,
        observation_delta_indices=None,
        image_features={},
    )
    metadata = SimpleNamespace(
        fps=50,
        features={ACTION: {}, OBS_STATE: {"names": ["state"]}},
        camera_keys=[],
    )

    timestamps = resolve_delta_timestamps(config, metadata)

    assert timestamps[ACTION] == pytest.approx([index / 50 for index in range(51)])


def test_ee_state_delta_uses_only_the_inference_time_state_anchor():
    config = SimpleNamespace(
        control_frequency_hz=50,
        mem_vit_enabled=False,
        mem_vit_num_frames=1,
        mem_vit_frame_interval_seconds=0.5,
        mem_vit_frame_stride=1,
        state_action_encoding="text",
        state_num_frames=1,
        state_history_frame_interval_seconds=0.04,
        state_history_frame_stride=1,
        action_history_enabled=False,
        io_schema_resolved=False,
        b2_action_representation="velocity",
        z1_action_representation="ee_state_delta",
        ee_supervision_source="control_action",
        b2_global_pose_state_indices=None,
        chunk_size=50,
        action_delta_indices=list(range(50)),
        reward_delta_indices=None,
        observation_delta_indices=None,
        image_features={},
    )
    metadata = SimpleNamespace(
        fps=50,
        features={ACTION: {}, OBS_STATE: {"names": ["state"]}},
        camera_keys=[],
    )

    timestamps = resolve_delta_timestamps(config, metadata)

    assert OBS_STATE not in timestamps
    assert timestamps[ACTION] == pytest.approx([index / 50 for index in range(50)])
