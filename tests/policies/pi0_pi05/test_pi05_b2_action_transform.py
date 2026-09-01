#!/usr/bin/env python

import math

import pytest
import torch

from lerobot.policies.pi05 import processor_pi05
from lerobot.policies.pi05.b2_action_transform import (
    CONTROL_EXTENDED_DATASET_ACTION_NAMES,
    DATASET_ACTION_NAMES,
    EE_DELTA_VALID_KEY,
    absolute_ee_pose_to_delta,
    absolute_ee_pose_to_reference_delta,
    action_label_multiplicity,
    action_schema_kwargs,
    b2_execution_action_names,
    b2_pose_delta_action_names,
    decode_b2_action_chunk,
    ee_delta_transition_validity,
    ee_pose_delta_to_absolute,
    ee_reference_delta_to_absolute,
    encode_b2_action_chunk,
    integrate_body_twist,
    integrate_body_twist_to_pose_delta,
    make_pi05_action_stats,
    se2_increment_to_body_twist,
    select_dataset_action_supervision,
)
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import (
    Pi05ActionRepresentationProcessorStep,
    _action_history_steps,
    make_pi05_pre_post_processors,
)
from lerobot.processor.pipeline import IdentityProcessorStep
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE


def _stored_control_action(action: torch.Tensor) -> torch.Tensor:
    stored = torch.zeros(action.shape[:-1] + (25,), dtype=action.dtype, device=action.device)
    stored[..., :16] = action
    stored[..., 16:25] = action[..., 5:14]
    return stored


def test_body_twist_round_trip_with_rotating_frame_and_unwrapped_yaw():
    twist = torch.randn(2, 50, 3, dtype=torch.float64)
    twist[..., 2] = 1.2
    trajectory = integrate_body_twist(twist, dt=0.1)
    torch.testing.assert_close(trajectory[..., 2], torch.full((2, 50), 0.12, dtype=torch.float64))
    torch.testing.assert_close(se2_increment_to_body_twist(trajectory, dt=0.1), twist, rtol=1e-9, atol=1e-9)


def test_b2_pose_delta_is_chunk_start_relative_and_round_trips_through_se2():
    twist = torch.tensor([[[1.0, 0.0, math.pi / 2], [1.0, 0.0, math.pi / 2]]], dtype=torch.float64)
    pose_delta = integrate_body_twist_to_pose_delta(twist, dt=0.5)
    assert pose_delta[0, 1, 0] > pose_delta[0, 0, 0]
    assert pose_delta[0, 1, 1] > pose_delta[0, 0, 1]
    action = torch.zeros(1, 2, 16, dtype=torch.float64)
    action[..., :3] = pose_delta
    decoded = decode_b2_action_chunk(action, dt=0.5, representation="pose_delta")
    torch.testing.assert_close(decoded[..., :3], twist, rtol=1e-9, atol=1e-9)


def test_ee_delta_first_point_is_current_to_next_and_round_trips():
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    poses = torch.zeros(1, 3, 9)
    poses[..., :6] = identity
    poses[0, :, 6:] = torch.tensor([[1.0, 2.0, 3.0], [1.1, 2.0, 3.0], [1.1, 2.2, 3.0]])
    deltas = absolute_ee_pose_to_delta(poses[:, 1:], poses[:, 0])
    torch.testing.assert_close(deltas[0, 0, 6:], torch.tensor([0.1, 0.0, 0.0]))
    torch.testing.assert_close(deltas[0, 1, 6:], torch.tensor([0.0, 0.2, 0.0]))
    torch.testing.assert_close(ee_pose_delta_to_absolute(deltas, poses[:, 0]), poses[:, 1:])


def test_rotvec_ee_delta_is_zero_centered_signed_and_round_trips():
    angles = torch.tensor([0.0, 0.2, -0.1], dtype=torch.float64)
    poses = torch.zeros(1, 3, 9, dtype=torch.float64)
    poses[0, :, 5] = torch.sin(angles)
    poses[0, :, 4] = torch.cos(angles)
    poses[0, :, 0] = 1.0
    deltas = absolute_ee_pose_to_delta(poses[:, 1:], poses[:, 0], rotation_representation="rotvec")
    assert deltas.shape == (1, 2, 6)
    torch.testing.assert_close(deltas[0, :, 0], torch.tensor([0.2, -0.3], dtype=torch.float64))
    torch.testing.assert_close(deltas[0, :, 1:3], torch.zeros(2, 2, dtype=torch.float64))
    reconstructed = ee_pose_delta_to_absolute(deltas, poses[:, 0], rotation_representation="rotvec")
    torch.testing.assert_close(reconstructed, poses[:, 1:], rtol=1e-7, atol=1e-7)


def test_ee_state_delta_uses_one_fixed_inference_time_reference():
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    reference = torch.cat((identity, torch.tensor([1.0, 2.0, 3.0]))).unsqueeze(0)
    targets = reference.unsqueeze(1).repeat(1, 2, 1)
    targets[0, 0, 6] += 0.1
    targets[0, 1, 6] += 0.3
    deltas = absolute_ee_pose_to_reference_delta(targets, reference, rotation_representation="rotvec")
    torch.testing.assert_close(deltas[0, :, 3], torch.tensor([0.1, 0.3]))
    reconstructed = ee_reference_delta_to_absolute(deltas, reference, rotation_representation="rotvec")
    torch.testing.assert_close(reconstructed, targets)


def test_encode_ee_delta_uses_51_absolute_targets_for_50_deltas():
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    action = torch.zeros(1, 51, 16)
    action[..., 5:11] = identity
    action[0, :, 11] = torch.arange(51, dtype=torch.float32) * 0.01
    encoded = encode_b2_action_chunk(action, dt=0.02, representation="velocity", z1_representation="ee_delta")
    assert encoded.shape == (1, 50, 16)
    torch.testing.assert_close(encoded[0, :, 11], torch.full((50,), 0.01))


def test_encode_ee_state_delta_uses_50_targets_and_one_measured_anchor():
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    action = torch.zeros(1, 50, 16)
    action[..., 5:11] = identity
    action[0, :, 11] = 1.0 + torch.arange(50, dtype=torch.float32) * 0.01
    anchor = torch.cat((identity, torch.tensor([1.0, 0.0, 0.0]))).unsqueeze(0)
    encoded = encode_b2_action_chunk(
        action,
        dt=0.02,
        representation="velocity",
        z1_representation="ee_state_delta",
        ee_delta_rotation_representation="rotvec",
        ee_state_anchor=anchor,
    )
    assert encoded.shape == (1, 50, 13)
    torch.testing.assert_close(encoded[0, :, 8], torch.arange(50, dtype=torch.float32) * 0.01)


def test_processor_reads_ee_state_anchor_without_using_it_as_supervision():
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    action = torch.zeros(1, 2, 25)
    action[..., 16:22] = identity
    action[0, :, 22] = torch.tensor([1.2, 1.4])
    state = torch.zeros(1, 58)
    state[..., 49:55] = identity
    state[..., 55] = 1.0
    step = Pi05ActionRepresentationProcessorStep(
        dt=0.02,
        representation="velocity",
        z1_representation="ee_state_delta",
        ee_delta_rotation_representation="rotvec",
        ee_supervision_source="control_action",
        ee_state_anchor_indices=tuple(range(49, 58)),
    )
    transformed = step(
        {
            TransitionKey.ACTION: action,
            TransitionKey.OBSERVATION: {OBS_STATE: state},
            TransitionKey.COMPLEMENTARY_DATA: {f"{ACTION}_is_pad": torch.zeros(1, 2, dtype=torch.bool)},
        }
    )[TransitionKey.ACTION]
    torch.testing.assert_close(transformed[0, :, 8], torch.tensor([0.2, 0.4]))


@pytest.mark.parametrize("b2_representation", ["velocity", "pose_delta"])
@pytest.mark.parametrize("z1_representation", ["ee_delta", "ee_state_delta"])
def test_all_supported_b2_z1_representation_combinations(b2_representation: str, z1_representation: str):
    horizon = 4
    action_count = horizon + int(z1_representation == "ee_delta")
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    action = torch.zeros(1, action_count, 16)
    action[..., :3] = torch.tensor([0.4, -0.1, 0.2])
    action[..., 5:11] = identity
    action[0, :, 11] = 0.5 + torch.arange(action_count) * 0.01
    anchor = torch.cat((identity, torch.tensor([0.4, 0.0, 0.0]))).unsqueeze(0)

    encoded = encode_b2_action_chunk(
        action,
        dt=0.02,
        representation=b2_representation,
        z1_representation=z1_representation,
        ee_delta_rotation_representation="rotvec",
        ee_state_anchor=anchor if z1_representation == "ee_state_delta" else None,
    )
    decoded = decode_b2_action_chunk(encoded, dt=0.02, representation=b2_representation)

    assert encoded.shape[-2] == horizon
    torch.testing.assert_close(decoded[..., :3], action[..., :horizon, :3], atol=1e-5, rtol=1e-5)
    expected_x = (
        torch.full((horizon,), 0.01)
        if z1_representation == "ee_delta"
        else 0.1 + torch.arange(horizon) * 0.01
    )
    torch.testing.assert_close(encoded[0, :, 8], expected_x)


@pytest.mark.parametrize("b2_representation", ["velocity", "pose_delta"])
@pytest.mark.parametrize("z1_representation", ["ee_delta", "ee_state_delta"])
def test_all_supported_combinations_construct_formal_processor_pipelines(
    b2_representation: str, z1_representation: str, monkeypatch
):
    monkeypatch.setattr(
        processor_pi05,
        "TokenizerProcessorStep",
        lambda **_kwargs: IdentityProcessorStep(),
    )
    config = PI05Config(
        device="cpu",
        b2_action_representation=b2_representation,
        z1_action_representation=z1_representation,
        ee_delta_rotation_representation="rotvec",
    )
    config.io_schema_resolved = True
    config.action_dt_seconds = 0.02
    config.state_feature_indices = list(range(19 if z1_representation == "ee_state_delta" else 10))
    config.ee_state_anchor_indices = list(range(49, 58)) if z1_representation == "ee_state_delta" else None
    name_fn = b2_pose_delta_action_names if b2_representation == "pose_delta" else b2_execution_action_names
    config.action_feature_names = name_fn(
        list(CONTROL_EXTENDED_DATASET_ACTION_NAMES), **action_schema_kwargs(config)
    )

    preprocessor, postprocessor = make_pi05_pre_post_processors(config, dataset_stats=None)

    assert any(isinstance(step, Pi05ActionRepresentationProcessorStep) for step in preprocessor.steps)
    assert any(isinstance(step, Pi05ActionRepresentationProcessorStep) for step in postprocessor.steps)


def test_velocity_mode_uses_dataset_command_directly():
    action = torch.zeros(1, 2, 16)
    action[..., :3] = torch.tensor([0.4, -0.2, 0.3])
    encoded = encode_b2_action_chunk(
        action,
        dt=0.02,
        representation="velocity",
        z1_representation="ee_delta",
    )
    torch.testing.assert_close(encoded[..., :3], action[..., :1, :3])


def test_ee_delta_padding_requires_both_adjacent_targets():
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    action = torch.zeros(1, 3, 16)
    action[..., 5:11] = identity
    transition = {
        TransitionKey.ACTION: _stored_control_action(action),
        TransitionKey.COMPLEMENTARY_DATA: {f"{ACTION}_is_pad": torch.tensor([[False, False, True]])},
    }
    step = Pi05ActionRepresentationProcessorStep(
        dt=0.02,
        representation="velocity",
        z1_representation="ee_delta",
    )
    transformed = step(transition)
    assert transformed[TransitionKey.ACTION].shape[-2] == 2
    assert transformed[TransitionKey.COMPLEMENTARY_DATA][f"{ACTION}_is_pad"].tolist() == [[False, True]]


def test_extended_action_supervision_selects_joint_control_ee_without_leaking_extra_dimensions():
    action = torch.zeros(1, 2, 25)
    action[..., 5:14] = -1.0
    action[..., 16:25] = 3.0

    selected = select_dataset_action_supervision(action, source="control_action")

    assert selected.shape == (1, 2, 16)
    torch.testing.assert_close(selected[..., 5:14], torch.full((1, 2, 9), 3.0))


def test_execution_action_names_accept_control_storage_schema():
    expected = b2_execution_action_names(list(DATASET_ACTION_NAMES))

    actual = b2_execution_action_names(list(CONTROL_EXTENDED_DATASET_ACTION_NAMES))

    assert actual == expected


def test_execution_action_names_reject_unknown_extended_storage_schema():
    names = list(CONTROL_EXTENDED_DATASET_ACTION_NAMES)
    names[-1] = "unknown_extra_channel"

    with pytest.raises(ValueError, match="unknown 25D extended dataset schema"):
        b2_execution_action_names(names)


def test_ee_delta_validity_requires_active_non_reset_at_both_endpoints():
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    action = torch.zeros(1, 4, 16)
    action[..., 5:11] = identity
    action[0, 2, 3] = 1.0
    transition = {
        TransitionKey.ACTION: _stored_control_action(action),
        TransitionKey.COMPLEMENTARY_DATA: {f"{ACTION}_is_pad": torch.zeros(1, 4, dtype=torch.bool)},
    }
    step = Pi05ActionRepresentationProcessorStep(
        dt=0.02,
        representation="velocity",
        z1_representation="ee_delta",
        ee_delta_rotation_representation="rotvec",
        ee_delta_supervision_mode="active_only",
    )
    transformed = step(transition)
    assert transformed[TransitionKey.ACTION].shape == (1, 3, 13)
    assert transformed[TransitionKey.COMPLEMENTARY_DATA][EE_DELTA_VALID_KEY].tolist() == [
        [True, False, False]
    ]


def test_new_action_schema_retains_explicit_gates_and_completion():
    action = torch.arange(32, dtype=torch.float32).view(1, 2, 16)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    action[..., 5:11] = identity
    encoded = encode_b2_action_chunk(action, dt=0.1, representation="velocity")
    assert encoded.shape == (1, 1, 16)
    torch.testing.assert_close(encoded[..., :5], action[..., :1, :5])
    torch.testing.assert_close(encoded[..., 14:], action[..., :1, 14:])

    without_optional_groups = encode_b2_action_chunk(
        action,
        dt=0.1,
        representation="velocity",
        predict_arm_teleop_inactive=False,
        predict_arm_reset=False,
        predict_ee_pose=False,
        predict_gripper=False,
        include_task_complete=False,
    )
    torch.testing.assert_close(without_optional_groups, action[..., :1, :3])


def test_b2_only_action_names_do_not_validate_disabled_ee_representation():
    names = b2_pose_delta_action_names(
        list(DATASET_ACTION_NAMES),
        z1_representation="ee_state_delta",
        ee_delta_rotation_representation="rotvec",
        predict_arm_teleop_inactive=False,
        predict_arm_reset=False,
        predict_ee_pose=False,
        predict_gripper=False,
        include_task_complete=False,
    )
    assert names == ["b2_delta_x", "b2_delta_y", "b2_delta_yaw"]


def test_processor_uses_new_schema_without_mutating_raw_transition():
    state = torch.arange(49, dtype=torch.float32).view(1, 49)
    action = torch.zeros(1, 2, 16)
    action[..., 0] = 1.0
    action[..., 3] = 1.0
    action[..., 15] = torch.tensor([[0.0, 1.0]])
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: state},
        TransitionKey.ACTION: _stored_control_action(action),
        TransitionKey.COMPLEMENTARY_DATA: {f"{ACTION}_is_pad": torch.zeros(1, 2, dtype=torch.bool)},
    }
    step = Pi05ActionRepresentationProcessorStep(
        dt=0.1,
        representation="velocity",
        z1_representation="ee_delta",
        state_indices=(0, 12, 37, 38, 39),
    )
    transformed = step(transition)
    assert transformed[TransitionKey.ACTION].shape == (1, 1, 16)
    torch.testing.assert_close(transformed[TransitionKey.ACTION][0, :, 0], torch.tensor([1.0]))
    torch.testing.assert_close(transformed[TransitionKey.ACTION][..., 3], action[..., :1, 3])
    torch.testing.assert_close(transformed[TransitionKey.ACTION][..., 15], action[..., :1, 15])
    assert transition[TransitionKey.OBSERVATION][OBS_STATE].shape == (1, 49)
    assert step.get_config()["include_task_complete"] is True


def test_action_multiplicity_can_cap_chunk_start_frames():
    assert action_label_multiplicity(5, [0, 1, 2], num_start_frames=4).tolist() == [1, 2, 3, 3, 2]


def test_all_ee_delta_supervision_ignores_legacy_modes_but_not_padding():
    action = torch.zeros(1, 4, 16)
    action[0, 1, 3] = 1.0
    action[0, 2, 4] = 1.0
    is_pad = torch.tensor([[False, False, False, True]])

    active_only = ee_delta_transition_validity(action, is_pad, supervision_mode="active_only")
    all_transitions = ee_delta_transition_validity(action, is_pad, supervision_mode="all")

    assert active_only.tolist() == [[False, False, False]]
    assert all_transitions.tolist() == [[True, True, False]]


def test_ee_delta_stats_reject_absolute_workspace_approximation():
    raw = {ACTION: {name: torch.zeros(16) for name in ("q01", "q99", "min", "max", "mean", "std")}}
    with pytest.raises(ValueError, match="measured after the temporal transform"):
        make_pi05_action_stats(
            raw,
            dt=0.02,
            chunk_size=50,
            z1_representation="ee_delta",
            ee_delta_rotation_representation="rotvec",
        )


def test_rotvec_ee_delta_builds_absolute_target_reference_step():
    config = PI05Config(
        z1_action_representation="ee_delta",
        ee_delta_rotation_representation="rotvec",
        action_history_enabled=False,
    )
    split, normalizer = _action_history_steps(config, dataset_stats=None)
    assert split.target_length == config.chunk_size + 1
    assert len(split.action_indices) == len(DATASET_ACTION_NAMES)
    assert normalizer is None
