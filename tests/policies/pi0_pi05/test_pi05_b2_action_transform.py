#!/usr/bin/env python

import math

import pytest
import torch

from lerobot.policies.pi05.b2_action_transform import (
    DATASET_ACTION_NAMES,
    EE_DELTA_VALID_KEY,
    absolute_ee_pose_to_delta,
    action_label_multiplicity,
    decode_b2_action_chunk,
    ee_pose_delta_to_absolute,
    encode_b2_action_chunk,
    global_pose_to_local_trajectory,
    integrate_body_twist,
    make_b2_trajectory_stats,
)
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import (
    Pi05B2LocalTrajectoryProcessorStep,
    _action_history_steps,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE


def test_body_twist_round_trip_with_rotating_frame_and_unwrapped_yaw():
    twist = torch.randn(2, 50, 3, dtype=torch.float64)
    twist[..., 2] = 1.2
    trajectory = integrate_body_twist(twist, dt=0.1)
    torch.testing.assert_close(trajectory[..., 2], torch.full((2, 50), 0.12, dtype=torch.float64))
    action = torch.zeros(2, 50, 16, dtype=torch.float64)
    action[..., :3] = trajectory
    decoded = decode_b2_action_chunk(action, dt=0.1)
    torch.testing.assert_close(decoded[..., :3], twist, rtol=1e-9, atol=1e-9)


def test_global_pose_trajectory_uses_current_body_frame_and_unwraps_yaw():
    global_pose = torch.tensor(
        [[[10.0, 20.0, 3.13], [10.0, 21.0, -3.13], [9.0, 21.0, -3.03]]], dtype=torch.float64
    )
    trajectory = global_pose_to_local_trajectory(global_pose)
    expected_first_yaw = 2 * math.pi - 6.26
    assert trajectory[0, 0, 2].item() == pytest.approx(expected_first_yaw)
    assert trajectory[0, 1, 2].item() == pytest.approx(0.1)


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


def test_encode_ee_delta_uses_51_absolute_targets_for_50_deltas():
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    action = torch.zeros(1, 51, 16)
    action[..., 5:11] = identity
    action[0, :, 11] = torch.arange(51, dtype=torch.float32) * 0.01
    encoded = encode_b2_action_chunk(action, dt=0.02, representation="velocity", z1_representation="ee_delta")
    assert encoded.shape == (1, 50, 16)
    torch.testing.assert_close(encoded[0, :, 11], torch.full((50,), 0.01))


def test_velocity_mode_uses_dataset_command_even_when_odom_is_available():
    action = torch.zeros(1, 2, 16)
    action[..., :3] = torch.tensor([0.4, -0.2, 0.3])
    global_pose = torch.tensor([[[0.0, 0.0, 0.0], [9.0, 8.0, 0.7], [7.0, 6.0, 1.2]]])
    encoded = encode_b2_action_chunk(
        action,
        dt=0.02,
        global_pose=global_pose,
        representation="velocity",
    )
    torch.testing.assert_close(encoded[..., :3], action[..., :3])


def test_ee_delta_padding_requires_both_adjacent_targets():
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    action = torch.zeros(1, 3, 16)
    action[..., 5:11] = identity
    transition = {
        TransitionKey.ACTION: action,
        TransitionKey.COMPLEMENTARY_DATA: {f"{ACTION}_is_pad": torch.tensor([[False, False, True]])},
    }
    step = Pi05B2LocalTrajectoryProcessorStep(
        dt=0.02,
        representation="velocity",
        z1_representation="ee_delta",
    )
    transformed = step(transition)
    assert transformed[TransitionKey.ACTION].shape[-2] == 2
    assert transformed[TransitionKey.COMPLEMENTARY_DATA][f"{ACTION}_is_pad"].tolist() == [[False, True]]


def test_ee_delta_validity_requires_active_non_reset_at_both_endpoints():
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    action = torch.zeros(1, 4, 16)
    action[..., 5:11] = identity
    action[0, 2, 3] = 1.0
    transition = {
        TransitionKey.ACTION: action,
        TransitionKey.COMPLEMENTARY_DATA: {f"{ACTION}_is_pad": torch.zeros(1, 4, dtype=torch.bool)},
    }
    step = Pi05B2LocalTrajectoryProcessorStep(
        dt=0.02,
        representation="velocity",
        z1_representation="ee_delta",
        ee_delta_rotation_representation="rotvec",
    )
    transformed = step(transition)
    assert transformed[TransitionKey.ACTION].shape == (1, 3, 13)
    assert transformed[TransitionKey.COMPLEMENTARY_DATA][EE_DELTA_VALID_KEY].tolist() == [
        [True, False, False]
    ]


def test_new_action_schema_retains_explicit_gates_and_completion():
    action = torch.arange(16, dtype=torch.float32).view(1, 1, 16)
    encoded = encode_b2_action_chunk(action, dt=0.1, representation="velocity")
    assert encoded.shape == (1, 1, 16)
    torch.testing.assert_close(encoded, action)

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
    torch.testing.assert_close(without_optional_groups, action[..., :3])


def test_processor_uses_new_schema_without_mutating_raw_transition():
    state = torch.arange(49, dtype=torch.float32).view(1, 49)
    action = torch.zeros(1, 2, 16)
    action[..., 0] = 1.0
    action[..., 3] = 1.0
    action[..., 15] = torch.tensor([[0.0, 1.0]])
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: state},
        TransitionKey.ACTION: action,
        TransitionKey.COMPLEMENTARY_DATA: {f"{ACTION}_is_pad": torch.zeros(1, 2, dtype=torch.bool)},
    }
    step = Pi05B2LocalTrajectoryProcessorStep(dt=0.1, state_indices=(0, 12, 37, 38, 39))
    transformed = step(transition)
    assert transformed[TransitionKey.ACTION].shape == (1, 2, 16)
    torch.testing.assert_close(transformed[TransitionKey.ACTION][0, :, 0], torch.tensor([0.1, 0.1]))
    torch.testing.assert_close(transformed[TransitionKey.ACTION][..., 3], action[..., 3])
    torch.testing.assert_close(transformed[TransitionKey.ACTION][..., 15], action[..., 15])
    assert transition[TransitionKey.OBSERVATION][OBS_STATE].shape == (1, 49)
    assert step.get_config()["include_task_complete"] is True


def test_action_multiplicity_can_cap_chunk_start_frames():
    assert action_label_multiplicity(5, [0, 1, 2], num_start_frames=4).tolist() == [1, 2, 3, 3, 2]


def test_trajectory_stats_keep_rare_positive_boolean_normalizable():
    raw = {
        ACTION: {
            "q01": torch.zeros(16),
            "q99": torch.zeros(16),
            "min": torch.zeros(16),
            "max": torch.zeros(16),
            "mean": torch.zeros(16),
            "std": torch.zeros(16),
        }
    }
    raw[ACTION]["min"][14] = -1.0
    transformed = make_b2_trajectory_stats(raw, dt=0.02, chunk_size=50)
    assert transformed is not None
    for name in ("arm_teleop_inactive", "arm_reset", "task_complete"):
        dim = [*DATASET_ACTION_NAMES].index(name)
        assert transformed[ACTION]["q01"][dim].item() == 0.0
        assert transformed[ACTION]["q99"][dim].item() == 1.0
        assert transformed[ACTION]["mean"][dim].item() == 0.5
        assert transformed[ACTION]["std"][dim].item() == 0.5


def test_ee_delta_stats_reject_absolute_workspace_approximation():
    raw = {ACTION: {name: torch.zeros(16) for name in ("q01", "q99", "min", "max", "mean", "std")}}
    with pytest.raises(ValueError, match="measured after the temporal action transform"):
        make_b2_trajectory_stats(
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
