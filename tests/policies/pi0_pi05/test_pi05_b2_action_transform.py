#!/usr/bin/env python

import math
from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.pi05.b2_action_transform import (
    completion_from_padding,
    decode_b2_action_chunk,
    encode_b2_action_chunk,
    integrate_body_twist,
    make_b2_trajectory_stats,
)
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.constants import ACTION


def test_body_twist_round_trip_with_rotating_frame_and_unwrapped_yaw():
    torch.manual_seed(7)
    twist = torch.randn(2, 50, 3, dtype=torch.float64)
    twist[..., :2] *= 0.4
    twist[..., 2] = 1.2  # 6 rad over the chunk: deliberately greater than pi.

    trajectory = integrate_body_twist(twist, dt=0.1)
    assert torch.all(trajectory[..., -1, 2] > math.pi)

    action = torch.zeros(2, 50, 17, dtype=torch.float64)
    action[..., 1:4] = trajectory
    decoded = decode_b2_action_chunk(action, dt=0.1)
    torch.testing.assert_close(decoded[..., 1:4], twist, rtol=1e-9, atol=1e-9)


def test_translation_follows_accumulated_yaw_instead_of_independent_sums():
    # Move forward while turning 90 degrees/s. The path must curve; naively
    # summing vx would instead produce y == 0 and x == 1.
    twist = torch.tensor([[[1.0, 0.0, math.pi / 2]] * 10])
    trajectory = integrate_body_twist(twist, dt=0.1)
    torch.testing.assert_close(
        trajectory[0, -1],
        torch.tensor([2 / math.pi, 2 / math.pi, math.pi / 2]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_episode_padding_derives_completion_and_stops_integration():
    action = torch.zeros(1, 5, 16)
    action[..., 1] = 1.0
    is_pad = torch.tensor([[False, False, False, True, True]])

    encoded = encode_b2_action_chunk(action, dt=0.1, is_pad=is_pad)
    torch.testing.assert_close(encoded[0, :, 1], torch.tensor([0.1, 0.2, 0.3, 0.3, 0.3]))
    torch.testing.assert_close(encoded[0, :, 16], torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]))
    assert completion_from_padding(is_pad).tolist() == [[False, False, True, True, True]]


def test_trajectory_stats_are_derived_without_mutating_dataset_stats():
    raw = {
        ACTION: {
            "q01": torch.tensor([0.0, -0.5, -0.2, -1.0] + [0.0] * 12),
            "q99": torch.tensor([1.0, 0.5, 0.2, 1.0] + [1.0] * 12),
            "min": torch.tensor([0.0, -0.6, -0.3, -1.2] + [0.0] * 12),
            "max": torch.tensor([1.0, 0.6, 0.3, 1.2] + [1.0] * 12),
        }
    }
    transformed = make_b2_trajectory_stats(raw, dt=0.1, chunk_size=50)

    assert raw[ACTION]["q01"].shape == (16,)
    assert transformed is not None
    assert transformed[ACTION]["q01"].shape == (17,)
    assert transformed[ACTION]["q99"][-1].item() == 1.0
    expected_xy_scale = math.hypot(0.5, 0.2) * 5.0
    torch.testing.assert_close(
        transformed[ACTION]["q99"][1:3], torch.tensor([expected_xy_scale, expected_xy_scale])
    )
    assert transformed[ACTION]["q99"][3].item() == 5.0


def test_gate_loss_trains_completion_and_masks_post_episode_actions():
    policy = PI05Policy.__new__(PI05Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        action_bool_loss_weight=4.0,
        action_continuous_loss_weight=1.0,
        action_masked_continuous_min_weight=0.0,
        action_bool_balance_eps=1e-3,
        action_gripper_target_true_side="negative",
        b2_local_trajectory_enabled=True,
    )
    losses = torch.ones(1, 5, 17)
    actions = -torch.ones(1, 5, 17)
    actions[..., 0] = 1.0
    actions[..., 4] = 1.0
    actions[..., 5] = -1.0
    # Final valid step and padding are complete after normalization.
    actions[..., 16] = torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0])
    is_pad = torch.tensor([[False, False, False, True, True]])

    loss, info = policy._b2_z1_gate_action_loss(losses, actions, "mean", is_pad)
    assert torch.isfinite(loss)
    assert info["gate_true_frac/task_complete"] == pytest.approx(0.6)
    assert info["continuous_mask_frac/b2_trajectory"] == pytest.approx(0.6)
