#!/usr/bin/env python

import json
import math
from types import SimpleNamespace

import pytest
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies import factory as policy_factory
from lerobot.policies.pi05.b2_action_transform import (
    action_label_multiplicity,
    action_sample_offsets,
    completion_from_padding,
    decode_b2_action_chunk,
    encode_b2_action_chunk,
    global_pose_to_local_trajectory,
    integrate_body_twist,
    make_b2_trajectory_stats,
    task_complete_class_counts,
)
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05.processor_pi05 import Pi05B2LocalTrajectoryProcessorStep
from lerobot.scripts.lerobot_train import configure_action_bool_balance
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE


def test_body_twist_round_trip_with_rotating_frame_and_unwrapped_yaw():
    torch.manual_seed(7)
    twist = torch.randn(2, 50, 3, dtype=torch.float64)
    twist[..., :2] *= 0.4
    twist[..., 2] = 1.2  # 6 rad over the chunk: deliberately greater than pi.

    trajectory = integrate_body_twist(twist, dt=0.1)
    assert torch.all(trajectory[..., -1, 2] > math.pi)

    action = torch.zeros(2, 50, 15, dtype=torch.float64)
    action[..., 0:3] = trajectory
    decoded = decode_b2_action_chunk(action, dt=0.1)
    torch.testing.assert_close(decoded[..., 0:3], twist, rtol=1e-9, atol=1e-9)


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


def test_fifty_hz_twist_round_trip_uses_twenty_millisecond_steps():
    twist = torch.tensor([[[0.6, -0.2, 0.4]] * 50], dtype=torch.float64)
    trajectory = integrate_body_twist(twist, dt=0.02)
    # Fifty 20 ms controls cover exactly one second.
    assert trajectory[0, -1, 2].item() == pytest.approx(0.4)

    action = torch.zeros(1, 50, 15, dtype=torch.float64)
    action[..., :3] = trajectory
    decoded = decode_b2_action_chunk(action, dt=0.02)
    torch.testing.assert_close(decoded[..., :3], twist, rtol=1e-9, atol=1e-9)


def test_global_pose_trajectory_uses_current_body_frame_and_unwraps_yaw():
    global_pose = torch.tensor(
        [[[10.0, 20.0, 3.13], [10.0, 21.0, -3.13], [9.0, 21.0, -3.03]]],
        dtype=torch.float64,
    )
    trajectory = global_pose_to_local_trajectory(global_pose)
    expected_first_yaw = 2 * math.pi - 6.26
    torch.testing.assert_close(
        trajectory[0, 0],
        torch.tensor([0.0115924, -0.9999328, expected_first_yaw], dtype=torch.float64),
        rtol=1e-5,
        atol=1e-5,
    )
    assert trajectory[0, 1, 2].item() == pytest.approx(expected_first_yaw + 0.1)


def test_global_pose_fields_request_future_state_without_exposing_it_to_policy():
    state_names = [f"state_{i}" for i in range(4)] + ["b2_position_x", "b2_position_y", "b2_yaw"]
    cfg = PI05Config(chunk_size=2, n_action_steps=2, device="cpu")
    meta = SimpleNamespace(
        fps=50,
        camera_keys=[],
        features={
            OBS_STATE: {"names": state_names},
            ACTION: {"names": [f"action_{i}" for i in range(16)]},
        },
    )
    delta_timestamps = resolve_delta_timestamps(cfg, meta)
    assert delta_timestamps[OBS_STATE] == [0.0, 0.02, 0.04]
    assert cfg.b2_global_pose_state_indices == [4, 5, 6]

    state = torch.zeros(1, 3, 7)
    state[0, :, :4] = torch.tensor([[1.0, 2.0, 3.0, 4.0]] * 3)
    state[0, :, 4:] = torch.tensor([[5.0, 6.0, 0.0], [6.0, 6.0, 0.0], [7.0, 6.0, 0.0]])
    action = torch.zeros(1, 2, 16)
    step = Pi05B2LocalTrajectoryProcessorStep(
        dt=0.02,
        state_indices=(0, 1, 2, 3),
        global_pose_state_indices=(4, 5, 6),
    )
    transformed = step(
        {
            TransitionKey.OBSERVATION: {OBS_STATE: state},
            TransitionKey.ACTION: action,
            TransitionKey.COMPLEMENTARY_DATA: {f"{ACTION}_is_pad": torch.zeros(1, 2, dtype=torch.bool)},
        }
    )
    assert transformed[TransitionKey.OBSERVATION][OBS_STATE].shape == (1, 4)
    torch.testing.assert_close(
        transformed[TransitionKey.ACTION][0, :, :3],
        torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )


def test_episode_padding_derives_completion_and_stops_integration():
    action = torch.zeros(1, 5, 16)
    action[..., 0] = 1.0  # b2_active: deliberately absent from the model target.
    action[..., 1] = 1.0
    action[..., 4] = 1.0  # arm_active: deliberately absent from the model target.
    action[..., 5] = 1.0  # arm_reset moves from dataset index 5 to model index 3.
    is_pad = torch.tensor([[False, False, False, True, True]])

    encoded = encode_b2_action_chunk(action, dt=0.1, is_pad=is_pad)
    assert encoded.shape == (1, 5, 15)
    torch.testing.assert_close(encoded[0, :, 0], torch.tensor([0.1, 0.2, 0.3, 0.3, 0.3]))
    torch.testing.assert_close(encoded[0, :, 3], torch.ones(5))
    torch.testing.assert_close(encoded[0, :, 14], torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]))
    assert completion_from_padding(is_pad).tolist() == [[False, False, True, True, True]]


def test_velocity_schema_can_retain_all_dataset_action_fields():
    action = torch.arange(16, dtype=torch.float32).view(1, 1, 16)
    encoded = encode_b2_action_chunk(
        action,
        dt=0.1,
        representation="velocity",
        predict_b2_active=True,
        predict_arm_active=True,
    )
    assert encoded.shape == (1, 1, 17)
    torch.testing.assert_close(encoded[..., :16], action)
    assert encoded[..., 16].item() == 0.0


def test_default_processor_selects_ten_state_dims_without_changing_dataset():
    state_indices = (0, 1, 2, 3, 4, 5, 12, 37, 38, 39)
    step = Pi05B2LocalTrajectoryProcessorStep(dt=0.1, state_indices=state_indices)
    state = torch.arange(46, dtype=torch.float32).view(1, 46)
    raw_action = torch.zeros(1, 2, 16)
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: state},
        TransitionKey.ACTION: raw_action,
        TransitionKey.COMPLEMENTARY_DATA: {f"{ACTION}_is_pad": torch.zeros(1, 2, dtype=torch.bool)},
    }
    transformed = step(transition)
    torch.testing.assert_close(
        transformed[TransitionKey.OBSERVATION][OBS_STATE], state[..., list(state_indices)]
    )
    assert transformed[TransitionKey.ACTION].shape == (1, 2, 15)
    # The caller-owned raw tensors and schema remain untouched.
    assert transition[TransitionKey.OBSERVATION][OBS_STATE].shape == (1, 46)
    assert transition[TransitionKey.ACTION].shape == (1, 2, 16)


def test_checkpoint_writes_human_readable_io_schema(tmp_path):
    config = PI05Config(
        b2_local_trajectory_dt=0.02,
        control_frequency_hz=50.0,
        dataset_frequency_hz=50.0,
        mem_vit_enabled=True,
        mem_vit_checkpoint="/data/checkpoints/mem_vit.pt",
        mem_vit_num_frames=6,
        mem_vit_frame_stride=25,
        n_action_steps=25,
    )
    config.io_schema_resolved = True
    config.dataset_camera_keys = ["observation.images.base", "observation.images.wrist"]
    config.dataset_state_feature_names = ["arm_q_1", "arm_gripper_feedback"]
    config.resolved_state_feature_names = ["arm_q_1"]
    config.state_feature_indices = [0]
    config.dataset_action_feature_names = [f"raw_{i}" for i in range(16)]
    config.action_feature_names = ["b2_local_x", "b2_local_y", "b2_local_yaw"]
    config._save_pretrained(tmp_path)
    metadata = json.loads((tmp_path / "pi05_deployment_metadata.json").read_text())
    assert metadata["format"] == "lerobot.pi05.deployment"
    assert metadata["version"] == 1
    assert metadata["timing"] == {
        "control_frequency_hz": 50.0,
        "action_dt_seconds": 0.02,
        "chunk_size_steps": 50,
        "chunk_horizon_seconds": 1.0,
        "execute_steps_per_inference": 25,
        "replan_interval_seconds": 0.5,
    }
    assert metadata["observation"]["state"]["selected_names"] == ["arm_q_1"]
    assert metadata["observation"]["state"]["switches"] == {
        "arm_joint_positions": True,
        "arm_joint_velocities": False,
        "arm_gripper_feedback": True,
        "b2_joint_positions": False,
        "b2_joint_velocities": False,
        "b2_trunk_pose": True,
        "b2_linear_velocity": False,
        "b2_angular_velocity": False,
    }
    assert metadata["action"]["representation"] == "local_trajectory"
    assert metadata["action"]["predict"]["b2_active"] is False
    assert metadata["memory"]["enabled"] is True
    assert metadata["memory"]["fixed_num_frames"] == 6
    assert metadata["memory"]["frame_interval_seconds"] == 0.5
    assert metadata["memory"]["max_history_span_seconds"] == 2.5
    assert "training_dataset_frequency_hz" not in metadata["timing"]
    assert "resampling" not in metadata["timing"]
    assert "training_action_semantics" not in metadata
    assert "initialization_checkpoint" not in metadata["memory"]
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.deployment_metadata() == metadata


def test_dataset_and_model_frequencies_are_resampled_independently(caplog):
    action_names = [f"action_{i}" for i in range(16)]
    state_names = ["b2_position_x", "b2_position_y", "b2_yaw"]

    high_rate_meta = SimpleNamespace(
        fps=50,
        camera_keys=[],
        features={OBS_STATE: {"names": state_names}, ACTION: {"names": action_names}},
    )
    ten_hz_policy = PI05Config(
        chunk_size=3,
        n_action_steps=3,
        control_frequency_hz=10,
        device="cpu",
    )
    high_to_low = resolve_delta_timestamps(ten_hz_policy, high_rate_meta)
    assert high_to_low[ACTION] == [0.0, 0.1, 0.2]
    assert high_to_low[OBS_STATE][-3:] == [0.1, 0.2, 0.3]

    low_rate_meta = SimpleNamespace(
        fps=10,
        camera_keys=[],
        features={OBS_STATE: {"names": state_names}, ACTION: {"names": action_names}},
    )
    fifty_hz_policy = PI05Config(
        chunk_size=6,
        n_action_steps=6,
        control_frequency_hz=50,
        device="cpu",
    )
    low_to_high = resolve_delta_timestamps(fifty_hz_policy, low_rate_meta)
    assert low_to_high[ACTION] == [0.0, 0.0, 0.0, 0.1, 0.1, 0.1]
    assert "cannot recover missing high-frequency behavior" in caplog.text


def test_low_rate_labels_are_counted_with_the_same_repetition_used_by_loader():
    offsets = action_sample_offsets(chunk_size=6, dataset_fps=10, control_frequency_hz=50)
    assert offsets == [0, 0, 0, 1, 1, 1]
    assert action_label_multiplicity(episode_length=3, offsets=offsets).tolist() == [3, 6, 6]


def test_policy_factory_resolves_default_io_schema(monkeypatch):
    state_names = [
        "arm_q_1",
        "arm_q_2",
        "arm_q_3",
        "arm_q_4",
        "arm_q_5",
        "arm_q_6",
        "arm_qd_1",
        "arm_qd_2",
        "arm_qd_3",
        "arm_qd_4",
        "arm_qd_5",
        "arm_qd_6",
        "arm_gripper_feedback",
        *[f"b2_joint_pos_{i}" for i in range(12)],
        *[f"b2_joint_vel_{i}" for i in range(12)],
        "b2_trunk_roll",
        "b2_trunk_pitch",
        "b2_body_height",
        "b2_body_vx",
        "b2_body_vy",
        "b2_body_vz",
        "b2_body_wx",
        "b2_body_wy",
        "b2_body_wz",
    ]
    action_names = [
        "b2_active",
        "b2_vx",
        "b2_vy",
        "b2_omega_z",
        "arm_active",
        "arm_reset",
        *[f"height_invariant_ee_{i}" for i in range(9)],
        "gripper_target",
    ]
    meta = SimpleNamespace(
        features={
            OBS_STATE: {"dtype": "float32", "shape": (46,), "names": state_names},
            ACTION: {"dtype": "float32", "shape": (16,), "names": action_names},
        },
        fps=50,
        camera_keys=[],
        stats={},
    )

    class DummyPolicy(torch.nn.Module):
        def __init__(self, config, **kwargs):
            super().__init__()
            self.config = config

    monkeypatch.setattr(policy_factory, "get_policy_class", lambda _: DummyPolicy)
    config = PI05Config(device="cpu")
    policy = policy_factory.make_policy(config, ds_meta=meta)
    assert policy.config.input_features[OBS_STATE].shape == (10,)
    assert policy.config.output_features[ACTION].shape == (15,)
    assert policy.config.state_feature_indices == [0, 1, 2, 3, 4, 5, 12, 37, 38, 39]
    assert "b2_active" not in policy.config.action_feature_names
    assert "arm_active" not in policy.config.action_feature_names
    assert policy.config.action_feature_names[-1] == "task_complete"
    assert policy.config.b2_local_trajectory_dt == pytest.approx(0.02)

    lower_rate_config = PI05Config(device="cpu", control_frequency_hz=10)
    lower_rate_policy = policy_factory.make_policy(lower_rate_config, ds_meta=meta)
    assert lower_rate_policy.config.dataset_frequency_hz == 50
    assert lower_rate_policy.config.control_frequency_hz == 10
    assert lower_rate_policy.config.b2_local_trajectory_dt == pytest.approx(0.1)

    velocity_config = PI05Config(
        device="cpu",
        state_use_arm_joint_positions=False,
        state_use_arm_joint_velocities=True,
        state_use_arm_gripper_feedback=False,
        state_use_b2_trunk_pose=False,
        state_use_b2_linear_velocity=True,
        state_use_b2_angular_velocity=True,
        b2_action_representation="velocity",
        action_predict_b2_active=True,
        action_predict_arm_active=True,
        action_predict_task_complete=False,
    )
    velocity_policy = policy_factory.make_policy(velocity_config, ds_meta=meta)
    assert velocity_policy.config.input_features[OBS_STATE].shape == (12,)
    assert velocity_policy.config.state_feature_indices == list(range(6, 12)) + list(range(40, 46))
    assert velocity_policy.config.output_features[ACTION].shape == (16,)
    assert velocity_policy.config.action_feature_names == action_names

    mismatched_config = PI05Config(device="cpu", b2_local_trajectory_dt=0.1)
    with pytest.raises(ValueError, match="does not match model control frequency"):
        policy_factory.make_policy(mismatched_config, ds_meta=meta)


@pytest.mark.parametrize("length", [1, 2, 4, 5, 6, 17])
def test_task_complete_class_counts_match_chunk_padding(length):
    chunk_size = 5
    positive, negative = task_complete_class_counts([length], chunk_size)
    brute_positive = 0
    brute_known = 0
    for start in range(length):
        is_pad = torch.tensor([[start + offset >= length for offset in range(chunk_size)]])
        target = completion_from_padding(is_pad)
        valid = torch.ones_like(target)
        valid[:, -1] = is_pad[:, -1]
        brute_positive += int((target & valid).sum())
        brute_known += int(valid.sum())
    assert (positive, negative) == (brute_positive, brute_known - brute_positive)


def test_all_enabled_bool_priors_are_precomputed_from_train_split():
    action_names = [
        "b2_active",
        "b2_vx",
        "b2_vy",
        "b2_omega_z",
        "arm_active",
        "arm_reset",
        *[f"height_invariant_ee_{i}" for i in range(9)],
        "gripper_target",
    ]
    actions = torch.zeros(4, 16)
    actions[:, 0] = torch.tensor([0.0, 1.0, 1.0, 0.0])
    actions[:, 4] = torch.tensor([0.0, 0.0, 1.0, 1.0])
    actions[:, 5] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    actions[:, 15] = torch.tensor([0.0, -1.0, 0.0, -1.0])

    class FakeHFDataset:
        def select_columns(self, _columns):
            return self

        def with_format(self, _format):
            return self

        def iter(self, batch_size):
            assert batch_size > 0
            yield {
                ACTION: actions.numpy(),
                "episode_index": torch.zeros(4, dtype=torch.int64).numpy(),
                "frame_index": torch.arange(4).numpy(),
            }

    policy_cfg = SimpleNamespace(
        action_predict_b2_active=True,
        action_predict_arm_active=True,
        action_predict_arm_reset=True,
        action_predict_gripper=True,
        action_predict_task_complete=True,
        action_gripper_target_true_side="negative",
        action_bool_loss_weight=4.0,
        action_bool_true_fractions={},
        chunk_size=3,
    )
    cfg = SimpleNamespace(trainable_config=policy_cfg, resume=False)
    dataset = SimpleNamespace(
        meta=SimpleNamespace(
            fps=10,
            features={ACTION: {"names": action_names}},
            episodes=[{"dataset_from_index": 0, "dataset_to_index": 4}],
        ),
        episodes=[0],
        num_episodes=1,
        hf_dataset=FakeHFDataset(),
    )

    stats = configure_action_bool_balance(cfg, dataset)
    assert stats is not None
    assert stats["b2_active"]["positive_labels"] == 5
    assert stats["b2_active"]["negative_labels"] == 4
    assert stats["arm_reset"]["positive_labels"] == 3
    assert stats["arm_reset"]["negative_labels"] == 6
    assert stats["gripper_target"]["positive_labels"] == 5
    assert stats["gripper_target"]["negative_labels"] == 4
    complete_positive, complete_negative = task_complete_class_counts([4], 3)
    assert stats["task_complete"]["positive_labels"] == complete_positive
    assert stats["task_complete"]["negative_labels"] == complete_negative
    assert policy_cfg.action_bool_true_fractions == {
        name: values["true_fraction"] for name, values in stats.items()
    }


def test_trajectory_stats_are_derived_without_mutating_dataset_stats():
    raw = {
        ACTION: {
            "q01": torch.tensor([0.0, -0.5, -0.2, -1.0] + [0.0] * 12),
            "q99": torch.tensor([1.0, 0.5, 0.2, 1.0] + [1.0] * 12),
            "min": torch.tensor([0.0, -0.6, -0.3, -1.2] + [0.0] * 12),
            "max": torch.tensor([1.0, 0.6, 0.3, 1.2] + [1.0] * 12),
        },
        OBS_STATE: {
            "q01": torch.arange(46, dtype=torch.float32),
            "q99": torch.arange(46, dtype=torch.float32) + 1,
        },
    }
    state_indices = (0, 1, 2, 3, 4, 5, 12, 37, 38, 39)
    transformed = make_b2_trajectory_stats(raw, dt=0.02, chunk_size=50, state_indices=state_indices)

    assert raw[ACTION]["q01"].shape == (16,)
    assert transformed is not None
    assert transformed[ACTION]["q01"].shape == (15,)
    assert transformed[ACTION]["q99"][-1].item() == 1.0
    expected_xy_scale = math.hypot(0.5, 0.2)
    torch.testing.assert_close(
        transformed[ACTION]["q99"][0:2], torch.tensor([expected_xy_scale, expected_xy_scale])
    )
    assert transformed[ACTION]["q99"][2].item() == 1.0
    assert transformed[OBS_STATE]["q01"].shape == (10,)
    torch.testing.assert_close(
        transformed[OBS_STATE]["q01"], torch.tensor(state_indices, dtype=torch.float32)
    )


def test_gate_loss_trains_completion_and_masks_post_episode_actions():
    policy = PI05Policy.__new__(PI05Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        action_bool_loss_weight=4.0,
        action_continuous_loss_weight=1.0,
        action_masked_continuous_min_weight=0.0,
        action_bool_balance_eps=1e-3,
        action_bool_true_fractions={
            "arm_reset": 0.2,
            "gripper_target": 0.5,
            "task_complete": 0.1,
        },
        action_gripper_target_true_side="negative",
        io_schema_resolved=True,
        b2_action_representation="local_trajectory",
        action_feature_names=[
            "b2_local_x",
            "b2_local_y",
            "b2_local_yaw",
            "arm_reset",
            "height_invariant_ee_rot6d_col0_x",
            "height_invariant_ee_rot6d_col0_y",
            "height_invariant_ee_rot6d_col0_z",
            "height_invariant_ee_rot6d_col1_x",
            "height_invariant_ee_rot6d_col1_y",
            "height_invariant_ee_rot6d_col1_z",
            "height_invariant_ee_x",
            "height_invariant_ee_y",
            "height_invariant_ee_z",
            "gripper_target",
            "task_complete",
        ],
    )
    losses = torch.ones(1, 5, 15)
    actions = -torch.ones(1, 5, 15)
    actions[..., 3] = -1.0
    # Final valid step and padding are complete after normalization.
    actions[..., 14] = torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0])
    is_pad = torch.tensor([[False, False, False, True, True]])

    loss, info = policy._b2_z1_gate_action_loss(losses, actions, "mean", is_pad)
    assert torch.isfinite(loss)
    assert info["gate_true_frac/task_complete"] == pytest.approx(0.6)
    assert info["continuous_mask_frac/b2_local_trajectory"] == pytest.approx(0.6)
    assert info["gate_global_true_frac/task_complete"] == pytest.approx(0.1)
    assert info["gate_weight/task_complete_true"] == pytest.approx(20.0)
    assert info["gate_weight/task_complete_false"] == pytest.approx(2.0 / 0.9)
    assert "gate_loss/b2_active" not in info
    assert "gate_loss/arm_active" not in info


def test_enabled_active_outputs_apply_their_continuous_loss_masks():
    policy = PI05Policy.__new__(PI05Policy)
    torch.nn.Module.__init__(policy)
    names = [
        "b2_active",
        "b2_vx",
        "b2_vy",
        "b2_omega_z",
        "arm_active",
        "arm_reset",
        *[f"height_invariant_ee_{i}" for i in range(9)],
        "gripper_target",
    ]
    policy.config = SimpleNamespace(
        action_bool_loss_weight=4.0,
        action_continuous_loss_weight=1.0,
        action_masked_continuous_min_weight=0.0,
        action_bool_balance_eps=1e-3,
        action_bool_true_fractions={
            "b2_active": 0.5,
            "arm_active": 0.5,
            "arm_reset": 0.25,
            "gripper_target": 0.5,
        },
        action_gripper_target_true_side="negative",
        io_schema_resolved=True,
        b2_action_representation="velocity",
        action_feature_names=names,
    )
    actions = -torch.ones(1, 4, 16)
    actions[0, :, 0] = torch.tensor([1.0, 1.0, -1.0, -1.0])
    actions[0, :, 4] = torch.tensor([1.0, -1.0, 1.0, -1.0])
    actions[0, :, 5] = torch.tensor([-1.0, -1.0, 1.0, -1.0])
    loss, info = policy._b2_z1_gate_action_loss(torch.ones_like(actions), actions, "mean")
    assert torch.isfinite(loss)
    assert info["continuous_mask_frac/b2_velocity"] == pytest.approx(0.5)
    assert info["continuous_mask_frac/ee_pose"] == pytest.approx(0.25)
    assert "gate_loss/b2_active" in info
    assert "gate_loss/arm_active" in info
    assert "gate_loss/arm_reset" in info
