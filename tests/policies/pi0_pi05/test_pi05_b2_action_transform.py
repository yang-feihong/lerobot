#!/usr/bin/env python

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies import factory as policy_factory
from lerobot.policies.pi05.b2_action_transform import (
    DATASET_ACTION_NAMES,
    action_label_multiplicity,
    decode_b2_action_chunk,
    encode_b2_action_chunk,
    global_pose_to_local_trajectory,
    integrate_body_twist,
    make_b2_trajectory_stats,
)
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05.processor_pi05 import Pi05B2LocalTrajectoryProcessorStep
from lerobot.scripts.lerobot_train import configure_action_bool_balance, resolve_task_complete_sampling
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE


def _state_names() -> list[str]:
    legs = [f"{leg}_{joint}" for leg in ("FR", "FL", "RR", "RL") for joint in ("hip", "thigh", "calf")]
    return [
        *[f"arm_q_{i}" for i in range(1, 7)],
        *[f"arm_qd_{i}" for i in range(1, 7)],
        "arm_gripper_feedback",
        *[f"b2_joint_pos_{name}" for name in legs],
        *[f"b2_joint_vel_{name}" for name in legs],
        "b2_trunk_roll",
        "b2_trunk_pitch",
        "b2_body_height",
        "b2_position_x",
        "b2_position_y",
        "b2_yaw",
        "b2_body_vx",
        "b2_body_vy",
        "b2_body_vz",
        "b2_body_wx",
        "b2_body_wy",
        "b2_body_wz",
    ]


def test_body_twist_round_trip_with_rotating_frame_and_unwrapped_yaw():
    twist = torch.randn(2, 50, 3, dtype=torch.float64)
    twist[..., 2] = 1.2
    trajectory = integrate_body_twist(twist, dt=0.1)
    assert torch.all(trajectory[..., -1, 2] > math.pi)
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
    assert trajectory[0, 1, 2].item() == pytest.approx(expected_first_yaw + 0.1)


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
    torch.testing.assert_close(transformed[TransitionKey.ACTION][0, :, 0], torch.tensor([0.1, 0.2]))
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


class _Episodes:
    def __init__(self, length: int):
        self.rows = [{"dataset_from_index": 0, "dataset_to_index": length}]

    def __getitem__(self, item):
        if isinstance(item, str):
            return [row[item] for row in self.rows]
        return self.rows[item]


class _FakeHFDataset:
    def __init__(self, actions: np.ndarray):
        self.actions = actions

    def select_columns(self, _columns):
        return self

    def with_format(self, _format):
        return self

    def iter(self, batch_size):
        assert batch_size > 0
        yield {
            ACTION: self.actions,
            "episode_index": np.zeros(len(self.actions), dtype=np.int64),
            "frame_index": np.arange(len(self.actions), dtype=np.int64),
        }


def _fake_dataset(actions: np.ndarray):
    return SimpleNamespace(
        meta=SimpleNamespace(
            fps=10,
            features={ACTION: {"names": list(DATASET_ACTION_NAMES)}},
            episodes=_Episodes(len(actions)),
        ),
        episodes=[0],
        num_episodes=1,
        hf_dataset=_FakeHFDataset(actions),
    )


def test_completion_tail_sampling_is_monotonic_and_physically_capped():
    actions = np.zeros((8, 16), dtype=np.float32)
    actions[4:, 15] = 1
    resolved = resolve_task_complete_sampling(
        _fake_dataset(actions), SimpleNamespace(task_complete_sample_tail_seconds=0.2)
    )
    assert resolved == ([7], {0: 7})  # first complete input plus two further 10 Hz tail inputs

    actions[6, 15] = 0
    with pytest.raises(ValueError, match="monotonic"):
        resolve_task_complete_sampling(
            _fake_dataset(actions), SimpleNamespace(task_complete_sample_tail_seconds=0.2)
        )


def test_all_boolean_priors_use_capped_starts_and_ignore_post_completion_controls():
    actions = np.zeros((5, 16), dtype=np.float32)
    actions[:, 3] = [1, 1, 0, 0, 0]
    actions[:, 4] = [0, 0, 1, 0, 0]
    actions[:, 14] = [0, -1, 0, -1, -1]
    actions[:, 15] = [0, 0, 0, 1, 1]
    policy_cfg = SimpleNamespace(
        action_predict_arm_teleop_inactive=True,
        action_predict_arm_reset=True,
        action_predict_gripper=True,
        action_predict_task_complete=True,
        action_gripper_target_true_side="negative",
        action_bool_loss_weight=4.0,
        action_bool_true_fractions={},
        chunk_size=3,
        control_frequency_hz=10,
    )
    stats = configure_action_bool_balance(
        SimpleNamespace(trainable_config=policy_cfg, resume=False), _fake_dataset(actions), {0: 4}
    )
    assert stats is not None
    assert (
        stats["arm_teleop_inactive"]["positive_labels"],
        stats["arm_teleop_inactive"]["negative_labels"],
    ) == (3, 3)
    assert (stats["arm_reset"]["positive_labels"], stats["arm_reset"]["negative_labels"]) == (3, 3)
    assert (stats["gripper_target"]["positive_labels"], stats["gripper_target"]["negative_labels"]) == (2, 4)
    assert (stats["task_complete"]["positive_labels"], stats["task_complete"]["negative_labels"]) == (5, 6)


def test_gate_loss_uses_inactive_reset_and_completion_ground_truth_masks():
    policy = PI05Policy.__new__(PI05Policy)
    torch.nn.Module.__init__(policy)
    names = ["b2_local_x", "b2_local_y", "b2_local_yaw", *DATASET_ACTION_NAMES[3:]]
    policy.config = SimpleNamespace(
        action_bool_loss_weight=4.0,
        action_continuous_loss_weight=1.0,
        action_masked_continuous_min_weight=0.0,
        action_bool_balance_eps=1e-3,
        action_bool_true_fractions=dict.fromkeys(
            ("arm_teleop_inactive", "arm_reset", "gripper_target", "task_complete"), 0.5
        ),
        action_gripper_target_true_side="negative",
        io_schema_resolved=True,
        b2_action_representation="local_trajectory",
        action_feature_names=names,
    )
    actions = -torch.ones(1, 5, 16)
    actions[0, :, 3] = torch.tensor([-1.0, 1.0, -1.0, -1.0, -1.0])
    actions[0, :, 4] = torch.tensor([-1.0, -1.0, 1.0, -1.0, -1.0])
    actions[0, :, 15] = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0])
    loss, info = policy._b2_z1_gate_action_loss(torch.ones_like(actions), actions, "mean")
    assert torch.isfinite(loss)
    assert info["continuous_mask_frac/b2_local_trajectory"] == pytest.approx(0.6)
    assert info["continuous_mask_frac/ee_pose"] == pytest.approx(0.2)
    assert info["gate_true_frac/task_complete"] == pytest.approx(0.4)
    assert "gate_loss/arm_teleop_inactive" in info


def test_disabling_inactive_prediction_removes_its_output_and_ee_mask():
    policy = PI05Policy.__new__(PI05Policy)
    torch.nn.Module.__init__(policy)
    names = ["b2_local_x", "b2_local_y", "b2_local_yaw", *DATASET_ACTION_NAMES[4:]]
    policy.config = SimpleNamespace(
        action_bool_loss_weight=4.0,
        action_continuous_loss_weight=1.0,
        action_masked_continuous_min_weight=0.0,
        action_bool_balance_eps=1e-3,
        action_bool_true_fractions=dict.fromkeys(("arm_reset", "gripper_target", "task_complete"), 0.5),
        action_gripper_target_true_side="negative",
        io_schema_resolved=True,
        b2_action_representation="local_trajectory",
        action_feature_names=names,
    )
    actions = -torch.ones(1, 3, 15)
    loss, info = policy._b2_z1_gate_action_loss(torch.ones_like(actions), actions, "mean")
    assert torch.isfinite(loss)
    assert info["continuous_mask_frac/ee_pose"] == pytest.approx(1.0)
    assert "gate_loss/arm_teleop_inactive" not in info


def test_policy_factory_resolves_49d_state_and_new_16d_action(monkeypatch):
    meta = SimpleNamespace(
        features={
            OBS_STATE: {"dtype": "float32", "shape": (49,), "names": _state_names()},
            ACTION: {"dtype": "float32", "shape": (16,), "names": list(DATASET_ACTION_NAMES)},
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
    policy = policy_factory.make_policy(PI05Config(device="cpu"), ds_meta=meta)
    assert policy.config.input_features[OBS_STATE].shape == (10,)
    assert policy.config.output_features[ACTION].shape == (16,)
    assert policy.config.state_feature_indices == [0, 1, 2, 3, 4, 5, 12, 37, 38, 39]
    assert policy.config.b2_global_pose_state_indices == [40, 41, 42]
    assert policy.config.action_feature_names[3] == "arm_teleop_inactive"
    assert policy.config.action_feature_names[-1] == "task_complete"

    disabled = policy_factory.make_policy(
        PI05Config(device="cpu", action_predict_arm_teleop_inactive=False), ds_meta=meta
    )
    assert disabled.config.output_features[ACTION].shape == (15,)
    assert "arm_teleop_inactive" not in disabled.config.action_feature_names


def test_checkpoint_metadata_describes_explicit_completion_protocol(tmp_path):
    config = PI05Config(b2_local_trajectory_dt=0.02, control_frequency_hz=50.0)
    config.io_schema_resolved = True
    config.dataset_camera_keys = []
    config.dataset_state_feature_names = _state_names()
    config.resolved_state_feature_names = _state_names()[:6]
    config.state_feature_indices = list(range(6))
    config.dataset_action_feature_names = list(DATASET_ACTION_NAMES)
    config.action_feature_names = ["b2_local_x", "b2_local_y", "b2_local_yaw", *DATASET_ACTION_NAMES[3:]]
    config._save_pretrained(tmp_path)
    metadata = json.loads((tmp_path / "pi05_deployment_metadata.json").read_text())
    assert metadata["action"]["predict"]["arm_teleop_inactive"] is True
    assert "b2_active" not in metadata["action"]["predict"]
    assert (
        metadata["action"]["task_complete_semantics"]
        == "explicit_true_in_the_post_task_tail_until_ros_bag_end"
    )
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.deployment_metadata() == metadata
