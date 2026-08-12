import json
from types import SimpleNamespace

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies import factory as policy_factory
from lerobot.policies.pi05.b2_action_transform import DATASET_ACTION_NAMES
from lerobot.policies.pi05.configuration_pi05 import PI05Config
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
    config.action_feature_names = ["b2_delta_x", "b2_delta_y", "b2_delta_yaw", *DATASET_ACTION_NAMES[3:]]
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


def test_historical_config_defaults_to_continuous_flow() -> None:
    assert PI05Config().discrete_action_training_mode == "continuous_flow"


def test_historical_saved_config_without_mode_loads_as_continuous_flow(tmp_path) -> None:
    PI05Config()._save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    raw = json.loads(config_path.read_text())
    raw.pop("discrete_action_training_mode")
    config_path.write_text(json.dumps(raw))

    restored = PreTrainedConfig.from_pretrained(tmp_path)

    assert restored.discrete_action_training_mode == "continuous_flow"


def test_structured_metadata_records_temporal_decoder_contract() -> None:
    metadata = PI05Config(discrete_action_training_mode="structured_temporal").deployment_metadata()
    assert metadata["version"] == 5
    assert metadata["action"]["discrete_training_mode"] == "structured_temporal"
    assert metadata["action"]["discrete_temporal_structure"]["arm_mode"]["allowed_transitions"] == "all"
    assert metadata["action"]["boolean_decoding"]["output_values"]["gripper_target"] == {
        "normalized_negative": -1.0471976,
        "normalized_nonnegative": 0.0,
    }
