import json
from types import SimpleNamespace

import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies import factory as policy_factory
from lerobot.policies.pi05.b2_action_transform import (
    CONTROL_EXTENDED_DATASET_ACTION_NAMES,
    DATASET_ACTION_NAMES,
    HEIGHT_INVARIANT_EE_STATE_NAMES,
)
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


def test_policy_factory_resolves_control_action_and_named_ee_state(monkeypatch, tmp_path):
    meta = SimpleNamespace(
        features={
            OBS_STATE: {
                "dtype": "float32",
                "shape": (58,),
                "names": [*_state_names(), *HEIGHT_INVARIANT_EE_STATE_NAMES],
            },
            ACTION: {
                "dtype": "float32",
                "shape": (25,),
                "names": list(CONTROL_EXTENDED_DATASET_ACTION_NAMES),
            },
        },
        fps=50,
        camera_keys=[],
        stats={},
        root=tmp_path,
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
    assert policy.config.b2_global_pose_state_indices is None
    assert policy.config.action_feature_names[3] == "arm_teleop_inactive"
    assert policy.config.action_feature_names[-1] == "task_complete"

    disabled = policy_factory.make_policy(
        PI05Config(
            device="cpu",
            action_predict_arm_teleop_inactive=False,
            action_predict_arm_reset=False,
        ),
        ds_meta=meta,
    )
    assert disabled.config.output_features[ACTION].shape == (14,)
    assert "arm_teleop_inactive" not in disabled.config.action_feature_names
    assert "arm_reset" not in disabled.config.action_feature_names


def test_policy_factory_resolves_joint_control_ee_supervision(monkeypatch, tmp_path):
    meta = SimpleNamespace(
        features={
            OBS_STATE: {
                "dtype": "float32",
                "shape": (58,),
                "names": [*_state_names(), *HEIGHT_INVARIANT_EE_STATE_NAMES],
            },
            ACTION: {
                "dtype": "float32",
                "shape": (25,),
                "names": list(CONTROL_EXTENDED_DATASET_ACTION_NAMES),
            },
        },
        fps=50,
        camera_keys=[],
        stats={},
        root=tmp_path,
    )

    class DummyPolicy(torch.nn.Module):
        def __init__(self, config, **kwargs):
            super().__init__()
            self.config = config

    monkeypatch.setattr(policy_factory, "get_policy_class", lambda _: DummyPolicy)
    policy = policy_factory.make_policy(
        PI05Config(
            device="cpu",
            b2_action_representation="velocity",
            z1_action_representation="ee_delta",
            ee_delta_rotation_representation="rotvec",
            action_predict_arm_teleop_inactive=False,
            action_predict_arm_reset=False,
            action_predict_task_complete=False,
            ee_supervision_source="control_action",
            ee_target_dataset_semantics="joint_control_inactive_interpolated",
        ),
        ds_meta=meta,
    )

    assert policy.config.ee_supervision_source == "control_action"
    assert policy.config.input_features[OBS_STATE].shape == (10,)
    assert policy.config.output_features[ACTION].shape == (10,)

    anchored = policy_factory.make_policy(
        PI05Config(
            device="cpu",
            z1_action_representation="ee_state_delta",
            ee_delta_rotation_representation="rotvec",
            action_predict_arm_teleop_inactive=False,
            action_predict_arm_reset=False,
            action_predict_task_complete=False,
        ),
        ds_meta=meta,
    )
    assert anchored.config.ee_state_anchor_indices == list(range(49, 58))
    assert anchored.config.resolved_state_feature_names[-9:] == list(HEIGHT_INVARIANT_EE_STATE_NAMES)
    assert anchored.config.input_features[OBS_STATE].shape == (19,)


def test_checkpoint_metadata_describes_explicit_completion_protocol(tmp_path):
    config = PI05Config(action_dt_seconds=0.02, control_frequency_hz=50.0)
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
    assert metadata["action"]["task_complete_semantics"] == "explicit_true_in_the_terminal_stage_hold"
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.deployment_metadata() == metadata


def test_deployment_metadata_preserves_explicit_dataset_camera_order() -> None:
    wrist_key = "observation.images.left_wrist_0_rgb"
    base_key = "observation.images.base_0_rgb"
    config = PI05Config(
        input_features={
            base_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            wrist_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        },
        dataset_camera_keys=[wrist_key, base_key],
    )

    observation = config.deployment_metadata()["observation"]

    assert observation["camera_keys_in_model_order"] == [wrist_key, base_key]
    assert list(observation["image_features"]) == [wrist_key, base_key]


BASE_CAMERA = "observation.images.base_0_rgb"
WRIST_CAMERA = "observation.images.left_wrist_0_rgb"


def _camera_config(keys, **kwargs):
    return PI05Config(
        device="cpu",
        input_features={key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)) for key in keys},
        **kwargs,
    )


@pytest.mark.parametrize("keys", [[WRIST_CAMERA, BASE_CAMERA], [BASE_CAMERA, WRIST_CAMERA]])
def test_new_camera_order_is_independent_of_dataset_keys(keys):
    config = _camera_config(keys)
    config.validate_features()
    assert list(config.image_features) == [BASE_CAMERA, WRIST_CAMERA]
    assert config.dataset_camera_keys == [BASE_CAMERA, WRIST_CAMERA]


def test_sorted_config_load_restores_checkpoint_camera_metadata(tmp_path):
    config = _camera_config([WRIST_CAMERA, BASE_CAMERA], dataset_camera_keys=[WRIST_CAMERA, BASE_CAMERA])
    config._save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(json.loads(config_path.read_text()), sort_keys=True))
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert list(restored.image_features) == [WRIST_CAMERA, BASE_CAMERA]
    assert list(restored.input_features)[:2] == [WRIST_CAMERA, BASE_CAMERA]


def test_hub_config_load_restores_checkpoint_metadata(tmp_path, monkeypatch):
    config = _camera_config([BASE_CAMERA, WRIST_CAMERA], dataset_camera_keys=[WRIST_CAMERA, BASE_CAMERA])
    config._save_pretrained(tmp_path)

    def download(*, filename, **kwargs):
        return str(tmp_path / filename)

    monkeypatch.setattr("lerobot.configs.policies.hf_hub_download", download)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    assert list(PreTrainedConfig.from_pretrained("test/model").image_features) == [WRIST_CAMERA, BASE_CAMERA]


def test_resolved_checkpoint_without_metadata_fails_closed(tmp_path):
    config = _camera_config([BASE_CAMERA, WRIST_CAMERA], io_schema_resolved=True)
    config._save_pretrained(tmp_path)
    (tmp_path / "pi05_deployment_metadata.json").unlink()
    with pytest.raises(FileNotFoundError, match="camera metadata"):
        PreTrainedConfig.from_pretrained(tmp_path)


def test_resume_restores_metadata_before_contract_comparison(tmp_path, monkeypatch):
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig

    config = _camera_config(
        [BASE_CAMERA, WRIST_CAMERA], dataset_camera_keys=[WRIST_CAMERA, BASE_CAMERA], io_schema_resolved=True
    )
    pipeline = TrainPipelineConfig(dataset=DatasetConfig(repo_id="local/test"), policy=config)
    config._save_pretrained(tmp_path)
    pipeline._save_pretrained(tmp_path)
    config.dataset_camera_keys = [BASE_CAMERA, WRIST_CAMERA]
    monkeypatch.setattr("lerobot.configs.train.parser.parse_arg", lambda _: str(tmp_path / "train_config.json"))
    pipeline._resolve_resume_checkpoint()
    assert list(pipeline.policy.image_features) == [WRIST_CAMERA, BASE_CAMERA]


@pytest.mark.parametrize(
    "keys", [[BASE_CAMERA], [BASE_CAMERA, BASE_CAMERA], [BASE_CAMERA, WRIST_CAMERA, "unknown"]]
)
def test_invalid_checkpoint_camera_contract_is_rejected(keys):
    config = _camera_config([WRIST_CAMERA, BASE_CAMERA])
    with pytest.raises(ValueError, match="camera metadata"):
        config.restore_camera_metadata({"observation": {"camera_keys_in_model_order": keys}})


def test_camera_contract_includes_synthetic_empty_cameras(tmp_path):
    config = _camera_config([WRIST_CAMERA, BASE_CAMERA], empty_cameras=1)
    config.validate_features()
    expected = [BASE_CAMERA, WRIST_CAMERA, "observation.images.empty_camera_0"]
    assert list(config.image_features) == expected
    config._save_pretrained(tmp_path)
    assert list(PreTrainedConfig.from_pretrained(tmp_path).image_features) == expected


@pytest.mark.parametrize("resume", [False, True])
def test_factory_uses_canonical_new_order_but_preserves_resume_metadata(monkeypatch, tmp_path, resume):
    meta = SimpleNamespace(
        features={
            WRIST_CAMERA: {
                "dtype": "video",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channels"],
            },
            BASE_CAMERA: {"dtype": "video", "shape": (224, 224, 3), "names": ["height", "width", "channels"]},
            OBS_STATE: {"dtype": "float32", "shape": (49,), "names": _state_names()},
            ACTION: {
                "dtype": "float32",
                "shape": (25,),
                "names": list(CONTROL_EXTENDED_DATASET_ACTION_NAMES),
            },
        },
        fps=50,
        camera_keys=[WRIST_CAMERA, BASE_CAMERA],
        stats={},
        root=tmp_path,
    )

    class DummyPolicy(torch.nn.Module):
        def __init__(self, config, **kwargs):
            super().__init__()
            self.config = config

        @classmethod
        def from_pretrained(cls, **kwargs):
            return cls(**kwargs)

    monkeypatch.setattr(policy_factory, "get_policy_class", lambda _: DummyPolicy)
    config = policy_factory.make_policy(PI05Config(device="cpu"), ds_meta=meta).config
    assert list(config.image_features) == [BASE_CAMERA, WRIST_CAMERA]
    if resume:
        config.dataset_camera_keys = [WRIST_CAMERA, BASE_CAMERA]
        config._save_pretrained(tmp_path)
        # Simulate a stale train_config and differently ordered dataset JSON.
        config.dataset_camera_keys = [BASE_CAMERA, WRIST_CAMERA]
        config.pretrained_path = tmp_path
        meta.camera_keys.reverse()
        resumed = policy_factory.make_policy(config, ds_meta=meta)
        assert list(resumed.config.image_features) == [WRIST_CAMERA, BASE_CAMERA]


@pytest.mark.parametrize("order", [[BASE_CAMERA, WRIST_CAMERA], [WRIST_CAMERA, BASE_CAMERA]])
def test_actual_image_preprocessing_matches_metadata_slots(order):
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    policy = PI05Policy.__new__(PI05Policy)
    torch.nn.Module.__init__(policy)
    policy.register_parameter("dummy", torch.nn.Parameter(torch.zeros(1)))
    policy.config = _camera_config([BASE_CAMERA, WRIST_CAMERA], dataset_camera_keys=order)
    # Distinct physical image contents, deliberately independent of dict order.
    batch = {WRIST_CAMERA: torch.ones(1, 3, 224, 224), BASE_CAMERA: torch.zeros(1, 3, 224, 224)}
    images, _, _ = policy._preprocess_images(batch)
    assert [image.mean().item() for image in images] == [-1.0 if key == BASE_CAMERA else 1.0 for key in order]


def test_current_config_defaults_to_control_action() -> None:
    assert PI05Config().discrete_action_training_mode == "continuous_flow"
    assert PI05Config().ee_target_dataset_semantics == "joint_control_inactive_interpolated"
    assert PI05Config().ee_delta_supervision_mode == "all"
    assert PI05Config().gripper_target_representation == "continuous_position"
    assert PI05Config().ee_supervision_source == "control_action"


def test_arm_mode_flow_channels_must_be_enabled_together() -> None:
    with pytest.raises(ValueError, match="must be enabled or disabled together"):
        PI05Config(action_predict_arm_teleop_inactive=True, action_predict_arm_reset=False)
    with pytest.raises(ValueError, match="must be enabled or disabled together"):
        PI05Config(action_predict_arm_teleop_inactive=False, action_predict_arm_reset=True)


def test_metadata_records_paired_flow_arm_mode_contract() -> None:
    action = PI05Config(
        discrete_action_training_mode="continuous_flow",
        action_predict_arm_teleop_inactive=True,
        action_predict_arm_reset=True,
    ).deployment_metadata()["action"]

    assert action["arm_mode_encoding"] == {
        "representation": "paired_flow_channels",
        "channels": ["arm_teleop_inactive", "arm_reset"],
        "states": {
            "teleop": {"arm_teleop_inactive": 0, "arm_reset": 0},
            "inactive": {"arm_teleop_inactive": 1, "arm_reset": 0},
            "reset": {"arm_teleop_inactive": 0, "arm_reset": 1},
        },
        "invalid": {"arm_teleop_inactive": 1, "arm_reset": 1},
        "training": "flow_matching_with_class_balanced_channel_loss",
    }


def test_saved_config_without_current_semantic_fields_loads_current_defaults(tmp_path) -> None:
    PI05Config()._save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    raw = json.loads(config_path.read_text())
    raw.pop("discrete_action_training_mode")
    raw.pop("ee_target_dataset_semantics")
    raw.pop("ee_delta_supervision_mode")
    raw.pop("gripper_target_representation")
    raw.pop("ee_supervision_source")
    config_path.write_text(json.dumps(raw))

    restored = PreTrainedConfig.from_pretrained(tmp_path)

    assert restored.discrete_action_training_mode == "continuous_flow"
    assert restored.ee_target_dataset_semantics == "joint_control_inactive_interpolated"
    assert restored.ee_delta_supervision_mode == "all"
    assert restored.gripper_target_representation == "continuous_position"
    assert restored.ee_supervision_source == "control_action"


def test_structured_metadata_records_temporal_decoder_contract() -> None:
    metadata = PI05Config(
        discrete_action_training_mode="structured_temporal",
        gripper_target_representation="binary_position",
        z1_action_representation="ee_delta",
        ee_delta_rotation_representation="rotvec",
    ).deployment_metadata()
    assert metadata["version"] == 10
    assert metadata["action"]["discrete_training_mode"] == "structured_temporal"
    assert metadata["action"]["ee_delta_rotation_representation"] == "rotvec"
    assert metadata["action"]["ee_target_dataset_semantics"] == "joint_control_inactive_interpolated"
    assert metadata["action"]["ee_delta_supervision_mode"] == "all"
    assert metadata["action"]["gripper_target_representation"] == "binary_position"
    assert metadata["action"]["discrete_temporal_structure"]["arm_mode"]["allowed_transitions"] == "all"
    assert metadata["action"]["boolean_decoding"]["output_values"]["gripper_target"] == {
        "normalized_negative": -1.0471976,
        "normalized_nonnegative": 0.0,
    }


@pytest.mark.parametrize("b2_representation", ["velocity", "pose_delta"])
@pytest.mark.parametrize("z1_representation", ["ee_delta", "ee_state_delta"])
def test_metadata_records_each_representation_and_its_runtime_anchor(
    b2_representation: str, z1_representation: str
) -> None:
    action = PI05Config(
        b2_action_representation=b2_representation,
        z1_action_representation=z1_representation,
    ).deployment_metadata()["action"]

    assert action["representation"] == b2_representation
    assert action["z1_representation"] == z1_representation
    assert action["b2_deployment_anchor"] == (
        "actual_world_pose_at_source_step" if b2_representation == "pose_delta" else None
    )
    assert action["ee_deployment_anchor"] == (
        "actual_ee_state_at_source_step"
        if z1_representation == "ee_state_delta"
        else "executed_ee_target_at_source_step"
    )
