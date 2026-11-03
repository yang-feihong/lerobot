import numpy as np
import pytest

from lerobot.utils.ee_frames import (
    EE_FRAME_CONTRACT, checkpoint_ee_frame, convert_ee_pose, convert_named_features, resume_ee_frame,
)


def test_roundtrip_and_tcp():
    pose = np.array([1., 0., 0., 0., 1., 0., .2, .3, .4])
    physical = convert_ee_pose(pose, "legacy", "physical")
    np.testing.assert_array_equal(physical, [1, 0, 0, 0, 0, -1, .2, .3, .4])
    np.testing.assert_array_equal(convert_ee_pose(physical, "physical", "legacy"), pose)
    np.testing.assert_array_equal(pose, [1, 0, 0, 0, 1, 0, .2, .3, .4])


def test_checkpoint_fallback_is_not_a_new_contract_default():
    assert checkpoint_ee_frame({}) == "legacy"
    assert checkpoint_ee_frame({"ee_frame_contract": EE_FRAME_CONTRACT, "model_ee_frame": "physical"}) == "physical"
    with pytest.raises((KeyError, ValueError)):
        checkpoint_ee_frame({"ee_frame_contract": EE_FRAME_CONTRACT})
    with pytest.raises(ValueError):
        checkpoint_ee_frame({"model_ee_frame": "physical"})


def test_incomplete_feature_block_rejected():
    with pytest.raises(ValueError):
        convert_named_features(np.zeros(1), ["height_invariant_ee_x"], "legacy", "physical")


def test_resume_never_changes_checkpoint_frame():
    assert resume_ee_frame(None, {}) == "legacy"
    assert resume_ee_frame("legacy", {}) == "legacy"
    with pytest.raises(ValueError, match="Resume cannot change"):
        resume_ee_frame("physical", {})
    explicit = {"version": 11, "ee_frame_contract": EE_FRAME_CONTRACT, "model_ee_frame": "physical"}
    assert resume_ee_frame(None, explicit) == "physical"
    with pytest.raises(ValueError, match="Resume cannot change"):
        resume_ee_frame("legacy", explicit)
    with pytest.raises(ValueError):
        checkpoint_ee_frame({"version": 11})


@pytest.mark.parametrize("frame", ["legacy", "physical"])
def test_new_pi05_metadata_records_choice(frame):
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    metadata = PI05Config(model_ee_frame=frame).deployment_metadata()
    assert metadata["version"] == 11
    assert checkpoint_ee_frame(metadata) == frame


def test_fresh_training_defaults_physical_without_changing_old_checkpoint_loading(tmp_path, monkeypatch):
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    policy = PI05Config(device="cpu", push_to_hub=False)
    assert checkpoint_ee_frame(policy.deployment_metadata()) == "legacy"
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="local/test"), policy=policy,
                              output_dir=tmp_path / "fresh")
    monkeypatch.setattr(cfg, "_resolve_pretrained_from_cli", lambda: None)
    cfg.validate()
    assert cfg.policy.model_ee_frame == "physical"
    assert checkpoint_ee_frame(cfg.policy.deployment_metadata()) == "physical"
