"""21D executed history: past provenance, modes, target masking and normalization."""

import math

import pytest
import torch

from lerobot.datasets.reset_presets import ResetPresetCatalog
from lerobot.policies.pi05.action_history import RESET_HISTORY_KEY, RESET_TARGET_NAMES, action_history_schema
from lerobot.policies.pi05.b2_action_transform import CONTROL_EXTENDED_DATASET_ACTION_NAMES
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.history_pause import augment_history_pause
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05.processor_pi05 import _action_history_steps
from lerobot.types import TransitionKey


def fixture(reverse=False):
    source = list(CONTROL_EXTENDED_DATASET_ACTION_NAMES)
    if reverse:
        source.reverse()
    cfg = PI05Config(device="cpu", action_history_enabled=True, state_action_encoding="continuous",
                     state_num_frames=6, state_history_frame_stride=2, chunk_size=4, n_action_steps=4,
                     z1_action_representation="ee_state_delta", dataset_action_feature_names=source)
    catalog = ResetPresetCatalog.__new__(ResetPresetCatalog)
    catalog.joint_targets = torch.tensor([entry["joint_target"] for entry in cfg.reset_preset_catalog])
    catalog.episodes = {7: [(94, 98, 0)]}
    catalog.unresolved = {7: [(98, 100)]}
    catalog.unknown_target_policy = "mask_audited_unknown"
    raw = torch.zeros(1, 9, len(source))
    raw[:, 1, source.index("arm_teleop_inactive")] = 1
    raw[:, 2:5, source.index("arm_reset")] = 1
    # Deliberately useless EE placeholders in hold/reset must never reach the model.
    for name in source:
        if name.startswith("control_height_invariant_ee_"):
            raw[..., source.index(name)] = .8
    batch = {"action": raw, "action_is_pad": torch.zeros(1, 9, dtype=torch.bool),
             "observation.state": torch.zeros(1, 6, 58),
             "episode_index": torch.tensor([7]), "frame_index": torch.tensor([100])}
    return cfg, catalog, batch


@pytest.mark.parametrize("reverse", [False, True])
def test_training_target_lookup_and_full_processor_use_strictly_past_frames(reverse):
    cfg, catalog, batch = fixture(reverse)
    original = batch["action"].clone()
    catalog.attach_history(batch, cfg)
    assert batch[RESET_HISTORY_KEY + "_is_pad"].tolist() == [[True, True, False, False, True]]
    torch.testing.assert_close(batch[RESET_HISTORY_KEY][0, 2:4], catalog.joint_targets[0].expand(2, 6))
    stats = {"action": {"q01": -torch.ones(25), "q99": torch.ones(25)}}
    split, normalizer = _action_history_steps(cfg, stats)
    result = split({TransitionKey.ACTION: batch["action"],
        TransitionKey.OBSERVATION: {RESET_HISTORY_KEY: batch[RESET_HISTORY_KEY]},
        TransitionKey.COMPLEMENTARY_DATA: {RESET_HISTORY_KEY + "_is_pad": batch[RESET_HISTORY_KEY + "_is_pad"]}})
    assert torch.equal(result[TransitionKey.ACTION], original[:, 5:])
    result = normalizer(result)
    prepared_batch = {**result[TransitionKey.OBSERVATION], **result[TransitionKey.COMPLEMENTARY_DATA]}
    policy = PI05Policy.__new__(PI05Policy)
    policy.config = cfg
    history = policy._prepare_action_history(prepared_batch, 5)
    names = action_history_schema(cfg)[0]
    assert len(names) == 21
    ee = [i for i, name in enumerate(names) if name.startswith("height_invariant_ee_")]
    qcols = [names.index(name) for name in RESET_TARGET_NAMES]
    assert not history[:, 1:, ee].any()
    assert not history[:, [0, 1, 4], :][:, :, qcols].any()
    torch.testing.assert_close(history[0, 2, qcols], catalog.joint_targets[0] / math.pi)
    assert history[0, 1, names.index("arm_teleop_inactive")] == 1
    assert history[0, 1, names.index("arm_reset")] == -1
    assert history[0, 2, names.index("arm_reset")] == 1
    assert torch.equal(batch["action"], original)


def test_unknown_is_masked_but_unaudited_missing_target_is_an_error():
    cfg, catalog, batch = fixture()
    catalog.unresolved = {}
    with pytest.raises(ValueError, match="Historical reset has no proven target"):
        catalog.attach_history(batch, cfg)


def test_pause_moves_targets_with_commands_and_invalidates_synthetic_hold(monkeypatch):
    cfg, catalog, batch = fixture()
    cfg.history_pause_max_steps = 2
    catalog.attach_history(batch, cfg)
    original = batch["action"][:, 5:].clone()
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor([2]))
    result = augment_history_pause(batch, cfg)
    assert result[RESET_HISTORY_KEY + "_is_pad"].tolist() == [[False, False, True, True, True]]
    torch.testing.assert_close(result[RESET_HISTORY_KEY][0, :2], catalog.joint_targets[0].expand(2, 6))
    assert result["action"][0, 3:5, 3].eq(1).all()
    assert result["action"][0, 3:5, 4].eq(0).all()
    assert torch.equal(result["action"][:, 5:], original)


@pytest.mark.parametrize("predict_arm", [True, False])
def test_full_preprocessor_keeps_target_history_and_mask_aligned(monkeypatch, predict_arm):
    from lerobot.policies.pi05 import processor_pi05
    from lerobot.policies.pi05.b2_action_transform import action_schema_kwargs, b2_pose_delta_action_names
    from lerobot.processor.pipeline import IdentityProcessorStep

    cfg, catalog, batch = fixture()
    cfg.action_predict_ee_pose = predict_arm
    cfg.action_predict_arm_reset = predict_arm
    cfg.action_predict_arm_teleop_inactive = predict_arm
    monkeypatch.setattr(processor_pi05, "TokenizerProcessorStep", lambda **kwargs: IdentityProcessorStep())
    cfg.io_schema_resolved = True
    cfg.action_dt_seconds = .02
    cfg.state_feature_indices = list(range(10)) + list(range(49, 58))
    cfg.ee_state_anchor_indices = list(range(49, 58))
    cfg.action_feature_names = b2_pose_delta_action_names(
        list(CONTROL_EXTENDED_DATASET_ACTION_NAMES), **action_schema_kwargs(cfg))
    dim = len(cfg.action_feature_names)
    batch["task"] = ["Open the door."]
    batch["observation.state"][..., [49, 53]] = 1
    # Supply a valid control rotation to the future action representation step.
    batch["action"][..., 16:22] = torch.tensor([1., 0., 0., 0., 1., 0.])
    catalog.attach_history(batch, cfg)
    stats = {key: {"q01": -torch.ones(size), "q99": torch.ones(size)}
             for key, size in [("action", 25), ("observation.state", 58)]}
    pre, _ = processor_pi05.make_pi05_pre_post_processors(cfg, stats,
        {"q01": -torch.ones(dim), "q99": torch.ones(dim)})
    prepared = pre(batch)
    assert prepared["observation.action_history"].shape == (1, 5, 21)
    policy = PI05Policy.__new__(PI05Policy)
    policy.config = cfg
    history = policy._prepare_action_history(prepared, 5)
    torch.testing.assert_close(history[0, 2, 15:], catalog.joint_targets[0] / math.pi)
    assert not history[0, 4, 15:].any()
    assert prepared["action"].shape == (1, 4, dim)


def test_v2_config_roundtrip_and_legacy_preservation(tmp_path):
    from lerobot.configs.policies import PreTrainedConfig

    cfg, _, _ = fixture()
    cfg.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.action_history_feature_names == cfg.action_history_feature_names
    assert restored.action_history_ee_semantics == "mode_reset_target_v2"
    assert len(restored.action_history_feature_names) == 21
