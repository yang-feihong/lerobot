"""Input-only pauses must never modify the observation anchor or future labels."""

from pathlib import Path

import pytest
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.action_history import action_history_schema
from lerobot.policies.pi05.b2_action_transform import CONTROL_EXTENDED_DATASET_ACTION_NAMES
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.history_pause import augment_history_pause


def config(**kwargs):
    from lerobot.policies.pi05.b2_action_transform import DATASET_ACTION_NAMES
    kwargs.setdefault("action_history_feature_names", [name for name in DATASET_ACTION_NAMES[:-1]
        if name not in {"arm_teleop_inactive", "arm_reset"}])
    return PI05Config(
        device="cpu", state_action_encoding="continuous", action_history_enabled=True,
        state_num_frames=4, chunk_size=4, n_action_steps=4,
        history_pause_max_steps=5, **kwargs,
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("representation", ["ee_delta", "ee_state_delta"])
def test_shift_hold_padding_and_future_invariance(monkeypatch, reverse, representation):
    source = list(CONTROL_EXTENDED_DATASET_ACTION_NAMES) + ["extra_command"]
    if reverse:
        source.reverse()
    cfg = config(dataset_action_feature_names=source, z1_action_representation=representation)
    target_length = 4 + int(representation == "ee_delta")
    action = torch.arange(4 * (3 + target_length) * len(source), dtype=torch.float32).reshape(
        4, 3 + target_length, len(source)
    )
    state = torch.arange(4 * 4 * 6, dtype=torch.float32).reshape(4, 4, 6)
    batch = {
        "observation.state": state,
        "action": action,
        "observation.state_is_pad": torch.tensor([[True, False, False, False]] * 4),
        "action_is_pad": torch.zeros(4, 3 + target_length, dtype=torch.bool),
        "observation.images.base": torch.randn(4, 6, 3, 2, 2),
        "reset_preset_target": torch.arange(4 * 4).reshape(4, 4),
        "reset_preset_mask": torch.ones(4, 4, dtype=torch.bool),
    }
    batch["action_is_pad"][2, 2] = True  # The last historical command is unknown.
    batch["action_is_pad"][1, -1] = True
    original = {key: value.clone() for key, value in batch.items()}
    shifts = torch.tensor([0, 1, 3, 5])
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: shifts)
    result = augment_history_pause(batch, cfg)
    names, columns, _ = action_history_schema(cfg)
    for row, shift in enumerate(shifts.tolist()):
        for slot in range(4):
            torch.testing.assert_close(result["observation.state"][row, slot], state[row, min(slot + shift, 3)])
        for slot in range(3):
            inserted = slot + shift >= 3
            for name, column in zip(names, columns, strict=True):
                expected = action[row, min(slot + shift, 2), column]
                if inserted and name in {"b2_vx", "b2_vy", "b2_omega_z", "arm_reset"}:
                    expected = torch.tensor(0.0)
                elif inserted and name == "arm_teleop_inactive":
                    expected = torch.tensor(1.0)
                torch.testing.assert_close(result["action"][row, slot, column], expected)
            assert result["action_is_pad"][row, slot] == batch["action_is_pad"][row, min(slot + shift, 2)]
    assert result["action_is_pad"][2, :3].all()
    torch.testing.assert_close(result["observation.state"][:, -1], state[:, -1], rtol=0, atol=0)
    torch.testing.assert_close(result["action"][:, 3:], action[:, 3:], rtol=0, atol=0)
    assert torch.equal(result["action_is_pad"][:, 3:], batch["action_is_pad"][:, 3:])
    for key in ("observation.images.base", "reset_preset_target", "reset_preset_mask"):
        assert result[key] is batch[key]
    for key in batch:
        assert torch.equal(batch[key], original[key]), key


def test_zero_is_identity_and_does_not_consume_rng():
    cfg = PI05Config(device="cpu")
    batch = {"arbitrary": object()}
    before = torch.get_rng_state()
    assert augment_history_pause(batch, cfg) is batch
    assert torch.equal(before, torch.get_rng_state())


def test_random_shifts_are_per_sample_bounded_and_reproducible():
    cfg = config()
    batch = {
        "observation.state": torch.arange(4.0)[None, :, None].expand(128, -1, -1),
        "action": torch.zeros(128, 8, len(CONTROL_EXTENDED_DATASET_ACTION_NAMES)),
    }
    torch.manual_seed(29)
    first = augment_history_pause(batch, cfg)
    torch.manual_seed(29)
    second = augment_history_pause(batch, cfg)
    assert torch.equal(first["observation.state"], second["observation.state"])
    starts = first["observation.state"][:, 0, 0]
    assert set(starts.tolist()) == {0, 1, 2, 3}
    assert torch.equal(first["observation.state"][:, -1], batch["observation.state"][:, -1])


def test_subset_command_schema_is_independent_of_output(monkeypatch):
    cfg = config(
        dataset_action_feature_names=["gripper_target", "b2_vx", "new_control"],
        action_history_feature_names=["new_control", "b2_vx", "gripper_target"],
        action_predict_ee_pose=False, action_predict_task_complete=False,
    )
    batch = {
        "observation.state": torch.zeros(1, 4, 2),
        "action": torch.arange(24.0).reshape(1, 8, 3),
    }
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor([5]))
    result = augment_history_pause(batch, cfg)
    assert torch.equal(result["action"][0, :3, 1], torch.zeros(3))
    for column in (0, 2):
        assert torch.equal(result["action"][0, :3, column], batch["action"][0, 2, column].expand(3))
    assert torch.equal(result["action"][:, 3:], batch["action"][:, 3:])


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_invalid_maximum_rejected(value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        PI05Config(device="cpu", history_pause_max_steps=value)


def test_positive_requires_command_history():
    with pytest.raises(ValueError, match="requires action_history_enabled"):
        PI05Config(device="cpu", history_pause_max_steps=1)


def test_saved_config_roundtrip(tmp_path):
    cfg = config()
    cfg.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.history_pause_max_steps == 5


def test_training_only_call_before_normalization():
    source = (Path(__file__).parents[3] / "src/lerobot/scripts/lerobot_train.py").read_text()
    assert source.count("batch = augment_history_pause(batch, cfg.policy)") == 1
    call = source.index("batch = augment_history_pause(batch, cfg.policy)")
    assert call < source.index("batch = preprocessor(batch)", call)
    assert "augment_history_pause" not in source[source.index("eval_batch = preprocessor(eval_batch)"):]
