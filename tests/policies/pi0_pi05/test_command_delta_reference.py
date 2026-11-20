"""Command-relative labels must use exactly the history visible to the model."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from datasets import Dataset

from lerobot.policies.pi05.b2_action_transform import (
    CONTROL_EXTENDED_DATASET_ACTION_NAMES,
    DATASET_ACTION_NAMES,
    action_schema_kwargs,
    b2_execution_action_names,
)
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import (
    Pi05ActionRepresentationProcessorStep,
    _action_history_steps,
    _ee_reference_kwargs,
)
from lerobot.policies.pi05.transformed_action_stats import compute_transformed_action_stats
from lerobot.types import TransitionKey


def command_config(**overrides):
    values = dict(device="cpu", state_action_encoding="continuous", state_num_frames=3,
                  action_history_enabled=True, action_history_feature_names=list(DATASET_ACTION_NAMES[:-1]),
                  action_history_ee_semantics="active_only_v1", z1_action_representation="ee_state_delta",
                  ee_delta_rotation_representation="rotvec", ee_delta_reference="previous_command",
                  chunk_size=2, n_action_steps=2)
    values.update(overrides)
    return PI05Config(**values)


@pytest.mark.parametrize("options,match", [
    ({"action_history_enabled": False}, "requires action_history_enabled"),
    ({"z1_action_representation": "ee_delta"}, "requires ee_state_delta"),
    ({"action_history_feature_names": ["b2_vx"]}, "all 9 EE history features"),
    ({"action_predict_ee_pose": False}, "with EE output"),
])
def test_command_reference_rejects_missing_model_information(options, match):
    with pytest.raises(ValueError, match=match):
        command_config(**options)


@pytest.mark.parametrize("source", ["recorded_action", "control_action"])
@pytest.mark.parametrize("unavailable", [None, "startup", "inactive", "reset", "inactive_all", "reset_all", "pause"])
def test_training_uses_last_visible_command_not_future_label(source, unavailable, monkeypatch):
    config = command_config(ee_supervision_source=source)
    raw = torch.zeros(1, 4, 25)
    identity = torch.tensor([1., 0., 0., 0., 1., 0.])
    raw[..., 5:11] = raw[..., 16:22] = identity
    raw[0, :, 11] = torch.tensor([1., 2., 4., 5.])
    raw[0, :, 22] = torch.tensor([10., 20., 40., 50.])
    state = torch.zeros(1, 3, 58)
    state[..., 49:55] = identity
    state[..., 55] = 3.
    pad = torch.zeros(1, 4, dtype=torch.bool)
    if unavailable == "startup":
        pad[:, :2] = True
    elif unavailable in {"inactive", "reset"}:
        raw[:, 1, 3 if unavailable == "inactive" else 4] = 1
    elif unavailable in {"inactive_all", "reset_all"}:
        raw[:, :2, 3 if unavailable == "inactive_all" else 4] = 1
    transition = {TransitionKey.ACTION: raw, TransitionKey.OBSERVATION: {"observation.state": state},
                  TransitionKey.COMPLEMENTARY_DATA: {"action_is_pad": pad}}
    if unavailable == "pause":
        from lerobot.policies.pi05.history_pause import augment_history_pause

        config.history_pause_max_steps = 1
        monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.ones(1, dtype=torch.long))
        augmented = augment_history_pause(
            {"action": raw, "observation.state": state, "action_is_pad": pad}, config
        )
        transition[TransitionKey.ACTION] = augmented["action"]
        transition[TransitionKey.OBSERVATION]["observation.state"] = augmented["observation.state"]
    split, _ = _action_history_steps(config, None)
    transition = split(transition)
    step = Pi05ActionRepresentationProcessorStep(
        dt=.02, z1_representation="ee_state_delta", ee_delta_rotation_representation="rotvec",
        ee_supervision_source=source, state_history_length=3, ee_state_anchor_indices=tuple(range(49, 58)),
        **_ee_reference_kwargs(config),
    )
    result = step(transition)[TransitionKey.ACTION]
    future = torch.tensor([4., 5.]) if source == "recorded_action" else torch.tensor([40., 50.])
    anchor = 3.
    if unavailable in {None, "pause"}:
        anchor = 2. if source == "recorded_action" else 20.
    elif unavailable in {"inactive", "reset"}:
        anchor = 1. if source == "recorded_action" else 10.
    torch.testing.assert_close(result[0, :, 8], future - anchor)


@pytest.mark.parametrize("source", ["recorded_action", "control_action"])
@pytest.mark.parametrize("masked_tail", [False, True])
def test_exact_statistics_use_visible_history_and_startup_fallback(tmp_path, source, masked_tail):
    config = command_config(ee_supervision_source=source)
    config.io_schema_resolved = True
    config.action_dt_seconds = .02
    config.control_frequency_hz = 50.
    config.state_history_frame_stride = 2
    config.dataset_action_feature_names = list(CONTROL_EXTENDED_DATASET_ACTION_NAMES)
    config.ee_state_anchor_indices = list(range(49, 58))
    config.action_feature_names = b2_execution_action_names(list(DATASET_ACTION_NAMES), **action_schema_kwargs(config))
    actions = np.zeros((8, 25), dtype=np.float32)
    states = np.zeros((8, 58), dtype=np.float32)
    for columns in (slice(5, 11), slice(16, 22)):
        actions[:, columns] = [1, 0, 0, 0, 1, 0]
    states[:, 49:55] = [1, 0, 0, 0, 1, 0]
    actions[:, 11] = np.arange(8) * .1 + 1
    actions[:, 22] = np.arange(8) * .2 + 2
    if masked_tail:
        actions[2:5, 3] = 1.
    dataset = SimpleNamespace(
        hf_dataset=Dataset.from_dict({"action": actions.tolist(), "observation.state": states.tolist(),
                                     "frame_index": list(range(8)), "episode_index": [0] * 8}),
        meta=SimpleNamespace(fps=50., features={"action": {"names": config.dataset_action_feature_names},
                                               "observation.state": {"shape": [58]}}),
        root=tmp_path, repo_id="test/commands", num_frames=8, num_episodes=1,
    )
    payload = compute_transformed_action_stats(dataset, config)
    column = 11 if source == "recorded_action" else 22
    expected = []
    for start in range(7):
        visible = [step for step in (start - 4, start - 2)
                   if step >= 0 and (actions[step, 3:5] < .5).all()]
        anchor = actions[visible[-1], column] if visible else 0.
        expected.extend(actions[start:start + 2, column] - anchor)
    index = config.action_feature_names.index("height_invariant_ee_delta_x")
    assert payload["stats"]["action"]["mean"][index] == pytest.approx(np.mean(expected), abs=1e-6)
    assert payload["schema"]["ee_delta_reference"] == "previous_command"
    assert payload["schema"]["ee_reference_history_stride"] == 2
    assert payload["schema"]["ee_reference_history_length"] == 2
    assert payload["schema"]["ee_reference_selection"] == "last_valid_visible_v1"
    from lerobot.policies.pi05.transformed_action_stats import validate_transformed_action_stats

    validate_transformed_action_stats(payload, dataset, config)
    for field in ("ee_reference_selection", "ee_reference_history_length"):
        value = payload["schema"].pop(field)
        with pytest.raises(ValueError, match="reference history selection"):
            validate_transformed_action_stats(payload, dataset, config)
        payload["schema"][field] = value


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_anchor_selects_each_batch_history_without_using_masked_values(device):
    from lerobot.policies.pi05.action_history import command_reference_anchor

    state = torch.zeros(2, 3, 9, device=device)
    commands = torch.arange(2 * 3 * 4 * 9, device=device).reshape(2, 3, 4, 9).float()
    valid = torch.tensor([[[True, True, False, False], [False, False, False, False],
                           [False, False, False, True]]] * 2, device=device)
    commands[~valid] = float("nan")
    result = command_reference_anchor(state, commands, valid)
    torch.testing.assert_close(result[:, 0], commands[:, 0, 1])
    torch.testing.assert_close(result[:, 1], state[:, 1])
    torch.testing.assert_close(result[:, 2], commands[:, 2, 3])


def test_reference_choice_survives_config_roundtrip_and_metadata(tmp_path):
    from lerobot.configs.policies import PreTrainedConfig

    config = command_config()
    config._save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.ee_delta_reference == "previous_command"
    action = restored.deployment_metadata()["action"]
    assert action["ee_deployment_anchor"] == "last_visible_history_ee_command_or_measured_state"
