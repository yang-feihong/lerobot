import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.image_frame_mask import mask_image_frames


def inputs(history=True, batch=64):
    shape = (batch, 6, 3, 4, 5) if history else (batch, 3, 4, 5)
    return [torch.full(shape, 0.25), torch.full(shape, 0.5)]


@pytest.mark.parametrize("probability,training", [(0, True), (1, False)])
def test_disabled_is_identity_and_does_not_consume_rng(probability, training):
    images = inputs()
    rng = torch.random.get_rng_state().clone()
    result = mask_image_frames(images, [], None, probability=probability, training=training)
    assert result is images
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("history", [False, True])
def test_full_probability_masks_exactly_one_camera_per_time_without_mutation(history):
    images = inputs(history)
    saved = [x.clone() for x in images]
    cameras = [torch.ones(64, dtype=torch.bool) for _ in images]
    result = mask_image_frames(images, cameras, None, probability=1, training=True)
    changed = [(x != y).flatten(-3).any(-1) for x, y in zip(result, images, strict=True)]
    assert torch.all(changed[0] ^ changed[1])
    assert all(c.any() for c in changed)
    for original, copy, augmented in zip(images, saved, result, strict=True):
        assert torch.equal(original, copy)
        assert augmented.dtype == original.dtype
        assert augmented.shape == original.shape
        assert augmented.min() >= -1 and augmented.max() <= 1


def test_missing_cameras_and_padded_history_do_not_get_masked():
    images = inputs(batch=2) + [torch.zeros(2, 6, 3, 4, 5)]
    cameras = [torch.ones(2, dtype=torch.bool), torch.tensor([True, False]), torch.zeros(2, dtype=torch.bool)]
    memory = [torch.tensor([[False, False, True, True, True, True]] * 2)] * 3
    result = mask_image_frames(images, cameras, memory, probability=1, training=True)
    assert torch.equal(result[0][:, :2], images[0][:, :2])
    assert torch.equal(result[1][:, :2], images[1][:, :2])
    assert torch.equal(result[0][1], images[0][1])  # sole available camera
    assert torch.equal(result[1][1], images[1][1])  # missing camera
    assert torch.equal(result[2], images[2])
    assert torch.equal(memory[0], torch.tensor([[False, False, True, True, True, True]] * 2))


def test_empirical_probability_and_reproducible_noise():
    images = inputs(batch=2048)
    cameras = [torch.ones(2048, dtype=torch.bool) for _ in images]
    torch.manual_seed(42)
    result = mask_image_frames(images, cameras, None, probability=0.3, training=True)
    torch.manual_seed(42)
    repeated = mask_image_frames(images, cameras, None, probability=0.3, training=True)
    changed = torch.stack([(x != y).flatten(-3).any(-1) for x, y in zip(result, images, strict=True)])
    assert not changed.all(0).any()
    assert changed.any(0).float().mean().item() == pytest.approx(0.3, abs=0.02)
    assert changed[0].float().mean().item() == pytest.approx(0.15, abs=0.02)
    assert changed[1].float().mean().item() == pytest.approx(0.15, abs=0.02)
    assert all(torch.equal(a, b) for a, b in zip(result, repeated, strict=True))


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_config_rejects_invalid_probabilities(value):
    with pytest.raises(ValueError, match="image_frame_mask_probability"):
        PI05Config(device="cpu", image_frame_mask_probability=value)


def test_config_default_and_serialization(tmp_path):
    from lerobot.configs import PreTrainedConfig

    assert PI05Config(device="cpu").image_frame_mask_probability == 0
    config = PI05Config(device="cpu", image_frame_mask_probability=0.3)
    config.save_pretrained(tmp_path)
    assert PreTrainedConfig.from_pretrained(tmp_path).image_frame_mask_probability == 0.3
    saved = tmp_path / "config.json"
    legacy = json.loads(saved.read_text())
    legacy.pop("image_frame_mask_probability")
    saved.write_text(json.dumps(legacy))
    assert PreTrainedConfig.from_pretrained(tmp_path).image_frame_mask_probability == 0


def test_augmentation_is_wired_only_to_training_loss_forward():
    source = Path(__file__).parents[3] / "src/lerobot/policies/pi05/modeling_pi05.py"
    tree = ast.parse(source.read_text())
    policy = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PI05Policy")
    owners = []
    for method in policy.body:
        if not isinstance(method, ast.FunctionDef):
            continue
        for node in ast.walk(method):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "mask_image_frames":
                owners.append(method.name)
                assert ast.unparse(next(k.value for k in node.keywords if k.arg == "training")) == "self.training"
    assert owners == ["forward"]


@pytest.mark.parametrize("training", [True, False])
def test_policy_forward_corrupts_only_training_images_not_labels(monkeypatch, training):
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.pi05 import modeling_pi05 as pi05
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    policy = pi05.PI05Policy.__new__(pi05.PI05Policy)
    torch.nn.Module.__init__(policy)
    policy.config = PI05Config(
        device="cpu", chunk_size=4, n_action_steps=4, max_action_dim=3,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        action_feature_names=["b2_vx", "b2_vy", "b2_omega_z"],
        action_loss_schema="off", action_predict_reset_preset=False,
        image_frame_mask_probability=1,
    )
    policy.model = SimpleNamespace(
        _zero_structured_discrete_channels=lambda x: x,
        sample_noise=lambda shape, device: torch.zeros(shape, device=device),
        sample_time=lambda size, device: torch.full((size,), 0.5, device=device),
        forward=Mock(return_value=torch.ones(1, 4, 3)),
    )
    images = inputs(batch=1)
    camera_masks = [torch.ones(1, dtype=torch.bool) for _ in images]
    monkeypatch.setattr(policy, "_mem_vit_window_plan", lambda *a, **kw: (6, None, None))
    monkeypatch.setattr(policy, "_preprocess_images", lambda *a, **kw: (images, camera_masks, None))
    monkeypatch.setattr(policy, "_prepare_state_history", lambda *a, **kw: None)
    actions = torch.ones(1, 4, 3)
    monkeypatch.setattr(policy, "prepare_action", lambda batch: batch[ACTION])
    batch = {ACTION: actions, OBS_LANGUAGE_TOKENS: torch.ones(1, 2, dtype=torch.long),
             OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 2, dtype=torch.bool)}
    policy.train(training)
    policy(batch)
    call = policy.model.forward.call_args
    actual = call.args[0]
    assert call.args[4] is actions
    assert torch.equal(batch[ACTION], torch.ones(1, 4, 3))
    if training:
        changed = [(a != b).flatten(-3).any(-1) for a, b in zip(actual, images, strict=True)]
        assert (changed[0] ^ changed[1]).all()
    else:
        assert actual is images
