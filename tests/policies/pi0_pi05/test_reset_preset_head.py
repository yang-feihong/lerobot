import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.reset_preset_head import ResetPresetHead, reset_preset_loss
from lerobot.utils.reset_presets import default_reset_presets, validate_selector_checkpoint


def test_candidate_expansion_and_permutation_do_not_resize_parameters():
    catalog = default_reset_presets()
    head = ResetPresetHead(8, catalog)
    features = torch.randn(2, 5, 8)
    logits = head(features)
    extended = copy.deepcopy(catalog) + [{"name": "future", "joint_target": [0.1] * 6}]
    other = ResetPresetHead(8, extended)
    other.load_state_dict(head.state_dict(), strict=True)
    assert sum(p.numel() for p in head.parameters()) == sum(p.numel() for p in other.parameters())
    torch.testing.assert_close(other(features)[..., :3], logits)
    permuted = ResetPresetHead(8, catalog[::-1])
    permuted.load_state_dict(head.state_dict(), strict=True)
    torch.testing.assert_close(permuted(features), logits.flip(-1))


def test_only_valid_nonpadded_reset_frames_contribute_ce():
    logits = torch.randn(2, 4, 3, requires_grad=True)
    labels = torch.tensor([[0, 1, -1, -1], [2, -1, -1, -1]])
    valid = labels >= 0
    pad = torch.zeros_like(valid)
    pad[0, 1] = True
    loss = reset_preset_loss(logits, labels, valid, pad)
    expected = torch.nn.functional.cross_entropy(logits[[0, 1], [0, 0]], torch.tensor([0, 2]))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert not logits.grad[~(valid & ~pad)].any()
    assert logits.grad[valid & ~pad].abs().sum() > 0


def test_reset_free_batch_has_differentiable_zero():
    logits = torch.randn(2, 4, 3, requires_grad=True)
    loss = reset_preset_loss(logits, torch.full((2, 4), -1), torch.zeros(2, 4, dtype=torch.bool))
    assert loss.item() == 0
    loss.backward()
    assert not logits.grad.any()


def test_invalid_target_rejected_and_none_reduction_supported():
    logits = torch.randn(2, 4, 3)
    labels, valid = torch.zeros(2, 4, dtype=torch.long), torch.ones(2, 4, dtype=torch.bool)
    assert reset_preset_loss(logits, labels, valid, reduction="none").shape == (2,)
    labels[0, 0] = 3
    with pytest.raises(ValueError, match="outside"):
        reset_preset_loss(logits, labels, valid)


def test_legacy_config_has_no_selector_or_flow_dimension_change():
    config = PI05Config(device="cpu")
    assert config.action_predict_reset_preset is False
    assert [p["name"] for p in config.reset_preset_catalog] == ["front", "middle", "back"]
    assert PI05Config(device="cpu", action_predict_reset_preset=True).max_action_dim == config.max_action_dim
    with pytest.raises(ValueError, match="reset gate"):
        PI05Config(device="cpu", action_predict_arm_reset=False, action_predict_reset_preset=True)


def test_peft_selector_roundtrip_and_missing_weights_rejected(tmp_path):
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.proj = nn.Linear(8, 8)
            self.model.reset_preset_head = ResetPresetHead(1024, default_reset_presets())
            self.config = {"tie_word_embeddings": False}

    model = get_peft_model(
        Tiny(), LoraConfig(r=2, target_modules=["proj"], modules_to_save=["reset_preset_head"])
    )
    model.save_pretrained(tmp_path)
    validate_selector_checkpoint(tmp_path, True)
    with pytest.raises(ValueError, match="mismatch"):
        validate_selector_checkpoint(tmp_path, False)
    state = get_peft_model_state_dict(model)
    shapes = {
        k.split(".reset_preset_head.", 1)[1]: list(v.shape)
        for k, v in state.items()
        if ".reset_preset_head." in k
    }
    assert len(shapes) == 6
    restored = get_peft_model(
        Tiny(), LoraConfig(r=2, target_modules=["proj"], modules_to_save=["reset_preset_head"])
    )
    set_peft_model_state_dict(restored, state)
    features = torch.randn(1, 3, 1024)
    head = model.get_base_model().model.reset_preset_head
    restored_head = restored.get_base_model().model.reset_preset_head
    torch.testing.assert_close(head(features), restored_head(features))
    assert all(p.requires_grad for p in head.modules_to_save.default.parameters())
    empty = tmp_path / "missing"
    empty.mkdir()
    with pytest.raises(ValueError, match="requires"):
        validate_selector_checkpoint(empty, True)


@pytest.mark.parametrize("rtc_prefix", [False, True])
def test_policy_forward_adds_selector_loss_and_optimizer_tracks_head(monkeypatch, rtc_prefix):
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
    from lerobot.utils.reset_presets import PRESET_ID_KEY, PRESET_VALID_KEY

    class Core(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature = nn.Parameter(torch.randn(1, 3, 8))
            self.reset_preset_head = ResetPresetHead(8, default_reset_presets())

        def _zero_structured_discrete_channels(self, x):
            return x

        def sample_noise(self, shape, device):
            return torch.zeros(shape, device=device)

        def sample_time(self, size, device):
            return torch.ones(size, device=device)

        def forward(self, *args, return_action_features=False, **kwargs):
            loss = self.feature.square().mean().expand(1, 3, 13)
            return (loss, self.feature) if return_action_features else loss

    policy = PI05Policy.__new__(PI05Policy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        state_action_encoding="text",
        mem_vit_enabled=False,
        action_history_enabled=False,
        training_rtc_config=None,
        discrete_action_training_mode="continuous_flow",
        action_predict_reset_preset=True,
        reset_preset_loss_weight=1.0,
        output_features={"action": SimpleNamespace(shape=(13,))},
        action_loss_schema="uniform_valid",
        optimizer_lr=1e-4,
        new_module_optimizer_lr_multiplier=1.0,
    )
    policy.model = Core()
    if rtc_prefix:
        from lerobot.policies.pi05 import modeling_pi05
        policy.config.training_rtc_config = SimpleNamespace(enabled=True)
        monkeypatch.setattr(modeling_pi05, "sample_training_rtc_delay", lambda *a, **kw: torch.tensor([1]))
    policy._mem_vit_window_plan = lambda *a, **kw: (1, None, None)
    policy._preprocess_images = lambda *a, **kw: ([], [], None)
    policy._prepare_state_history = lambda *a, **kw: None
    policy.prepare_action = lambda batch: batch["action"]
    batch = {
        "action": torch.zeros(1, 3, 13),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 1),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 1),
        PRESET_ID_KEY: torch.tensor([[0, -1, 2]]),
        PRESET_VALID_KEY: torch.tensor([[True, False, True]]),
    }
    loss, info = policy.forward(batch)
    assert loss.item() > policy.model.feature.square().mean().item()
    assert info["reset_preset_loss"] > 0
    with torch.no_grad():
        expected = reset_preset_loss(
            policy.model.reset_preset_head(policy.model.feature),
            batch[PRESET_ID_KEY], batch[PRESET_VALID_KEY],
            torch.tensor([[True, False, False]]) if rtc_prefix else None,
        )
    assert info["reset_preset_loss"] == pytest.approx(expected.item())
    loss.backward()
    assert policy.model.reset_preset_head.query.weight.grad.abs().sum() > 0
    params = {id(p) for group in policy.get_optim_params() for p in group["params"]}
    assert all(id(p) in params for p in policy.model.reset_preset_head.parameters())


def test_real_tiny_pi05_peft_selector_training_and_resume(monkeypatch, tmp_path):
    from peft import PeftModel

    from tests.policies.pi0_pi05.test_pi05_mem_peft import tiny_policy_factory

    make = tiny_policy_factory.__wrapped__(monkeypatch)
    policy = make(action_predict_reset_preset=True, gradient_checkpointing=True)
    wrapped = policy.wrap_with_peft(peft_cli_overrides={"r": 2, "lora_alpha": 2})
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-3)
    losses, features = policy.model(
        images=[torch.randn(2, 6, 3, 8, 8)],
        img_masks=[torch.ones(2, dtype=torch.bool)],
        tokens=torch.randint(0, 32, (2, 3)),
        masks=torch.ones(2, 3, dtype=torch.bool),
        actions=torch.randn(2, 2, 4),
        noise=torch.randn(2, 2, 4),
        time=torch.tensor([0.3, 0.7]),
        state=torch.randn(2, 2, 4),
        action_history=torch.randn(2, 1, 4),
        image_memory_masks=[torch.ones(2, 6, dtype=torch.bool)],
        return_action_features=True,
    )
    logits = policy.model.reset_preset_head(features)
    loss = losses.mean() + reset_preset_loss(
        logits, torch.tensor([[0, 1], [2, 1]]), torch.ones(2, 2, dtype=torch.bool)
    )
    loss.backward()
    head = policy.model.reset_preset_head.modules_to_save.default
    assert head.query.weight.grad.abs().sum() > 0
    optimizer.step()
    wrapped.save_pretrained(tmp_path)
    validate_selector_checkpoint(tmp_path, True)
    fresh = make(action_predict_reset_preset=True)
    PeftModel.from_pretrained(fresh, tmp_path, is_trainable=True)
    fresh._restore_peft_trainability()
    probe = torch.randn(1, 2, 16)
    torch.testing.assert_close(fresh.model.reset_preset_head(probe), policy.model.reset_preset_head(probe))
