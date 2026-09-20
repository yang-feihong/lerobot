"""Exercise PEFT on the real PI05 forward with tiny CPU transformer configs."""

import re

import pytest
import torch

pytest.importorskip("peft")
pytest.importorskip("transformers")

from peft import LoraConfig, PeftModel, get_peft_model_state_dict
from peft.utils.other import ModulesToSaveWrapper

from lerobot.policies.pi05 import modeling_pi05
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@pytest.fixture
def tiny_policy_factory(monkeypatch):
    original_paligemma = modeling_pi05.PaliGemmaForConditionalGenerationWithPiGemma
    original_expert = modeling_pi05.PiGemmaForCausalLM

    def tiny_paligemma(config):
        config.text_config.vocab_size = 32
        config.vision_config.hidden_size = 16
        config.vision_config.projection_dim = 16
        config.vision_config.intermediate_size = 32
        config.vision_config.num_hidden_layers = 2
        config.vision_config.num_attention_heads = 4
        config.vision_config.patch_size = 4
        return original_paligemma(config)

    def tiny_expert(config):
        config.vocab_size = 32
        return original_expert(config)

    monkeypatch.setattr(modeling_pi05, "PaliGemmaForConditionalGenerationWithPiGemma", tiny_paligemma)
    monkeypatch.setattr(modeling_pi05, "PiGemmaForCausalLM", tiny_expert)
    monkeypatch.setattr(
        modeling_pi05, "get_gemma_config", lambda _: modeling_pi05.GemmaConfig(16, 2, 32, 8, 1, 2)
    )

    def make_policy(**overrides):
        config_kwargs = {
            "device": "cpu",
            "dtype": "float32",
            "pretrained_path": "tiny-test-base",
            "peft_train_active_modules_only": True,
            "image_resolution": (8, 8),
            "chunk_size": 2,
            "n_action_steps": 2,
            "max_action_dim": 4,
            "max_state_dim": 4,
            "mem_vit_enabled": True,
            "mem_vit_num_frames": 6,
            "mem_vit_temporal_every": 1,
            "state_action_encoding": "continuous",
            "state_num_frames": 2,
            "action_history_enabled": True,
        }
        config_kwargs.update(overrides)
        return modeling_pi05.PI05Policy(PI05Config(**config_kwargs))

    return make_policy


def _training_loss(policy):
    images = torch.randn(2, 6, 3, 8, 8) if policy.config.mem_vit_enabled else torch.randn(2, 3, 8, 8)
    return policy.model(
        images=[images],
        img_masks=[torch.ones(2, dtype=torch.bool)],
        tokens=torch.randint(0, 32, (2, 3)),
        masks=torch.ones(2, 3, dtype=torch.bool),
        actions=torch.randn(2, 2, 4),
        noise=torch.randn(2, 2, 4),
        time=torch.tensor([0.3, 0.7]),
        state=torch.randn(2, 2, 4),
        action_history=torch.randn(2, 1, 4),
        image_memory_masks=[torch.ones(2, 6, dtype=torch.bool)] if policy.config.mem_vit_enabled else None,
    ).mean()


def test_mem_finetune_config_preserves_old_default_and_validates_modes():
    assert PI05Config(device="cpu").mem_vit_finetune_mode == "full"
    with pytest.raises(ValueError, match="mem_vit_finetune_mode"):
        PI05Config(device="cpu", mem_vit_finetune_mode="invalid")


@pytest.mark.parametrize(
    "mem_enabled,mode,freeze,checkpointing",
    [
        (True, "full", False, False),
        (True, "lora", False, False),
        (True, "lora", False, True),
        (True, "frozen", False, True),
        (True, "full", True, False),
        (True, "lora", True, False),
        (False, "full", True, False),
        (False, "full", False, False),
    ],
)
def test_mem_peft_trainability_forward_backward(
    tiny_policy_factory, mem_enabled, mode, freeze, checkpointing
):
    policy = tiny_policy_factory(
        mem_vit_enabled=mem_enabled,
        mem_vit_finetune_mode=mode,
        freeze_vision_encoder=freeze,
        gradient_checkpointing=checkpointing,
    )
    wrapped = policy.wrap_with_peft(peft_cli_overrides={"r": 2, "lora_alpha": 2})
    wrapped.train()
    backbone = policy.model.paligemma_with_expert
    vision = backbone.paligemma.model.vision_tower
    frozen = freeze or (mem_enabled and mode == "frozen")
    vision_lora = not frozen and (not mem_enabled or mode == "lora")

    assert any("lora_" in name for name, _ in vision.named_parameters()) == vision_lora
    if frozen:
        assert not vision.training
        assert not any(param.requires_grad for param in vision.parameters())
    elif vision_lora:
        assert all(param.requires_grad == ("lora_" in name) for name, param in vision.named_parameters())
        assert hasattr(vision.vision_model.encoder.layers[0].self_attn.out_proj, "lora_A")
    else:
        assert all(param.requires_grad for param in vision.parameters())

    expert = backbone.gemma_expert
    assert isinstance(expert, ModulesToSaveWrapper)
    assert expert.model is expert.modules_to_save["default"].model
    assert not any(param.requires_grad for param in expert.original_module.parameters())
    assert all(param.requires_grad for param in expert.model.parameters())
    assert not expert.lm_head.weight.requires_grad

    loss = _training_loss(policy)
    assert torch.isfinite(loss)
    loss.backward()
    missing_gradients = [
        name for name, param in policy.named_parameters() if param.requires_grad and param.grad is None
    ]
    assert missing_gradients == []
    assert any(
        param.grad is not None for name, param in backbone.paligemma.named_parameters() if "lora_" in name
    )
    assert all(param.grad is None for param in expert.original_module.parameters())

    # The exact copy used by the handwritten joint forward is the one PEFT saves.
    active_weight = expert.model.layers[0].self_attn.q_proj.weight
    original_weight = expert.original_module.model.layers[0].self_attn.q_proj.weight
    original_before = original_weight.detach().clone()
    active_before = active_weight.detach().clone()
    torch.optim.SGD((param for param in policy.parameters() if param.requires_grad), lr=0.1).step()
    assert not torch.equal(active_weight, active_before)
    assert torch.equal(original_weight, original_before)
    adapter_state = get_peft_model_state_dict(wrapped, save_embedding_layers=False)
    saved_key = next(
        key for key in adapter_state if key.endswith("gemma_expert.model.layers.0.self_attn.q_proj.weight")
    )
    assert torch.equal(adapter_state[saved_key], active_weight)


@pytest.mark.parametrize("mode,freeze", [("frozen", False), ("lora", True)])
def test_explicit_peft_config_cannot_add_lora_to_frozen_vision(tiny_policy_factory, mode, freeze):
    policy = tiny_policy_factory(mem_vit_finetune_mode=mode, freeze_vision_encoder=freeze)
    peft_config = LoraConfig(
        r=2,
        target_modules=policy._get_default_peft_targets()["target_modules"],
        modules_to_save=policy._get_default_peft_targets()["modules_to_save"],
        exclude_modules={"unused_projection"},
    )
    policy.wrap_with_peft(peft_config=peft_config)
    assert re.fullmatch(peft_config.exclude_modules, "model.unused_projection")
    vision = policy.model.paligemma_with_expert.paligemma.model.vision_tower
    assert not any("lora_" in name for name, _ in vision.named_parameters())
    assert not any(param.requires_grad for param in vision.parameters())


def test_restoring_mem_lora_does_not_enable_base_or_original_weights(tiny_policy_factory):
    policy = tiny_policy_factory(mem_vit_finetune_mode="lora")
    policy.wrap_with_peft(peft_cli_overrides={"r": 2})
    before = {name for name, param in policy.named_parameters() if param.requires_grad}
    policy._enable_mem_vit_full_finetuning()  # Historical callers cannot undo the selected mode.
    policy._restore_peft_trainability()
    assert {name for name, param in policy.named_parameters() if param.requires_grad} == before


def test_legacy_full_mem_keeps_optimizer_parameter_groups(tiny_policy_factory):
    assert PI05Config(device="cpu").peft_train_active_modules_only is False
    policy = tiny_policy_factory(peft_train_active_modules_only=False)
    policy.wrap_with_peft(peft_cli_overrides={"r": 2})
    expert = policy.model.paligemma_with_expert.gemma_expert
    assert all(param.requires_grad for param in expert.parameters())


def test_mem_lora_rejects_custom_targets_that_skip_vision(tiny_policy_factory):
    policy = tiny_policy_factory(mem_vit_finetune_mode="lora")
    with pytest.raises(ValueError, match="requires trainable vision LoRA adapters"):
        policy.wrap_with_peft(peft_cli_overrides={"target_modules": r".*language_model\..*\.q_proj", "r": 2})


@pytest.mark.parametrize("mode", ["lora", "frozen", "full"])
def test_mem_peft_disk_roundtrip_preserves_updated_output_and_trainability(
    tiny_policy_factory, tmp_path, mode
):
    from lerobot.common.train_utils import _append_peft_base_weights

    # Reconstructing the same base models mirrors loading the same pretrained
    # PI05 and MEM checkpoints before applying the saved adapters.
    torch.manual_seed(7)
    policy = tiny_policy_factory(mem_vit_finetune_mode=mode, gradient_checkpointing=True)
    wrapped = policy.wrap_with_peft(peft_cli_overrides={"r": 2, "lora_alpha": 2})
    optimizer = torch.optim.AdamW((param for param in policy.parameters() if param.requires_grad), lr=1e-3)
    _training_loss(policy).backward()
    optimizer.step()
    expected_trainable = {name for name, param in policy.named_parameters() if param.requires_grad}
    wrapped.eval()
    with torch.no_grad():
        torch.manual_seed(11)
        expected_loss = _training_loss(policy)

    wrapped.save_pretrained(tmp_path, save_embedding_layers=False)
    if mode == "full":
        assert _append_peft_base_weights(tmp_path, wrapped.state_dict(), key_fragment=".vision_tower.") > 0

    torch.manual_seed(7)
    restored_policy = tiny_policy_factory(mem_vit_finetune_mode=mode, gradient_checkpointing=True)
    restored = PeftModel.from_pretrained(restored_policy, tmp_path, is_trainable=True, torch_device="cpu")
    restored_policy._restore_peft_trainability()

    assert {
        name for name, param in restored_policy.named_parameters() if param.requires_grad
    } == expected_trainable
    restored_vision = restored_policy.model.paligemma_with_expert.paligemma.model.vision_tower
    if mode != "full":
        assert all(
            not param.requires_grad
            for name, param in restored_vision.named_parameters()
            if "lora_" not in name
        )
    # Compare every tensor as well as the result of the real six-frame forward.
    expected_state = policy.state_dict()
    restored_state = restored_policy.state_dict()
    assert restored_state.keys() == expected_state.keys()
    for name, expected_tensor in expected_state.items():
        torch.testing.assert_close(restored_state[name], expected_tensor, rtol=0, atol=0, msg=name)
    restored.eval()
    with torch.no_grad():
        torch.manual_seed(11)
        torch.testing.assert_close(_training_loss(restored_policy), expected_loss, rtol=0, atol=0)
