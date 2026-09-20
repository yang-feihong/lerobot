"""Additional behavioral checks for the unpushed MEM LoRA commit."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("peft")

from peft import PeftModel

from lerobot.common.train_utils import load_training_state, save_checkpoint
from tests.policies.pi0_pi05.test_pi05_mem_peft import (
    _training_loss,
    tiny_policy_factory as _tiny_policy_fixture,
)


@pytest.fixture
def tiny_policy_factory(monkeypatch):
    return _tiny_policy_fixture.__wrapped__(monkeypatch)


@pytest.mark.parametrize("mode", ["full", "lora", "frozen"])
@pytest.mark.parametrize("gathered", [False, True])
@pytest.mark.parametrize("active_only", [False, True])
def test_checkpoint_resume_matches_next_optimizer_step(
    tiny_policy_factory, tmp_path, mode, gathered, active_only
):
    base_path = tmp_path / "base"
    base_path.mkdir()

    def create():
        torch.manual_seed(7)
        return tiny_policy_factory(
            mem_vit_finetune_mode=mode,
            peft_train_active_modules_only=active_only,
            gradient_checkpointing=True,
            pretrained_path=str(base_path),
        )

    def step(policy, optimizer):
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(31)
        loss = _training_loss(policy)
        loss.backward()
        optimizer.step()
        return loss.detach()

    policy = create()
    wrapped = policy.wrap_with_peft(peft_cli_overrides={"r": 2, "lora_alpha": 2})
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-3)
    step(policy, optimizer)
    save_checkpoint(
        tmp_path,
        1,
        SimpleNamespace(peft=object(), save_pretrained=Mock()),
        wrapped,
        optimizer,
        model_state_dict=wrapped.state_dict() if gathered else None,
    )
    expected_loss = step(policy, optimizer)

    restored_policy = create()
    restored = PeftModel.from_pretrained(restored_policy, tmp_path / "pretrained_model", is_trainable=True)
    restored_policy._restore_peft_trainability()
    restored.train()
    restored_optimizer = torch.optim.AdamW(restored_policy.get_optim_params(), lr=1e-3)
    saved_step, _, _ = load_training_state(tmp_path, restored_optimizer, None)
    assert saved_step == 1
    torch.testing.assert_close(step(restored_policy, restored_optimizer), expected_loss, rtol=0, atol=0)
    for name, expected in policy.state_dict().items():
        torch.testing.assert_close(restored_policy.state_dict()[name], expected, rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("dropout", [0.0, 0.2])
@pytest.mark.parametrize("target", ["vision", "language", "both"])
def test_lora_checkpointing_preserves_gradients(tiny_policy_factory, dropout, target):
    """Recomputation must use the same stochastic forward as the original loss."""
    gradients = []
    losses = []
    rng_states = []
    for checkpointing in (False, True):
        torch.manual_seed(7)
        policy = tiny_policy_factory(
            mem_vit_finetune_mode="frozen" if target == "language" else "lora",
            gradient_checkpointing=checkpointing,
        )
        target_modules = {
            "vision": r".*vision_tower\..*\.(q_proj|v_proj)",
            "language": r".*language_model\..*\.(q_proj|v_proj)",
            "both": policy._get_default_peft_targets()["target_modules"],
        }[target]
        policy.wrap_with_peft(
            peft_cli_overrides={
                "r": 2,
                "lora_alpha": 2,
                "lora_dropout": dropout,
                "target_modules": target_modules,
            }
        )
        # Exercise nonzero adapters as in an ongoing run, not only zero-initialized LoRA B.
        with torch.no_grad():
            for name, parameter in policy.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(std=0.05)
        torch.manual_seed(31)
        loss = _training_loss(policy)
        losses.append(loss.detach())
        loss.backward()
        rng_states.append(torch.get_rng_state())
        gradients.append(
            {name: p.grad.clone() for name, p in policy.named_parameters() if p.grad is not None}
        )
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
    assert gradients[0].keys() == gradients[1].keys()
    for name, expected in gradients[0].items():
        torch.testing.assert_close(
            gradients[1][name],
            expected,
            rtol=1e-5,
            atol=1e-7,
            msg=lambda detail, name=name: f"{name}\n{detail}",
        )
    # Backward recomputation must not consume random numbers from the next training step.
    torch.testing.assert_close(rng_states[0], rng_states[1], rtol=0, atol=0)
