"""CPU regressions for prefix-conditioned flow training and cached sampling."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies.pi05 import modeling_pi05 as pi05
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi_gemma import PiGemmaRMSNorm
from lerobot.policies.rtc.configuration_rtc import RTCConfig, TrainingRTCConfig
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def test_adarms_per_token_condition_matches_independent_tokens():
    torch.manual_seed(11)
    norm = PiGemmaRMSNorm(8, cond_dim=8)
    torch.nn.init.normal_(norm.dense.weight)
    x, cond = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
    output, gate = norm(x, cond)
    independent = [norm(x[:, k : k + 1], cond[:, k]) for k in range(4)]
    torch.testing.assert_close(output, torch.cat([item[0] for item in independent], dim=1))
    torch.testing.assert_close(gate, torch.cat([item[1] for item in independent], dim=1))
    global_output, global_gate = norm(x, cond[:, 0])
    repeated_output, repeated_gate = norm(x, cond[:, :1].expand_as(cond))
    torch.testing.assert_close(global_output, repeated_output)
    torch.testing.assert_close(global_gate.expand_as(repeated_gate), repeated_gate)


@pytest.fixture
def tiny_model(monkeypatch):
    """Use real Gemma attention/cache/AdaRMS paths with small local random weights."""
    from transformers import GemmaConfig, PaliGemmaConfig

    def small_paligemma():
        return PaliGemmaConfig(
            vision_config={
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "patch_size": 14,
            },
        )

    def small_gemma(**kwargs):
        kwargs["vocab_size"] = 128
        return GemmaConfig(**kwargs)

    original_paligemma = pi05.PaliGemmaForConditionalGenerationWithPiGemma

    def small_paligemma_model(config):
        config.projection_dim = 32
        config.vision_config.projection_dim = 32
        config.text_config.vocab_size = 128
        config.image_token_index = 127
        return original_paligemma(config=config)

    monkeypatch.setattr(pi05, "CONFIG_MAPPING", {"paligemma": small_paligemma, "gemma": small_gemma})
    monkeypatch.setattr(pi05, "get_gemma_config", lambda _: pi05.GemmaConfig(32, 1, 64, 8, 1, 4))
    monkeypatch.setattr(pi05, "PaliGemmaForConditionalGenerationWithPiGemma", small_paligemma_model)
    config = PI05Config(
        device="cpu",
        chunk_size=4,
        n_action_steps=4,
        max_action_dim=2,
        image_resolution=(28, 28),
        num_inference_steps=2,
        training_rtc_config=TrainingRTCConfig(enabled=True, simulated_delay=3),
        rtc_config=RTCConfig(mode="training"),
    )
    return pi05.PI05Pytorch(config)


def _inputs():
    return {
        "images": [torch.randn(2, 3, 28, 28)],
        "img_masks": [torch.ones(2, dtype=torch.bool)],
        "tokens": torch.ones(2, 2, dtype=torch.long),
        "masks": torch.ones(2, 2, dtype=torch.bool),
    }


def test_training_rtc_config_checkpoint_round_trip(tmp_path):
    config = PI05Config(
        training_rtc_config=TrainingRTCConfig(
            enabled=True,
            simulated_delay=8,
            delay_distribution="uniform",
        )
    )
    config.save_pretrained(tmp_path)

    restored = PreTrainedConfig.from_pretrained(tmp_path)

    assert restored.training_rtc_config == config.training_rtc_config


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
def test_real_forward_clean_prefix_and_backward(tiny_model, monkeypatch, checkpointing, precision):
    model = tiny_model.train()
    model.paligemma_with_expert.to_bfloat16_for_selected_params(precision)
    if checkpointing:
        model.gradient_checkpointing_enable()
    actions, noise = torch.randn(2, 4, 2), torch.randn(2, 4, 2)
    time = torch.tensor([[0.0, 0.0, 0.4, 0.4], [0.0, 0.7, 0.7, 0.7]])
    original_embed = model.embed_suffix
    observed = {}

    def capture(x, t):
        observed.update(x=x.detach().clone(), time=t.detach().clone())
        return original_embed(x, t)

    monkeypatch.setattr(model, "embed_suffix", capture)
    losses = model(**_inputs(), actions=actions, noise=noise, time=time)
    expected = time[..., None] * noise + (1 - time[..., None]) * actions
    torch.testing.assert_close(observed["x"], expected)
    assert torch.equal(observed["time"], time)
    assert torch.equal(losses[time == 0], torch.zeros_like(losses[time == 0]))
    assert losses.shape == actions.shape
    losses.sum().backward()
    assert model.action_out_proj.weight.grad is not None
    assert torch.isfinite(model.action_out_proj.weight.grad).all()
    adaptive_norm = model.paligemma_with_expert.gemma_expert.model.layers[0].input_layernorm
    assert torch.isfinite(adaptive_norm.dense.weight.grad).all()


def test_disabled_objective_preserves_ordinary_flow_and_rng(tiny_model):
    model = tiny_model.eval()
    model.config.training_rtc_config.enabled = False
    inputs = _inputs()
    actions, noise, time = torch.randn(2, 4, 2), torch.randn(2, 4, 2), torch.tensor([0.3, 0.7])
    rng = torch.random.get_rng_state()
    scalar = model(**inputs, actions=actions, noise=noise, time=time)
    assert torch.equal(torch.random.get_rng_state(), rng)
    per_token = model(**inputs, actions=actions, noise=noise, time=time[:, None].expand(-1, 4))
    torch.testing.assert_close(scalar, per_token)
    assert torch.count_nonzero(scalar[:, :2]) > 0


def test_cached_training_sampler_freezes_prefix_each_step(tiny_model, monkeypatch):
    model = tiny_model.eval()
    noise, target = torch.randn(2, 4, 2), torch.randn(2, 4, 2)
    prefix = torch.randn(2, 2, 2)
    delay = torch.tensor([2, 1])
    prefix_mask = torch.arange(4)[None, :] < delay[:, None]
    calls = []

    def constant_velocity(*, x_t, timestep, **kwargs):
        calls.append((x_t.clone(), timestep.clone()))
        return noise - target

    monkeypatch.setattr(model, "denoise_step", constant_velocity)
    model.rtc_processor = Mock()
    model.rtc_processor.is_debug_enabled.return_value = False
    result = model.sample_actions(
        **_inputs(), noise=noise, prev_chunk_left_over=prefix, inference_delay=delay
    )
    assert len(calls) == 2
    for x, time in calls:
        assert time.shape == (2, 4)
        assert (time[prefix_mask] == 0).all()
        torch.testing.assert_close(x[0, :2], prefix[0])
        torch.testing.assert_close(x[1, :1], prefix[1, :1])
    torch.testing.assert_close(result[~prefix_mask], target[~prefix_mask])
    assert torch.equal(result[0, :2], prefix[0])
    assert torch.equal(result[1, :1], prefix[1, :1])
    model.rtc_processor.denoise_step.assert_not_called()


def test_real_cached_sampler_zero_delay_matches_legacy_and_inference_dispatch(tiny_model):
    model = tiny_model.eval()
    inputs, noise = _inputs(), torch.randn(2, 4, 2)
    training = model.sample_actions(**inputs, noise=noise, inference_delay=0)
    model.config.rtc_config.enabled = False
    baseline = model.sample_actions(**inputs, noise=noise)
    torch.testing.assert_close(training, baseline, rtol=1e-5, atol=1e-6)
    model.config.rtc_config.enabled = True
    model.config.rtc_config.mode = "inference"
    model.rtc_processor = Mock()
    model.rtc_processor.is_debug_enabled.return_value = False
    model.rtc_processor.denoise_step.side_effect = lambda **kw: kw["original_denoise_step_partial"](kw["x_t"])
    inference = model.sample_actions(**inputs, noise=noise)
    assert model.rtc_processor.denoise_step.call_count == model.config.num_inference_steps
    torch.testing.assert_close(inference, baseline)


@pytest.mark.parametrize("schema", ["off", "uniform_valid", "always"])
@pytest.mark.parametrize("reduction", ["mean", "none"])
def test_policy_reduction_excludes_prefix_and_padding(monkeypatch, schema, reduction):
    policy = pi05.PI05Policy.__new__(pi05.PI05Policy)
    torch.nn.Module.__init__(policy)
    policy.config = PI05Config(
        device="cpu",
        chunk_size=4,
        n_action_steps=4,
        max_action_dim=3,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        action_loss_schema=schema,
        action_feature_names=["b2_vx", "b2_vy", "b2_omega_z"],
        action_masked_continuous_min_weight=0.2,
        training_rtc_config=TrainingRTCConfig(enabled=True, simulated_delay=3),
    )
    losses = torch.tensor([100.0, 4.0, 8.0, 200.0]).reshape(1, 4, 1).expand(1, 4, 3).requires_grad_()
    policy.model = SimpleNamespace(
        _zero_structured_discrete_channels=lambda x: x,
        sample_noise=lambda shape, device: torch.zeros(shape, device=device),
        sample_time=lambda size, device: torch.full((size,), 0.5, device=device),
        forward=Mock(return_value=losses),
    )
    monkeypatch.setattr(pi05, "sample_training_rtc_delay", lambda *a, **kw: torch.tensor([1]))
    monkeypatch.setattr(policy, "_mem_vit_window_plan", lambda *a, **kw: (1, None, None))
    monkeypatch.setattr(policy, "_preprocess_images", lambda *a, **kw: ([], [], None))
    monkeypatch.setattr(policy, "_prepare_state_history", lambda *a: None)
    batch = {
        ACTION: torch.zeros(1, 4, 3),
        f"{ACTION}_is_pad": torch.tensor([[False, False, False, True]]),
        pi05.EE_DELTA_VALID_KEY: torch.ones(1, 4, dtype=torch.bool),
        OBS_LANGUAGE_TOKENS: torch.ones(1, 2, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 2, dtype=torch.bool),
    }
    loss, info = policy(batch, reduction=reduction)
    assert loss.item() == pytest.approx(6.0)
    assert info["training_rtc_delay_mean"] == 1.0
    assert info["loss_per_dim"] == pytest.approx([6.0] * 3)
    assert policy.model.forward.call_args.args[6].tolist() == [[0.0, 0.5, 0.5, 0.5]]
    loss.sum().backward()
    assert (losses.grad[:, [0, 3]] == 0).all()
    assert (losses.grad[:, 1:3] > 0).all()
