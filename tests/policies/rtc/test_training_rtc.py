"""Training RTC's time convention, supported delays, and checkpoint contract."""

import json

import pytest
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.rtc.configuration_rtc import RTCConfig, TrainingRTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.rtc.training_rtc import (
    make_training_rtc_time,
    prepare_training_rtc_prefix,
    sample_training_rtc_delay,
)


@pytest.mark.parametrize("distribution", ["exponential", "uniform"])
def test_sample_delay_has_reference_support_and_distribution(distribution):
    torch.manual_seed(251)
    config = TrainingRTCConfig(enabled=True, simulated_delay=5, delay_distribution=distribution)
    delays = sample_training_rtc_delay(config, 20000, "cpu")
    counts = delays.bincount(minlength=5).float() / len(delays)
    expected = torch.full((5,), 0.2) if distribution == "uniform" else (-torch.arange(5).float()).softmax(0)
    assert delays.dtype == torch.long
    assert delays.min() == 0
    assert delays.max() == 4
    torch.testing.assert_close(counts, expected, atol=0.012, rtol=0)


def test_delay_preserves_supervised_suffix_near_episode_end(monkeypatch):
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor([4, 4, 4, 4]))
    is_pad = torch.tensor(
        [[False] * 5, [False, False, True, True, True], [False, True, True, True, True], [True] * 5]
    )
    delays = sample_training_rtc_delay(
        TrainingRTCConfig(delay_distribution="uniform"), 4, "cpu", action_is_pad=is_pad
    )
    assert delays.tolist() == [4, 1, 0, 0]


def test_per_token_time_marks_clean_prefix_without_changing_suffix():
    times, mask = make_training_rtc_time(torch.tensor([0.2, 0.7]), torch.tensor([0, 2]), 4)
    torch.testing.assert_close(times, torch.tensor([[0.2] * 4, [0, 0, 0.7, 0.7]]))
    assert mask.tolist() == [[False] * 4, [True, True, False, False]]


def test_prefix_broadcasts_batch_and_pads_action_dimensions():
    prefix = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    content, mask = prepare_training_rtc_prefix(
        torch.randn(2, 5, 4), prefix, torch.tensor([1, 3]), TrainingRTCConfig()
    )
    assert mask.tolist() == [[True, False, False, False, False], [True, True, True, False, False]]
    torch.testing.assert_close(content[0, :3, :2], prefix)
    torch.testing.assert_close(content[0], content[1])
    assert not content[:, 3:].any()
    assert not content[:, :, 2:].any()


def test_first_chunk_uses_no_prefix():
    content, mask = prepare_training_rtc_prefix(torch.randn(2, 5, 4), None, None, TrainingRTCConfig())
    assert not content.any()
    assert not mask.any()


@pytest.mark.parametrize("delay", [-1, 5, 1.5, True, torch.tensor(1.5), torch.tensor([0, -1])])
def test_inference_delay_never_silently_clips(delay):
    with pytest.raises(ValueError):
        prepare_training_rtc_prefix(
            torch.zeros(2, 8, 4),
            torch.zeros(8, 4),
            delay,
            TrainingRTCConfig(simulated_delay=5),
        )


@pytest.mark.parametrize("prefix", [None, torch.zeros(1, 4)])
def test_positive_delay_requires_enough_prefix(prefix):
    with pytest.raises(ValueError, match="prefix"):
        prepare_training_rtc_prefix(torch.zeros(2, 8, 4), prefix, 2, TrainingRTCConfig())


@pytest.mark.parametrize(
    "prefix", [torch.zeros(3, 5, 4), torch.zeros(5, 6), torch.full((5, 4), float("nan"))]
)
def test_invalid_prefix_shape_and_nonfinite_values_fail(prefix):
    with pytest.raises(ValueError):
        prepare_training_rtc_prefix(torch.zeros(2, 8, 4), prefix, 2, TrainingRTCConfig())


@pytest.mark.parametrize("delay", [0, -2, 2.5, True])
def test_training_delay_config_requires_positive_integer(delay):
    with pytest.raises(ValueError, match="positive integer"):
        TrainingRTCConfig(simulated_delay=delay)


def test_config_keeps_old_runtime_mode_and_requires_trained_checkpoint():
    assert RTCConfig().mode == "inference"
    assert PI05Config(device="cpu").training_rtc_config is None
    with pytest.raises(ValueError, match="RTC mode"):
        RTCConfig(mode="typo")
    with pytest.raises(ValueError, match="delay_distribution"):
        TrainingRTCConfig(delay_distribution="typo")
    with pytest.raises(ValueError, match="requires an enabled"):
        PI05Config(device="cpu", rtc_config=RTCConfig(mode="training"))
    with pytest.raises(ValueError, match="chunk_size"):
        PI05Config(device="cpu", chunk_size=4, n_action_steps=4, training_rtc_config=TrainingRTCConfig(True))
    with pytest.raises(ValueError, match="continuous_flow"):
        PI05Config(
            device="cpu",
            discrete_action_training_mode="structured_temporal",
            training_rtc_config=TrainingRTCConfig(True),
        )


def test_training_config_and_deployment_contract_roundtrip(tmp_path):
    config = PI05Config(
        device="cpu",
        rtc_config=RTCConfig(mode="training"),
        training_rtc_config=TrainingRTCConfig(True, 5, "uniform"),
    )
    config._save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.training_rtc_config == config.training_rtc_config
    assert restored.rtc_config.mode == "training"
    assert restored.deployment_metadata() == config.deployment_metadata()
    assert restored.deployment_metadata()["policy"]["training_rtc"] == {
        "enabled": True,
        "simulated_delay": 5,
        "delay_distribution": "uniform",
    }
    # Old checkpoint configs continue loading with training RTC disabled.
    payload = json.loads((tmp_path / "config.json").read_text())
    payload.pop("training_rtc_config")
    payload["rtc_config"].pop("mode")
    (tmp_path / "config.json").write_text(json.dumps(payload))
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.training_rtc_config is None
    assert restored.rtc_config.mode == "inference"


def test_default_deployment_metadata_preserves_old_resume_contract():
    default = PI05Config(device="cpu", rtc_config=RTCConfig()).deployment_metadata()
    disabled = PI05Config(
        device="cpu", rtc_config=RTCConfig(), training_rtc_config=TrainingRTCConfig(enabled=False)
    ).deployment_metadata()
    assert default == disabled
    assert default["version"] == 10
    assert "training_rtc" not in default["policy"]
    assert "mode" not in default["policy"]["rtc"]


def test_checkpoint_cli_overrides_create_optional_training_rtc_config(tmp_path):
    PI05Config(device="cpu")._save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(
        tmp_path,
        cli_overrides=[
            "--training_rtc_config.enabled=true",
            "--training_rtc_config.simulated_delay=8",
            "--training_rtc_config.delay_distribution=uniform",
        ],
    )
    assert restored.training_rtc_config == TrainingRTCConfig(True, 8, "uniform")
    assert restored.rtc_config is None


def test_guidance_processor_rejects_unsupported_training_mode():
    processor = RTCProcessor(RTCConfig(mode="training"))
    with pytest.raises(ValueError, match="requires a policy"):
        processor.denoise_step(
            x_t=torch.zeros(1, 5, 4),
            prev_chunk_left_over=None,
            inference_delay=0,
            time=0.5,
            original_denoise_step_partial=lambda value: value,
        )


def test_training_rtc_defaults_cover_deployment_latency() -> None:
    config = TrainingRTCConfig()

    assert config.simulated_delay == 16
    assert config.delay_distribution == "uniform"
