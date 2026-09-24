"""Exercise training RTC launcher switches without launching training."""

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "train_vla_pi05.sh"


def dry_run(tmp_path, *args):
    env = os.environ.copy()
    env.pop("LEROBOT_RUNTIME_BIN", None)
    return subprocess.run(
        ["bash", str(SCRIPT), "--dry-run=true", f"--output-root={tmp_path / 'outputs'}", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )


def flags(result):
    assert result.returncode == 0, result.stderr
    return dict(arg.split("=", 1) for arg in shlex.split(result.stdout) if arg.startswith("--"))


@pytest.mark.parametrize("enabled", ["true", "false"])
@pytest.mark.parametrize("distribution", ["exponential", "uniform"])
def test_training_rtc_launch_flags(tmp_path, enabled, distribution):
    result = flags(
        dry_run(
            tmp_path,
            f"--training-rtc={enabled}",
            "--training-rtc-simulated-delay=8",
            f"--training-rtc-delay-distribution={distribution}",
        )
    )
    assert result["--policy.training_rtc_config.enabled"] == enabled
    assert result["--policy.training_rtc_config.simulated_delay"] == "8"
    assert result["--policy.training_rtc_config.delay_distribution"] == distribution
    assert not any(key.startswith("--policy.rtc_config") for key in result)


def test_training_rtc_default_is_disabled(tmp_path):
    resolved = flags(dry_run(tmp_path))
    assert resolved["--policy.training_rtc_config.enabled"] == "false"
    assert resolved["--policy.training_rtc_config.simulated_delay"] == "16"
    assert resolved["--policy.training_rtc_config.delay_distribution"] == "uniform"


def test_task_complete_is_an_orthogonal_optional_output(tmp_path):
    resolved = flags(dry_run(tmp_path, "--predict-task-complete=true"))
    assert resolved["--policy.action_predict_task_complete"] == "true"
    assert resolved["--policy.action_predict_arm_teleop_inactive"] == "true"
    assert resolved["--policy.action_predict_arm_reset"] == "true"


@pytest.mark.parametrize(
    "args",
    [
        ["--training-rtc=yes"],
        ["--training-rtc-simulated-delay=0"],
        ["--training-rtc-simulated-delay=1.5"],
        ["--training-rtc-delay-distribution=invalid"],
        ["--training-rtc=true", "--training-rtc-simulated-delay=51"],
        [
            "--training-rtc=true",
            "--action-semantics-profile=custom",
            "--discrete-action-training-mode=structured_temporal",
        ],
    ],
)
def test_bad_training_rtc_flags_fail_before_launch(tmp_path, args):
    assert dry_run(tmp_path, *args).returncode == 2


def test_resume_preserves_training_rtc_and_validates_explicit_assertions(tmp_path):
    checkpoint = tmp_path / "run/checkpoints/000500"
    pretrained = checkpoint / "pretrained_model"
    pretrained.mkdir(parents=True)
    (pretrained / "train_config.json").write_text(
        json.dumps(
            {
                "policy": {
                    "training_rtc_config": {
                        "enabled": True,
                        "simulated_delay": 8,
                        "delay_distribution": "uniform",
                    }
                }
            }
        )
    )
    for extra in [[], ["--training-rtc=true", "--training-rtc-simulated-delay=8"]]:
        result = flags(dry_run(tmp_path, f"--resume-checkpoint={checkpoint}", *extra))
        assert not any(key.startswith("--policy.training_rtc_config") for key in result)
    result = dry_run(tmp_path, f"--resume-checkpoint={checkpoint}", "--training-rtc=false")
    assert result.returncode == 2
    assert "Resume preserves checkpoint" in result.stderr
    assert "--base-policy" in result.stderr
