"""New-run defaults must not reinterpret historical models or resume configs."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def dry_run(tmp_path, *args):
    return subprocess.run(
        ["bash", str(ROOT / "train_vla_pi05.sh"), "--dry-run=true", *args],
        cwd=ROOT,
        env={**os.environ, "VLA_STORAGE_ROOT": str(tmp_path), "LEROBOT_RUNTIME_BIN": ""},
        text=True,
        capture_output=True,
        check=False,
    )


def policy_arguments(output):
    return dict(
        arg.removeprefix("--policy.").split("=", 1)
        for arg in shlex.split(output)
        if arg.startswith("--policy.")
    )


@pytest.mark.parametrize("dataset", ["pull_door", "all_scenes", "external_grasp", "future_task"])
def test_direct_new_training_defaults_are_task_independent_and_valid(tmp_path, dataset):
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    result = dry_run(tmp_path, f"--dataset-repo-id=local/{dataset}")
    assert result.returncode == 0, result.stderr
    args = policy_arguments(result.stdout)
    assert args["action_history_enabled"] == "true"
    assert args["ee_delta_reference"] == "previous_command"
    assert args["state_action_encoding"] == "continuous"
    assert args["z1_action_representation"] == "ee_state_delta"
    assert args["history_pause_max_steps"] == "0"
    config = PI05Config(
        device="cpu",
        action_history_enabled=True,
        ee_delta_reference=args["ee_delta_reference"],
        state_action_encoding=args["state_action_encoding"],
        z1_action_representation=args["z1_action_representation"],
    )
    assert config.action_history_enabled


def test_explicit_no_history_state_delta_remains_available(tmp_path):
    result = dry_run(tmp_path, "--action-history-enabled=false", "--ee-delta-reference=current_state")
    assert result.returncode == 0, result.stderr
    args = policy_arguments(result.stdout)
    assert args["action_history_enabled"] == "false"
    assert args["ee_delta_reference"] == "current_state"


def test_command_delta_without_history_still_fails(tmp_path):
    result = dry_run(tmp_path, "--action-history-enabled=false")
    assert result.returncode != 0
    assert "requires action history" in result.stderr


def test_resume_does_not_pass_fresh_history_or_reference_defaults(tmp_path):
    checkpoint = tmp_path / "old_run/checkpoints/010000"
    model = checkpoint / "pretrained_model"
    model.mkdir(parents=True)
    saved = model / "train_config.json"
    original = '{"policy":{"action_history_enabled":false,"ee_delta_reference":"current_state"}}'
    saved.write_text(original)
    result = dry_run(tmp_path, f"--resume-checkpoint={checkpoint}")
    assert result.returncode == 0, result.stderr
    args = policy_arguments(result.stdout)
    assert "action_history_enabled" not in args
    assert "ee_delta_reference" not in args
    assert "state_action_encoding" not in args
    assert f"--config_path={saved}" in shlex.split(result.stdout)
    assert saved.read_text() == original


def test_plan_environment_and_missing_variable_fallbacks_agree():
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; printf "%s %s" "$ACTION_HISTORY_ENABLED" "$EE_DELTA_REFERENCE"',
         "bash", str(ROOT / "training_plan/.env")],
        text=True, capture_output=True, check=True,
    )
    assert result.stdout == "true previous_command"
    launch = (ROOT / "training_plan/launch.sh").read_text()
    assert '--action-history-enabled="${ACTION_HISTORY_ENABLED:-true}"' in launch
    assert '--ee-delta-reference="${EE_DELTA_REFERENCE:-previous_command}"' in launch


def test_missing_fields_in_historical_policy_keep_legacy_interpretation():
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    config = PI05Config(device="cpu")
    assert not config.action_history_enabled
    assert config.ee_delta_reference == "current_state"
