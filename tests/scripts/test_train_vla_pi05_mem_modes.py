"""Validate launch arguments without starting training or requiring a GPU."""

import os
import shlex
import subprocess
import tempfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "train_vla_pi05.sh"


def dry_run(*args):
    env = os.environ.copy()
    env.pop("LEROBOT_RUNTIME_BIN", None)
    with tempfile.TemporaryDirectory() as output_root:
        return subprocess.run(
            ["bash", str(SCRIPT), "--dry-run=true", f"--output-root={output_root}", *args],
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )


@pytest.mark.parametrize("finetune", ["lora", "expert", "full"])
@pytest.mark.parametrize("vision", [None, "full", "lora", "frozen"])
@pytest.mark.parametrize("freeze", [False, True])
def test_mem_launch_mode_contract(tmp_path, finetune, vision, freeze):
    checkpoint = tmp_path / "mem.pt"
    checkpoint.touch()
    args = [
        "--enable-mem=true",
        f"--mem-vit-checkpoint={checkpoint}",
        f"--finetune-mode={finetune}",
        f"--freeze-vision-encoder={str(freeze).lower()}",
        "--video-backend=pyav",
    ]
    if vision is not None:
        args.append(f"--mem-vit-finetune-mode={vision}")
    result = dry_run(*args)
    if vision == "lora" and finetune != "lora" and not freeze:
        assert result.returncode == 2
        assert "MEM LoRA requires" in result.stderr
        return
    assert result.returncode == 0, result.stderr
    flags = dict(arg.split("=", 1) for arg in shlex.split(result.stdout) if arg.startswith("--"))
    expected = (
        "frozen" if freeze or vision == "frozen" else vision or ("lora" if finetune == "lora" else "full")
    )
    assert flags["--policy.mem_vit_finetune_mode"] == expected
    assert flags["--policy.freeze_vision_encoder"] == str(expected == "frozen").lower()
    assert flags["--dataset.video_backend"] == "pyav"
    assert ("--peft.method_type" in flags) == (finetune == "lora")
    if finetune == "lora":
        assert flags["--policy.peft_train_active_modules_only"] == "true"


def test_resume_does_not_override_saved_vision_or_peft_strategy(tmp_path):
    checkpoint = tmp_path / "run/checkpoints/000500"
    pretrained = checkpoint / "pretrained_model"
    pretrained.mkdir(parents=True)
    (pretrained / "train_config.json").write_text("{}")
    result = dry_run(f"--resume-checkpoint={checkpoint}", "--enable-mem=true")
    assert result.returncode == 0, result.stderr
    flags = dict(arg.split("=", 1) for arg in shlex.split(result.stdout) if arg.startswith("--"))
    assert flags["--resume"] == "true"
    assert flags["--config_path"] == str(pretrained / "train_config.json")
    assert "--policy.mem_vit_finetune_mode" not in flags
    assert "--policy.freeze_vision_encoder" not in flags
    assert "--policy.peft_train_active_modules_only" not in flags
    assert "--peft.method_type" not in flags


@pytest.mark.parametrize(
    "arg",
    [
        "--mem-vit-finetune-mode=invalid",
        "--freeze-vision-encoder=yes",
        "--lora-rank=0",
        "--lora-alpha=-1",
        "--video-backend=invalid",
    ],
)
def test_invalid_new_launch_options_fail_early(arg):
    assert dry_run(arg).returncode == 2
