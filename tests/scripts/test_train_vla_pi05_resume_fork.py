"""Exercise forked-resume launcher semantics without starting training."""

import json
import os
import shlex
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "train_vla_pi05.sh"


def test_resume_can_fork_checkpoint_into_new_output_and_wandb_run(tmp_path: Path) -> None:
    checkpoint = tmp_path / "old_run/checkpoints/050000"
    pretrained = checkpoint / "pretrained_model"
    pretrained.mkdir(parents=True)
    (pretrained / "train_config.json").write_text(json.dumps({"policy": {}}))
    output_root = tmp_path / "outputs"
    new_name = "boundary_corrected_static_horizon_resume50k"

    env = os.environ.copy()
    env.pop("LEROBOT_RUNTIME_BIN", None)
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run=true",
            f"--output-root={output_root}",
            f"--resume-checkpoint={checkpoint}",
            f"--resume-new-run-name={new_name}",
            "--resume-with-updated-dataset=true",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    flags = dict(arg.split("=", 1) for arg in shlex.split(result.stdout) if arg.startswith("--"))
    assert flags["--resume"] == "true"
    assert flags["--resume_with_updated_dataset"] == "true"
    assert flags["--output_dir"] == str(output_root / new_name)
    assert flags["--job_name"] == new_name
    assert flags["--wandb.resume_training_run"] == "false"


def test_resume_fork_rejects_existing_output(tmp_path: Path) -> None:
    checkpoint = tmp_path / "old_run/checkpoints/050000"
    pretrained = checkpoint / "pretrained_model"
    pretrained.mkdir(parents=True)
    (pretrained / "train_config.json").write_text(json.dumps({"policy": {}}))
    output_root = tmp_path / "outputs"
    (output_root / "already_exists").mkdir(parents=True)

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run=true",
            f"--output-root={output_root}",
            f"--resume-checkpoint={checkpoint}",
            "--resume-new-run-name=already_exists",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 1
    assert "Forked resume output already exists" in result.stderr
