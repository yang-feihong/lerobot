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


def test_resume_fork_can_add_task_complete_output(tmp_path: Path) -> None:
    checkpoint = tmp_path / "old_run/checkpoints/050000"
    pretrained = checkpoint / "pretrained_model"
    pretrained.mkdir(parents=True)
    (pretrained / "train_config.json").write_text(
        json.dumps({"policy": {"action_predict_task_complete": False}})
    )
    output_root = tmp_path / "outputs"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run=true",
            f"--output-root={output_root}",
            f"--resume-checkpoint={checkpoint}",
            "--resume-new-run-name=task_complete_from50k",
            "--resume-with-updated-dataset=true",
            "--dataset-root=/data/three_stage_task_complete",
            "--predict-task-complete=true",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    flags = dict(arg.split("=", 1) for arg in shlex.split(result.stdout) if arg.startswith("--"))
    assert flags["--policy.action_predict_task_complete"] == "true"
    assert flags["--resume"] == "true"
    assert flags["--resume_with_updated_dataset"] == "true"
    assert flags["--dataset.root"] == "/data/three_stage_task_complete"


def test_resume_task_complete_extension_requires_fork_and_updated_dataset(tmp_path: Path) -> None:
    checkpoint = tmp_path / "old_run/checkpoints/050000"
    pretrained = checkpoint / "pretrained_model"
    pretrained.mkdir(parents=True)
    (pretrained / "train_config.json").write_text(
        json.dumps({"policy": {"action_predict_task_complete": False}})
    )

    for extra_arg, expected_error in (
        ("--resume-new-run-name=task_complete_from50k", "updated-dataset=true"),
        ("--resume-with-updated-dataset=true", "resume-new-run-name"),
    ):
        result = subprocess.run(
            [
                "bash",
                str(SCRIPT),
                "--dry-run=true",
                f"--resume-checkpoint={checkpoint}",
                extra_arg,
                "--predict-task-complete=true",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 2
        assert expected_error in result.stderr


def test_resume_restores_semantic_sampling_and_can_explicitly_replace_it(tmp_path: Path) -> None:
    checkpoint = tmp_path / "old_run/checkpoints/050000"
    pretrained = checkpoint / "pretrained_model"
    pretrained.mkdir(parents=True)
    saved_dataset = {
        "semantic_views_path": "/data/staff1/meta/semantic_views.json",
        "semantic_phases": ["approach", "handle_press", "traversal"],
        "semantic_phase_weights": {"approach": 1.0, "handle_press": 1.0, "traversal": 1.0},
        "semantic_source_weights": {"full_episode": 1.0, "independent_stage2": 1.0},
    }
    (pretrained / "train_config.json").write_text(
        json.dumps({"policy": {}, "dataset": saved_dataset})
    )

    restored = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run=true",
            f"--resume-checkpoint={checkpoint}",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert restored.returncode == 0, restored.stderr
    restored_flags = dict(
        arg.split("=", 1) for arg in shlex.split(restored.stdout) if arg.startswith("--")
    )
    assert restored_flags["--config_path"] == str(pretrained / "train_config.json")
    assert "--dataset.semantic_phases" not in restored_flags
    assert "--dataset.semantic_source_weights" not in restored_flags

    replacement_root = tmp_path / "replacement"
    replacement_sidecar = replacement_root / "meta/semantic_views.json"
    replacement_sidecar.parent.mkdir(parents=True)
    replacement_sidecar.write_text("{}")
    replaced = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run=true",
            f"--output-root={tmp_path / 'outputs'}",
            f"--resume-checkpoint={checkpoint}",
            "--resume-new-run-name=approach_only",
            "--resume-with-updated-dataset=true",
            f"--dataset-root={replacement_root}",
            f"--semantic-views-path={replacement_sidecar}",
            '--semantic-phases=["approach"]',
            '--semantic-phase-weights={"approach":1.0}',
            "--semantic-source-weights={}",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert replaced.returncode == 0, replaced.stderr
    replaced_flags = dict(
        arg.split("=", 1) for arg in shlex.split(replaced.stdout) if arg.startswith("--")
    )
    assert json.loads(replaced_flags["--dataset.semantic_phases"]) == ["approach"]
    assert json.loads(replaced_flags["--dataset.semantic_phase_weights"]) == {"approach": 1.0}
    assert json.loads(replaced_flags["--dataset.semantic_source_weights"]) == {}
