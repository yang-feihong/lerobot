import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(tmp_path, *options):
    result = subprocess.run(
        ["bash", str(ROOT / "train_vla_pi05.sh"), "--dry-run=true", *options],
        cwd=ROOT, env={**os.environ, "VLA_STORAGE_ROOT": str(tmp_path), "LEROBOT_RUNTIME_BIN": ""},
        text=True, capture_output=True, check=True,
    )
    return dict(arg.removeprefix("--policy.").split("=", 1) for arg in shlex.split(result.stdout)
                if arg.startswith("--policy."))


def test_default_disabled_and_explicit_probability_reaches_policy(tmp_path):
    assert run(tmp_path)["image_frame_mask_probability"] == "0"
    assert run(tmp_path, "--image-frame-mask-probability=0.3")["image_frame_mask_probability"] == "0.3"


def test_resume_does_not_override_saved_mask_configuration(tmp_path):
    checkpoint = tmp_path / "old_run/checkpoints/010000"
    model = checkpoint / "pretrained_model"
    model.mkdir(parents=True)
    saved = model / "train_config.json"
    original = '{"policy":{"image_frame_mask_probability":0.3}}'
    saved.write_text(original)
    args = run(tmp_path, f"--resume-checkpoint={checkpoint}")
    assert "image_frame_mask_probability" not in args
    assert saved.read_text() == original


def test_plan_exposes_zero_default_and_passes_environment_value():
    env = (ROOT / "training_plan/.env").read_text()
    launch = (ROOT / "training_plan/launch.sh").read_text()
    assert "IMAGE_FRAME_MASK_PROBABILITY=0" in env.splitlines()
    assert '--image-frame-mask-probability="${IMAGE_FRAME_MASK_PROBABILITY:-0}"' in launch
