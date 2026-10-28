"""Exercise the actual resume preflight without loading a model or allocating GPUs."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PLAN = Path(__file__).resolve().parents[2] / "training_plan"
PRESETS = json.loads((PLAN / "semantic_training_presets.json").read_text())["presets"]
PROFILE = json.loads((PLAN / "training_profile.json").read_text())
VALIDATOR = (PLAN / "launch.sh").read_text().split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]


@pytest.mark.parametrize("change", ["add", "remove", "rewrite", "new_state", "memory_drift"])
@pytest.mark.parametrize("allow", [False, True])
def test_same_run_instruction_extension_requires_opt_in_and_keeps_old_contracts(tmp_path, change, allow):
    old = copy.deepcopy(PRESETS["three_goal_conditioned"]["state_instruction_matrix"])
    new = copy.deepcopy(PRESETS["three_goal_plus_full"]["state_instruction_matrix"])
    policy = copy.deepcopy(PROFILE["policy"])
    if change == "remove":
        del new["approach"]["goal_open"]
    elif change == "rewrite":
        new["approach"]["goal_open"]["action_supervision"] = "hold"
    elif change == "new_state":
        new["extra_state"] = copy.deepcopy(new["approach"])
    elif change == "memory_drift":
        policy["mem_vit_max_num_frames"] = 2
    dataset = {
        "root": "/data/dataset",
        "semantic_phases": list(old),
        "semantic_phase_weights": dict.fromkeys(old, 1.0),
        "semantic_state_instruction_matrix": old,
    }
    config = tmp_path / "train_config.json"
    config.write_text(json.dumps({"wandb": {"run_id": "existing-run"}, "dataset": dataset, "policy": policy}))
    args = [
        str(config),
        dataset["root"],
        json.dumps(list(old)),
        json.dumps(dataset["semantic_phase_weights"]),
        json.dumps(new),
        "full_mem",
        "pose_delta",
        "ee_state_delta",
        "true",
        "false",
        "resume-dry-run",
        "false",
        "1",
        "6",
        "0.5",
        "true",
        "0.15",
        "0.05",
        "0.25",
        "1.0",
        "false",
        *[
            PROFILE["policy"][key]
            for key in (
                "policy_base_artifact_id",
                "policy_base_sha256",
                "mem_vit_artifact_id",
                "mem_vit_sha256",
                "policy_tokenizer_artifact_id",
                "policy_tokenizer_sha256",
            )
        ],
        str(allow).lower(),
    ]
    result = subprocess.run(
        [sys.executable, "-", *args],
        input=VALIDATOR,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(PLAN)},
        check=False,
    )
    assert (result.returncode == 0) == (allow and change == "add"), result.stderr
    if result.returncode:
        assert "checkpoint configuration mismatch" in result.stderr
