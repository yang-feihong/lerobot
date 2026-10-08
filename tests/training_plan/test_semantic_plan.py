import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "training_plan/semantic_plan.py"
PRESETS_PATH = REPO_ROOT / "training_plan/semantic_training_presets.json"
JOBS_PATH = REPO_ROOT / "training_plan/training_jobs.json"
SPEC = importlib.util.spec_from_file_location("semantic_plan", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
semantic_plan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(semantic_plan)


def _sidecar(tmp_path: Path, intent: str = "goal_approach") -> Path:
    path = tmp_path / "semantic_views.json"
    path.write_text(
        json.dumps(
            {
                "format": "lerobot.semantic_views",
                "version": 1,
                "intents": {
                    "ep0:primary": {
                        "canonical_intent": intent,
                        "instructions": ["Approach the door."],
                    },
                    "ep0:other": {
                        "canonical_intent": "goal_open",
                        "instructions": ["Open the door."],
                    },
                },
                "episodes": {
                    "0": {
                        "segments": [
                            {
                                "phase": "approach",
                                "frame_start": 0,
                                "frame_stop": 10,
                                "source_group": "full_episode",
                            }
                        ],
                        "views": [
                            {
                                "intent_id": "ep0:primary",
                                "frame_start": 0,
                                "frame_stop": 10,
                                "completion_frame": 10,
                                "status": "active",
                                "action_supervision": "demonstrated",
                                "view_kind": "primary",
                            },
                            {
                                "intent_id": "ep0:other",
                                "frame_start": 0,
                                "frame_stop": 10,
                                "completion_frame": 10,
                                "status": "active",
                                "action_supervision": "demonstrated",
                                "view_kind": "compatible_goal",
                            },
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_stage1_contract_is_explicit_and_validates_sidecar(tmp_path):
    matrix = semantic_plan.load_plan(PRESETS_PATH, JOBS_PATH)
    task = semantic_plan.resolve_task(matrix, "staff1_stage1_full_mem")
    assert task["training_preset"] == "stage1_only"
    assert semantic_plan.state_weights(matrix, "staff1_stage1_full_mem") == {"approach": 1.0}
    assert semantic_plan.selected_matrix(matrix, "staff1_stage1_full_mem") == {
        "approach": {
            "goal_approach": {
                "status": "active",
                "action_supervision": "demonstrated",
                "completion_boundary": "state_end",
            }
        }
    }
    assert semantic_plan.validate_sidecar(
        matrix,
        "staff1_stage1_full_mem",
        _sidecar(tmp_path),
    ) == {"episodes": 1, "segments": 1, "frames": 10}


def test_stage1_contract_rejects_wrong_language(tmp_path):
    matrix = semantic_plan.load_plan(PRESETS_PATH, JOBS_PATH)
    with pytest.raises(ValueError, match="exactly one configured state-instruction relation"):
        semantic_plan.validate_sidecar(
            matrix,
            "staff1_stage1_full_mem",
            _sidecar(tmp_path, intent="goal_open"),
        )


def test_inline_task_contract_is_equivalent_to_preset(tmp_path):
    payload = semantic_plan.load_plan(PRESETS_PATH, JOBS_PATH)
    payload["tasks"]["inline_stage1"] = {
        "dataset_family": "staff1",
        "memory_mode": "standard",
        "resources": {
            "node": "f",
            "gpu_ids": [0, 1, 2, 3],
            "main_process_port": 29500,
        },
        "state_weights": {"approach": 1.0},
        "state_instruction_matrix": {
            "approach": {
                "goal_approach": {
                    "status": "active",
                    "action_supervision": "demonstrated",
                    "completion_boundary": "state_end",
                }
            }
        },
    }
    inline_path = tmp_path / "jobs.json"
    inline_path.write_text(
        json.dumps({"format": "lerobot.training_jobs", "version": 1, "jobs": payload["tasks"]}),
        encoding="utf-8",
    )
    loaded = semantic_plan.load_plan(PRESETS_PATH, inline_path)
    assert semantic_plan.state_weights(loaded, "inline_stage1") == semantic_plan.state_weights(
        loaded, "staff1_stage1_full_mem"
    )
    assert semantic_plan.selected_matrix(
        loaded, "inline_stage1"
    ) == semantic_plan.selected_matrix(loaded, "staff1_stage1_full_mem")


def test_task_cannot_mix_preset_and_inline_matrix(tmp_path):
    payload = semantic_plan.load_plan(PRESETS_PATH, JOBS_PATH)
    payload["tasks"]["invalid"] = {
        "dataset_family": "staff1",
        "memory_mode": "standard",
        "resources": {
            "node": "f",
            "gpu_ids": [0, 1, 2, 3],
            "main_process_port": 29500,
        },
        "training_preset": "stage1_only",
        "state_weights": {"approach": 1.0},
        "state_instruction_matrix": payload["training_presets"]["stage1_only"][
            "state_instruction_matrix"
        ],
    }
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(
        json.dumps({"format": "lerobot.training_jobs", "version": 1, "jobs": payload["tasks"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one semantic configuration form"):
        semantic_plan.load_plan(PRESETS_PATH, invalid_path)
