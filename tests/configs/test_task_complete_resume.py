from copy import deepcopy

from lerobot.configs.train import (
    is_task_complete_deployment_metadata_extension,
    is_task_status_deployment_metadata_extension,
)


def test_task_complete_metadata_extension_only_allows_appended_completion() -> None:
    saved = {
        "policy": {"type": "pi05"},
        "action": {
            "model_names": ["b2_vx", "gripper_target"],
            "predict": {"gripper": True, "task_complete": False},
            "task_complete_deployment_behavior": None,
            "task_complete_semantics": None,
        },
    }
    extended = deepcopy(saved)
    extended["action"]["model_names"].append("task_complete")
    extended["action"]["predict"]["task_complete"] = True
    extended["action"]["task_complete_deployment_behavior"] = (
        "stop_before_executing_later_chunk_elements_at_first_true"
    )
    extended["action"]["task_complete_semantics"] = "explicit_true_in_the_terminal_stage_hold"

    assert is_task_complete_deployment_metadata_extension(saved, extended)

    unresolved = deepcopy(extended)
    unresolved["action"]["model_names"] = list(saved["action"]["model_names"])
    assert is_task_complete_deployment_metadata_extension(saved, unresolved)

    reordered = deepcopy(extended)
    reordered["action"]["model_names"] = ["task_complete", "b2_vx", "gripper_target"]
    assert not is_task_complete_deployment_metadata_extension(saved, reordered)

    changed_policy = deepcopy(extended)
    changed_policy["policy"]["type"] = "different"
    assert not is_task_complete_deployment_metadata_extension(saved, changed_policy)


def test_task_status_metadata_extension_appends_both_flow_channels() -> None:
    saved = {
        "policy": {"type": "pi05"},
        "action": {
            "model_names": ["b2_vx"],
            "predict": {"task_complete": False, "task_blocked": False},
            "task_complete_deployment_behavior": None,
            "task_complete_semantics": None,
            "task_blocked_semantics": None,
            "boolean_decoding": {
                "true_side": {"task_complete": "positive"},
                "output_values": {"task_complete": {"false": 0.0, "true": 1.0}},
            },
        },
    }
    current = deepcopy(saved)
    action = current["action"]
    action["model_names"] += ["task_complete", "task_blocked"]
    action["predict"].update(task_complete=True, task_blocked=True)
    action["task_complete_deployment_behavior"] = (
        "stop_before_executing_later_chunk_elements_at_first_true"
    )
    action["task_complete_semantics"] = "explicit_true_in_the_terminal_stage_hold"
    action["task_blocked_semantics"] = (
        "instruction_precondition_is_not_satisfied_in_the_current_state"
    )
    action["boolean_decoding"]["true_side"]["task_blocked"] = "positive"
    action["boolean_decoding"]["output_values"]["task_blocked"] = {
        "false": 0.0,
        "true": 1.0,
    }

    assert is_task_status_deployment_metadata_extension(saved, current)

    reordered = deepcopy(current)
    reordered["action"]["model_names"][-2:] = ["task_blocked", "task_complete"]
    assert not is_task_status_deployment_metadata_extension(saved, reordered)
