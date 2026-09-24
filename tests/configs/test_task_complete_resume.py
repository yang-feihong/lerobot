from copy import deepcopy

from lerobot.configs.train import is_task_complete_deployment_metadata_extension


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
