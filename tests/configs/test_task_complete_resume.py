from copy import deepcopy

from lerobot.configs.train import is_task_complete_deployment_metadata_extension


def test_task_complete_metadata_extension_only_allows_appended_completion() -> None:
    saved = {
        "policy": {"type": "pi05"},
        "action": {
            "model_names": ["b2_vx", "gripper_target"],
            "predict": {"gripper": True, "task_complete": False},
        },
    }
    extended = deepcopy(saved)
    extended["action"]["model_names"].append("task_complete")
    extended["action"]["predict"]["task_complete"] = True

    assert is_task_complete_deployment_metadata_extension(saved, extended)

    reordered = deepcopy(extended)
    reordered["action"]["model_names"] = ["task_complete", "b2_vx", "gripper_target"]
    assert not is_task_complete_deployment_metadata_extension(saved, reordered)

    changed_policy = deepcopy(extended)
    changed_policy["policy"]["type"] = "different"
    assert not is_task_complete_deployment_metadata_extension(saved, changed_policy)
