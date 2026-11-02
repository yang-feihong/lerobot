"""Joint + full presets select only diagonal goals and the full command column."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("joint_plus_full_plan", ROOT / "training_plan/semantic_plan.py")
plan_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plan_module)


@pytest.mark.parametrize(
    "name,parent,diagonal,full",
    [
        (
            "three_stage_plus_full",
            "three_goal_plus_full",
            {"approach": "goal_approach", "handle_press": "goal_open", "traversal": "goal_enter"},
            "full_episode_command",
        ),
        (
            "pull_joint_plus_full",
            "three_goal_plus_full",
            {"approach": "goal_approach", "handle_press": "goal_open", "traversal": "goal_enter"},
            "full_episode_command",
        ),
        (
            "external_grasp_joint_plus_full",
            "external_grasp_goal_plus_full",
            {"object_approach": "object_approach", "external_object_grasp": "external_object_grasp"},
            "full_episode_approach_grasp",
        ),
    ],
)
def test_joint_plus_full_exact_allowlist_and_shared_entry(name, parent, diagonal, full):
    plan = plan_module.load_plan(
        ROOT / "training_plan/semantic_training_presets.json", ROOT / "training_plan/training_jobs.json"
    )
    definition = plan["training_presets"][name]
    matrix = definition["state_instruction_matrix"]
    coupled = plan["training_presets"][parent]["state_instruction_matrix"]
    assert definition["state_weights"] == dict.fromkeys(diagonal, 1.0)
    assert set(matrix) == set(diagonal)
    assert sum(map(len, matrix.values())) == 2 * len(diagonal)
    for index, (state, goal) in enumerate(diagonal.items()):
        assert set(matrix[state]) == {goal, full}
        assert matrix[state][goal] == {
            "status": "active",
            "action_supervision": "demonstrated",
            "completion_boundary": "goal_end",
        }
        assert matrix[state][full] == {
            "status": "active",
            "action_supervision": "demonstrated",
            "completion_boundary": "goal_end" if index == len(diagonal) - 1 else "future_goal_end",
        }
        assert all(rule == coupled[state][intent] for intent, rule in matrix[state].items())
    # New presets must not narrow the existing coupled variants.
    assert sum(map(len, coupled.values())) == len(diagonal) * (len(diagonal) + 1)
    plan["tasks"] = {"new": {"training_preset": name, "data_recipe": "test_recipe"}}
    assert plan_module.selected_matrix(plan, "new") == matrix
    assert plan_module.state_weights(plan, "new") == definition["state_weights"]
