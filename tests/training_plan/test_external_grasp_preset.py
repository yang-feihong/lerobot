"""External grasp uses the same goal-conditioned machinery as the door preset."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from lerobot.datasets.semantic_views import apply_semantic_views_to_batch, load_semantic_views
from lerobot.utils.constants import ACTION

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("external_preset_plan", ROOT / "training_plan/semantic_plan.py")
plan_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plan_module)
NAME = "external_grasp_goal_plus_full"


@pytest.fixture
def definition():
    # Run the real preset parser over the maintained library, not a duplicate matrix.
    plan = plan_module.load_plan(ROOT / "training_plan/semantic_training_presets.json", ROOT / "training_plan/training_jobs.json")
    return plan["training_presets"][NAME]


@pytest.fixture
def sidecar(tmp_path):
    intents = {
        "approach": {"canonical_intent": "object_approach", "instructions": ["Approach the bottle."]},
        "grasp": {"canonical_intent": "external_object_grasp", "instructions": ["Grasp the bottle."]},
        "full": {"canonical_intent": "full_episode_approach_grasp", "instructions": ["Go to the bottle and pick it up."]},
    }
    views = [
        {"intent_id": "approach", "frame_start": 0, "frame_stop": 10, "completion_frame": 6,
         "status": "active", "action_supervision": "demonstrated"},
        {"intent_id": "approach", "frame_start": 10, "frame_stop": 20,
         "status": "complete", "action_supervision": "hold"},
        *[{"intent_id": intent, "frame_start": 0, "frame_stop": 20, "completion_frame": 16,
           "status": "active", "action_supervision": "demonstrated"} for intent in ("grasp", "full")],
    ]
    path = tmp_path / "views.json"
    path.write_text(json.dumps({"format": "lerobot.semantic_views", "version": 1, "intents": intents,
        "episodes": {"0": {"segments": [
            {"phase": "object_approach", "frame_start": 0, "frame_stop": 10, "source_group": "external_object_grasp"},
            {"phase": "external_object_grasp", "frame_start": 10, "frame_stop": 20, "source_group": "external_object_grasp"}],
            "views": views}}}))
    return path


def test_exact_two_by_three_definition_and_completed_relation(definition):
    matrix = definition["state_instruction_matrix"]
    assert definition["state_weights"] == {"object_approach": 1.0, "external_object_grasp": 1.0}
    assert sum(map(len, matrix.values())) == 6
    assert all(set(row) == {"object_approach", "external_object_grasp", "full_episode_approach_grasp"} for row in matrix.values())
    assert matrix["external_object_grasp"]["object_approach"] == {
        "status": "complete", "action_supervision": "hold", "completion_boundary": "none"}
    for intent in ("external_object_grasp", "full_episode_approach_grasp"):
        assert matrix["object_approach"][intent] == {
            "status": "active", "action_supervision": "demonstrated", "completion_boundary": "future_goal_end"}
        assert matrix["external_object_grasp"][intent] == {
            "status": "active", "action_supervision": "demonstrated", "completion_boundary": "goal_end"}


def test_all_six_cells_validate_and_missing_completed_cell_fails(definition, sidecar):
    payload = {"training_presets": {NAME: definition}, "tasks": {"external": {"training_preset": NAME}},
               "training_profile": {"memory_mode": "full_mem"}}
    assert plan_module.validate_sidecar(payload, "external", sidecar) == {"episodes": 1, "segments": 2, "frames": 20}
    data = json.loads(sidecar.read_text())
    data["episodes"]["0"]["views"] = [v for v in data["episodes"]["0"]["views"] if v["status"] != "complete"]
    sidecar.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="configured cells with no sidecar supervision"):
        plan_module.validate_sidecar(payload, "external", sidecar)


@pytest.mark.parametrize("frame,intent,complete", [(12, "object_approach", True), (2, "external_object_grasp", False),
    (2, "full_episode_approach_grasp", False), (12, "external_object_grasp", False),
    (12, "full_episode_approach_grasp", False), (7, "object_approach", True),
    (9, "external_object_grasp", False), (9, "full_episode_approach_grasp", False),
    (17, "external_object_grasp", True), (17, "full_episode_approach_grasp", True)])
def test_runtime_supervision_matches_preset(definition, sidecar, frame, intent, complete):
    action = torch.randn(1, 3, 25)
    action[..., 15] = 0
    before = action.clone()
    batch = {ACTION: action, "episode_index": torch.tensor([0]), "frame_index": torch.tensor([frame]),
             "index": torch.tensor([frame]), "semantic_sampling_intent": [intent], "task": [""]}
    apply_semantic_views_to_batch(batch, load_semantic_views(sidecar), step=0, seed=0, randomize=False,
                                  state_instruction_matrix=definition["state_instruction_matrix"])
    if complete:
        torch.testing.assert_close(batch[ACTION][0, :, :3], torch.zeros(3, 3))
        torch.testing.assert_close(batch[ACTION][0, :, 15], torch.ones(3))
        torch.testing.assert_close(batch[ACTION][0, :, 16:25], before[0, 0, 16:25].expand(3, -1))
        torch.testing.assert_close(batch[ACTION][0, :, 14], before[0, 0, 14].expand(3))
    else:
        torch.testing.assert_close(batch[ACTION], before)
