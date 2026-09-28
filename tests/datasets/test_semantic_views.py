import json

import pytest
import torch

from lerobot.datasets.factory import semantic_source_split_keys
from lerobot.datasets.semantic_views import (
    apply_semantic_views_to_batch,
    load_semantic_views,
    semantic_phase_sampling_groups,
    semantic_status_priors,
)
from lerobot.policies.pi05.b2_action_transform import TASK_BLOCKED_KEY
from lerobot.utils.constants import ACTION


def _catalog(tmp_path, views):
    path = tmp_path / "views.json"
    path.write_text(
        json.dumps(
            {
                "format": "lerobot.semantic_views",
                "version": 1,
                "intents": {
                    "go": {"instructions": ["Open the door.", "Push the door open."]},
                    "pass": {"instructions": ["Enter the room."]},
                },
                "episodes": {"0": {"views": views}},
            }
        )
    )
    return load_semantic_views(path)


def test_selected_episodes_are_validated_and_exposed(tmp_path):
    path = tmp_path / "selected.json"
    path.write_text(
        json.dumps(
            {
                "format": "lerobot.semantic_views",
                "version": 1,
                "selected_episodes": [3],
                "intents": {"go": {"instructions": ["Go to the door."]}},
                "episodes": {
                    "3": {
                        "views": [
                            {
                                "intent_id": "go",
                                "status": "active",
                                "action_supervision": "demonstrated",
                            }
                        ]
                    }
                },
            }
        )
    )
    assert load_semantic_views(path).selected_episodes == (3,)


def test_semantic_source_split_keys_preserve_each_physical_source(tmp_path):
    path = tmp_path / "sources.json"
    path.write_text(
        json.dumps(
            {
                "format": "lerobot.semantic_views",
                "version": 1,
                "intents": {"go": {"instructions": ["Go."]}},
                "episodes": {
                    "0": {
                        "segments": [
                            {
                                "phase": "approach",
                                "frame_start": 0,
                                "frame_stop": 5,
                                "source_group": "full_episode",
                            },
                            {
                                "phase": "handle_press",
                                "frame_start": 5,
                                "frame_stop": 9,
                                "source_group": "full_episode",
                            },
                        ],
                        "views": [
                            {
                                "intent_id": "go",
                                "frame_start": 0,
                                "frame_stop": 9,
                                "status": "active",
                                "action_supervision": "demonstrated",
                            }
                        ],
                    },
                    "1": {
                        "segments": [
                            {
                                "phase": "handle_press",
                                "frame_start": 0,
                                "frame_stop": 7,
                                "source_group": "independent_stage2",
                            }
                        ],
                        "views": [
                            {
                                "intent_id": "go",
                                "frame_start": 0,
                                "frame_stop": 7,
                                "status": "active",
                                "action_supervision": "demonstrated",
                            }
                        ],
                    },
                },
            }
        )
    )
    catalog = load_semantic_views(path)
    assert semantic_source_split_keys(catalog, [0, 1]) == {
        0: "full_episode",
        1: "independent_stage2",
    }


def test_semantic_source_split_keys_reject_mixed_source_episode(tmp_path):
    path = tmp_path / "mixed_sources.json"
    path.write_text(
        json.dumps(
            {
                "format": "lerobot.semantic_views",
                "version": 1,
                "intents": {"go": {"instructions": ["Go."]}},
                "episodes": {
                    "0": {
                        "segments": [
                            {
                                "phase": "approach",
                                "frame_start": 0,
                                "frame_stop": 5,
                                "source_group": "full_episode",
                            },
                            {
                                "phase": "handle_press",
                                "frame_start": 5,
                                "frame_stop": 9,
                                "source_group": "other",
                            },
                        ],
                        "views": [
                            {
                                "intent_id": "go",
                                "frame_start": 0,
                                "frame_stop": 9,
                                "status": "active",
                                "action_supervision": "demonstrated",
                            }
                        ],
                    }
                },
            }
        )
    )
    with pytest.raises(ValueError, match="one physical source group"):
        semantic_source_split_keys(load_semantic_views(path), [0])


def _batch():
    action = torch.zeros((1, 4, 25))
    action[..., :3] = 0.4
    action[..., 3] = 0.0
    action[..., 4] = 1.0
    action[..., 14] = -1.047
    action[:, 0, 16:25] = torch.arange(9)
    action[:, 1:, 16:25] = 99.0
    return {
        ACTION: action,
        "episode_index": torch.tensor([0]),
        "frame_index": torch.tensor([7]),
        "index": torch.tensor([7]),
        "task": [""],
    }


def test_complete_view_materializes_hold_without_masking_action(tmp_path):
    catalog = _catalog(
        tmp_path,
        [{"intent_id": "go", "status": "complete", "action_supervision": "hold"}],
    )
    batch = _batch()
    counts = apply_semantic_views_to_batch(batch, catalog, step=3, seed=4, randomize=False)
    assert counts == {"active": 0, "complete": 1, "blocked": 0, "hold": 1}
    assert batch["task"] == ["Open the door."]
    assert torch.equal(batch[ACTION][0, :, :3], torch.zeros((4, 3)))
    assert torch.equal(batch[ACTION][0, :, 3], torch.ones(4))
    assert torch.equal(batch[ACTION][0, :, 4], torch.zeros(4))
    assert torch.equal(batch[ACTION][0, :, 16:25], torch.arange(9).repeat(4, 1))
    assert torch.equal(batch[ACTION][0, :, 15], torch.ones(4))
    assert torch.equal(batch[TASK_BLOCKED_KEY], torch.zeros((1, 4)))


def test_blocked_and_complete_are_mutually_exclusive(tmp_path):
    catalog = _catalog(
        tmp_path,
        [{"intent_id": "pass", "status": "blocked", "action_supervision": "hold"}],
    )
    batch = _batch()
    apply_semantic_views_to_batch(batch, catalog, step=0, seed=0, randomize=False)
    assert torch.equal(batch[ACTION][0, :, 15], torch.zeros(4))
    assert torch.equal(batch[TASK_BLOCKED_KEY], torch.ones((1, 4)))


def test_active_primary_preserves_recorded_actions(tmp_path):
    catalog = _catalog(
        tmp_path,
        [
            {
                "intent_id": "go",
                "status": "active",
                "action_supervision": "demonstrated",
                "completion_source": "recorded",
            }
        ],
    )
    batch = _batch()
    expected = batch[ACTION].clone()
    apply_semantic_views_to_batch(batch, catalog, step=0, seed=0, randomize=False)
    assert torch.equal(batch[ACTION], expected)


def test_rejects_non_hold_counterfactual(tmp_path):
    with pytest.raises(ValueError, match="must use deterministic hold"):
        _catalog(
            tmp_path,
            [{"intent_id": "go", "status": "blocked", "action_supervision": "demonstrated"}],
        )


def test_status_priors_follow_view_probability_not_language_count(tmp_path):
    catalog = _catalog(
        tmp_path,
        [
            {"intent_id": "go", "status": "active", "action_supervision": "demonstrated"},
            {"intent_id": "pass", "status": "blocked", "action_supervision": "hold", "weight": 3},
        ],
    )
    assert semantic_status_priors(catalog, {0: 5}) == {
        "task_complete": 0.0,
        "task_blocked": 0.75,
    }


def test_status_priors_include_active_view_completion_suffix(tmp_path):
    catalog = _catalog(
        tmp_path,
        [
            {
                "intent_id": "go",
                "status": "active",
                "action_supervision": "demonstrated",
                "frame_start": 0,
                "frame_stop": 10,
                "completion_frame": 10,
            }
        ],
    )
    priors = semantic_status_priors(catalog, {0: 10}, action_horizon=4)
    # Input frames 7, 8 and 9 contain respectively 1, 2 and 3 positive suffix labels.
    assert priors["task_complete"] == pytest.approx(6 / 40)


def test_composite_goal_switches_to_hold_at_its_own_terminal_boundary(tmp_path):
    catalog = _catalog(
        tmp_path,
        [
            {
                "intent_id": "go",
                "status": "active",
                "action_supervision": "demonstrated",
                "completion_frame": 9,
            }
        ],
    )
    batch = _batch()  # starts at physical frame 7
    apply_semantic_views_to_batch(batch, catalog, step=0, seed=0, randomize=False)
    assert batch[ACTION][0, :, 15].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert torch.equal(batch[ACTION][0, :2, :3], torch.full((2, 3), 0.4))
    assert torch.equal(batch[ACTION][0, 2:, :3], torch.zeros((2, 3)))
    assert torch.equal(batch[ACTION][0, 2:, 3], torch.ones(2))


def test_phase_sampling_uses_full_episode_segments_and_hierarchical_weights(tmp_path):
    path = tmp_path / "phases.json"
    path.write_text(
        json.dumps(
            {
                "format": "lerobot.semantic_views",
                "version": 1,
                "intents": {"go": {"instructions": ["Go."]}},
                "episodes": {
                    "0": {
                        "segments": [
                            {"phase": "approach", "frame_start": 1, "frame_stop": 4},
                            {"phase": "handle_press", "frame_start": 4, "frame_stop": 6},
                        ],
                        "views": [
                            {
                                "intent_id": "go",
                                "status": "active",
                                "action_supervision": "demonstrated",
                            }
                        ],
                    },
                    "1": {
                        "segments": [
                            {
                                "phase": "handle_press",
                                "frame_start": 0,
                                "frame_stop": 3,
                                "source_group": "independent",
                            }
                        ],
                        "views": [
                            {
                                "intent_id": "go",
                                "status": "active",
                                "action_supervision": "demonstrated",
                            }
                        ],
                    },
                },
            }
        )
    )
    catalog = load_semantic_views(path)
    eligible, groups, weights, names = semantic_phase_sampling_groups(
        catalog,
        [0, 10],
        [0, 1],
        phases=["approach", "handle_press"],
        phase_weights={"approach": 1.0, "handle_press": 1.0},
        source_weights={"default": 1.0, "independent": 1.0},
    )
    assert eligible.tolist() == [1, 2, 3, 4, 5, 10, 11, 12]
    assert names == ["approach/default", "handle_press/default", "handle_press/independent"]
    assert weights == pytest.approx([0.5, 0.25, 0.25])
    assert [group.tolist() for group in groups] == [[1, 2, 3], [4, 5], [10, 11, 12]]

    eligible, groups, weights, names = semantic_phase_sampling_groups(
        catalog,
        [0, 10],
        [0, 1],
        phases=["approach", "handle_press"],
        phase_weights={"approach": 1.0, "handle_press": 1.0},
    )
    assert eligible.tolist() == [1, 2, 3, 4, 5, 10, 11, 12]
    assert names == ["approach/*", "handle_press/*"]
    assert weights == pytest.approx([0.5, 0.5])
    assert [group.tolist() for group in groups] == [[1, 2, 3], [4, 5, 10, 11, 12]]

    with pytest.raises(ValueError, match="exactly match selected source groups"):
        semantic_phase_sampling_groups(
            catalog,
            [0, 10],
            [0, 1],
            phases=["approach", "handle_press"],
            source_weights={"default": 1.0},
        )
