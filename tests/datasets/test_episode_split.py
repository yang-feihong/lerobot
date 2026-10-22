import json

import pytest

from lerobot.datasets.episode_split import (
    load_recording_ids,
    recording_is_validation,
    split_episode_indices,
    split_manifest,
    validate_resume_split,
)
from lerobot.datasets.semantic_views import load_semantic_views


def memberships(ids, fraction=0.1):
    train, val = split_episode_indices(
        episode_indices=list(ids), recording_id_by_episode=ids, eval_split=fraction
    )
    return {ids[ep]: ep in val for ep in train + val}


def test_append_reorder_renumber_filter_and_annotation_independent():
    original = {i: f"/acquisition/session-{i % 7}/recording-{i}.bag" for i in range(1000)}
    wanted = memberships(original)
    extended = {**original, **{i: f"/new/recording-{i}" for i in range(1000, 1400)}}
    assert all(memberships(extended)[identity] == side for identity, side in wanted.items())
    reordered = {9000 + i: identity for i, identity in enumerate(reversed(list(original.values())))}
    assert memberships(reordered) == wanted
    subset = dict(list(original.items())[::3])
    assert memberships(subset) == {identity: wanted[identity] for identity in subset.values()}


def test_same_recording_crops_and_views_cannot_leak():
    groups = {0: "/staff1/session/a.bag", 1: "/staff1/session/a.bag", 2: "/staff2/session/a.bag"}
    train, val = split_episode_indices(
        episode_indices=list(groups), recording_id_by_episode=groups, eval_split=0.5
    )
    assert (0 in val) == (1 in val)
    assert (0 in train) == (1 in train)
    assert "/staff1/session/a.bag" != "/staff2/session/a.bag"


def test_hash_distributes_interleaved_objects_and_sessions_without_tail_bias():
    for obj in range(3):
        for session in range(4):
            flags = [
                recording_is_validation(f"/session-{session}/object-{obj}/recording-{i}", 0.1)
                for i in range(2500)
            ]
            assert abs(sum(flags) / len(flags) - 0.1) < 0.025


@pytest.mark.parametrize("fraction", [-0.1, 1.0, float("nan"), float("inf")])
def test_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        memberships({0: "a"}, fraction)


def test_missing_identity_is_not_silently_replaced_by_an_episode_number(tmp_path):
    with pytest.raises(ValueError, match="missing recording"):
        split_episode_indices(episode_indices=[0], recording_id_by_episode={}, eval_split=0.1)
    path = tmp_path / "views.json"
    path.write_text(
        json.dumps(
            {
                "format": "lerobot.semantic_views",
                "version": 1,
                "intents": {"go": {"instructions": ["Go."]}},
                "episodes": {
                    "0": {
                        "views": [
                            {"intent_id": "go", "status": "active", "action_supervision": "demonstrated"}
                        ]
                    }
                },
            }
        )
    )
    with pytest.raises(ValueError, match="Missing immutable"):
        load_recording_ids(tmp_path, [0], load_semantic_views(path))


def test_dataset_move_and_explicit_acquisition_identity(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/episode_split_groups.json").write_text(
        json.dumps(
            {
                "format": "lerobot.episode_split_groups",
                "version": 1,
                "recording_ids": {"2": "robot/session/recording-uuid"},
            }
        )
    )
    assert load_recording_ids(tmp_path, [2]) == {2: "robot/session/recording-uuid"}


def test_manifest_and_resume_cannot_silently_change_holdout():
    current = split_manifest({0: "a", 1: "b"}, 0.1)
    validate_resume_split(current, current, updated_dataset=False)
    for previous in ({}, split_manifest({0: "a", 1: "b"}, 0.2), split_manifest({0: "a"}, 0.1)):
        with pytest.raises(ValueError):
            validate_resume_split(previous, current, updated_dataset=False)
        validate_resume_split(previous, current, updated_dataset=True)


def test_split_disabled_and_small_groups_are_not_forcibly_reassigned():
    assert memberships({0: "a", 1: "b"}, 0.0) == {"a": False, "b": False}
    assert memberships({0: "a"})["a"] == memberships({0: "a", 1: "b"})["a"]
