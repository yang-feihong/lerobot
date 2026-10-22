from dataclasses import replace
from pathlib import Path
from random import Random
from types import SimpleNamespace

import pytest

from lerobot.datasets.dataset_reader import DatasetReader
from lerobot.datasets.semantic_views import (
    SemanticSegment,
    SemanticView,
    SemanticViewCatalog,
    semantic_sample_views,
    with_semantic_temporal_filter,
)
from lerobot.datasets.temporal_history import RandomHistorySamplingConfig
from lerobot.policies.pi05.transformed_action_stats import _semantic_start_frames


def fixture():
    history = RandomHistorySamplingConfig(("base", "wrist"), 6, 50, 0.5, 0.15, 0.05, 0.25, 1.0)
    reader = DatasetReader.__new__(DatasetReader)
    reader._random_history_sampling = history
    reader._random_history_rng = Random(123)
    reader._valid_frame_ranges = None
    reader._semantic_catalog = None
    reader.delta_indices = {
        "base": [-125, -100, -75, -50, -25, 0],
        "wrist": [-125, -100, -75, -50, -25, 0],
        "action": list(range(50)),
        "state": [-24, 0],
    }
    reader._meta = SimpleNamespace(episodes={0: {"dataset_from_index": 0, "dataset_to_index": 400}})
    dataset = SimpleNamespace(
        reader=reader, episodes=[0], meta=SimpleNamespace(total_episodes=1, episodes=reader._meta.episodes)
    )
    view = SemanticView(
        "enter",
        "goal_enter",
        ("Enter.",),
        "active",
        "demonstrated",
        frame_start=0,
        frame_stop=400,
        completion_frame=380,
    )
    catalog = SemanticViewCatalog(
        Path("unused"),
        {0: (view,)},
        "sha",
        1,
        (),
        (0,),
        {0: (SemanticSegment("approach", 0, 100), SemanticSegment("traversal", 200, 400))},
    )
    return catalog, dataset


def test_invalid_middle_stage_only_excludes_intersecting_windows():
    original, dataset = fixture()
    catalog = with_semantic_temporal_filter(original, dataset)
    assert catalog.eligible_start_ranges[0] == ((0, 100), (265, 400))
    assert semantic_sample_views(catalog, 0, 50)
    assert not semantic_sample_views(catalog, 0, 51)
    assert not semantic_sample_views(catalog, 0, 264)
    assert semantic_sample_views(catalog, 0, 265)
    assert semantic_sample_views(catalog, 0, 399)  # Episode-edge padding remains legal.
    starts = _semantic_start_frames(catalog, 0, 400, ["approach", "traversal"])
    assert starts.tolist() == list(range(51)) + list(range(265, 400))


def test_crossing_adjacent_valid_stages_remains_legal():
    original, dataset = fixture()
    catalog = replace(
        original, segments={0: (*original.segments[0], SemanticSegment("handle_press", 100, 200))}
    )
    filtered = with_semantic_temporal_filter(catalog, dataset)
    assert filtered.eligible_start_ranges[0] == ((0, 400),)
    assert semantic_sample_views(filtered, 0, 99)
    assert semantic_sample_views(filtered, 0, 200)


def test_history_uses_physical_validity_not_presence_of_goal_annotations():
    original, dataset = fixture()
    catalog = replace(
        original,
        physical_segments={
            0: (
                SemanticSegment("approach", 0, 100),
                SemanticSegment("handle_press", 100, 200),
                SemanticSegment("traversal", 200, 400),
            )
        },
    )
    filtered = with_semantic_temporal_filter(catalog, dataset)
    assert filtered.eligible_start_ranges[0] == ((0, 400),)
    assert semantic_sample_views(filtered, 0, 200)


def test_real_reader_conditions_shared_mem_clock_on_valid_history():
    original, dataset = fixture()
    with_semantic_temporal_filter(original, dataset)
    for _ in range(50):
        indices, _ = dataset.reader._get_query_indices(265, 0)
        assert indices["base"] == indices["wrist"]
        assert min(indices["base"]) >= 200
        assert indices["action"][-1] == 314
    with pytest.raises(ValueError, match="instruction window"):
        dataset.reader._get_query_indices(264, 0)
    with pytest.raises(ValueError, match="invalid physical stage"):
        dataset.reader._get_query_indices(51, 0)


def test_completed_instruction_can_hold_without_consuming_rejected_future_actions():
    original, dataset = fixture()
    done = replace(original.episodes[0][0], canonical_intent="goal_approach", completion_frame=40)
    original = replace(original, episodes={0: (*original.episodes[0], done)})
    catalog = with_semantic_temporal_filter(original, dataset)
    assert [v.canonical_intent for v in semantic_sample_views(catalog, 0, 90)] == ["goal_approach"]
    indices, _ = dataset.reader._get_query_indices(90, 0, semantic_intent="goal_approach")
    assert max(indices["action"]) == 99
    with pytest.raises(ValueError, match="instruction window"):
        dataset.reader._get_query_indices(90, 0, semantic_intent="goal_enter")


def test_zero_jitter_mem_does_not_silently_shorten_its_fixed_interval():
    original, dataset = fixture()
    dataset.reader._random_history_sampling = replace(
        dataset.reader._random_history_sampling, global_interval_std_seconds=0, local_interval_std_seconds=0
    )
    catalog = with_semantic_temporal_filter(original, dataset)
    assert catalog.eligible_start_ranges[0] == ((0, 100), (325, 400))
    assert not semantic_sample_views(catalog, 0, 324)
    with pytest.raises(ValueError, match="MEM frames"):
        dataset.reader._random_history_sampling.sample_delta_indices(Random(0), max_history_frames=65)


def test_stats_skip_incomplete_static_starts_but_keep_complete_holds():
    original, _ = fixture()
    catalog = replace(original, static_first_action_frames=frozenset({(0, 0), (0, 380)}))
    starts = _semantic_start_frames(catalog, 0, 400, ["approach", "traversal"])
    assert 0 not in starts
    assert 380 in starts
