import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.semantic_views import (
    SemanticSegment,
    SemanticView,
    SemanticViewCatalog,
    apply_semantic_views_to_batch,
    semantic_relation_sampling_groups,
    semantic_status_priors,
)

PRESETS = json.loads(
    (Path(__file__).resolve().parents[2] / "training_plan/semantic_training_presets.json").read_text()
)["presets"]


def mixed_domain_catalog():
    from dataclasses import replace

    catalog, _, length, matrix = make_plan("three_stage_joint")
    # Unequal lengths/source sizes must not change either marginal.
    return replace(catalog, episodes={0: catalog.episodes[0], 1: catalog.episodes[0]},
                   segments={0: catalog.segments[0], 1: catalog.segments[0]},
                   episode_domains={0: "simulation", 1: "real"}), length, matrix


@pytest.mark.parametrize("weights,expected", [
    ({"simulation": 3, "real": 1}, {"simulation": .75, "real": .25}),
    ({"simulation": 0, "real": 1}, {"real": 1}),
    ({"simulation": 1}, {"simulation": 1}),
    ({"simulation": 1, "real": 5}, {"simulation": 1/6, "real": 5/6}),
])
def test_domain_ratio_is_conditional_on_each_state_instruction_cell(weights, expected):
    catalog, length, matrix = mixed_domain_catalog()
    plan = semantic_relation_sampling_groups(catalog, [0, length], [0, 1],
        state_instruction_matrix=matrix, domain_weights=weights)
    masses = Counter()
    for name, mass, frames in zip(plan.names, plan.group_weights, plan.frame_groups, strict=True):
        state, intent, domain, _ = name.split("/")
        masses[state, intent] += mass
        assert mass == pytest.approx(expected[domain] / 3)
        assert all((frame < length) == (domain == "simulation") for frame in frames)
    assert list(masses.values()) == pytest.approx([1/3] * 3)
    sampler = EpisodeAwareSampler([0, length], [length, 2*length], seed=17, shuffle=True,
        eligible_frame_indices=plan.eligible_frames, sampling_group_frame_indices=plan.frame_groups,
        sampling_group_weights=plan.group_weights, sampling_group_labels=plan.intents,
        sampling_group_frame_weights=plan.frame_weights)
    samples = list(sampler)
    sampler.load_state_dict({"epoch": 0, "start_index": 7})
    assert list(sampler) == samples[7:]


def test_domain_ratio_missing_cell_fails_instead_of_silent_renormalization():
    from dataclasses import replace

    catalog, length, matrix = mixed_domain_catalog()
    catalog = replace(catalog, segments={0: catalog.segments[0], 1: catalog.segments[1][:1]})
    with pytest.raises(ValueError, match="domain has no eligible samples"):
        semantic_relation_sampling_groups(catalog, [0, length], [0, 1],
            state_instruction_matrix=matrix, domain_weights={"simulation": 3, "real": 1})


def test_domain_ratio_precedes_source_weights_and_static_frame_weights():
    from dataclasses import replace

    catalog, length, matrix = mixed_domain_catalog()
    segments = {ep: tuple(replace(s, source_group="a" if ep % 2 == 0 else "b")
                          for s in catalog.segments[0]) for ep in range(4)}
    catalog = replace(catalog, episodes=dict.fromkeys(range(4), catalog.episodes[0]), segments=segments,
                      episode_domains={0: "simulation", 1: "simulation", 2: "real", 3: "real"})
    plan = semantic_relation_sampling_groups(catalog, [ep*length for ep in range(4)], list(range(4)),
        state_instruction_matrix=matrix, domain_weights={"simulation": 3, "real": 1},
        source_weights={"a": 1, "b": 3}, interior_static_frames=[0], interior_static_weight=.1)
    for name, mass in zip(plan.names, plan.group_weights, strict=True):
        _, _, domain, source = name.split("/")
        assert mass == pytest.approx((.75 if domain == "simulation" else .25) *
                                    (.25 if source == "a" else .75) / 3)


@pytest.mark.parametrize("weights", [{"sim": 1}, {"real": -1}, {"real": 0}, {"real": float("nan")}, {"real": True}])
def test_domain_ratio_rejects_invalid_weights(weights):
    catalog, length, matrix = mixed_domain_catalog()
    with pytest.raises(ValueError, match="Domain weights"):
        semantic_relation_sampling_groups(catalog, [0, length], [0, 1],
            state_instruction_matrix=matrix, domain_weights=weights)


def make_plan(preset):
    matrix = PRESETS[preset]["state_instruction_matrix"]
    lengths = [81] if len(matrix) == 1 else [[11, 53, 17][i % 3] for i in range(len(matrix))]
    views, segments = [], []
    start = 0
    for (state, instructions), length in zip(matrix.items(), lengths, strict=True):
        stop = start + length
        segments.append(SemanticSegment(state, start, stop))
        for intent, relation in instructions.items():
            boundary = relation["completion_boundary"]
            completion = (
                None
                if boundary == "none"
                else stop
                if boundary == "state_end"
                else stop + 1
                if boundary == "future_goal_end"
                else stop - 2
            )
            views.append(
                SemanticView(
                    f"{state}:{intent}",
                    intent,
                    (f"{state}:{intent}",),
                    relation["status"],
                    relation["action_supervision"],
                    frame_start=start,
                    frame_stop=stop,
                    completion_frame=completion,
                )
            )
        start = stop
    catalog = SemanticViewCatalog(Path("unused"), {0: tuple(views)}, "sha", 1, (), (0,), {0: tuple(segments)})
    plan = semantic_relation_sampling_groups(
        catalog, [0], [0], state_instruction_matrix=matrix, phase_weights=PRESETS[preset]["state_weights"]
    )
    return catalog, plan, start, matrix


def sampler_for(plan, length):
    return EpisodeAwareSampler(
        [0],
        [length],
        shuffle=True,
        seed=11,
        eligible_frame_indices=plan.eligible_frames,
        sampling_group_frame_indices=plan.frame_groups,
        sampling_group_weights=plan.group_weights,
        sampling_group_labels=plan.intents,
        sampling_group_frame_weights=plan.frame_weights,
    )


def test_pull_joint_plus_full_mixes_independent_stages_only_into_diagonal_cells():
    from dataclasses import replace
    import numpy as np

    catalog, _, full_length, matrix = make_plan('pull_joint_plus_full')
    episodes, segments = dict(catalog.episodes), dict(catalog.segments)
    offsets = [0, full_length]
    for episode, (phase, goal) in enumerate([
            ('approach', 'goal_approach'), ('handle_press', 'goal_open'), ('traversal', 'goal_enter')], 1):
        length = 10 + episode
        episodes[episode] = (SemanticView(f'independent:{goal}', goal, (goal,), 'active', 'demonstrated',
                                        frame_start=0, frame_stop=length, completion_frame=length-2),)
        segments[episode] = (SemanticSegment(phase, 0, length, 'independent_pull'),)
        offsets.append(offsets[-1] + length)
    catalog = replace(catalog, episodes=episodes, segments=segments, selected_episodes=(0, 1, 2, 3))
    plan = semantic_relation_sampling_groups(catalog, offsets[:-1], [0, 1, 2, 3],
                                            state_instruction_matrix=matrix)
    assert len(plan.frame_groups) == 6
    assert plan.group_weights == pytest.approx([1/6] * 6)
    for state, intent, frames in zip(plan.states, plan.intents, plan.frame_groups, strict=True):
        if intent == 'full_episode_command':
            assert np.all(frames < full_length)
        else:
            assert np.any(frames >= full_length)
    with pytest.raises(ValueError, match='no eligible samples'):
        semantic_relation_sampling_groups(catalog, offsets[:-1], [1, 2, 3], state_instruction_matrix=matrix)


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_every_preset_balances_actual_state_instruction_pairs_and_preserves_resume(preset):
    catalog, plan, length, matrix = make_plan(preset)
    cells = sum(len(row) for row in matrix.values())
    assert plan.group_weights == pytest.approx([1 / cells] * cells)
    sampler = sampler_for(plan, length)
    samples = list(sampler)
    counts = Counter(
        (next(s.phase for s in catalog.segments[0] if s.frame_start <= frame < s.frame_stop), intent)
        for frame, intent in samples
    )
    assert set(counts) == {(state, intent) for state, row in matrix.items() for intent in row}
    assert max(counts.values()) - min(counts.values()) <= 1
    replay = sampler_for(plan, length)
    replay.load_state_dict({"epoch": 0, "start_index": 13})
    assert list(replay) == samples[13:]

    dataset = LeRobotDataset.__new__(LeRobotDataset)
    dataset.writer = None

    def get_item(index, **kwargs):
        return {
            "episode_index": torch.tensor(0),
            "frame_index": torch.tensor(index),
            "index": torch.tensor(index),
            "action": torch.zeros(1, 25),
        }

    dataset.reader = SimpleNamespace(hf_dataset=object(), get_item=get_item, num_frames=length)
    loader = torch.utils.data.DataLoader(dataset, batch_size=7, sampler=sampler_for(plan, length))
    for batch in loader:
        selected = list(batch["semantic_sampling_intent"])
        apply_semantic_views_to_batch(
            batch, catalog, step=17, seed=42, randomize=True, state_instruction_matrix=matrix
        )
        assert "semantic_sampling_intent" not in batch
        assert all(task.endswith(":" + intent) for task, intent in zip(batch["task"], selected, strict=True))


def test_static_rejection_can_shrink_one_cell_without_changing_cell_prior():
    from dataclasses import replace

    catalog, _, length, matrix = make_plan("three_goal_conditioned")
    # Frames in handle_press keep completed Approach but exclude incomplete Open/Enter.
    catalog = replace(catalog, static_first_action_frames=frozenset((0, frame) for frame in range(11, 40)))
    plan = semantic_relation_sampling_groups(catalog, [0], [0], state_instruction_matrix=matrix)
    sizes = {name: len(frames) for name, frames in zip(plan.names, plan.frame_groups, strict=True)}
    assert sizes["handle_press/goal_approach/*"] == 53
    assert sizes["handle_press/goal_open/*"] == 24
    assert plan.group_weights == pytest.approx([1 / 9] * 9)
    priors = semantic_status_priors(catalog, {0: length}, action_horizon=1, relation_sampling=plan)
    assert priors["task_complete"] > 1 / 3  # Three always-complete cells plus active completion tails.


def test_empty_cell_is_reported_instead_of_fabricating_supervision():
    from dataclasses import replace

    catalog, _, _, matrix = make_plan("three_goal_conditioned")
    catalog = replace(
        catalog, episodes={0: tuple(v for v in catalog.episodes[0] if v.canonical_intent != "goal_enter")}
    )
    with pytest.raises(ValueError, match="cell has no eligible samples"):
        semantic_relation_sampling_groups(catalog, [0], [0], state_instruction_matrix=matrix)


def test_unequal_number_of_instructions_per_state_still_balances_cells():
    catalog, _, _, full_matrix = make_plan("three_goal_conditioned")
    matrix = {
        state: {intent: full_matrix[state][intent] for intent in goals}
        for state, goals in (
            ("approach", ("goal_approach",)),
            ("handle_press", ("goal_open", "goal_enter")),
            ("traversal", ("goal_enter",)),
        )
    }
    plan = semantic_relation_sampling_groups(catalog, [0], [0], state_instruction_matrix=matrix)
    assert plan.group_weights == pytest.approx([0.25] * 4)


def test_labeled_indices_are_sharded_correctly_across_four_training_ranks():
    from accelerate.data_loader import BatchSamplerShard

    _, plan, length, _ = make_plan("three_goal_conditioned")

    def batches():
        return torch.utils.data.BatchSampler(sampler_for(plan, length), batch_size=3, drop_last=False)

    expected = list(batches())
    ranks = [list(BatchSamplerShard(batches(), num_processes=4, process_index=rank)) for rank in range(4)]
    merged = [batch for step in zip(*ranks, strict=True) for batch in step]
    assert merged[: len(expected)] == expected
    assert merged[len(expected) :] == [expected[0]]


def test_completed_holds_are_not_removed_by_raw_interior_static_classification():
    catalog, _, length, matrix = make_plan("three_goal_conditioned")
    plan = semantic_relation_sampling_groups(
        catalog,
        [0],
        [0],
        state_instruction_matrix=matrix,
        interior_static_frames=list(range(11, 40)),
        interior_static_weight=0,
    )
    approach = plan.names.index("handle_press/goal_approach/*")
    assert set(range(11, 40)).issubset(set(plan.frame_groups[approach]))
