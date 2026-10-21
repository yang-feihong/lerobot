from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.datasets.semantic_views import (
    SemanticSegment, SemanticView, SemanticViewCatalog,
    apply_semantic_views_to_batch, semantic_phase_sampling_groups,
    semantic_sample_views, semantic_status_priors, semantic_view_start_allowed,
    with_incomplete_static_start_filter,
)

NAMES = ['b2_vx', 'b2_vy', 'b2_omega_z', 'arm_teleop_inactive', 'arm_reset'] + [f'ee_{i}' for i in range(9)] + ['gripper_target', 'task_complete'] + [f'control_{i}' for i in range(9)]


@pytest.fixture
def data():
    actions = np.zeros((10, 25), dtype=np.float32)
    actions[:, 3] = 1
    actions[3, 14] = -1.047  # Gripper movement must NOT exempt the sample.
    actions[4, 0] = .2
    actions[5, 3] = 0  # Active arm must remain even if B2=0.
    actions[6, 1] = 1e-8  # Exact zero, not a new arbitrary velocity tolerance.
    actions[7, 4] = 1  # User's predicate does not add a reset condition.
    actions[0, 15] = 1  # Raw completion is irrelevant for boundary-defined goals.
    batch = {'episode_index': np.zeros(10, dtype=np.int64), 'frame_index': np.arange(10), 'action': actions}

    class Columns:
        def select_columns(self, names): return self
        def with_format(self, name): return self
        def iter(self, batch_size): yield batch

    dataset = SimpleNamespace(meta=SimpleNamespace(features={'action': {'names': NAMES}}), hf_dataset=Columns())
    views = (
        SemanticView('open', 'goal_open', ('Open the door.',), 'active', 'demonstrated', completion_frame=2),
        SemanticView('enter', 'goal_enter', ('Enter the room.',), 'active', 'demonstrated', completion_frame=8),
    )
    catalog = SemanticViewCatalog(Path('unused'), {0: views}, 'sha', 1, (), (0,),
                                  {0: (SemanticSegment('handle_press', 0, 6), SemanticSegment('traversal', 6, 10))})
    return catalog, with_incomplete_static_start_filter(catalog, dataset), actions


def test_filter_is_per_instruction_and_preserves_boundary_complete(data):
    original, filtered, _ = data
    assert len(semantic_sample_views(original, 0, 2)) == 2
    assert [v.canonical_intent for v in semantic_sample_views(filtered, 0, 2)] == ['goal_open']
    assert semantic_sample_views(filtered, 0, 0) == ()
    assert len(semantic_sample_views(filtered, 0, 8)) == 2


def test_only_b2_zero_and_arm_inactive_define_static(data):
    _, filtered, _ = data
    enter = filtered.episodes[0][1]
    assert not semantic_view_start_allowed(filtered, enter, 0, 3)  # Gripper irrelevant.
    assert semantic_view_start_allowed(filtered, enter, 0, 4)  # B2 moving.
    assert semantic_view_start_allowed(filtered, enter, 0, 5)  # Arm active.
    assert semantic_view_start_allowed(filtered, enter, 0, 6)  # Tiny but nonzero B2.
    assert not semantic_view_start_allowed(filtered, enter, 0, 7)  # Reset not an exemption.


def test_sampler_removes_frames_with_no_eligible_instruction(data):
    _, filtered, _ = data
    eligible, groups, weights, _ = semantic_phase_sampling_groups(filtered, [100], [0], phases=['handle_press'])
    assert eligible.tolist() == [102, 103, 104, 105]
    assert groups[0].tolist() == eligible.tolist()
    assert weights == [1]


def test_view_selection_never_reintroduces_excluded_enter_pair(data):
    _, filtered, actions = data
    for seed in range(30):
        batch = {'episode_index': torch.tensor([0]), 'frame_index': torch.tensor([2]),
                 'action': torch.from_numpy(actions[2:6].copy()).unsqueeze(0)}
        apply_semantic_views_to_batch(batch, filtered, step=0, seed=seed, randomize=True)
        assert batch['task'] == ['Open the door.']
        assert batch['action'][0, :, 15].tolist() == [1, 1, 1, 1]
    batch = {'episode_index': torch.tensor([0]), 'frame_index': torch.tensor([0]),
             'action': torch.from_numpy(actions[:4].copy()).unsqueeze(0)}
    with pytest.raises(ValueError, match='excluded incomplete static start'):
        apply_semantic_views_to_batch(batch, filtered, step=0, seed=0, randomize=False)


def test_status_priors_use_the_same_filtered_pairs(data):
    _, filtered, _ = data
    priors = semantic_status_priors(filtered, {0: 10}, phases=['handle_press'])
    assert priors['task_complete'] == pytest.approx(.75)


def test_recorded_completion_and_hold_views(data):
    _, filtered, _ = data
    view = replace(filtered.episodes[0][1], completion_frame=None, completion_source='recorded')
    assert semantic_view_start_allowed(filtered, view, 0, 0)
    assert not semantic_view_start_allowed(filtered, view, 0, 1)
    view = replace(view, completion_source='zero', action_supervision='hold')
    assert not semantic_view_start_allowed(filtered, view, 0, 4)
    assert semantic_view_start_allowed(filtered, replace(view, status='complete'), 0, 4)
