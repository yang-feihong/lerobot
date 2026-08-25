from types import SimpleNamespace

import numpy as np
import pytest
import torch
from datasets import Dataset

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.pi05.b2_action_transform import DATASET_ACTION_NAMES
from lerobot.processor import UnnormalizerProcessorStep
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.scripts.openloop_vla_eval import (
    _compute_metrics,
    _current_b2_pose_from_raw_state,
    _manipulation_onset_windows,
    _openloop_model_action_names,
    _unnormalize_model_action,
    _world_trajectory_plot_limits,
)


def test_discrete_metrics_expose_all_false_rare_event_predictions():
    names = ["arm_reset", "gripper_target", "task_complete"]
    expert = np.array(
        [[0.0, 0.0, 0.0], [1.0, -1.0471976, 0.0], [0.0, -1.0471976, 1.0]],
        dtype=np.float32,
    )
    pred = np.zeros_like(expert)

    metrics = _compute_metrics("all", expert, pred, names)

    assert metrics["discrete_pred_true_frac/arm_reset"] == 0.0
    assert metrics["discrete_recall/arm_reset"] == 0.0
    assert metrics["discrete_supervision_true_frac/gripper_target"] == pytest.approx(2 / 3)
    assert metrics["discrete_recall/gripper_target"] == 0.0
    assert metrics["discrete_recall/task_complete"] == 0.0


def test_openloop_reference_stops_after_checkpoint_unnormalization():
    normalized_supervision = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    unnormalizer = UnnormalizerProcessorStep(
        features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        norm_map={FeatureType.ACTION: NormalizationMode.MIN_MAX},
        stats={"action": {"min": np.array([-1.0, 1.0]), "max": np.array([1.0, 5.0])}},
    )

    def deployment_transform(transition):
        pytest.fail("deployment transform must not run")

    postprocessor = SimpleNamespace(
        steps=[unnormalizer, deployment_transform],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    model_space = _unnormalize_model_action(normalized_supervision, postprocessor)

    expected = torch.tensor([[[1.0, 7.0], [3.0, 11.0]]])
    torch.testing.assert_close(model_space, expected)


@pytest.mark.parametrize(
    ("representation", "expected_b2_names"),
    [
        ("velocity", ["b2_vx", "b2_vy", "b2_omega_z"]),
        ("pose_delta", ["b2_delta_x", "b2_delta_y", "b2_delta_yaw"]),
    ],
)
def test_openloop_action_names_match_checkpoint_b2_representation(representation, expected_b2_names):
    names = _openloop_model_action_names(list(DATASET_ACTION_NAMES), representation=representation)

    assert names[:3] == expected_b2_names


def test_current_b2_pose_is_read_from_raw_non_mem_state():
    raw_state = torch.arange(2 * 49, dtype=torch.float32).reshape(2, 49)

    pose = _current_b2_pose_from_raw_state(raw_state, [40, 41, 42], history_length=1)

    torch.testing.assert_close(pose, raw_state[:, [40, 41, 42]])


def test_current_b2_pose_uses_end_of_history_prefix():
    raw_state = torch.arange(2 * 8 * 49, dtype=torch.float32).reshape(2, 8, 49)

    pose = _current_b2_pose_from_raw_state(raw_state, [40, 41, 42], history_length=3)

    torch.testing.assert_close(pose, raw_state[:, 2, [40, 41, 42]])


def test_world_xy_plot_limits_share_one_physical_scale():
    expert = np.array([[[1.0, -2.0, 0.1], [3.0, 4.0, 0.2]]], dtype=np.float32)
    pred = np.array([[[0.0, -1.0, -0.2], [2.0, 5.0, 0.3]]], dtype=np.float32)

    limits = _world_trajectory_plot_limits(expert, pred)

    assert limits[0] == limits[1]
    assert limits[0][0] < -2.0
    assert limits[0][1] > 5.0
    assert limits[2][0] < -0.2
    assert limits[2][1] > 0.3


def test_onset_windows_are_kept_even_across_episode_boundaries():
    actions = np.zeros((8, 16), dtype=np.float32)
    actions[:, 3] = 1.0
    actions[2:4, 3] = 0.0
    actions[6:8, 3] = 0.0
    dataset = type("DatasetStub", (), {})()
    dataset.hf_dataset = Dataset.from_dict(
        {
            "action": actions.tolist(),
            "episode_index": [0, 0, 0, 0, 1, 1, 1, 1],
            "frame_index": [0, 1, 2, 3, 0, 1, 2, 3],
        }
    )

    indices, episode_frames = _manipulation_onset_windows(dataset, pre_frames=1, post_frames=1)

    assert indices == {1, 2, 3, 5, 6, 7}
    assert episode_frames == {(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)}
