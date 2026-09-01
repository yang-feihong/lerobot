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
    _current_state_slice,
    _episode_anchor_plot_names,
    _episode_b2_command_anchors,
    _manipulation_onset_windows,
    _openloop_model_action_names,
    _reanchor_episode_plot_actions,
    _resolve_image_source,
    _unnormalize_model_action,
    _world_trajectory_plot_limits,
)


def test_openloop_image_source_reproduces_checkpoint_sim_pairing(tmp_path):
    dataset_root = tmp_path / "staff1_approach"
    dataset_root.mkdir()
    paired_root = tmp_path / "staff1_approach_sim"
    paired_root.mkdir()
    manifest = paired_root / "paired_image_manifest.jsonl"
    manifest.write_text("", encoding="utf-8")
    policy_path = tmp_path / "checkpoint"
    policy_path.mkdir()
    (policy_path / "train_config.json").write_text(
        """{
          "dataset": {
            "image_source": "sim",
            "sim_image_manifest": "/old/mount/staff1_approach_sim/paired_image_manifest.jsonl",
            "sim_image_root": "/old/mount",
            "mixed_sim_probability": 0.25,
            "image_source_seed": 17
          }
        }""",
        encoding="utf-8",
    )

    resolved = _resolve_image_source(
        policy_path=policy_path,
        dataset_root=dataset_root,
        requested_source="checkpoint",
        sim_image_manifest=None,
        sim_image_root=None,
        mixed_sim_probability=None,
        image_source_seed=None,
    )

    assert resolved == {
        "image_source": "sim",
        "sim_image_manifest": manifest,
        "sim_image_root": tmp_path,
        "mixed_sim_probability": 0.25,
        "image_source_seed": 17,
    }


def test_openloop_image_source_defaults_to_real_for_legacy_checkpoint(tmp_path):
    resolved = _resolve_image_source(
        policy_path=tmp_path,
        dataset_root=tmp_path / "dataset",
        requested_source="checkpoint",
        sim_image_manifest=None,
        sim_image_root=None,
        mixed_sim_probability=None,
        image_source_seed=None,
    )

    assert resolved == {"image_source": "real"}


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


def test_current_state_slice_is_read_from_raw_non_mem_state():
    raw_state = torch.arange(2 * 49, dtype=torch.float32).reshape(2, 49)

    pose = _current_state_slice(raw_state, [40, 41, 42], history_length=1)

    torch.testing.assert_close(pose, raw_state[:, [40, 41, 42]])


def test_current_state_slice_uses_end_of_history_prefix():
    raw_state = torch.arange(2 * 8 * 49, dtype=torch.float32).reshape(2, 8, 49)

    pose = _current_state_slice(raw_state, [40, 41, 42], history_length=3)

    torch.testing.assert_close(pose, raw_state[:, 2, [40, 41, 42]])


def test_episode_plot_reanchors_b2_pose_delta_to_first_world_pose():
    actions = np.zeros((2, 2, 3), dtype=np.float32)
    actions[..., 0] = 1.0
    current_pose = np.array(
        [[10.0, 0.0, np.pi / 2], [10.0, 1.0, np.pi / 2]],
        dtype=np.float32,
    )

    actual = _reanchor_episode_plot_actions(
        actions,
        ["b2_delta_x", "b2_delta_y", "b2_delta_yaw"],
        b2_representation="pose_delta",
        z1_representation="ee_delta",
        ee_rotation_representation="rot6d",
        b2_episode_anchors=current_pose,
        current_ee_anchor=None,
    )

    np.testing.assert_allclose(actual[0, :, :2], [[1.0, 0.0], [1.0, 0.0]], atol=1.0e-6)
    np.testing.assert_allclose(actual[1, :, :2], [[2.0, 0.0], [2.0, 0.0]], atol=1.0e-6)


def test_episode_b2_anchors_integrate_commands_from_first_frame():
    dataset = SimpleNamespace(
        meta=SimpleNamespace(
            episodes=[{"dataset_from_index": 0, "dataset_to_index": 4}],
        ),
        absolute_to_relative_idx=None,
        hf_dataset=Dataset.from_dict(
            {
                "action": [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                ]
            }
        ),
    )

    actual = _episode_b2_command_anchors(
        dataset,
        0,
        np.array([0, 1, 2, 3]),
        dataset_dt_seconds=1.0,
    )

    np.testing.assert_allclose(actual[:3], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    np.testing.assert_allclose(actual[3], [2.0, 0.0, 1.0])


def test_episode_plot_reanchors_ee_state_delta_to_first_measured_pose():
    ee_names = [
        "height_invariant_ee_rot6d_col0_x",
        "height_invariant_ee_rot6d_col0_y",
        "height_invariant_ee_rot6d_col0_z",
        "height_invariant_ee_rot6d_col1_x",
        "height_invariant_ee_rot6d_col1_y",
        "height_invariant_ee_rot6d_col1_z",
        "height_invariant_ee_delta_x",
        "height_invariant_ee_delta_y",
        "height_invariant_ee_delta_z",
    ]
    identity = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    actions = np.zeros((2, 1, 12), dtype=np.float32)
    actions[..., 3:9] = identity
    actions[..., 9] = 0.5
    anchors = np.zeros((2, 9), dtype=np.float32)
    anchors[:, :6] = identity
    anchors[:, 6] = [0.0, 1.0]

    actual = _reanchor_episode_plot_actions(
        actions,
        ["b2_vx", "b2_vy", "b2_omega_z", *ee_names],
        b2_representation="velocity",
        z1_representation="ee_state_delta",
        ee_rotation_representation="rot6d",
        b2_episode_anchors=None,
        current_ee_anchor=anchors,
    )

    np.testing.assert_allclose(actual[:, 0, 9], [0.5, 1.5], atol=1.0e-6)
    np.testing.assert_allclose(actual[:, 0, 3:9], np.tile(identity, (2, 1)), atol=1.0e-6)


def test_step_delta_episode_plot_is_unchanged():
    actions = np.arange(24, dtype=np.float32).reshape(2, 2, 6)
    actual = _reanchor_episode_plot_actions(
        actions,
        ["b2_vx", "b2_vy", "b2_omega_z", "arm_reset", "gripper_target", "task_complete"],
        b2_representation="velocity",
        z1_representation="ee_delta",
        ee_rotation_representation="rotvec",
        b2_episode_anchors=None,
        current_ee_anchor=None,
    )
    np.testing.assert_array_equal(actual, actions)


def test_b2_only_episode_plot_does_not_require_an_ee_block():
    actions = np.zeros((2, 3, 3), dtype=np.float32)
    actions[:, :, 0] = 0.1
    anchors = np.zeros((2, 3), dtype=np.float32)

    actual = _reanchor_episode_plot_actions(
        actions,
        ["b2_delta_x", "b2_delta_y", "b2_delta_yaw"],
        b2_representation="pose_delta",
        z1_representation="ee_state_delta",
        ee_rotation_representation="rot6d",
        b2_episode_anchors=anchors,
        current_ee_anchor=None,
    )

    np.testing.assert_array_equal(actual, actions)


def test_episode_anchor_plot_names_only_change_anchored_dimensions():
    names = ["b2_delta_x", "b2_delta_y", "b2_delta_yaw", "height_invariant_ee_delta_x"]
    actual = _episode_anchor_plot_names(
        names,
        b2_representation="pose_delta",
        z1_representation="ee_state_delta",
    )
    assert actual == [
        "b2_episode_anchor_x",
        "b2_episode_anchor_y",
        "b2_episode_anchor_yaw",
        "episode_anchor_ee_delta_x",
    ]


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
