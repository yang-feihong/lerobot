from types import SimpleNamespace

import numpy as np
import pytest
from datasets import Dataset

from lerobot.policies.pi05.b2_action_transform import (
    CONTROL_EXTENDED_DATASET_ACTION_NAMES,
    DATASET_ACTION_NAMES,
    HEIGHT_INVARIANT_EE_STATE_NAMES,
    action_schema_kwargs,
    b2_execution_action_names,
    b2_pose_delta_action_names,
    make_pi05_action_stats,
)
from lerobot.policies.pi05.transformed_action_stats import (
    assert_transformed_action_stats_equal,
    compute_transformed_action_stats,
    load_transformed_action_stats,
    save_transformed_action_stats,
    transformed_action_stats_ee_valid_count,
    validate_transformed_action_stats,
)
from lerobot.utils.constants import ACTION, OBS_STATE


def test_transformed_stats_follow_episode_deltas_and_active_endpoint_mask(tmp_path) -> None:
    actions = np.zeros((6, 16), dtype=np.float32)
    actions[:, 5] = 1.0
    actions[:, 9] = 1.0
    actions[:3, 11] = [0.0, 0.01, 0.02]
    actions[3:, 11] = [10.0, 10.01, 10.02]
    actions[2, 3] = 1.0
    actions[:, 14] = [-1.0471976, 0.0, -1.0471976, 0.0, -1.0471976, 0.0]
    stored_actions = np.zeros((6, 25), dtype=np.float32)
    stored_actions[:, :16] = actions
    stored_actions[:, 16:25] = actions[:, 5:14]
    hf_dataset = Dataset.from_dict(
        {
            ACTION: stored_actions.tolist(),
            "episode_index": [0, 0, 0, 1, 1, 1],
            "frame_index": [0, 1, 2, 0, 1, 2],
        }
    )
    meta = SimpleNamespace(
        fps=50.0,
        features={ACTION: {"names": list(CONTROL_EXTENDED_DATASET_ACTION_NAMES)}},
    )
    dataset = SimpleNamespace(
        hf_dataset=hf_dataset,
        meta=meta,
        root=tmp_path,
        repo_id="local/synthetic",
        num_frames=6,
        num_episodes=2,
    )
    config = SimpleNamespace(
        io_schema_resolved=True,
        action_dt_seconds=0.02,
        control_frequency_hz=50.0,
        b2_action_representation="velocity",
        z1_action_representation="ee_delta",
        ee_delta_rotation_representation="rotvec",
        action_predict_arm_teleop_inactive=True,
        action_predict_arm_reset=True,
        action_predict_ee_pose=True,
        action_predict_gripper=True,
        action_predict_task_complete=True,
        dataset_action_feature_names=list(CONTROL_EXTENDED_DATASET_ACTION_NAMES),
        b2_global_pose_state_indices=None,
        ee_supervision_source="control_action",
        ee_target_dataset_semantics="joint_control_inactive_interpolated",
        ee_delta_supervision_mode="active_only",
        gripper_target_representation="binary_position",
    )
    config.action_feature_names = b2_execution_action_names(
        list(DATASET_ACTION_NAMES), **action_schema_kwargs(config)
    )

    payload = compute_transformed_action_stats(dataset, config)

    assert payload["counts"]["all_transitions"] == 4
    assert payload["counts"]["ee_active_non_reset_both_endpoints"] == 3
    assert transformed_action_stats_ee_valid_count(payload) == 3
    names = payload["schema"]["action_names"]
    x_index = names.index("height_invariant_ee_delta_x")
    assert payload["stats"][ACTION]["q01"][x_index] == pytest.approx(0.01, abs=1e-6)
    assert payload["stats"][ACTION]["q99"][x_index] == pytest.approx(0.01, abs=1e-6)

    path = save_transformed_action_stats(payload, tmp_path / "stats.json")
    loaded = load_transformed_action_stats(path)
    validate_transformed_action_stats(loaded, dataset, config)
    assert_transformed_action_stats_equal(loaded, payload)

    raw_stats = {
        ACTION: {
            key: np.zeros(16, dtype=np.float32)
            for key in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")
        }
    }
    raw_stats[ACTION]["count"] = np.asarray([6])
    normalized_stats = make_pi05_action_stats(
        raw_stats,
        transformed_action_stats=loaded["stats"][ACTION],
        dt=0.02,
        chunk_size=50,
        **action_schema_kwargs(config),
    )
    assert len(normalized_stats[ACTION]["q01"]) == 13
    assert normalized_stats[ACTION]["q01"][x_index] == pytest.approx(0.01, abs=1e-6)


def test_transformed_stats_valid_count_supports_all_transition_schema() -> None:
    assert transformed_action_stats_ee_valid_count({"counts": {"ee_all_transitions": 17}}) == 17


@pytest.mark.parametrize("b2_representation", ["velocity", "pose_delta"])
@pytest.mark.parametrize("z1_representation", ["ee_delta", "ee_state_delta"])
def test_transformed_stats_support_all_formal_representation_pairs(
    tmp_path, b2_representation: str, z1_representation: str
) -> None:
    frames = 8
    canonical = np.zeros((frames, 16), dtype=np.float32)
    canonical[:, :3] = np.asarray([0.3, -0.1, 0.2], dtype=np.float32)
    canonical[:, 5] = 1.0
    canonical[:, 9] = 1.0
    canonical[:, 11] = np.arange(frames, dtype=np.float32) * 0.01
    canonical[:, 14] = np.where(np.arange(frames) % 2, 0.0, -1.0471976)
    stored = np.zeros((frames, 25), dtype=np.float32)
    stored[:, :16] = canonical
    stored[:, 16:25] = canonical[:, 5:14]
    state = np.zeros((frames, 9), dtype=np.float32)
    state[:, 0] = 1.0
    state[:, 4] = 1.0
    dataset = SimpleNamespace(
        hf_dataset=Dataset.from_dict(
            {
                ACTION: stored.tolist(),
                OBS_STATE: state.tolist(),
                "episode_index": [0] * frames,
                "frame_index": list(range(frames)),
            }
        ),
        meta=SimpleNamespace(
            fps=50.0,
            features={
                ACTION: {"names": list(CONTROL_EXTENDED_DATASET_ACTION_NAMES)},
                OBS_STATE: {"names": list(HEIGHT_INVARIANT_EE_STATE_NAMES)},
            },
        ),
        root=tmp_path,
        repo_id="local/all-modes",
        num_frames=frames,
        num_episodes=1,
    )
    config = SimpleNamespace(
        io_schema_resolved=True,
        action_dt_seconds=0.02,
        control_frequency_hz=50.0,
        chunk_size=4,
        b2_action_representation=b2_representation,
        z1_action_representation=z1_representation,
        ee_delta_rotation_representation="rotvec",
        action_predict_arm_teleop_inactive=True,
        action_predict_arm_reset=True,
        action_predict_ee_pose=True,
        action_predict_gripper=True,
        action_predict_task_complete=True,
        dataset_action_feature_names=list(CONTROL_EXTENDED_DATASET_ACTION_NAMES),
        ee_supervision_source="control_action",
        ee_state_anchor_indices=list(range(9)),
        ee_target_dataset_semantics="joint_control_inactive_interpolated",
        ee_delta_supervision_mode="all",
        gripper_target_representation="binary_position",
    )
    name_fn = b2_execution_action_names
    if b2_representation == "pose_delta":
        name_fn = b2_pose_delta_action_names
    config.action_feature_names = name_fn(
        list(CONTROL_EXTENDED_DATASET_ACTION_NAMES), **action_schema_kwargs(config)
    )

    payload = compute_transformed_action_stats(dataset, config)

    assert payload["schema"]["representation"] == b2_representation
    assert payload["schema"]["z1_representation"] == z1_representation
    assert len(payload["stats"][ACTION]["q01"]) == 13
