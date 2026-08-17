from types import SimpleNamespace

import numpy as np
import pytest
from datasets import Dataset

from lerobot.policies.pi05.b2_action_transform import (
    DATASET_ACTION_NAMES,
    action_schema_kwargs,
    b2_execution_action_names,
    make_b2_trajectory_stats,
)
from lerobot.policies.pi05.transformed_action_stats import (
    assert_transformed_action_stats_equal,
    compute_transformed_action_stats,
    load_transformed_action_stats,
    save_transformed_action_stats,
    validate_transformed_action_stats,
)
from lerobot.utils.constants import ACTION


def test_transformed_stats_follow_episode_deltas_and_active_endpoint_mask(tmp_path) -> None:
    actions = np.zeros((6, 16), dtype=np.float32)
    actions[:, 5] = 1.0
    actions[:, 9] = 1.0
    actions[:3, 11] = [0.0, 0.01, 0.02]
    actions[3:, 11] = [10.0, 10.01, 10.02]
    actions[2, 3] = 1.0
    actions[:, 14] = [-1.0471976, 0.0, -1.0471976, 0.0, -1.0471976, 0.0]
    hf_dataset = Dataset.from_dict(
        {
            ACTION: actions.tolist(),
            "episode_index": [0, 0, 0, 1, 1, 1],
            "frame_index": [0, 1, 2, 0, 1, 2],
        }
    )
    meta = SimpleNamespace(
        fps=50.0,
        features={ACTION: {"names": list(DATASET_ACTION_NAMES)}},
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
        b2_local_trajectory_dt=0.02,
        control_frequency_hz=50.0,
        b2_action_representation="velocity",
        z1_action_representation="ee_delta",
        ee_delta_rotation_representation="rotvec",
        action_predict_arm_teleop_inactive=True,
        action_predict_arm_reset=True,
        action_predict_ee_pose=True,
        action_predict_gripper=True,
        action_predict_task_complete=True,
        dataset_action_feature_names=list(DATASET_ACTION_NAMES),
        b2_global_pose_state_indices=None,
    )
    config.action_feature_names = b2_execution_action_names(
        list(DATASET_ACTION_NAMES), **action_schema_kwargs(config)
    )

    payload = compute_transformed_action_stats(dataset, config)

    assert payload["counts"]["all_transitions"] == 4
    assert payload["counts"]["ee_active_non_reset_both_endpoints"] == 3
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
    normalized_stats = make_b2_trajectory_stats(
        raw_stats,
        transformed_action_stats=loaded["stats"][ACTION],
        dt=0.02,
        chunk_size=50,
        **action_schema_kwargs(config),
    )
    assert len(normalized_stats[ACTION]["q01"]) == 13
    assert normalized_stats[ACTION]["q01"][x_index] == pytest.approx(0.01, abs=1e-6)
