import math

import numpy as np
import pytest
import torch
from datasets import Dataset

from lerobot.scripts.openloop_vla_eval import (
    _current_b2_pose_from_raw_state,
    _local_trajectory_to_world,
    _manipulation_onset_windows,
    _world_trajectory_plot_limits,
)


def test_current_b2_pose_is_read_from_raw_non_mem_state():
    raw_state = torch.arange(2 * 49, dtype=torch.float32).reshape(2, 49)

    pose = _current_b2_pose_from_raw_state(raw_state, [40, 41, 42], history_length=1)

    torch.testing.assert_close(pose, raw_state[:, [40, 41, 42]])


def test_current_b2_pose_uses_end_of_history_prefix():
    raw_state = torch.arange(2 * 8 * 49, dtype=torch.float32).reshape(2, 8, 49)

    pose = _current_b2_pose_from_raw_state(raw_state, [40, 41, 42], history_length=3)

    torch.testing.assert_close(pose, raw_state[:, 2, [40, 41, 42]])


def test_local_trajectory_to_world_uses_each_inference_start_pose():
    local = np.array(
        [
            [[1.0, 0.0, 0.1], [1.0, 2.0, 0.2]],
            [[2.0, 0.0, -0.3], [0.0, 1.0, -0.4]],
        ],
        dtype=np.float32,
    )
    starts = np.array([[10.0, 20.0, math.pi / 2], [-3.0, 4.0, 0.0]], dtype=np.float32)

    world = _local_trajectory_to_world(local, starts)

    np.testing.assert_allclose(world[0, :, :2], [[10.0, 21.0], [7.910158, 21.795338]], atol=1e-6)
    np.testing.assert_allclose(world[1, :, :2], [[-1.0, 4.0], [-0.70448, 4.955337]], atol=1e-6)
    np.testing.assert_allclose(
        world[..., 2],
        [[math.pi / 2 + 0.1, math.pi / 2 + 0.3], [-0.3, -0.7]],
        atol=1e-6,
    )


def test_local_trajectory_to_world_rejects_unmatched_start_poses():
    with pytest.raises(ValueError, match="incompatible"):
        _local_trajectory_to_world(np.zeros((2, 50, 3)), np.zeros((3, 3)))


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
