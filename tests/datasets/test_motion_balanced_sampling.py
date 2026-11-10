import numpy as np
import pytest

from lerobot.datasets.motion_balanced_sampling import motion_priority_masks


def _identity_ee(num_frames: int) -> np.ndarray:
    state = np.zeros((num_frames, 9), dtype=np.float64)
    state[:, 0] = 1.0
    state[:, 4] = 1.0
    return state


def test_motion_priority_masks_detect_future_translation_without_crossing_horizon():
    state = _identity_ee(8)
    state[5:, 6] = 0.03
    translation, rotation, gripper = motion_priority_masks(
        state,
        np.zeros(8),
        horizon_frames=2,
        translation_threshold_m=0.02,
        rotation_threshold_rad=0.1,
        gripper_change_threshold=0.5,
    )
    assert np.flatnonzero(translation).tolist() == [3, 4]
    assert not rotation.any()
    assert not gripper.any()


def test_motion_priority_masks_detect_rotation_and_gripper_change():
    state = _identity_ee(6)
    angle = np.deg2rad(20.0)
    state[4:, :6] = [np.cos(angle), np.sin(angle), 0.0, -np.sin(angle), np.cos(angle), 0.0]
    gripper_values = np.array([0.0, 0.0, 0.0, -1.047, -1.047, -1.047])
    translation, rotation, gripper = motion_priority_masks(
        state,
        gripper_values,
        horizon_frames=2,
        translation_threshold_m=0.02,
        rotation_threshold_rad=np.deg2rad(10.0),
        gripper_change_threshold=0.5,
    )
    assert not translation.any()
    assert np.flatnonzero(rotation).tolist() == [2, 3]
    assert np.flatnonzero(gripper).tolist() == [1, 2]


@pytest.mark.parametrize("source,expected", [("control_action", []), ("recorded_action", [3, 4])])
def test_motion_pool_follows_selected_action_source(source, expected):
    from types import SimpleNamespace

    from datasets import Dataset

    from lerobot.datasets.motion_balanced_sampling import build_motion_priority_pool
    from lerobot.policies.pi05.b2_action_transform import CONTROL_EXTENDED_DATASET_ACTION_NAMES

    actions = np.zeros((8, 25))
    actions[:, 5:14] = _identity_ee(8)
    actions[:, 16:25] = _identity_ee(8)
    actions[5:, 11] = 0.03
    dataset = SimpleNamespace(
        hf_dataset=Dataset.from_dict({"action": actions.tolist(), "index": list(range(8)),
                                     "episode_index": [0] * 8}),
        meta=SimpleNamespace(features={"action": {"names": list(CONTROL_EXTENDED_DATASET_ACTION_NAMES)}}),
    )
    pool = build_motion_priority_pool(dataset, horizon_frames=2, translation_threshold_m=.02,
                                     rotation_threshold_rad=.1, gripper_change_threshold=.5,
                                     ee_supervision_source=source)
    assert pool.frame_indices.tolist() == expected
