import numpy as np

from lerobot.datasets.static_horizon_sampling import static_horizon_masks


def _static_actions(length: int) -> np.ndarray:
    actions = np.zeros((length, 16), dtype=np.float32)
    actions[:, 3] = 1.0
    actions[:, 5:11] = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    actions[:, 11:14] = np.asarray([0.4, 0.0, 0.6], dtype=np.float32)
    return actions


def test_static_horizons_distinguish_interior_wait_from_terminal_hold():
    actions = _static_actions(12)
    actions[5, 0] = 0.4
    interior, terminal = static_horizon_masks(
        actions,
        horizon_frames=4,
        b2_indices=(0, 1, 2),
        inactive_index=3,
        reset_index=4,
        gripper_index=14,
        b2_tolerance=1.0e-6,
        gripper_tolerance=1.0e-6,
    )
    assert np.flatnonzero(interior).tolist() == [0, 1]
    # Starts 9--11 rely on a repeated terminal hold to complete four targets.
    assert np.flatnonzero(terminal).tolist() == [6, 7, 8, 9, 10, 11]


def test_gripper_transition_prevents_false_static_chunk():
    actions = _static_actions(6)
    actions[3:, 14] = -1.047
    interior, terminal = static_horizon_masks(
        actions,
        horizon_frames=3,
        b2_indices=(0, 1, 2),
        inactive_index=3,
        reset_index=4,
        gripper_index=14,
        b2_tolerance=1.0e-6,
        gripper_tolerance=1.0e-6,
    )
    assert np.flatnonzero(interior).tolist() == [0]
    assert np.flatnonzero(terminal).tolist() == [3, 4, 5]
