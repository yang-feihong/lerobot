import numpy as np
import pytest

from lerobot.policies.pi05.manipulation_metrics import compute_manipulation_onset_metrics


def test_manipulation_onset_metrics_cover_gate_translation_sign_and_so3() -> None:
    names = [
        "b2_vx",
        "b2_vy",
        "b2_omega_z",
        "arm_teleop_inactive",
        "arm_reset",
        "height_invariant_ee_delta_rotvec_x",
        "height_invariant_ee_delta_rotvec_y",
        "height_invariant_ee_delta_rotvec_z",
        "height_invariant_ee_delta_x",
        "height_invariant_ee_delta_y",
        "height_invariant_ee_delta_z",
        "gripper_target",
        "task_complete",
    ]
    expert = np.zeros((5, 50, len(names)), dtype=np.float32)
    predicted = np.zeros_like(expert)
    expert[:, :, names.index("arm_teleop_inactive")] = 1.0
    predicted[:, :, names.index("arm_teleop_inactive")] = 1.0
    expert[2:, :, names.index("arm_teleop_inactive")] = 0.0
    predicted[3:, :, names.index("arm_teleop_inactive")] = 0.0
    predicted[2, 5:, names.index("arm_teleop_inactive")] = 0.0
    expert[2, :, names.index("height_invariant_ee_delta_x")] = 0.001
    predicted[2, :, names.index("height_invariant_ee_delta_x")] = 0.001
    expert[2, :, names.index("height_invariant_ee_delta_rotvec_x")] = 0.01
    predicted[2, :, names.index("height_invariant_ee_delta_rotvec_x")] = 0.01

    metrics = compute_manipulation_onset_metrics(
        frame_indices=np.arange(5),
        expert_chunks=expert,
        predicted_chunks=predicted,
        action_names=names,
        control_frequency_hz=50.0,
        dataset_frequency_hz=50.0,
        dataset_group="staff1",
    )

    assert metrics["manipulation_onset_count"] == 1
    assert metrics["manipulation_onset_detection_rate"] == 1.0
    assert metrics["arm_mode_precision"] == 1.0
    assert metrics["arm_mode_recall"] == pytest.approx(2 / 3)
    assert metrics["predicted_active_ee_translation_endpoint_error_m"] == pytest.approx(0.005, abs=1e-7)
    assert metrics["wrist_twist_sign_accuracy"] == 1.0
    assert metrics["cumulative_so3_error_1s_rad"] == pytest.approx(0.05, abs=1e-6)
