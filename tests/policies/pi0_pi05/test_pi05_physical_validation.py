import math

import pytest
import torch

from lerobot.policies.pi05.physical_validation import physical_action_metric_sums


def _mean(metrics, name):
    total, count = metrics[name]
    return total / count


def test_pose_delta_metrics_are_reported_in_physical_units():
    names = ["b2_delta_x", "b2_delta_y", "b2_delta_yaw"]
    expert = torch.zeros(1, 2, 3)
    predicted = torch.tensor([[[0.10, 0.0, 0.10], [0.20, 0.0, 0.20]]])
    metrics = physical_action_metric_sums(
        expert,
        predicted,
        torch.ones(1, 2, dtype=torch.bool),
        action_names=names,
        b2_representation="pose_delta",
        z1_representation="ee_state_delta",
        ee_rotation_representation="rotvec",
        action_dt_seconds=0.02,
    )

    assert _mean(metrics, "b2_x_mae_m") == pytest.approx(0.15)
    assert _mean(metrics, "b2_xy_ade_m") == pytest.approx(0.15)
    assert _mean(metrics, "b2_xy_fde_m") == pytest.approx(0.20)
    assert _mean(metrics, "b2_yaw_fde_rad") == pytest.approx(0.20)


def test_ee_metrics_exclude_inactive_steps_and_use_so3_error():
    names = [
        "b2_delta_x",
        "b2_delta_y",
        "b2_delta_yaw",
        "arm_teleop_inactive",
        "arm_reset",
        "height_invariant_ee_delta_rotvec_x",
        "height_invariant_ee_delta_rotvec_y",
        "height_invariant_ee_delta_rotvec_z",
        "height_invariant_ee_delta_x",
        "height_invariant_ee_delta_y",
        "height_invariant_ee_delta_z",
        "gripper_target",
    ]
    expert = torch.zeros(1, 2, len(names))
    predicted = expert.clone()
    expert[0, 1, names.index("arm_teleop_inactive")] = 1.0
    predicted[0, 0, names.index("height_invariant_ee_delta_rotvec_z")] = math.pi / 2
    predicted[0, 0, names.index("height_invariant_ee_delta_x")] = 0.03
    # This much larger inactive-step error must not enter the EE metric.
    predicted[0, 1, names.index("height_invariant_ee_delta_x")] = 1.0
    metrics = physical_action_metric_sums(
        expert,
        predicted,
        torch.ones(1, 2, dtype=torch.bool),
        action_names=names,
        b2_representation="pose_delta",
        z1_representation="ee_state_delta",
        ee_rotation_representation="rotvec",
        action_dt_seconds=0.02,
    )

    assert _mean(metrics, "ee_translation_ade_m") == pytest.approx(0.03)
    assert _mean(metrics, "ee_rotation_ade_rad") == pytest.approx(math.pi / 2)
    assert metrics["ee_translation_ade_m"][1] == 1


def test_velocity_metrics_integrate_se2_before_comparison():
    names = ["b2_vx", "b2_vy", "b2_omega_z"]
    expert = torch.zeros(1, 2, 3)
    predicted = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    metrics = physical_action_metric_sums(
        expert,
        predicted,
        torch.ones(1, 2, dtype=torch.bool),
        action_names=names,
        b2_representation="velocity",
        z1_representation="ee_state_delta",
        ee_rotation_representation="rotvec",
        action_dt_seconds=0.02,
    )

    assert _mean(metrics, "b2_xy_ade_m") == pytest.approx(0.03)
    assert _mean(metrics, "b2_xy_fde_m") == pytest.approx(0.04)
