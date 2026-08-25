from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.pi05.b2_action_transform import DATASET_ACTION_NAMES
from lerobot.policies.pi05.modeling_pi05 import (
    LinearChainCRF,
    PI05Policy,
    PI05Pytorch,
    absorbing_bool_decode,
    absorbing_hazard_nll,
)


def test_gate_loss_uses_inactive_reset_and_completion_ground_truth_masks():
    policy = PI05Policy.__new__(PI05Policy)
    torch.nn.Module.__init__(policy)
    names = ["b2_delta_x", "b2_delta_y", "b2_delta_yaw", *DATASET_ACTION_NAMES[3:]]
    policy.config = SimpleNamespace(
        action_bool_loss_weight=4.0,
        action_continuous_loss_weight=1.0,
        action_masked_continuous_min_weight=0.0,
        action_bool_balance_eps=1e-3,
        action_bool_true_fractions=dict.fromkeys(
            ("arm_teleop_inactive", "arm_reset", "gripper_target", "task_complete"), 0.5
        ),
        action_gripper_target_true_side="negative",
        io_schema_resolved=True,
        b2_action_representation="pose_delta",
        z1_action_representation="ee_delta",
        action_feature_names=names,
    )
    actions = -torch.ones(1, 5, 16)
    actions[0, :, 3] = torch.tensor([-1.0, 1.0, -1.0, -1.0, -1.0])
    actions[0, :, 4] = torch.tensor([-1.0, -1.0, 1.0, -1.0, -1.0])
    actions[0, :, 15] = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0])
    loss, info = policy._b2_z1_gate_action_loss(
        torch.ones_like(actions), actions, "mean", ee_delta_is_valid=torch.ones(1, 5, dtype=torch.bool)
    )
    assert torch.isfinite(loss)
    assert info["continuous_mask_frac/b2_pose_delta"] == pytest.approx(0.6)
    assert info["continuous_mask_frac/ee_pose"] == pytest.approx(0.2)
    assert info["gate_true_frac/task_complete"] == pytest.approx(0.4)
    assert "gate_loss/arm_teleop_inactive" in info


def test_disabling_inactive_prediction_removes_its_output_and_ee_mask():
    policy = PI05Policy.__new__(PI05Policy)
    torch.nn.Module.__init__(policy)
    names = ["b2_delta_x", "b2_delta_y", "b2_delta_yaw", *DATASET_ACTION_NAMES[4:]]
    policy.config = SimpleNamespace(
        action_bool_loss_weight=4.0,
        action_continuous_loss_weight=1.0,
        action_masked_continuous_min_weight=0.0,
        action_bool_balance_eps=1e-3,
        action_bool_true_fractions=dict.fromkeys(("arm_reset", "gripper_target", "task_complete"), 0.5),
        action_gripper_target_true_side="negative",
        io_schema_resolved=True,
        b2_action_representation="pose_delta",
        z1_action_representation="ee_delta",
        action_feature_names=names,
    )
    actions = -torch.ones(1, 3, 15)
    loss, info = policy._b2_z1_gate_action_loss(
        torch.ones_like(actions), actions, "mean", ee_delta_is_valid=torch.ones(1, 3, dtype=torch.bool)
    )
    assert torch.isfinite(loss)
    assert info["continuous_mask_frac/ee_pose"] == pytest.approx(1.0)
    assert "gate_loss/arm_teleop_inactive" not in info


def test_uniform_valid_loss_uses_every_non_padding_continuous_element() -> None:
    losses = torch.tensor([[[1.0, 3.0], [100.0, 100.0], [5.0, 7.0]]])
    is_pad = torch.tensor([[False, True, False]])

    mean_loss = PI05Policy._uniform_valid_action_loss(losses, "mean", is_pad)
    per_sample = PI05Policy._uniform_valid_action_loss(losses, "none", is_pad)

    assert mean_loss.item() == pytest.approx(4.0)
    assert per_sample.tolist() == pytest.approx([4.0])


def test_uniform_valid_loss_ignores_empty_episode_tail_samples() -> None:
    losses = torch.tensor(
        [
            [[1.0, 3.0], [5.0, 7.0]],
            [[100.0, 100.0], [100.0, 100.0]],
        ]
    )
    is_pad = torch.tensor([[False, False], [True, True]])

    mean_loss = PI05Policy._uniform_valid_action_loss(losses, "mean", is_pad)
    per_sample = PI05Policy._uniform_valid_action_loss(losses, "none", is_pad)

    assert mean_loss.item() == pytest.approx(4.0)
    assert per_sample.tolist() == pytest.approx([4.0, 0.0])


def test_uniform_valid_loss_rejects_an_entirely_empty_batch() -> None:
    losses = torch.ones(2, 3, 4)
    is_pad = torch.ones(2, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="batch containing only padding"):
        PI05Policy._uniform_valid_action_loss(losses, "mean", is_pad)


def test_arm_mode_crf_allows_reset_to_reappear() -> None:
    crf = LinearChainCRF(3)
    targets = torch.tensor([[0, 2, 2, 0, 1, 2, 0]])
    emissions = torch.full((1, targets.shape[1], 3), -20.0)
    emissions.scatter_(2, targets.unsqueeze(-1), 20.0)
    decoded = crf.decode(emissions)
    torch.testing.assert_close(decoded, targets)
    assert torch.isfinite(crf.nll(emissions, targets, torch.ones_like(targets, dtype=torch.bool))).all()


def test_crf_initial_persistence_prior_removes_isolated_weak_spike() -> None:
    emissions = torch.zeros((1, 5, 2))
    emissions[:, :, 0] = 2.0
    emissions[0, 2, 1] = 3.0

    without_persistence = LinearChainCRF(2).decode(emissions)
    with_persistence = LinearChainCRF(2, initial_stay_bias=4.0).decode(emissions)

    assert without_persistence.tolist() == [[0, 0, 1, 0, 0]]
    assert with_persistence.tolist() == [[0, 0, 0, 0, 0]]


def test_crf_initial_persistence_prior_keeps_sustained_state_change() -> None:
    emissions = torch.zeros((1, 8, 2))
    emissions[:, :, 0] = 2.0
    emissions[0, 3:, 1] = 3.0

    crf = LinearChainCRF(2, initial_stay_bias=4.0)
    decoded = crf.decode(emissions)

    assert decoded.tolist() == [[0, 0, 0, 1, 1, 1, 1, 1]]
    torch.testing.assert_close(crf.transitions.detach(), torch.eye(2) * 4.0)


def test_task_complete_hazard_is_monotonic_without_masking_its_tail() -> None:
    logits = torch.tensor([[-3.0, -2.0, 4.0, -5.0]])
    target = torch.tensor([[False, False, True, True]])
    mask = torch.ones_like(target)
    decoded = absorbing_bool_decode(logits)
    assert decoded.tolist() == [[False, False, True, True]]
    assert torch.isfinite(absorbing_hazard_nll(logits, target, mask)).all()


def test_newly_initialized_modules_use_a_separate_optimizer_group() -> None:
    policy = PI05Policy.__new__(PI05Policy)
    torch.nn.Module.__init__(policy)
    policy.model = torch.nn.Module()
    policy.model.action_out_proj = torch.nn.Linear(2, 2)
    policy.model.state_memory_proj = torch.nn.Linear(2, 2)
    policy.model.arm_mode_head = torch.nn.Linear(2, 3)
    policy.config = SimpleNamespace(
        optimizer_lr=2.5e-5,
        new_module_optimizer_lr_multiplier=40.0,
    )

    groups = policy.get_optim_params()

    assert [group["name"] for group in groups] == ["pretrained_and_action_expert", "new_modules"]
    assert "lr" not in groups[0]
    assert groups[1]["lr"] == pytest.approx(1e-3)
    assert set(groups[0]["params"]) == set(policy.model.action_out_proj.parameters())
    assert set(groups[1]["params"]) == {
        *policy.model.state_memory_proj.parameters(),
        *policy.model.arm_mode_head.parameters(),
    }


def test_structured_loss_excludes_discrete_dimensions_from_flow_matching() -> None:
    policy = PI05Policy.__new__(PI05Policy)
    torch.nn.Module.__init__(policy)
    policy.model = torch.nn.Module()
    policy.model.arm_mode_crf = LinearChainCRF(3)
    policy.model.gripper_state_crf = LinearChainCRF(2)
    names = ["b2_delta_x", "b2_delta_y", "b2_delta_yaw", *DATASET_ACTION_NAMES[3:]]
    policy.config = SimpleNamespace(
        action_feature_names=names,
        action_gripper_target_true_side="negative",
        action_bool_true_fractions={
            "arm_teleop_inactive": 0.25,
            "arm_reset": 0.1,
            "gripper_target": 0.5,
            "task_complete": 0.2,
        },
        action_bool_loss_weight=4.0,
        action_continuous_loss_weight=1.0,
        action_masked_continuous_min_weight=0.0,
        b2_action_representation="pose_delta",
        z1_action_representation="ee_delta",
    )
    actions = -torch.ones((1, 5, len(names)))
    actions[:, :, names.index("gripper_target")] = torch.tensor([[-1.0, -1.0, 1.0, 1.0, 1.0]])
    actions[:, :, names.index("task_complete")] = torch.tensor([[-1.0, -1.0, -1.0, 1.0, 1.0]])
    logits = {
        "arm_mode": torch.zeros((1, 5, 3)),
        "gripper_state": torch.zeros((1, 5, 2)),
        "task_complete": torch.zeros((1, 5)),
    }
    baseline_losses = torch.ones_like(actions)
    changed_losses = baseline_losses.clone()
    for name in ("arm_teleop_inactive", "arm_reset", "gripper_target", "task_complete"):
        changed_losses[:, :, names.index(name)] = 1_000_000.0

    ee_valid = torch.ones(1, 5, dtype=torch.bool)
    baseline, _ = policy._structured_temporal_action_loss(
        baseline_losses, logits, actions, "mean", None, ee_valid
    )
    changed, _ = policy._structured_temporal_action_loss(
        changed_losses, logits, actions, "mean", None, ee_valid
    )

    torch.testing.assert_close(changed, baseline)


def test_structured_ee_delta_loss_uses_both_endpoint_validity_mask() -> None:
    policy = PI05Policy.__new__(PI05Policy)
    torch.nn.Module.__init__(policy)
    policy.model = torch.nn.Module()
    policy.model.arm_mode_crf = LinearChainCRF(3)
    policy.model.gripper_state_crf = LinearChainCRF(2)
    names = ["b2_vx", "b2_vy", "b2_omega_z", *DATASET_ACTION_NAMES[3:]]
    policy.config = SimpleNamespace(
        action_feature_names=names,
        action_gripper_target_true_side="negative",
        action_bool_true_fractions={
            "arm_teleop_inactive": 0.25,
            "arm_reset": 0.1,
            "gripper_target": 0.5,
            "task_complete": 0.2,
        },
        action_bool_loss_weight=4.0,
        action_continuous_loss_weight=1.0,
        action_masked_continuous_min_weight=0.0,
        b2_action_representation="velocity",
        z1_action_representation="ee_delta",
    )
    actions = -torch.ones((1, 3, len(names)))
    logits = {
        "arm_mode": torch.zeros((1, 3, 3)),
        "gripper_state": torch.zeros((1, 3, 2)),
        "task_complete": torch.full((1, 3), -10.0),
    }
    losses = torch.ones_like(actions)
    validity = torch.tensor([[True, False, True]])
    _, info = policy._structured_temporal_action_loss(losses, logits, actions, "mean", None, validity)
    assert info["continuous_mask_frac/ee_pose"] == pytest.approx(2 / 3)


def test_structured_mode_removes_discrete_targets_from_flow_input() -> None:
    model = PI05Pytorch.__new__(PI05Pytorch)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        discrete_action_training_mode="structured_temporal",
        action_feature_names=["b2_vx", "arm_teleop_inactive", "arm_reset", "gripper_target", "task_complete"],
    )
    actions = torch.arange(10, dtype=torch.float32).reshape(2, 5)

    sanitized = model._zero_structured_discrete_channels(actions)

    torch.testing.assert_close(sanitized[:, 0], actions[:, 0])
    torch.testing.assert_close(sanitized[:, 1:], torch.zeros_like(sanitized[:, 1:]))
    torch.testing.assert_close(actions, torch.arange(10, dtype=torch.float32).reshape(2, 5))
