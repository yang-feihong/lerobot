from types import SimpleNamespace

import numpy as np
import torch

from lerobot.datasets.mixture_sampling import DatasetMixtureSource
from lerobot.scripts.lerobot_train import (
    _is_eval_policy_metric,
    _wandb_eval_metrics,
    _wandb_train_metrics,
    attach_loss_semantics,
    build_episode_semantic_lookup,
    select_balanced_eval_indices,
)


def test_train_metrics_are_grouped_and_action_dimensions_use_names() -> None:
    tracker = {
        "loss": 0.4,
        "steps": 20,
        "epochs": 1.5,
        "lr": 2.5e-5,
        "gpu_mem_gb": 12.3,
        "loss_contribution/train/handle_press/z1": 0.07,
    }
    policy = {
        "loss": 0.4,
        "loss_per_dim": [0.1, None, 0.3],
        "continuous_loss": 0.2,
        "discrete_loss/arm_mode": 0.5,
        "discrete_accuracy/arm_mode": 0.8,
        "state_num_frames": 13.0,
        "loss_contribution/b2": 0.12,
        "loss_contribution/handle_press/z1": 0.08,
        "loss_weight_fraction/b2": 0.4,
        "loss_semantic_fraction/approach": 1.0,
        "custom_scalar": 7.0,
    }

    grouped = _wandb_train_metrics(tracker, policy, ["b2_vx", "arm_reset", "ee_x"])

    assert grouped["overview/train_loss"] == 0.4
    assert grouped["optimization/learning_rate"] == 2.5e-5
    assert grouped["performance/gpu_memory_gb"] == 12.3
    assert grouped["continuous_action/loss"] == 0.2
    assert grouped["discrete_action/loss/arm_mode"] == 0.5
    assert grouped["discrete_action/accuracy/arm_mode"] == 0.8
    assert grouped["memory/state/num_frames"] == 13.0
    assert grouped["loss_contribution/train/b2"] == 0.12
    assert grouped["loss_contribution/train/handle_press/z1"] == 0.07
    assert grouped["loss_weight_fraction/train/b2"] == 0.4
    assert grouped["loss_semantic_fraction/train/approach"] == 1.0
    assert grouped["action_dimensions/train/b2_vx"] == 0.1
    assert grouped["action_dimensions/train/ee_x"] == 0.3
    assert "action_dimensions/train/arm_reset" not in grouped
    assert grouped["policy_diagnostics/custom_scalar"] == 7.0
    assert all(not key.startswith("train/") for key in grouped)


def test_eval_metrics_share_overview_and_domain_groups() -> None:
    grouped = _wandb_eval_metrics(
        0.3,
        25,
        {
            "continuous_loss": 0.2,
            "discrete_loss/task_complete": 0.4,
            "discrete_accuracy/task_complete": 0.9,
            "task_complete_pred_frac/true": 0.08,
            "task_complete_precision/true": 0.7,
            "task_complete_recall/true": 0.6,
            "loss_contribution/b2": 0.12,
            "loss_contribution/traversal/z1": 0.08,
            "loss_weight_fraction/b2": 0.4,
            "loss_semantic_fraction/approach": 1.0,
        },
        {"b2_vx": 0.11, "ee_x": 0.22},
    )

    assert grouped == {
        "overview/val_loss": 0.3,
        "data/val_batches": 25,
        "continuous_action/val_loss": 0.2,
        "discrete_action/val_loss/task_complete": 0.4,
        "discrete_action/val_accuracy/task_complete": 0.9,
        "discrete_action/val_class_metrics/task_complete_pred_frac/true": 0.08,
        "discrete_action/val_class_metrics/task_complete_precision/true": 0.7,
        "discrete_action/val_class_metrics/task_complete_recall/true": 0.6,
        "loss_contribution/val/b2": 0.12,
        "loss_contribution/val/traversal/z1": 0.08,
        "loss_weight_fraction/val/b2": 0.4,
        "loss_semantic_fraction/val/approach": 1.0,
        "action_dimensions/val/b2_vx": 0.11,
        "action_dimensions/val/ee_x": 0.22,
    }


def test_eval_policy_metric_filter_keeps_class_diagnostics() -> None:
    assert _is_eval_policy_metric("continuous_loss", 0.2)
    assert _is_eval_policy_metric("discrete_loss/arm_mode", 0.3)
    assert _is_eval_policy_metric("discrete_accuracy/gripper_target", 0.8)
    assert _is_eval_policy_metric("arm_mode_pred_frac/reset", 0.1)
    assert _is_eval_policy_metric("task_complete_precision/true", 0.7)
    assert _is_eval_policy_metric("gripper_target_recall/closed", 0.6)
    assert _is_eval_policy_metric("loss_contribution/b2", 0.1)
    assert _is_eval_policy_metric("loss_weight_fraction/b2", 0.4)
    assert _is_eval_policy_metric("loss_semantic_fraction/approach", 1.0)
    assert not _is_eval_policy_metric("loss_per_dim", [0.1])
    assert not _is_eval_policy_metric("custom_scalar", 1.0)


def test_eval_selection_spreads_samples_across_time_and_sources() -> None:
    tasks = np.zeros(12, dtype=np.int64)
    sources = np.asarray([0] * 6 + [1] * 6, dtype=np.int64)

    selected = select_balanced_eval_indices(tasks, 6, source_indices=sources)

    assert selected == [0, 2, 5, 6, 8, 11]

    weighted = select_balanced_eval_indices(
        tasks,
        8,
        source_indices=sources,
        source_weights={0: 0.75, 1: 0.25},
    )
    assert weighted == [0, 1, 2, 3, 4, 5, 6, 11]


def test_semantic_loss_lookup_aggregates_sources_with_the_same_meaning() -> None:
    dataset = SimpleNamespace(
        meta=SimpleNamespace(total_episodes=5, info=SimpleNamespace(semantic_type=None))
    )
    sources = [
        DatasetMixtureSource("staff1_press", "handle_press", 0.4, 0, 2),
        DatasetMixtureSource("independent_press", "handle_press", 0.4, 2, 4),
        DatasetMixtureSource("new_task", "handover", 0.2, 4, 5),
    ]

    lookup, names = build_episode_semantic_lookup(dataset, sources)
    batch = {"episode_index": torch.tensor([0, 3, 4])}
    attach_loss_semantics(batch, lookup, names)

    assert names == ("handle_press", "handover")
    assert batch["loss_semantic_id"].tolist() == [0, 0, 1]
    assert batch["loss_semantic_names"] == names


def test_single_source_dataset_uses_its_declared_semantic_type() -> None:
    dataset = SimpleNamespace(
        meta=SimpleNamespace(total_episodes=3, info=SimpleNamespace(semantic_type="approach"))
    )

    lookup, names = build_episode_semantic_lookup(dataset, None)

    assert lookup.tolist() == [0, 0, 0]
    assert names == ("approach",)
