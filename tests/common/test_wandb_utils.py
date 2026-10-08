from types import SimpleNamespace

from lerobot.common.wandb_utils import WandBLogger, cfg_to_wandb_tags


class FakeWandB:
    def __init__(self):
        self.defined = []
        self.logged = []

    def define_metric(self, name, **kwargs):
        self.defined.append((name, kwargs))

    def log(self, data, step=None):
        self.logged.append((data, step))


def test_grouped_logger_preserves_top_level_groups() -> None:
    logger = WandBLogger.__new__(WandBLogger)
    logger._wandb = FakeWandB()
    logger._wandb_custom_step_key = None
    logger._define_default_metrics()

    logger.log_grouped_dict({"overview/train_loss": 1.0, "discrete_action/accuracy/arm_mode": 0.75}, step=12)

    data, step = logger._wandb.logged[-1]
    assert step == 12
    assert data == {
        "overview/train_loss": 1.0,
        "discrete_action/accuracy/arm_mode": 0.75,
        "optimizer_step": 12,
    }
    assert "train/overview/train_loss" not in data
    assert (
        "overview/train_loss",
        {"step_metric": "optimizer_step", "summary": "min"},
    ) in logger._wandb.defined
    assert (
        "overview/val_loss",
        {"step_metric": "optimizer_step", "summary": "min"},
    ) in logger._wandb.defined


def test_grouped_logger_ignores_non_scalar_values() -> None:
    logger = WandBLogger.__new__(WandBLogger)
    logger._wandb = FakeWandB()

    logger.log_grouped_dict({"overview/train_loss": 1.0, "ignored": SimpleNamespace()}, step=3)

    assert logger._wandb.logged[-1][0] == {"overview/train_loss": 1.0, "optimizer_step": 3}


def test_wandb_tags_capture_stage_membership_and_model_semantics() -> None:
    cfg = SimpleNamespace(
        is_reward_model_training=False,
        seed=1000,
        dataset=SimpleNamespace(
            repo_id="local/staff1",
            semantic_phases=["approach", "handle_press"],
            semantic_state_instruction_matrix={"approach": {}, "handle_press": {}},
            semantic_view_kind_weights={},
            image_source="real",
        ),
        env=None,
        policy=SimpleNamespace(
            type="pi05",
            mem_vit_enabled=True,
            state_action_encoding="continuous",
            b2_action_representation="pose_delta",
            z1_action_representation="ee_state_delta",
            action_predict_arm_teleop_inactive=True,
            action_predict_arm_reset=True,
            action_predict_task_complete=True,
            action_predict_task_blocked=False,
        ),
        wandb=SimpleNamespace(tags=["experiment:control"]),
    )

    tags = cfg_to_wandb_tags(cfg)

    assert "stage:stage1" in tags
    assert "stage:stage2" in tags
    assert "stage:stage3" not in tags
    assert "stage-set:stage1+stage2" in tags
    assert "memory:mem-vit" in tags
    assert "arm-mode-output:true" in tags
    assert "task-complete-output:true" in tags
    assert "task-blocked-output:false" in tags
    assert "experiment:control" in tags


def test_wandb_tags_do_not_guess_stages_from_dataset_name() -> None:
    cfg = SimpleNamespace(
        is_reward_model_training=False,
        seed=1,
        dataset=SimpleNamespace(
            repo_id="local/dataset_named_stage1",
            semantic_phases=[],
            semantic_state_instruction_matrix={},
            semantic_view_kind_weights={},
            image_source="real",
        ),
        env=None,
        policy=SimpleNamespace(type="pi05"),
        wandb=SimpleNamespace(tags=[]),
    )

    tags = cfg_to_wandb_tags(cfg)

    assert "stage-set:unspecified" in tags
    assert not any(tag.startswith("stage:stage") for tag in tags)
