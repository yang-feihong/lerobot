from types import SimpleNamespace

from lerobot.common.wandb_utils import WandBLogger


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
