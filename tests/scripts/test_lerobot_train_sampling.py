from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.policies.pi05.b2_action_transform import DATASET_ACTION_NAMES
from lerobot.scripts.lerobot_train import configure_action_bool_balance, resolve_task_complete_sampling
from lerobot.utils.constants import ACTION


class _Episodes:
    def __init__(self, length: int):
        self.rows = [{"dataset_from_index": 0, "dataset_to_index": length}]

    def __getitem__(self, item):
        if isinstance(item, str):
            return [row[item] for row in self.rows]
        return self.rows[item]


class _FakeHFDataset:
    def __init__(self, actions: np.ndarray):
        self.actions = actions

    def select_columns(self, _columns):
        return self

    def with_format(self, _format):
        return self

    def iter(self, batch_size):
        assert batch_size > 0
        yield {
            ACTION: self.actions,
            "episode_index": np.zeros(len(self.actions), dtype=np.int64),
            "frame_index": np.arange(len(self.actions), dtype=np.int64),
        }


def _fake_dataset(actions: np.ndarray):
    return SimpleNamespace(
        meta=SimpleNamespace(
            fps=10,
            features={ACTION: {"names": list(DATASET_ACTION_NAMES)}},
            episodes=_Episodes(len(actions)),
        ),
        episodes=[0],
        num_episodes=1,
        hf_dataset=_FakeHFDataset(actions),
    )


def test_completion_tail_sampling_is_monotonic_and_physically_capped():
    actions = np.zeros((8, 16), dtype=np.float32)
    actions[4:, 15] = 1
    resolved = resolve_task_complete_sampling(
        _fake_dataset(actions),
        SimpleNamespace(type="pi05", task_complete_sample_tail_seconds=0.2, action_predict_task_complete=True),
    )
    assert resolved == ([7], {0: 7})

    actions[6, 15] = 0
    with pytest.raises(ValueError, match="monotonic"):
        resolve_task_complete_sampling(
            _fake_dataset(actions),
            SimpleNamespace(type="pi05", task_complete_sample_tail_seconds=0.2, action_predict_task_complete=True),
        )


def test_completion_tail_sampling_excludes_episode_without_completion():
    actions = np.zeros((8, 16), dtype=np.float32)
    resolved = resolve_task_complete_sampling(
        _fake_dataset(actions),
        SimpleNamespace(type="pi05", task_complete_sample_tail_seconds=0.2, action_predict_task_complete=True),
    )
    assert resolved == ([0], {0: 0})


def test_completion_sampling_is_disabled_when_completion_is_not_an_output():
    actions = np.zeros((8, 16), dtype=np.float32)
    assert (
        resolve_task_complete_sampling(
            _fake_dataset(actions),
            SimpleNamespace(type="pi05", task_complete_sample_tail_seconds=0.2, action_predict_task_complete=False),
        )
        is None
    )


def test_all_boolean_priors_use_capped_starts_and_ignore_post_completion_controls():
    actions = np.zeros((5, 16), dtype=np.float32)
    actions[:, 3] = [1, 1, 0, 0, 0]
    actions[:, 4] = [0, 0, 1, 0, 0]
    actions[:, 14] = [0, -1, 0, -1, -1]
    actions[:, 15] = [0, 0, 0, 1, 1]
    policy_cfg = SimpleNamespace(
        type="pi05",
        action_predict_arm_teleop_inactive=True,
        action_predict_arm_reset=True,
        action_predict_gripper=True,
        action_predict_task_complete=True,
        gripper_target_representation="binary_position",
        action_gripper_target_true_side="negative",
        action_bool_loss_weight=4.0,
        action_bool_true_fractions={},
        chunk_size=3,
        control_frequency_hz=10,
    )
    stats = configure_action_bool_balance(
        SimpleNamespace(trainable_config=policy_cfg, resume=False), _fake_dataset(actions), {0: 4}
    )
    assert stats is not None
    assert (
        stats["arm_teleop_inactive"]["positive_labels"],
        stats["arm_teleop_inactive"]["negative_labels"],
    ) == (3, 3)
    assert (stats["arm_reset"]["positive_labels"], stats["arm_reset"]["negative_labels"]) == (3, 3)
    assert (stats["gripper_target"]["positive_labels"], stats["gripper_target"]["negative_labels"]) == (2, 4)
    assert (stats["task_complete"]["positive_labels"], stats["task_complete"]["negative_labels"]) == (5, 6)


def test_continuous_gripper_does_not_create_a_boolean_prior():
    actions = np.zeros((5, 16), dtype=np.float32)
    policy_cfg = SimpleNamespace(
        type="pi05",
        action_predict_arm_teleop_inactive=False,
        action_predict_arm_reset=False,
        action_predict_gripper=True,
        action_predict_task_complete=False,
        gripper_target_representation="continuous_position",
    )

    stats = configure_action_bool_balance(
        SimpleNamespace(trainable_config=policy_cfg, resume=False),
        _fake_dataset(actions),
    )

    assert stats is None
