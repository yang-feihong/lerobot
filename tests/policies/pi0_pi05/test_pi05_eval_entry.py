from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


class Core(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mode_changes = 0
        self.calls = 0

    def train(self, mode=True):
        self.mode_changes += 1
        return super().train(mode)

    def sample_actions(self, *_args, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        return torch.arange(12).reshape(1, 3, 4).float()


def test_caller_owned_eval_mode_preserves_outputs_and_conditioning():
    policy = PI05Policy.__new__(PI05Policy)
    torch.nn.Module.__init__(policy)
    policy.model = Core()
    policy.config = SimpleNamespace(
        state_action_encoding="discrete",
        mem_vit_enabled=False,
        action_history_enabled=False,
        discrete_action_training_mode="continuous_flow",
        action_predict_reset_preset=False,
        output_features={ACTION: SimpleNamespace(shape=(2,))},
    )
    policy._mem_vit_window_plan = lambda *_args, **_kwargs: (1, None, None)
    policy._preprocess_images = lambda *_args, **_kwargs: (
        [torch.zeros(1, 3, 8, 8)],
        [torch.ones(1, dtype=torch.bool)],
        [None],
    )
    batch = {
        OBS_LANGUAGE_TOKENS: torch.ones(1, 3, dtype=torch.int64),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 3, dtype=torch.bool),
    }
    with pytest.raises(RuntimeError, match="requires policy.eval"):
        policy.predict_action_chunk_in_eval(batch)
    noise = torch.ones(1, 3, 4)
    regular = policy.predict_action_chunk(batch, noise=noise, inference_delay=2)
    changes = policy.model.mode_changes
    direct = policy.predict_action_chunk_in_eval(batch, noise=noise, inference_delay=2)
    assert torch.equal(regular, direct) and direct.shape == (1, 3, 2)
    assert policy.model.mode_changes == changes
    assert policy.model.kwargs == {"noise": noise, "inference_delay": 2}
    policy.train()
    with pytest.raises(RuntimeError):
        policy.predict_action_chunk_in_eval(batch)
    assert torch.equal(policy.predict_action_chunk(batch), regular)
