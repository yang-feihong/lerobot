from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from lerobot.policies.pi05.modeling_pi05 import PI05Policy


class TinyPolicy(PI05Policy):
    constructions = []

    def __init__(self, config, initialize_backbone=True):
        torch.nn.Module.__init__(self)
        self.config = config
        self.constructions.append(initialize_backbone)
        self.model = torch.nn.Module()
        self.model.paligemma_with_expert = torch.nn.Module()
        self.model.paligemma_with_expert.paligemma = torch.nn.Linear(2, 2)
        self.model.extra = torch.nn.Parameter(torch.full((2,), 7.0))
        if initialize_backbone:
            with torch.no_grad():
                self.model.paligemma_with_expert.paligemma.weight.fill_(3)
                self.model.paligemma_with_expert.paligemma.bias.fill_(4)

    @staticmethod
    def _fix_pytorch_state_dict_keys(state, config):
        return state


def load_tiny(tmp_path, tensors):
    TinyPolicy.constructions = []
    save_file(tensors, tmp_path / "model.safetensors")
    return TinyPolicy.from_pretrained(
        tmp_path,
        config=SimpleNamespace(mem_vit_checkpoint=None),
        restore_checkpoint_camera_metadata=False,
        initialize_backbone=False,
        strict=False,
    )


def test_fast_loading_preserves_initialized_non_backbone_parameters(tmp_path):
    prefix = "model.paligemma_with_expert.paligemma."
    policy = load_tiny(tmp_path, {
        prefix + "weight": torch.ones(2, 2),
        prefix + "bias": torch.ones(2),
    })
    assert TinyPolicy.constructions == [False]
    assert torch.equal(policy.model.extra, torch.full((2,), 7.0))
    assert torch.equal(policy.model.paligemma_with_expert.paligemma.weight, torch.ones(2, 2))


def test_missing_backbone_weights_fall_back_to_normal_initialization(tmp_path):
    policy = load_tiny(tmp_path, {
        "model.paligemma_with_expert.paligemma.weight": torch.ones(2, 2),
    })
    assert TinyPolicy.constructions == [False, True]
    assert torch.equal(policy.model.paligemma_with_expert.paligemma.bias, torch.full((2,), 4.0))
    assert torch.equal(policy.model.paligemma_with_expert.paligemma.weight, torch.ones(2, 2))
