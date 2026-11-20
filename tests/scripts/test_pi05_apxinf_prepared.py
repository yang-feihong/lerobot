from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from lerobot.scripts import pi05_apxinf as module
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def test_shared_boundary_uses_delivery_specific_conditioning(monkeypatch):
    policy = module.ApxInfPreparedPolicy()
    policy.config = SimpleNamespace()
    policy._rtc_inputs = Mock(return_value={"prefix":"explicit"})
    policy.runner = SimpleNamespace(_infer_patches=Mock(return_value=np.ones((50,32), np.float32)))
    images = [torch.arange(3*224*224, dtype=torch.float32).reshape(1,3,224,224)]*2
    monkeypatch.setattr(module, "preprocess_pi05_images", lambda *args: (images, [torch.tensor([True])]*2, None))
    batch = {OBS_LANGUAGE_TOKENS:torch.tensor([[9,8,0]]), OBS_LANGUAGE_ATTENTION_MASK:torch.tensor([[True,True,False]])}
    out = policy.predict_action_chunk(batch, noise=torch.zeros(1,50,32), inference_delay=5, prev_chunk_left_over="previous")
    policy._rtc_inputs.assert_called_once_with("previous", 5)
    call = policy.runner._infer_patches.call_args
    assert call.kwargs == {"prefix":"explicit"}
    np.testing.assert_array_equal(call.args[1], [9,8])
    patches = call.args[0]
    assert patches.shape == (512,588) and patches.flags.c_contiguous
    np.testing.assert_array_equal(patches[0], images[0][0,:,:14,:14].reshape(-1).numpy())
    assert out.shape == (1,50,10) and out.eq(1).all()


def test_no_legacy_loader_or_implicit_rtc():
    assert not hasattr(module, "ApxInfPolicy")
    with pytest.raises(NotImplementedError):
        module.ApxInfPreparedPolicy()._rtc_inputs(None, 0)
