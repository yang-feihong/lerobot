from types import SimpleNamespace

import pytest
import torch

from lerobot.scripts.pi05_processor_cache import B2ReferenceCapture, TokenizedPromptCache, bind_b2_reference_capture
from lerobot.types import TransitionKey


def tokenizer_step():
    step = SimpleNamespace(input_tokenizer=object(), max_length=512, truncation=True,
                           padding="max_length", padding_side="right", calls=0)

    def tokenize(text):
        step.calls += 1
        texts = [text] if isinstance(text, str) else text
        return dict(input_ids=torch.tensor([[len(t), step.max_length] for t in texts]),
                    attention_mask=torch.ones((len(texts), 2), dtype=torch.int64))

    step._tokenize_text = tokenize
    return step


def test_text_cache_is_owned_and_keys_include_complete_prompt_and_config():
    step = tokenizer_step()
    cache = TokenizedPromptCache(step)
    first = cache(["拿起水瓶"])
    saved = first["input_ids"].clone()
    first["input_ids"].zero_()
    torch.testing.assert_close(cache(["拿起水瓶"])["input_ids"], saved)
    assert step.calls == 1
    cache(["拿起水瓶, State: 1"])
    cache(["拿起水瓶, State: 2"])
    step.padding_side = "left"
    cache(["拿起水瓶"])
    step.max_length = 256
    cache(["拿起水瓶"])
    step.input_tokenizer = object()
    cache(["拿起水瓶"])
    assert step.calls == 6


def test_text_cache_is_bounded_and_preserves_batch_order():
    step = tokenizer_step()
    cache = TokenizedPromptCache(step)
    values = cache(["a", "long"])
    assert values["input_ids"][:, 0].tolist() == [1, 4]
    assert cache(["long", "a"])["input_ids"][:, 0].tolist() == [4, 1]
    for i in range(30):
        cache([str(i)])
    assert len(cache.entries) <= 16 and cache.bytes <= 1024 * 1024
    before = step.calls
    cache(["x" * (1024 * 1024 + 1)])
    cache(["x" * (1024 * 1024 + 1)])
    assert step.calls == before + 2


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reference_capture_precedes_execution_transform_and_does_not_alias(dtype):
    normalizer = object()
    hooks = []
    pipeline = SimpleNamespace(steps=[object(), normalizer, object()],
                               register_after_step_hook=hooks.append)
    names = ("extra", "b2_delta_yaw", "b2_delta_x", "b2_delta_y")
    capture = B2ReferenceCapture(pipeline, normalizer, names)
    action = torch.arange(40, dtype=dtype).reshape(1, 10, 4)
    expected = action[0, :, [2, 3, 1]].float().clone()
    hooks[0](0, {TransitionKey.ACTION: action})
    assert capture.reference is None
    hooks[0](1, {TransitionKey.ACTION: action})
    action.zero_()
    torch.testing.assert_close(capture.reference, expected, rtol=0, atol=0)
    capture.reference = None
    hooks[0](1, {TransitionKey.ACTION: action})
    assert not capture.reference.any()


def test_replacement_pipeline_has_independent_capture():
    def pipeline():
        step = object()
        hooks = []
        return SimpleNamespace(steps=[step], register_after_step_hook=hooks.append), step, hooks

    names = ("b2_delta_x", "b2_delta_y", "b2_delta_yaw")
    first, step, hooks = pipeline()
    old = B2ReferenceCapture(first, step, names)
    second, step, new_hooks = pipeline()
    new = B2ReferenceCapture(second, step, names)
    hooks[0](0, {TransitionKey.ACTION: torch.zeros(1, 2, 3)})
    new_hooks[0](0, {TransitionKey.ACTION: torch.ones(1, 2, 3)})
    assert not old.reference.any() and new.reference.all()


def test_shared_reference_is_only_bound_when_raw_actions_enter_unnormalizer_first():
    step = object()
    hooks = []
    names = ("b2_delta_x", "b2_delta_y", "b2_delta_yaw")
    pipeline = SimpleNamespace(steps=[object(), step], register_after_step_hook=hooks.append)
    assert bind_b2_reference_capture(pipeline, step, names) is None
    assert not hooks
    pipeline.steps = [step]
    assert bind_b2_reference_capture(pipeline, step, names) is hooks[0]
