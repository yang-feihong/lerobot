"""Worker-local reuse of tokenized prompts and physical B2 action references."""

from collections import OrderedDict

import torch

from lerobot.processor.tokenizer_processor import TokenizerProcessorStep
from lerobot.types import TransitionKey


class TokenizedPromptCache:
    def __init__(self, step):
        self.step = step
        self.tokenize = step._tokenize_text
        self.tokenizer = step.input_tokenizer
        self.entries = OrderedDict()
        self.bytes = 0

    def __call__(self, text):
        step = self.step
        if step.input_tokenizer is not self.tokenizer:
            self.entries.clear()
            self.bytes = 0
            self.tokenizer = step.input_tokenizer
        texts = (text,) if isinstance(text, str) else tuple(text)
        key = (step.max_length, step.truncation,
               step.padding, step.padding_side, isinstance(text, str), texts)
        if key in self.entries:
            values, _ = self.entries[key]
            self.entries.move_to_end(key)
        else:
            values = self.tokenize(text)
            size = sum(v.numel() * v.element_size() for v in values.values())
            size += sum(len(v.encode("utf-8")) for v in texts)
            if size <= 1024 * 1024:
                values = {k: v.clone() for k, v in values.items()}
                self.entries[key] = (values, size)
                self.bytes += size
                while len(self.entries) > 16 or self.bytes > 1024 * 1024:
                    _, (_, evicted) = self.entries.popitem(last=False)
                    self.bytes -= evicted
        return {k: v.clone() for k, v in values.items()}


def cache_tokenized_prompts(processor):
    for step in processor.steps:
        if isinstance(step, TokenizerProcessorStep):
            step._tokenize_text = TokenizedPromptCache(step)


def bind_b2_reference_capture(processor, unnormalizer, action_names):
    if processor.steps[0] is not unnormalizer:
        return None
    return B2ReferenceCapture(processor, unnormalizer, action_names)


class B2ReferenceCapture:
    def __init__(self, processor, unnormalizer, action_names):
        self.index = next(i for i, step in enumerate(processor.steps) if step is unnormalizer)
        self.columns = [action_names.index(name) for name in ("b2_delta_x", "b2_delta_y", "b2_delta_yaw")]
        self.reference = None
        processor.register_after_step_hook(self)

    def __call__(self, index, transition):
        if index == self.index:
            # Capture before execution-space transforms can replace or mutate the action.
            self.reference = (transition[TransitionKey.ACTION][..., self.columns]
                              .detach().cpu().to(torch.float32).reshape(-1, 3))
