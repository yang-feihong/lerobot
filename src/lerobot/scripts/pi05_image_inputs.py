"""Bounded, pixel-exact image preparation for serialized inference workers."""

from collections import OrderedDict
from collections.abc import Callable

import numpy as np
import torch

from lerobot.configs import FeatureType, NormalizationMode
from lerobot.processor import NormalizerProcessorStep


def images_are_unchanged(preprocessor) -> bool:
    allowed = {
        "rename_observations_processor",
        "to_batch_processor",
        "relative_actions_processor",
        "pi05_action_representation_processor",
        "normalizer_processor",
        "pi05_prepare_state_tokenizer_processor_step",
        "tokenizer_processor",
        "device_processor",
        "pi05_split_action_history_processor",
    }
    for step in preprocessor.steps:
        if type(step).__dict__.get("_registry_name") not in allowed:
            return False
        if type(step).__dict__.get("_registry_name") == "device_processor" and step.float_dtype not in {
            None,
            "float",
            "float32",
            "double",
            "float64",
        }:
            return False
        if isinstance(step, NormalizerProcessorStep):
            visual = any(feature.type == FeatureType.VISUAL for feature in step.features.values())
            if visual and step.norm_map[FeatureType.VISUAL] != NormalizationMode.IDENTITY:
                return False
    return True


class ImageTensorCache:
    def __init__(
        self,
        camera_keys: tuple[str, ...],
        frames: int,
        resolution: tuple[int, int],
        device: torch.device,
        decode: Callable,
        resize: Callable,
        max_cache_bytes: int = 64 * 1024 * 1024,
        max_pinned_bytes: int = 32 * 1024 * 1024,
    ):
        if frames < 1 or min(resolution) < 1 or min(max_cache_bytes, max_pinned_bytes) < 0:
            raise ValueError("Invalid image cache dimensions or budgets")
        self.camera_keys, self.frames, self.resolution = camera_keys, frames, resolution
        self.device, self.decode, self.resize = torch.device(device), decode, resize
        self.capacity = min(64, max_cache_bytes // (3 * resolution[0] * resolution[1] * 4))
        self.max_pinned_bytes = max_pinned_bytes
        self.cache = OrderedDict()
        self.buffers = {}
        # A CPU-built lookup preserves CPU float32 division rounding on CUDA.
        self.levels = torch.arange(256, dtype=torch.float32).div_(255.0).to(self.device)
        self.device = self.levels.device
        self.ready = torch.cuda.Event() if self.device.type == "cuda" else None

    def clear(self):
        self.cache.clear()

    def _upload(self, key: str, images: list[np.ndarray]) -> torch.Tensor:
        shape = images[0].shape
        needed = self.frames * int(np.prod(shape))
        if self.device.type == "cuda" and needed <= self.max_pinned_bytes:
            previous = self.buffers.get(key)
            if previous is not None:
                previous[1].synchronize()
            if previous is None or tuple(previous[0].shape[1:]) != shape:
                host = torch.empty((self.frames, *shape), dtype=torch.uint8, pin_memory=True)
                copied = torch.cuda.Event()
                self.buffers[key] = host, copied
            else:
                host, copied = previous
            host = host[: len(images)]
            for slot, image in enumerate(images):
                np.copyto(host.numpy()[slot], image)
            uploaded = host.to(self.device, non_blocking=True)
            copied.record(torch.cuda.current_stream(self.device))
        else:
            uploaded = torch.from_numpy(np.stack(images)).to(self.device)
        return uploaded.permute(0, 3, 1, 2).contiguous()

    @torch.no_grad()
    def prepare(self, windows: dict[str, list[np.ndarray]], *, current=(), decoded=None) -> dict[str, torch.Tensor]:
        stream = torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
        if stream is not None:
            stream.wait_event(self.ready)
        try:
            result = {}
            contents_by_key, available_by_key, missing_by_key = {}, {}, {}
            known = set()
            decode_requests = {}
            for key in self.camera_keys:
                if len(windows[key]) != self.frames:
                    raise ValueError("Image window frame count differs from the configured history")
                contents = [image.tobytes() for image in windows[key]]
                available = {}
                for content in contents:
                    identity = key, content
                    if identity in self.cache:
                        available[content] = self.cache[identity]
                        self.cache.move_to_end(identity)
                missing = list(dict.fromkeys(content for content in contents if content not in available))
                contents_by_key[key], available_by_key[key], missing_by_key[key] = contents, available, missing
                known.update(available)
                for content in missing:
                    decode_requests.setdefault(content, key)
            for image in current:
                content = image.tobytes()
                if content not in known:
                    decode_requests.setdefault(content, "current")
            decoded_by_content = {} if decoded is None else dict(decoded)
            pending = [(content, key) for content, key in decode_requests.items() if content not in decoded_by_content]
            if pending:
                decoded_by_content.update(zip((content for content, _key in pending), self.decode(
                    [(np.frombuffer(content, dtype=np.uint8), key) for content, key in pending]
                ), strict=True))
            for key in self.camera_keys:
                contents, available, missing = contents_by_key[key], available_by_key[key], missing_by_key[key]
                if missing:
                    new_images = [decoded_by_content[content] for content in missing]
                    shapes = {image.shape for image in new_images}
                    shapes.update(entry[0] for entry in available.values())
                    if len(shapes) != 1:
                        raise ValueError("Image shapes differ within one camera window")
                    pixels = self.levels[self._upload(key, new_images).long()]
                    resized = self.resize(pixels.permute(0, 2, 3, 1), *self.resolution).permute(0, 3, 1, 2)
                    for content, image in zip(missing, resized, strict=True):
                        entry = new_images[0].shape, image.clone()
                        available[content] = entry
                        if self.capacity:
                            self.cache[key, content] = entry
                            self.cache.move_to_end((key, content))
                            while len(self.cache) > self.capacity:
                                self.cache.popitem(last=False)
                elif len({entry[0] for entry in available.values()}) != 1:
                    raise ValueError("Image shapes differ within one camera window")
                frames = [available[content][1] for content in contents]
                if stream is not None:
                    for image in frames:
                        image.record_stream(stream)
                result[key] = torch.stack(frames).unsqueeze(0)
            return result
        finally:
            if stream is not None:
                self.ready.record(stream)
