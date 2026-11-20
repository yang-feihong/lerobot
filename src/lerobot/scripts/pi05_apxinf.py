"""Shared prepared-tensor boundary for version-specific PI0.5 native adapters.

Model loading and RTC API selection belong to the verified delivery adapter.
Image normalization, patch ordering, token selection and output slicing stay here.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

from lerobot.policies.pi05.modeling_pi05 import preprocess_pi05_images
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


class ApxInfPreparedPolicy:
    def eval(self):
        return self

    def _rtc_inputs(self, prefix, delay):
        raise NotImplementedError("A delivery adapter must implement explicit RTC conditioning")

    def _infer_prepared(self, patches, tokens, mask, noise, conditioning):
        return self.runner._infer_patches(
            patches, np.ascontiguousarray(tokens[0][mask[0]], dtype=np.uint32),
            noise[0], **conditioning,
        )

    def prepare_images_async(self, batch):
        """Prepare only immutable CPU images; native runner stays on its worker thread."""
        if getattr(self, "_image_executor", None) is None:
            self._image_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pi05-cpu-images")
        snapshot = dict(batch)

        def prepare():
            with torch.inference_mode():
                return preprocess_pi05_images(snapshot, self.config, torch.device("cpu"))

        return self._image_executor.submit(prepare)

    def predict_action_chunk(self, batch, *, noise=None, inference_delay=0, prev_chunk_left_over=None,
                             prepared_images=None):
        try:
            conditioning = self._rtc_inputs(prev_chunk_left_over, inference_delay)
            if self.runner is None:
                self.runner = self._runner_factory()
                if (self.runner.action_horizon, self.runner.action_dim, self.runner.num_views) != (50, 32, 2):
                    raise RuntimeError("Loaded ApxInf network dimensions differ from profile")
            images, masks, _ = (preprocess_pi05_images(batch, self.config, torch.device("cpu"))
                                if prepared_images is None else prepared_images.result())
        except BaseException:
            if prepared_images is not None:
                prepared_images.cancel()
            raise
        if len(images) != 2 or any(not bool(mask.all()) for mask in masks):
            raise ValueError("This ApxInf handoff requires both cameras")
        values = torch.stack(images, dim=1).detach().numpy()
        if values.shape != (1, 2, 3, 224, 224):
            raise ValueError("Expected batch size one and two 224x224 views")
        patches = values[0].reshape(2, 3, 16, 14, 16, 14).transpose(0, 2, 4, 1, 3, 5).reshape(512, 588)
        mask = batch[OBS_LANGUAGE_ATTENTION_MASK].bool().cpu().numpy()
        tokens = batch[OBS_LANGUAGE_TOKENS].cpu().numpy()
        if noise is None:
            noise = torch.randn(1, 50, 32)
        latent = noise.detach().cpu().numpy()
        if latent.shape != (1, 50, 32):
            raise ValueError("Expected flow noise [1,50,32]")
        result = np.asarray(
            self._infer_prepared(
                np.ascontiguousarray(patches, dtype=np.float32),
                np.ascontiguousarray(tokens, dtype=np.int64),
                np.ascontiguousarray(mask, dtype=np.bool_),
                np.ascontiguousarray(latent, dtype=np.float32),
                conditioning,
            ),
            dtype=np.float32,
        )
        if result.shape != (50, 32) or not np.isfinite(result).all():
            raise RuntimeError("Invalid ApxInf action tensor")
        return torch.from_numpy(result[None, :, :10].copy())

    predict_action_chunk_in_eval = predict_action_chunk

    def close(self):
        executor = getattr(self, "_image_executor", None)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
            self._image_executor = None
        self.runner = None
