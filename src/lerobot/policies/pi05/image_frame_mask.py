"""Training-only frame corruption, keeping an uncorrupted camera at each time."""

import torch
from torch import Tensor


def mask_image_frames(
    images: list[Tensor],
    camera_masks: list[Tensor],
    memory_masks: list[Tensor | None] | None,
    *,
    probability: float,
    training: bool,
) -> list[Tensor]:
    """With probability p per (sample, time), replace one valid camera by noise.

    Images have already been normalized to [-1, 1]. Padding/missing cameras
    remain untouched; a sole available camera is never corrupted. This is pixel
    augmentation, not an attention mask: all existing validity masks stay intact.
    """
    if not training or probability == 0 or len(images) < 2:
        return images

    valid = []
    for index, (image, camera_mask) in enumerate(zip(images, camera_masks, strict=True)):
        frames = image.shape[1] if image.ndim == 5 else 1
        present = camera_mask[:, None].expand(-1, frames)
        if memory_masks is not None and memory_masks[index] is not None:
            present = present & memory_masks[index]
        valid.append(present)
    available = torch.stack(valid, dim=-1)  # [B, K, cameras]
    scores = torch.rand(available.shape, device=images[0].device).masked_fill(~available, -1)
    camera = scores.argmax(dim=-1)
    corrupt = (torch.rand(camera.shape, device=camera.device) < probability) & (
        available.sum(dim=-1) >= 2
    )

    result = []
    for index, image in enumerate(images):
        selected = corrupt & (camera == index)
        if image.ndim == 4:
            selected = selected[:, 0]
        augmented = image.clone()
        augmented[selected] = torch.rand_like(image[selected]) * 2 - 1
        result.append(augmented)
    return result
