"""A shared candidate scorer: parameter count is independent of preset count."""

import math

import torch
from torch import nn
from torch.nn import functional

from lerobot.utils.reset_presets import validate_reset_presets


class ResetPresetHead(nn.Module):
    def __init__(self, feature_dim, catalog, hidden_dim=64):
        super().__init__()
        validate_reset_presets(catalog)
        self.query = nn.Linear(feature_dim, hidden_dim)
        self.candidate = nn.Sequential(
            nn.Linear(12, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        # The checkpoint config owns the catalog; do not introduce N-dependent weights/buffers.
        self.register_buffer("targets", torch.tensor([p["joint_target"] for p in catalog]), persistent=False)

    def forward(self, features):
        q = self.query(features)
        targets = self.targets.to(dtype=q.dtype)
        k = self.candidate(torch.cat((targets.sin(), targets.cos()), dim=-1))
        return q @ k.T / math.sqrt(q.shape[-1])


def reset_preset_loss(logits, labels, valid, action_is_pad=None, reduction="mean"):
    if labels.shape != logits.shape[:-1] or valid.shape != labels.shape:
        raise ValueError("Reset preset targets must align with the action horizon")
    labels, valid = labels.to(logits.device).long(), valid.to(logits.device).bool()
    if action_is_pad is not None:
        valid = valid & ~action_is_pad.to(logits.device).bool()
    if ((labels[valid] < 0) | (labels[valid] >= logits.shape[-1])).any():
        raise ValueError("Reset target class is outside the checkpoint catalog")
    ce = functional.cross_entropy(logits.transpose(1, 2), labels.masked_fill(~valid, 0), reduction="none")
    # Preserve a differentiable zero on reset-free batches, including DDP ranks.
    sums = (ce * valid).sum(dim=1)
    counts = valid.sum(dim=1)
    if reduction == "none":
        return sums / counts.clamp_min(1)
    if reduction != "mean":
        raise ValueError(f"Unknown reduction {reduction!r}")
    return sums.sum() / counts.sum().clamp_min(1)
