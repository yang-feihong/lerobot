"""Clean action prefixes for training-time real-time chunking.

LeRobot's flow convention is noise at t=1 and clean actions at t=0. Prefix
tokens therefore receive t=0 while uncommitted tokens keep the sampled time.
"""

import torch
from torch import Tensor

from .configuration_rtc import TrainingRTCConfig


def sample_training_rtc_delay(
    config: TrainingRTCConfig,
    batch_size: int,
    device: torch.device | str,
    action_is_pad: Tensor | None = None,
) -> Tensor:
    """Sample per-example delays, retaining supervision near episode ends."""
    if config.delay_distribution == "uniform":
        delay = torch.randint(config.simulated_delay, (batch_size,), device=device)
    else:
        logits = -torch.arange(config.simulated_delay, device=device, dtype=torch.float32)
        delay = torch.multinomial(logits.softmax(dim=0), batch_size, replacement=True)
    if action_is_pad is not None:
        if action_is_pad.ndim != 2 or action_is_pad.shape[0] != batch_size:
            raise ValueError("action_is_pad must have shape [batch_size, horizon]")
        if action_is_pad.dtype != torch.bool:
            raise ValueError("action_is_pad must be boolean")
        # Stop at the first padded token; a conditioned prefix must consist of
        # real actions, and at least one real target must remain when available.
        valid_prefix = (~action_is_pad.to(device=device)).long().cumprod(dim=1).sum(dim=1)
        delay = torch.minimum(delay, (valid_prefix - 1).clamp_min(0))
    return delay


def _validate_delay(delay: Tensor, batch_size: int, horizon: int) -> None:
    if delay.ndim != 1 or delay.shape[0] != batch_size:
        raise ValueError("RTC delay must have shape [batch_size]")
    if delay.dtype not in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}:
        raise ValueError("RTC delay must contain integers")
    if torch.any(delay < 0) or torch.any(delay >= horizon):
        raise ValueError(f"RTC delay must be in [0, {horizon - 1}] for this action horizon")


def make_training_rtc_time(time: Tensor, delay: Tensor, horizon: int) -> tuple[Tensor, Tensor]:
    """Return per-token flow times and the clean prefix mask, both [B, H]."""
    if time.ndim != 1:
        raise ValueError("Flow time must have shape [batch_size]")
    _validate_delay(delay, time.shape[0], horizon)
    prefix_mask = torch.arange(horizon, device=time.device)[None, :] < delay.to(time.device)[:, None]
    return time[:, None].expand(-1, horizon).masked_fill(prefix_mask, 0), prefix_mask


def prepare_training_rtc_prefix(
    noise: Tensor,
    prefix: Tensor | None,
    inference_delay: int | Tensor | None,
    config: TrainingRTCConfig,
) -> tuple[Tensor, Tensor]:
    """Validate and pad normalized committed actions without clipping delays.

    A [T, D] prefix or a singleton batch broadcasts across the noise batch.
    Short action vectors are zero-padded to the model action dimension.
    """
    if noise.ndim != 3:
        raise ValueError("Action noise must have shape [batch_size, horizon, action_dim]")
    batch_size, horizon, action_dim = noise.shape
    if inference_delay is None:
        inference_delay = 0
    if isinstance(inference_delay, Tensor):
        delay = inference_delay.to(device=noise.device)
        if delay.ndim == 0:
            delay = delay.expand(batch_size)
    elif type(inference_delay) is int:
        delay = torch.full((batch_size,), inference_delay, device=noise.device, dtype=torch.long)
    else:
        raise ValueError("inference_delay must be an integer or integer tensor")
    _validate_delay(delay, batch_size, horizon)
    if torch.any(delay >= config.simulated_delay):
        raise ValueError(
            f"inference_delay exceeds trained support [0, {config.simulated_delay - 1}]; "
            "fine-tune with a larger training_rtc_config.simulated_delay"
        )
    prefix_mask = torch.arange(horizon, device=noise.device)[None, :] < delay[:, None]
    content = torch.zeros_like(noise)
    if prefix is None:
        if torch.any(delay > 0):
            raise ValueError("A positive inference_delay requires a supplied action prefix")
        return content, prefix_mask
    if prefix.ndim == 2:
        prefix = prefix.unsqueeze(0)
    if prefix.ndim != 3 or prefix.shape[0] not in {1, batch_size}:
        raise ValueError("Action prefix must have shape [T, D], [1, T, D], or [batch_size, T, D]")
    if not 0 < prefix.shape[2] <= action_dim:
        raise ValueError("Action prefix dimension must be positive and no larger than model action_dim")
    if torch.any(delay > prefix.shape[1]):
        raise ValueError("Action prefix is shorter than inference_delay")
    prefix = prefix.to(device=noise.device, dtype=noise.dtype)
    if not torch.isfinite(prefix).all():
        raise ValueError("Action prefix must contain only finite values")
    copied_steps = min(horizon, prefix.shape[1])
    content[:, :copied_steps, : prefix.shape[2]] = prefix[:, :copied_steps]
    return content, prefix_mask
