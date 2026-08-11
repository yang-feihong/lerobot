#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import builtins
import logging
import math
import types
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available, require_package

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers.cache_utils import DynamicCache
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma

    from ..pi_gemma import (
        PaliGemmaForConditionalGenerationWithPiGemma,
        PiGemmaForCausalLM,
        _gated_residual,
        layernorm_forward,
    )
else:
    CONFIG_MAPPING = None
    DynamicCache = None
    modeling_gemma = None
    PiGemmaForCausalLM = None
    _gated_residual = None
    layernorm_forward = None
    PaliGemmaForConditionalGenerationWithPiGemma = None
from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    OPENPI_ATTENTION_MASK_VALUE,
)

from ..pretrained import PreTrainedPolicy, T
from ..rtc.modeling_rtc import RTCProcessor
from .configuration_pi05 import DEFAULT_IMAGE_SIZE, PI05Config

OBS_ACTION_HISTORY = "observation.action_history"


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None
    state: Tensor | None
    action_history: Tensor | None
    image_memory_masks: list[Tensor | None] | None
    action_history_mask: Tensor | None
    state_memory_mask: Tensor | None


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):  # see openpi `sample_beta` (exact copy)
    # Beta sampling uses _sample_dirichlet which isn't implemented for MPS, so sample on CPU
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)


def make_att_2d_masks(pad_masks, att_masks):  # see openpi `make_att_2d_masks` (exact copy)
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def clone_past_key_values(past_key_values):
    """Clone the DynamicCache returned by prefix prefill for compiled denoising."""
    return DynamicCache(
        tuple(
            (keys.clone(), values.clone(), sliding_window) for keys, values, sliding_window in past_key_values
        )
    )


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad_torch(  # see openpi `resize_with_pad_torch` (exact copy)
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else 0.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    return padded_images


# Define the complete layer computation function for gradient checkpointing
def compute_layer_complete(inputs_embeds, attention_mask, position_ids, adarms_cond, layers, rotary_emb):
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = layers[i]
        hidden_states, gate = layernorm_forward(layer.input_layernorm, hidden_states, adarms_cond[i])
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    # Concatenate and process attention
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    paligemma_layer = layers[0]
    scaling = paligemma_layer.self_attn.scaling
    # Attention computation
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma_layer.self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    # Get head_dim from the current layer, not from the model
    head_dim = paligemma_layer.self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = layers[i]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        out_emb = _gated_residual(hidden_states, out_emb, gates[i])
        after_first_residual = out_emb.clone()
        out_emb, gate = layernorm_forward(layer.post_attention_layernorm, out_emb, adarms_cond[i])
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = _gated_residual(after_first_residual, out_emb, gate)
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


class GemmaConfig:  # see openpi `gemma.py: Config`
    """Configuration for Gemma model variants."""

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


MEM_VIT_NUM_FRAMES = 1
MEM_VIT_TEMPORAL_EVERY = 4
MEM_VIT_USE_ORIGINAL_FOR_K1 = False


def _mem_temporal_pos_emb(
    num_frames: int,
    dim: int,
    device,
    dtype,
    *,
    step_scale: float = 1.0,
    include_current: bool = True,
):
    """Fixed sinusoidal temporal embedding for MEM-style memory tokens.

    Frame order is assumed to be:
        [oldest_history, ..., current]

    The current frame is t=0 and receives exactly zero temporal embedding.
    This introduces no learnable parameters.
    """
    if dim % 2 != 0:
        raise ValueError(f"hidden dim must be even, got {dim}")

    stop = 1 if include_current else 0
    start = -(num_frames - 1) if include_current else -num_frames
    pos = torch.arange(start, stop, device=device, dtype=torch.float32) * step_scale

    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = pos[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    # Make current frame t=0 exactly zero.
    if include_current:
        emb = emb - emb[-1:, :]
    return emb.to(dtype=dtype)


def _mem_sparse_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
):
    """MEM-style sparse spatiotemporal attention for SigLIP ViT.

    For K frames, this implements the sparse attention semantics:

        E  O  O
        I  E  O
        I  I  E

    where:
        E = same-frame all-patch spatial attention
        I = same-patch causal temporal attention
        O = no connection

    The implementation avoids dense (K*N) x (K*N) attention by computing:
        spatial logits:  O(K * N^2)
        temporal logits: O(N * K^2)

    Spatial and temporal logits are concatenated before one shared softmax.
    Therefore this is equivalent to sparse big-mask attention, not two
    independent attentions.

    No new learnable parameters are introduced. It reuses this SigLIP attention
    module's original q_proj/k_proj/v_proj/out_proj.
    """
    num_frames = getattr(self, "mem_num_frames", MEM_VIT_NUM_FRAMES)

    # Original-forward escape hatch for K=1.
    # Default is False because we want to test whether the new forward naturally
    # degenerates to original SigLIP attention when K=1.
    if num_frames <= 1 and getattr(self, "mem_use_original_for_k1", MEM_VIT_USE_ORIGINAL_FOR_K1):
        return self._original_forward(hidden_states, attention_mask=attention_mask, **kwargs)

    if attention_mask is not None:
        raise NotImplementedError(
            "MEM sparse vision attention currently expects attention_mask=None. "
            "Supporting arbitrary patch masks requires folding them into both spatial "
            "and temporal sparse logits."
        )

    bk, num_patches, embed_dim = hidden_states.shape
    if bk % num_frames != 0:
        raise ValueError(
            f"Batch dimension {bk} is not divisible by mem_num_frames={num_frames}. "
            "For multi-frame use, pass frames flattened as (B*K, C, H, W)."
        )

    batch_size = bk // num_frames
    num_heads = self.num_heads
    head_dim = self.head_dim
    frame_mask = getattr(self, "mem_frame_mask", None)
    if frame_mask is not None:
        frame_mask = frame_mask.to(device=hidden_states.device, dtype=torch.bool)
        if tuple(frame_mask.shape) != (batch_size, num_frames):
            raise ValueError(
                f"MEM frame mask shape {tuple(frame_mask.shape)} does not match "
                f"(batch_size, num_frames)=({batch_size}, {num_frames})."
            )

    # Shape: (B, K, N, D)
    x = hidden_states.view(batch_size, num_frames, num_patches, embed_dim)

    # Fixed temporal embedding, current frame gets zero.
    # For K=1 this is exactly zero.
    t_emb = _mem_temporal_pos_emb(
        num_frames,
        embed_dim,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    x = x + t_emb[None, :, None, :]

    # Reuse original SigLIP projections.
    flat_x = x.reshape(bk, num_patches, embed_dim)
    input_shape = flat_x.shape[:-1]
    hidden_shape = (*input_shape, num_heads, head_dim)

    q = self.q_proj(flat_x).view(hidden_shape).transpose(1, 2)
    k = self.k_proj(flat_x).view(hidden_shape).transpose(1, 2)
    v = self.v_proj(flat_x).view(hidden_shape).transpose(1, 2)

    # (B*K, H, N, Dh) -> (B, H, K, N, Dh)
    q = q.view(batch_size, num_frames, num_heads, num_patches, head_dim).permute(0, 2, 1, 3, 4)
    k = k.view(batch_size, num_frames, num_heads, num_patches, head_dim).permute(0, 2, 1, 3, 4)
    v = v.view(batch_size, num_frames, num_heads, num_patches, head_dim).permute(0, 2, 1, 3, 4)

    # Spatial logits:
    # query (b,h,t,p), key (b,h,t,q)
    # Shape: (B, H, K, N, N)
    spatial_logits = torch.einsum("bhtpd,bhtqd->bhtpq", q, k)

    # Temporal logits:
    # query (b,h,t,p), key (b,h,s,p)
    # Shape: (B, H, K, N, K)
    temporal_logits = torch.einsum("bhtpd,bhspd->bhtps", q, k)

    # Only allow s < t.
    # s == t is already included in the same-frame spatial E block.
    # For K=1 this masks the whole temporal branch, leaving pure spatial attention.
    time_mask = torch.tril(
        torch.ones(num_frames, num_frames, device=hidden_states.device, dtype=torch.bool),
        diagonal=-1,
    )
    temporal_logits = temporal_logits.masked_fill(
        ~time_mask[None, None, :, None, :],
        torch.finfo(temporal_logits.dtype).min,
    )
    if frame_mask is not None:
        temporal_logits = temporal_logits.masked_fill(
            ~frame_mask[:, None, None, None, :],
            torch.finfo(temporal_logits.dtype).min,
        )

    # One shared softmax over spatial keys + temporal keys.
    logits = torch.cat([spatial_logits, temporal_logits], dim=-1)
    logits = logits * self.scale

    weights = torch.softmax(logits, dim=-1, dtype=torch.float32).to(q.dtype)
    weights = torch.nn.functional.dropout(
        weights,
        p=0.0 if not self.training else self.dropout,
        training=self.training,
    )

    spatial_weights = weights[..., :num_patches]
    temporal_weights = weights[..., num_patches:]

    spatial_out = torch.einsum("bhtpq,bhtqd->bhtpd", spatial_weights, v)
    temporal_out = torch.einsum("bhtps,bhspd->bhtpd", temporal_weights, v)

    out = spatial_out + temporal_out

    # (B, H, K, N, Dh) -> (B*K, N, D)
    out = out.permute(0, 2, 3, 1, 4).contiguous()
    out = out.view(bk, num_patches, embed_dim)

    # Reuse original SigLIP output projection.
    out = self.out_proj(out)
    return out, None


def _patch_siglip_vision_tower_for_mem_vit(
    vision_tower,
    num_frames: int = MEM_VIT_NUM_FRAMES,
    temporal_every: int = MEM_VIT_TEMPORAL_EVERY,
    use_original_for_k1: bool = MEM_VIT_USE_ORIGINAL_FOR_K1,
):
    """Patch every temporal_every-th SigLIP vision attention layer.

    By default, K=1 still uses the new sparse attention forward, so we can test
    whether it naturally degenerates to original ViT attention.
    """
    vision_model = vision_tower.vision_model if hasattr(vision_tower, "vision_model") else vision_tower

    encoder = vision_model.encoder

    for layer_idx, layer in enumerate(encoder.layers):
        # 1-indexed "every 4-th layer": 4, 8, 12, ...
        if (layer_idx + 1) % temporal_every != 0:
            continue

        attn = layer.self_attn

        if not hasattr(attn, "_original_forward"):
            attn._original_forward = attn.forward

        attn.mem_num_frames = num_frames
        attn.mem_use_original_for_k1 = use_original_for_k1
        attn.forward = types.MethodType(_mem_sparse_attention_forward, attn)


def _load_mem_vit_vision_checkpoint(vision_tower: nn.Module, checkpoint_path: str | Path) -> None:
    path = Path(checkpoint_path).expanduser()
    if path.is_dir():
        path = path / "mem_vit_distill_latest.pt"
    if not path.is_file():
        raise FileNotFoundError(f"MEM-ViT checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "student_vision_tower" not in checkpoint:
        raise RuntimeError(f"Invalid MEM-ViT checkpoint, missing 'student_vision_tower': {path}")

    missing_keys, unexpected_keys = vision_tower.load_state_dict(
        checkpoint["student_vision_tower"],
        strict=True,
    )
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Failed to load MEM-ViT checkpoint {path}: "
            f"{len(missing_keys)} missing keys, {len(unexpected_keys)} unexpected keys"
        )
    logging.info("Loaded MEM-ViT vision tower from %s (distill step=%s)", path, checkpoint.get("step"))


class PaliGemmaWithExpertModel(
    nn.Module
):  # see openpi `gemma_pytorch.py: PaliGemmaWithExpertModel` this class is almost a exact copy of PaliGemmaWithExpertModel in openpi
    """PaliGemma model with action expert for PI05."""

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        image_size: int = DEFAULT_IMAGE_SIZE,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
        mem_vit_enabled: bool = False,
        mem_vit_num_frames: int = 1,
        mem_vit_temporal_every: int = 4,
        mem_vit_use_original_for_k1: bool = True,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.mem_vit_enabled = mem_vit_enabled
        self.mem_vit_num_frames = mem_vit_num_frames

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.image_size = image_size
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config_hf)
        self.gemma_expert = PiGemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        if self.mem_vit_enabled:
            _patch_siglip_vision_tower_for_mem_vit(
                self.paligemma.model.vision_tower,
                num_frames=self.mem_vit_num_frames,
                temporal_every=mem_vit_temporal_every,
                use_original_for_k1=mem_vit_use_original_for_k1,
            )

        self.to_bfloat16_for_selected_params(precision)
        self._set_requires_grad()

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        # Keep full vision path in float32 so we never toggle (toggle causes optimizer
        # "same dtype" error). Saves memory vs full float32; more memory than only 3 params.
        params_to_keep_float32 = [
            "vision_tower",
            "multi_modal_projector",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def _set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
            for param in self.paligemma.model.vision_tower.parameters():
                param.requires_grad = False
        if self.train_expert_only:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False
            if self.mem_vit_enabled and not self.freeze_vision_encoder:
                self.paligemma.model.vision_tower.train()
                for param in self.paligemma.model.vision_tower.parameters():
                    param.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
        if self.train_expert_only:
            self.paligemma.eval()
            if mode and self.mem_vit_enabled and not self.freeze_vision_encoder:
                self.paligemma.model.vision_tower.train()

    def embed_image(self, image: torch.Tensor, memory_mask: torch.Tensor | None = None):
        # Vision tower and multi_modal_projector are kept in float32 (params_to_keep_float32).
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        if image.ndim == 5:
            if not self.mem_vit_enabled:
                image = image[:, -1]
            else:
                if memory_mask is None:
                    raise ValueError("MEM-ViT 5D image input requires a [B,K] memory mask.")
                if memory_mask.shape != image.shape[:2]:
                    raise ValueError(
                        f"MEM-ViT image memory mask shape {tuple(memory_mask.shape)} does not match "
                        f"image window shape {tuple(image.shape[:2])}."
                    )
                if not memory_mask.bool().any(dim=1).all():
                    raise ValueError("Every MEM-ViT sample must have at least one valid frame.")
                batch_size, num_frames = image.shape[:2]
                self.set_mem_vit_num_frames(num_frames)
                self.set_mem_vit_frame_mask(memory_mask)
                image = image.flatten(0, 1)
                try:
                    image_outputs = self.paligemma.model.get_image_features(image)
                    features = image_outputs.pooler_output.view(
                        batch_size, num_frames, *image_outputs.pooler_output.shape[1:]
                    )[:, -1]
                finally:
                    self.set_mem_vit_frame_mask(None)
                if features.dtype != out_dtype:
                    features = features.to(out_dtype)
                return features

        num_frames = self.mem_vit_num_frames if self.mem_vit_enabled else 1
        if num_frames > 1 and image.shape[0] % num_frames != 0:
            raise ValueError(
                f"MEM-ViT image batch dimension {image.shape[0]} is not divisible by "
                f"mem_vit_num_frames={num_frames}. The dataloader must provide exactly that many frames "
                "per sample for each image feature."
            )
        image_outputs = self.paligemma.model.get_image_features(image)
        features = image_outputs.pooler_output
        if num_frames > 1:
            batch_size = image.shape[0] // num_frames
            features = features.view(batch_size, num_frames, *features.shape[1:])[:, -1]
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    def set_mem_vit_num_frames(self, num_frames: int) -> None:
        if not self.mem_vit_enabled:
            return
        if num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {num_frames}")
        self.mem_vit_num_frames = num_frames
        vision_tower = self.paligemma.model.vision_tower
        vision_model = vision_tower.vision_model if hasattr(vision_tower, "vision_model") else vision_tower
        for layer in vision_model.encoder.layers:
            attn = layer.self_attn
            if hasattr(attn, "_original_forward"):
                attn.mem_num_frames = num_frames

    def set_mem_vit_frame_mask(self, frame_mask: torch.Tensor | None) -> None:
        if not self.mem_vit_enabled:
            return
        vision_tower = self.paligemma.model.vision_tower
        vision_model = vision_tower.vision_model if hasattr(vision_tower, "vision_model") else vision_tower
        for layer in vision_model.encoder.layers:
            attn = layer.self_attn
            if hasattr(attn, "_original_forward"):
                attn.mem_frame_mask = frame_mask

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.model.language_model.get_input_embeddings()(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.model.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            paligemma_layers = self.paligemma.model.language_model.layers
            gemma_expert_layers = self.gemma_expert.model.layers
            rotary_emb = self.paligemma.model.language_model.rotary_emb

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Process all layers with gradient checkpointing if enabled
            for layers in zip(paligemma_layers, gemma_expert_layers, strict=True):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        layers=layers,
                        rotary_emb=rotary_emb,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        layers=layers,
                        rotary_emb=rotary_emb,
                    )

            # final norm
            final_norms = (
                self.paligemma.model.language_model.norm,
                self.gemma_expert.model.norm,
            )

            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = layernorm_forward(final_norms[i], hidden_states, adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values


class PI05Pytorch(nn.Module):  # see openpi `PI0Pytorch`
    """Core PI05 PyTorch model."""

    def __init__(self, config: PI05Config, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(
                f"PaliGemma expects square image resolution, invalid resolution: {config.image_resolution}"
            )

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
            mem_vit_enabled=config.mem_vit_enabled,
            mem_vit_num_frames=config.mem_vit_num_frames,
            mem_vit_temporal_every=config.mem_vit_temporal_every,
            mem_vit_use_original_for_k1=config.mem_vit_use_original_for_k1,
        )

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        if config.state_action_encoding == "continuous":
            self.state_memory_proj = nn.Linear(config.max_state_dim, paligemma_config.width)
        if config.action_history_enabled:
            self.action_memory_proj = nn.Linear(config.max_action_dim, paligemma_config.width)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            # Also compile the main forward pass used during training
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI05Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI05Pytorch model")

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def embed_state_memory(
        self, state: torch.Tensor | None, state_memory_mask: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        if self.config.state_action_encoding != "continuous":
            return None
        if state is None:
            raise ValueError("Continuous PI0.5 state encoding requires observation.state.")
        if state.ndim == 2:
            state = state[:, None, :]
        if state.ndim != 3:
            raise ValueError(f"Expected MEM state memory as [B,K,D] or [B,D], got {tuple(state.shape)}")
        num_frames = state.shape[1]
        if state_memory_mask is not None and state_memory_mask.shape != state.shape[:2]:
            raise ValueError(
                f"State memory mask shape {tuple(state_memory_mask.shape)} does not match "
                f"state window shape {tuple(state.shape[:2])}."
            )
        if num_frames > self.config.state_num_frames:
            raise ValueError(
                f"State memory has {num_frames} frames per sample, but "
                f"configured state_num_frames={self.config.state_num_frames}."
            )

        state = pad_vector(state, self.config.max_state_dim)
        if self.state_memory_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        def state_memory_proj_func(state):
            return self.state_memory_proj(state)

        state_emb = self._apply_checkpoint(state_memory_proj_func, state)
        temporal_emb = _mem_temporal_pos_emb(
            num_frames,
            state_emb.shape[-1],
            device=state_emb.device,
            dtype=state_emb.dtype,
            step_scale=self.config.state_history_frame_interval_seconds,
        )
        return state_emb + temporal_emb[None, :, :]

    def embed_action_memory(
        self, action_history: torch.Tensor | None, action_history_mask: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        if not self.config.action_history_enabled:
            return None
        if action_history is None:
            raise ValueError(
                "action_history_enabled=true requires observation.action_history. "
                "Training obtains it from the dataset; deployment must provide previously executed actions."
            )
        if action_history.ndim == 2:
            action_history = action_history[:, None, :]
        if action_history.ndim != 3:
            raise ValueError(
                f"Expected action history as [B,K,D] or [B,D], got {tuple(action_history.shape)}"
            )
        num_frames = action_history.shape[1]
        max_frames = self.config.state_num_frames - 1
        if num_frames > max_frames:
            raise ValueError(f"Action history has {num_frames} frames, configured maximum is {max_frames}.")
        if action_history_mask is not None and action_history_mask.shape != action_history.shape[:2]:
            raise ValueError(
                f"Action history mask shape {tuple(action_history_mask.shape)} does not match "
                f"history shape {tuple(action_history.shape[:2])}."
            )
        action_history = pad_vector(action_history, self.config.max_action_dim)
        if self.action_memory_proj.weight.dtype == torch.float32:
            action_history = action_history.to(torch.float32)
        action_emb = self._apply_checkpoint(self.action_memory_proj, action_history)
        temporal_emb = _mem_temporal_pos_emb(
            num_frames,
            action_emb.shape[-1],
            device=action_emb.device,
            dtype=action_emb.dtype,
            step_scale=self.config.state_history_frame_interval_seconds,
            include_current=False,
        )
        return action_emb + temporal_emb[None, :, :]

    def embed_prefix(
        self,
        images,
        img_masks,
        tokens,
        masks,
        state: torch.Tensor | None = None,
        action_history: torch.Tensor | None = None,
        image_memory_masks=None,
        state_memory_mask: torch.Tensor | None = None,
        action_history_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer."""
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        if image_memory_masks is None:
            image_memory_masks = [None] * len(images)

        for img, img_mask, image_memory_mask in zip(images, img_masks, image_memory_masks, strict=True):

            def image_embed_func(img, image_memory_mask=image_memory_mask):
                return self.paligemma_with_expert.embed_image(img, memory_mask=image_memory_mask)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        state_emb = self.embed_state_memory(state, state_memory_mask=state_memory_mask)
        if state_emb is not None:
            bsize, num_state_tokens = state_emb.shape[:2]
            embs.append(state_emb)
            if state_memory_mask is None:
                state_memory_mask = torch.ones(
                    bsize, num_state_tokens, dtype=torch.bool, device=state_emb.device
                )
            else:
                state_memory_mask = state_memory_mask.to(device=state_emb.device, dtype=torch.bool)
            pad_masks.append(state_memory_mask)
            att_masks += [0] * num_state_tokens

        action_memory_emb = self.embed_action_memory(action_history, action_history_mask=action_history_mask)
        if action_memory_emb is not None:
            bsize, num_action_tokens = action_memory_emb.shape[:2]
            embs.append(action_memory_emb)
            if action_history_mask is None:
                action_history_mask = torch.ones(
                    bsize, num_action_tokens, dtype=torch.bool, device=action_memory_emb.device
                )
            else:
                action_history_mask = action_history_mask.to(
                    device=action_memory_emb.device, dtype=torch.bool
                )
            pad_masks.append(action_history_mask)
            att_masks += [0] * num_action_tokens

        # Process language tokens
        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            return lang_emb

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(
        self,
        images,
        img_masks,
        tokens,
        masks,
        actions,
        noise,
        time,
        state=None,
        action_history=None,
        image_memory_masks=None,
        state_memory_mask=None,
        action_history_mask=None,
    ) -> Tensor:
        """Do a full training forward pass and compute the loss."""
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            tokens,
            masks,
            state,
            action_history,
            image_memory_masks=image_memory_masks,
            state_memory_mask=state_memory_mask,
            action_history_mask=action_history_mask,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise = self.sample_noise(actions_shape, device)

        state = kwargs.get("state")
        action_history = kwargs.get("action_history")
        image_memory_masks = kwargs.get("image_memory_masks")
        state_memory_mask = kwargs.get("state_memory_mask")
        action_history_mask = kwargs.get("action_history_mask")
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            tokens,
            masks,
            state,
            action_history,
            image_memory_masks=image_memory_masks,
            state_memory_mask=state_memory_mask,
            action_history_mask=action_history_mask,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        past_key_values = clone_past_key_values(past_key_values)
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


class PI05Policy(PreTrainedPolicy):
    """PI05 Policy for LeRobot."""

    config_class = PI05Config
    name = "pi05"

    def __init__(
        self,
        config: PI05Config,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        require_package("transformers", extra="pi")
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize the core PI05 model
        self.init_rtc_processor()
        self.model = PI05Pytorch(config, rtc_processor=self.rtc_processor)

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        self.reset()

    def _enable_lora_full_finetuning_modules(self) -> None:
        """Keep full-trained VLA adaptation modules trainable after PEFT freezes the base policy."""
        full_train_modules = [
            self.model.paligemma_with_expert.gemma_expert,
            self.model.action_in_proj,
            self.model.action_out_proj,
            self.model.time_mlp_in,
            self.model.time_mlp_out,
        ]
        if self.config.state_action_encoding == "continuous":
            full_train_modules.append(self.model.state_memory_proj)
        if self.config.action_history_enabled:
            full_train_modules.append(self.model.action_memory_proj)
        for module in full_train_modules:
            module.train()
            for param in module.parameters():
                param.requires_grad = True

    def _enable_mem_vit_full_finetuning(self) -> None:
        """Keep MEM-ViT trainable when PEFT freezes the rest of the base policy."""
        vision_tower = self.model.paligemma_with_expert.paligemma.model.vision_tower
        vision_tower.train()
        for param in vision_tower.parameters():
            param.requires_grad = True

    def wrap_with_peft(
        self,
        peft_config=None,
        peft_cli_overrides: dict | None = None,
    ) -> PreTrainedPolicy:
        peft_model = super().wrap_with_peft(
            peft_config=peft_config,
            peft_cli_overrides=peft_cli_overrides,
        )
        self._enable_lora_full_finetuning_modules()
        if self.config.mem_vit_enabled and not self.config.freeze_vision_encoder:
            self._enable_mem_vit_full_finetuning()
        return peft_model

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Override the from_pretrained method to handle key remapping and display important disclaimer."""
        print(
            "The PI05 model is a direct port of the OpenPI implementation. \n"
            "This implementation follows the original OpenPI structure for compatibility. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        # Initialize model without loading weights
        # Check if dataset_stats were provided in kwargs
        model = cls(config, **kwargs)

        # Load state dict (expects keys with "model." prefix)
        try:
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=kwargs.get("cache_dir"),
                    force_download=kwargs.get("force_download", False),
                    resume_download=kwargs.get("resume_download"),
                    proxies=kwargs.get("proxies"),
                    token=kwargs.get("token"),
                    revision=kwargs.get("revision"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
                from safetensors.torch import load_file

                original_state_dict = load_file(resolved_file)
                print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                print(f"Could not load state dict from remote files: {e}")
                print("Returning model without loading pretrained weights")
                return model

            # First, fix any key differences (see openpi model.py, _fix_pytorch_state_dict_keys)
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            # Then add "model." prefix for all keys that don't already have it
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            # MEM mode adds a small continuous state-memory projection that is not present
            # in the base PI0.5 checkpoint, so allow those new parameters to initialize
            # from scratch while still reporting all missing/unexpected keys.
            has_new_observation_modules = (
                getattr(model.config, "mem_vit_enabled", False)
                or getattr(model.config, "state_action_encoding", "text") == "continuous"
                or getattr(model.config, "action_history_enabled", False)
            )
            effective_strict = strict and not has_new_observation_modules
            missing_keys, unexpected_keys = model.load_state_dict(
                remapped_state_dict, strict=effective_strict
            )

            if missing_keys:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")

        except Exception as e:
            print(f"Warning: Could not load state dict: {e}")

        if getattr(model.config, "mem_vit_checkpoint", None) is not None:
            _load_mem_vit_vision_checkpoint(
                model.model.paligemma_with_expert.paligemma.model.vision_tower,
                model.config.mem_vit_checkpoint,
            )

        return model

    def _fix_pytorch_state_dict_keys(
        self, state_dict, model_config
    ):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture."""
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            # Handle MLP naming changes for pi05
            # pi05 model expects time_mlp_*, but checkpoint might have action_time_mlp_*
            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
            # Also handle state_proj which shouldn't exist in pi05
            if key.startswith("state_proj."):
                logging.warning(f"Skipping state_proj key in pi05 mode: {key}")
                continue

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might have this, but current model expects different structure
                logging.warning(f"Vision embedding key might need handling: {key}")

            if (
                key == "model.paligemma_with_expert.paligemma.lm_head.weight"
                or key == "paligemma_with_expert.paligemma.lm_head.weight"
            ):
                fixed_state_dict[
                    "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
                ] = value.clone()

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Create processor if config provided
        # If RTC is not enabled - we can still track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _mem_vit_window_plan(
        self, batch: dict[str, Tensor], *, randomize: bool
    ) -> tuple[int, Tensor, Tensor]:
        if not self.config.mem_vit_enabled:
            device = next(self.parameters()).device
            batch_size = batch[OBS_STATE].shape[0]
            lengths = torch.ones(batch_size, dtype=torch.long, device=device)
            mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
            return 1, lengths, mask

        image_key = next((key for key in self.config.image_features if key in batch), None)
        if image_key is None:
            raise ValueError("MEM-ViT PI05 requires at least one configured image for window planning.")
        image = batch[image_key]
        batch_size = image.shape[0]
        device = image.device
        max_frames = self.config.mem_vit_num_frames

        if randomize and self.config.mem_vit_min_num_frames is not None:
            min_frames = self.config.mem_vit_min_num_frames
            target_lengths = torch.randint(min_frames, max_frames + 1, (batch_size,), device=device)
        else:
            target_lengths = torch.full((batch_size,), max_frames, dtype=torch.long, device=device)

        if image.ndim < 5:
            available_lengths = torch.ones(batch_size, dtype=torch.long, device=device)
        else:
            available_lengths = torch.full((batch_size,), image.shape[1], dtype=torch.long, device=device)
            image_pad = batch.get(f"{image_key}_is_pad")
            if image_pad is not None:
                available_lengths = (~image_pad.to(device=device).bool()).sum(dim=1)

        lengths = torch.minimum(target_lengths, available_lengths).clamp_min(1)
        window_num_frames = int(lengths.max().item())
        positions = torch.arange(window_num_frames, device=device)[None, :]
        memory_mask = positions >= (window_num_frames - lengths[:, None])
        return window_num_frames, lengths, memory_mask

    def _fixed_memory_window_plan(
        self,
        batch: dict[str, Tensor],
        *,
        key: str,
        max_frames: int,
        allow_empty: bool = False,
    ) -> tuple[int, Tensor, Tensor]:
        value = batch.get(key)
        if value is None:
            raise ValueError(f"MEM input requires {key}.")
        batch_size = value.shape[0]
        device = value.device
        available = torch.ones(batch_size, dtype=torch.long, device=device)
        if value.ndim >= 3:
            available.fill_(value.shape[1])
            pad = batch.get(f"{key}_is_pad")
            if pad is not None:
                available = (~pad.to(device=device).bool()).sum(dim=1)
        minimum = 0 if allow_empty else 1
        lengths = available.clamp(min=minimum, max=max_frames)
        window_num_frames = max(1, int(lengths.max().item()))
        positions = torch.arange(window_num_frames, device=device)[None, :]
        mask = positions >= (window_num_frames - lengths[:, None])
        return window_num_frames, lengths, mask

    def _prepare_state_history(self, batch: dict[str, Tensor], window_num_frames: int) -> Tensor | None:
        if self.config.state_action_encoding != "continuous":
            return None
        state = batch[OBS_STATE]
        if state.ndim == 2:
            return state[:, None, :]
        if state.ndim != 3:
            raise ValueError(f"Expected observation.state as [B,K,D] or [B,D], got {tuple(state.shape)}")
        return state[:, -window_num_frames:, :]

    def _prepare_action_history(self, batch: dict[str, Tensor], window_num_frames: int) -> Tensor | None:
        if not self.config.action_history_enabled:
            return None
        action_history = batch[OBS_ACTION_HISTORY]
        if action_history.ndim == 2:
            return action_history[:, None, :]
        if action_history.ndim != 3:
            raise ValueError(
                f"Expected {OBS_ACTION_HISTORY} as [B,K,D] or [B,D], got {tuple(action_history.shape)}"
            )
        return action_history[:, -window_num_frames:, :]

    def _preprocess_images(
        self,
        batch: dict[str, Tensor],
        *,
        mem_vit_window_num_frames: int | None = None,
        mem_vit_memory_mask: Tensor | None = None,
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor | None]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images = []
        img_masks = []
        image_memory_masks = []

        # Get device from model parameters
        device = next(self.parameters()).device

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]
            if img.ndim not in (4, 5):
                raise ValueError(
                    f"Expected image feature {key} to be 4D [B,C,H,W]/[B,H,W,C] or "
                    f"5D [B,K,C,H,W]/[B,K,H,W,C], got {tuple(img.shape)}"
                )

            original_batch_size = img.shape[0]
            if img.ndim == 5:
                num_frames = img.shape[1]
                if not self.config.mem_vit_enabled:
                    img = img[:, -1]
                elif mem_vit_window_num_frames is None or mem_vit_memory_mask is None:
                    raise ValueError(
                        "mem_vit_window_num_frames and mem_vit_memory_mask must be provided "
                        "when MEM-ViT image windows are used."
                    )
                elif mem_vit_window_num_frames > num_frames:
                    raise ValueError(
                        f"Requested {mem_vit_window_num_frames} MEM-ViT frames for image feature {key}, "
                        f"but the batch only contains {num_frames} frames."
                    )
                elif num_frames != mem_vit_window_num_frames:
                    img = img[:, -mem_vit_window_num_frames:]
                    num_frames = mem_vit_window_num_frames
                if self.config.mem_vit_enabled and num_frames != mem_vit_window_num_frames:
                    raise ValueError(
                        f"Image feature {key} has {num_frames} frames per sample, but "
                        f"runtime window has {mem_vit_window_num_frames}."
                    )
                if not (img.shape[2] == 3 or img.shape[-1] <= 4):
                    raise ValueError(
                        f"Could not infer channel dimension for image feature {key}: {tuple(img.shape)}"
                    )

            # Ensure tensor is on the same device as the model
            if img.device != device:
                img = img.to(device)

            # Ensure float32 dtype for consistency
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            restore_mem_window_shape = img.ndim == 5 and self.config.mem_vit_enabled
            if restore_mem_window_shape:
                window_shape = img.shape[:2]
                img = img.flatten(0, 1)

            # from openpi preprocess_observation_pytorch: Handle both [B, C, H, W] and [B, H, W, C] formats
            is_channels_first = img.shape[1] == 3  # Check if channels are in dimension 1

            if is_channels_first:
                # Convert [B, C, H, W] to [B, H, W, C] for processing
                img = img.permute(0, 2, 3, 1)

            # from openpi preprocess_observation_pytorch: Resize with padding if needed
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            # Normalize from [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            # from openpi preprocess_observation_pytorch: Convert back to [B, C, H, W] format if it was originally channels-first
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            if restore_mem_window_shape:
                img = img.view(*window_shape, *img.shape[1:])

            images.append(img)
            # Create mask (all ones for real images)
            mask = torch.ones(original_batch_size, dtype=torch.bool, device=device)
            img_masks.append(mask)
            image_memory_masks.append(
                mem_vit_memory_mask.to(device=device) if restore_mem_window_shape else None
            )

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # Padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # Mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)
            image_memory_masks.append(
                mem_vit_memory_mask.to(device=device)
                if self.config.mem_vit_enabled and img.ndim == 5 and mem_vit_memory_mask is not None
                else None
            )

        return images, img_masks, image_memory_masks

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def _use_b2_z1_gate_action_loss(self, action_dim: int) -> bool:
        schema = getattr(self.config, "action_loss_schema", "auto")
        if schema == "off":
            return False
        if schema == "always":
            return True
        if schema != "auto":
            raise ValueError(
                f"Unsupported action_loss_schema={schema!r}. Expected one of: 'auto', 'always', 'off'."
            )
        return bool(getattr(self.config, "io_schema_resolved", False)) and action_dim == len(
            self.config.action_feature_names or []
        )

    @staticmethod
    def _normalized_bool_mask(actions: Tensor, dim: int, true_side: str = "positive") -> Tensor:
        """Infer a bool label from a normalized two-valued action dimension.

        PI0.5 receives actions after the policy preprocessor normalization step. For a raw 0/1 gate,
        identity/min-max/quantile normalization keeps the true class on the positive side. This is more
        robust than a batch-local midpoint because a small batch may contain only one class.

        Some binary targets are stored as 0 vs negative raw command values. For those, quantile
        normalization puts the active/event class on the negative side, so callers can pass
        true_side="negative".
        """
        if true_side == "positive":
            return actions[:, :, dim] > 0
        if true_side == "negative":
            return actions[:, :, dim] < 0
        raise ValueError(f"Unsupported true_side={true_side!r}. Expected 'positive' or 'negative'.")

    def _balanced_bool_weights(self, target_true: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        """Return per-element weights that give true/false classes comparable total mass."""
        target_true = target_true.to(dtype=torch.bool)
        if valid_mask is None:
            balance_target = target_true
        else:
            valid_mask = valid_mask.to(device=target_true.device, dtype=torch.bool)
            balance_target = target_true[valid_mask]
            if balance_target.numel() == 0:
                return torch.zeros_like(target_true, dtype=torch.float32)
        eps = getattr(self.config, "action_bool_balance_eps", 1e-3)
        true_frac = balance_target.float().mean().clamp(min=eps, max=1.0 - eps)
        false_frac = 1.0 - true_frac
        true_weight = 0.5 / true_frac
        false_weight = 0.5 / false_frac
        weights = torch.where(target_true, true_weight, false_weight)
        if valid_mask is not None:
            weights = weights * valid_mask
        return weights

    def _global_bool_weights(self, name: str, target_true: Tensor, valid_mask: Tensor) -> Tensor:
        """Use a fixed train-split class prior for an enabled boolean action."""
        true_fraction = getattr(self.config, "action_bool_true_fractions", {}).get(name)
        if true_fraction is None:
            return self._balanced_bool_weights(target_true, valid_mask)
        true_weight = 0.5 / true_fraction
        false_weight = 0.5 / (1.0 - true_fraction)
        return torch.where(target_true, true_weight, false_weight).to(dtype=torch.float32) * valid_mask.to(
            dtype=torch.float32
        )

    def _masked_dim_loss(
        self,
        losses: Tensor,
        dim_indices: list[int],
        active_mask: Tensor,
        base_weight: float,
    ) -> tuple[Tensor, Tensor]:
        """Return weighted losses and weights for selected continuous dimensions."""
        if not dim_indices:
            empty = losses.new_zeros((0,))
            return empty, empty
        dim_losses = losses[:, :, dim_indices]
        weights = active_mask.to(dtype=dim_losses.dtype).unsqueeze(-1).expand_as(dim_losses)
        min_weight = getattr(self.config, "action_masked_continuous_min_weight", 0.0)
        if min_weight > 0:
            weights = torch.clamp(weights, min=min_weight)
        weights = weights * base_weight
        return dim_losses * weights, weights

    def _b2_z1_gate_action_loss(
        self,
        losses: Tensor,
        actions: Tensor,
        reduction: str,
        action_is_pad: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        """Reduce PI0.5 action losses with the B2+Z1 gate-aware schema."""
        names = list(self.config.action_feature_names or [])
        if actions.shape[-1] != len(names):
            raise ValueError(
                f"B2+Z1 action names ({len(names)}) do not match model action width {actions.shape[-1]}"
            )
        name_to_dim = {name: i for i, name in enumerate(names)}
        bool_weight = float(getattr(self.config, "action_bool_loss_weight", 4.0))
        continuous_weight = float(getattr(self.config, "action_continuous_loss_weight", 1.0))

        if action_is_pad is None:
            valid_mask = torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device)
        else:
            valid_mask = ~action_is_pad.to(device=actions.device, dtype=torch.bool)
            if valid_mask.shape != actions.shape[:2]:
                raise ValueError(
                    f"action_is_pad shape {tuple(valid_mask.shape)} does not match action chunk "
                    f"shape {tuple(actions.shape[:2])}"
                )

        weighted_parts: list[Tensor] = []
        weight_parts: list[Tensor] = []
        bool_dim_stats: dict[str, float] = {}
        bool_targets: dict[str, Tensor] = {}
        completion_target = None
        if "task_complete" in name_to_dim:
            completion_target = self._normalized_bool_mask(actions, name_to_dim["task_complete"])
        execution_valid = valid_mask if completion_target is None else (valid_mask & ~completion_target)

        for name in ("arm_teleop_inactive", "arm_reset", "gripper_target"):
            if name not in name_to_dim:
                continue
            true_side = (
                getattr(self.config, "action_gripper_target_true_side", "negative")
                if name == "gripper_target"
                else "positive"
            )
            dim = name_to_dim[name]
            target = self._normalized_bool_mask(actions, dim, true_side=true_side)
            bool_targets[name] = target
            weights = self._global_bool_weights(name, target, execution_valid) * bool_weight
            dim_losses = losses[:, :, dim]
            weighted_parts.append(dim_losses * weights)
            weight_parts.append(weights)
            bool_dim_stats[f"gate_true_frac/{name}"] = float(
                target[execution_valid].float().mean().detach().cpu().item()
            )
            bool_dim_stats[f"gate_loss/{name}"] = float(
                ((dim_losses * weights).sum() / weights.sum().clamp_min(1e-6)).detach().cpu().item()
            )
            global_true_fraction = getattr(self.config, "action_bool_true_fractions", {}).get(name)
            if global_true_fraction is not None:
                bool_dim_stats[f"gate_global_true_frac/{name}"] = float(global_true_fraction)
                bool_dim_stats[f"gate_weight/{name}_true"] = float(bool_weight * 0.5 / global_true_fraction)
                bool_dim_stats[f"gate_weight/{name}_false"] = float(
                    bool_weight * 0.5 / (1.0 - global_true_fraction)
                )

        arm_teleop_inactive = bool_targets.get("arm_teleop_inactive")
        arm_reset = bool_targets.get("arm_reset")
        b2_continuous_mask = execution_valid
        ee_continuous_mask = execution_valid
        if arm_teleop_inactive is not None:
            ee_continuous_mask = ee_continuous_mask & ~arm_teleop_inactive
        if arm_reset is not None:
            ee_continuous_mask = ee_continuous_mask & ~arm_reset
        b2_names = (
            ["b2_delta_x", "b2_delta_y", "b2_delta_yaw"]
            if self.config.b2_action_representation == "local_trajectory"
            else ["b2_vx", "b2_vy", "b2_omega_z"]
        )
        b2_dims = [name_to_dim[name] for name in b2_names]
        ee_dims = [i for i, name in enumerate(names) if name.startswith("height_invariant_ee_")]
        for dim_losses, weights in [
            self._masked_dim_loss(losses, b2_dims, b2_continuous_mask, continuous_weight),
            self._masked_dim_loss(
                losses,
                ee_dims,
                ee_continuous_mask,
                continuous_weight,
            ),
        ]:
            weighted_parts.append(dim_losses)
            weight_parts.append(weights)

        if completion_target is not None:
            completion_dim = name_to_dim["task_complete"]
            completion_valid = valid_mask
            completion_weights = (
                self._global_bool_weights("task_complete", completion_target, completion_valid) * bool_weight
            )
            completion_losses = losses[:, :, completion_dim]
            weighted_parts.append(completion_losses * completion_weights)
            weight_parts.append(completion_weights)
            bool_dim_stats["gate_true_frac/task_complete"] = float(
                completion_target[completion_valid].float().mean().detach().cpu().item()
            )
            bool_dim_stats["gate_loss/task_complete"] = float(
                ((completion_losses * completion_weights).sum() / completion_weights.sum().clamp_min(1e-6))
                .detach()
                .cpu()
                .item()
            )
            global_true_fraction = getattr(self.config, "action_bool_true_fractions", {}).get("task_complete")
            if global_true_fraction is not None:
                bool_dim_stats["gate_global_true_frac/task_complete"] = float(global_true_fraction)
                bool_dim_stats["gate_weight/task_complete_true"] = float(
                    bool_weight * 0.5 / global_true_fraction
                )
                bool_dim_stats["gate_weight/task_complete_false"] = float(
                    bool_weight * 0.5 / (1.0 - global_true_fraction)
                )

        weighted = torch.cat([part.reshape(losses.shape[0], -1) for part in weighted_parts], dim=1)
        weights = torch.cat([part.reshape(losses.shape[0], -1) for part in weight_parts], dim=1)
        per_sample_loss = weighted.sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6)

        loss_dict: dict[str, float] = {
            "gate_aware_action_loss": 1.0,
            f"continuous_mask_frac/b2_{self.config.b2_action_representation}": float(
                b2_continuous_mask.float().mean().detach().cpu().item()
            ),
            "continuous_mask_frac/ee_pose": float(ee_continuous_mask.float().mean().detach().cpu().item()),
        }
        loss_dict.update(bool_dim_stats)
        if completion_target is not None:
            loss_dict["gate_true_frac/task_complete"] = bool_dim_stats["gate_true_frac/task_complete"]
            loss_dict["gate_loss/task_complete"] = bool_dim_stats["gate_loss/task_complete"]
            for key in (
                "gate_global_true_frac/task_complete",
                "gate_weight/task_complete_true",
                "gate_weight/task_complete_false",
            ):
                if key in bool_dim_stats:
                    loss_dict[key] = bool_dim_stats[key]
        if reduction == "none":
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        loss = per_sample_loss.mean()
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Prepare inputs
        mem_vit_window_num_frames, _mem_vit_lengths, mem_vit_memory_mask = self._mem_vit_window_plan(
            batch, randomize=False
        )
        state_window_num_frames, _state_lengths, state_memory_mask = (
            self._fixed_memory_window_plan(
                batch,
                key=OBS_STATE,
                max_frames=self.config.state_num_frames,
            )
            if self.config.state_action_encoding == "continuous"
            else (1, None, None)
        )
        images, img_masks, image_memory_masks = self._preprocess_images(
            batch,
            mem_vit_window_num_frames=mem_vit_window_num_frames,
            mem_vit_memory_mask=mem_vit_memory_mask,
        )
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        # Sample actions using the model (pass through RTC kwargs, no separate state needed for PI05)
        if self.config.state_action_encoding == "continuous":
            kwargs["state"] = self._prepare_state_history(batch, state_window_num_frames)
            kwargs["state_memory_mask"] = state_memory_mask
        if self.config.mem_vit_enabled:
            kwargs["image_memory_masks"] = image_memory_masks
        if self.config.action_history_enabled:
            action_window, _action_lengths, action_mask = self._fixed_memory_window_plan(
                batch,
                key=OBS_ACTION_HISTORY,
                max_frames=self.config.state_num_frames - 1,
                allow_empty=True,
            )
            kwargs["action_history"] = self._prepare_action_history(batch, action_window)
            kwargs["action_history_mask"] = action_mask
        actions = self.model.sample_actions(images, img_masks, tokens, masks, **kwargs)

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        Args:
            batch: Training batch containing observations and actions.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        # Prepare inputs
        mem_vit_window_num_frames, mem_vit_lengths, mem_vit_memory_mask = self._mem_vit_window_plan(
            batch, randomize=self.training
        )
        state_window_num_frames, state_history_lengths, state_memory_mask = (
            self._fixed_memory_window_plan(
                batch,
                key=OBS_STATE,
                max_frames=self.config.state_num_frames,
            )
            if self.config.state_action_encoding == "continuous"
            else (1, None, None)
        )
        images, img_masks, image_memory_masks = self._preprocess_images(
            batch,
            mem_vit_window_num_frames=mem_vit_window_num_frames,
            mem_vit_memory_mask=mem_vit_memory_mask,
        )
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.prepare_action(batch)
        state = self._prepare_state_history(batch, state_window_num_frames)
        action_history = None
        action_history_mask = None
        action_history_lengths = None
        if self.config.action_history_enabled:
            action_window, action_history_lengths, action_history_mask = self._fixed_memory_window_plan(
                batch,
                key=OBS_ACTION_HISTORY,
                max_frames=self.config.state_num_frames - 1,
                allow_empty=True,
            )
            action_history = self._prepare_action_history(batch, action_window)

        noise = self.model.sample_noise(actions.shape, actions.device)
        time = self.model.sample_time(actions.shape[0], actions.device)

        # Compute loss (no separate state needed for PI05)
        losses = self.model.forward(
            images,
            img_masks,
            tokens,
            masks,
            actions,
            noise,
            time,
            state=state,
            action_history=action_history,
            image_memory_masks=image_memory_masks,
            state_memory_mask=state_memory_mask,
            action_history_mask=action_history_mask,
        )

        # Truncate losses to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]

        loss_dict = {
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }
        if self.config.mem_vit_enabled:
            loss_dict["mem_image_num_frames"] = float(mem_vit_lengths.float().mean().item())
            loss_dict["mem_image_window_num_frames"] = float(mem_vit_window_num_frames)
        if state_history_lengths is not None:
            loss_dict["state_num_frames"] = float(state_history_lengths.float().mean().item())
            loss_dict["state_window_num_frames"] = float(state_window_num_frames)
        if action_history_lengths is not None:
            loss_dict["action_history_num_frames"] = float(action_history_lengths.float().mean().item())

        if self._use_b2_z1_gate_action_loss(original_action_dim):
            loss, gate_loss_dict = self._b2_z1_gate_action_loss(
                losses,
                actions[:, :, :original_action_dim],
                reduction,
                batch.get(f"{ACTION}_is_pad"),
            )
            loss_dict.update(gate_loss_dict)
            return loss, loss_dict

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for PI0.5 VLA fine-tuning.

        LoRA is the low-rank adaptation on top of expert fine-tuning: the
        action expert and PI0.5 action/state/time projection layers are full
        fine-tuned and saved, while the PaliGemma/VLM backbone receives LoRA
        adapters. In non-MEM mode this includes ViT LoRA. In MEM mode, MEM-ViT
        is full fine-tuned instead of receiving LoRA adapters.
        """
        transformer_projections = (
            "q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|fc1|fc2|patch_embedding|linear"
        )
        target_modules = rf".*\.paligemma_with_expert\.paligemma\..*\.({transformer_projections})"
        modules_to_save = [
            "model.paligemma_with_expert.gemma_expert",
            "model.action_in_proj",
            "model.action_out_proj",
            "model.time_mlp_in",
            "model.time_mlp_out",
        ]
        if self.config.state_action_encoding == "continuous":
            modules_to_save.append("model.state_memory_proj")
        if self.config.action_history_enabled:
            modules_to_save.append("model.action_memory_proj")
        peft_targets = {
            "target_modules": target_modules,
            "modules_to_save": modules_to_save,
        }
        if self.config.mem_vit_enabled:
            peft_targets["exclude_modules"] = r".*\.paligemma_with_expert\.paligemma\.model\.vision_tower\..*"
        return peft_targets
