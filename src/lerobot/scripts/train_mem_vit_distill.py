#!/usr/bin/env python3
"""
Distill Pi0.5 original single-frame ViT features into a MEM-style sparse-history ViT.

This script intentionally does NOT feed consecutive frames by default.  For each
sample, it chooses a current frame and samples historical frames with random
second-scale gaps, e.g. K=4, gap=3s -> [t-9s, t-6s, t-3s, t].

Teacher: original Pi0.5 image path, frozen, K=1, original SigLIP forward for K=1.
Student: same checkpoint, MEM-style ViT, K sparse frames; distill current-frame token.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.video_utils import decode_video_frames
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: F401
from lerobot.policies.pi05.modeling_pi05 import (
    _patch_siglip_vision_tower_for_mem_vit,
    resize_with_pad_torch,
)

logger = logging.getLogger("train_mem_vit_distill")


def torch_save_atomic(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def update_latest_checkpoint(checkpoint_path: Path, latest_path: Path) -> None:
    """Atomically point latest_path at checkpoint_path without duplicating a multi-GB file."""
    temporary = latest_path.with_suffix(latest_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    os.link(checkpoint_path, temporary)
    os.replace(temporary, latest_path)


def prune_step_checkpoints(output_dir: Path, keep: int) -> None:
    if keep <= 0:
        return
    checkpoints = sorted(
        output_dir.glob("mem_vit_distill_step_*.pt"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    for checkpoint in checkpoints[:-keep]:
        checkpoint.unlink()


class Pi05ImageFeaturePath(nn.Module):
    def __init__(self, vision_tower: nn.Module, multi_modal_projector: nn.Module):
        super().__init__()
        self.vision_tower = vision_tower
        self.multi_modal_projector = multi_modal_projector

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_outputs = self.vision_tower(pixel_values)
        return self.multi_modal_projector(image_outputs.last_hidden_state)


class RandomSparseHistoryImageDataset(Dataset):
    """Read sparse K-frame windows directly from LeRobot video files.

    This bypasses LeRobotDataset(delta_timestamps), because delta_timestamps is
    fixed at construction time. MEM-ViT distillation needs random temporal gaps.
    """

    def __init__(
        self,
        *,
        repo_id: str,
        root: str | Path | None,
        revision: str | None,
        image_key: str,
        num_frames: int,
        min_gap_sec: float,
        max_gap_sec: float,
        min_gap_frames: int | None,
        max_gap_frames: int | None,
        gap_sampling: str,
        num_samples: int,
        start_sample_index: int,
        episode_indices: set[int] | None,
        seed: int | None,
        allow_short_history: bool,
        force_cache_sync: bool,
        video_backend: str | None,
        tolerance_s: float,
    ):
        super().__init__()
        if num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {num_frames}")
        if min_gap_sec < 0 or max_gap_sec < min_gap_sec:
            raise ValueError(f"Invalid gap range: [{min_gap_sec}, {max_gap_sec}]")
        if (min_gap_frames is None) != (max_gap_frames is None):
            raise ValueError("--min-gap-frames and --max-gap-frames must be provided together")
        if min_gap_frames is not None and (min_gap_frames < 0 or max_gap_frames < min_gap_frames):
            raise ValueError(f"Invalid frame gap range: [{min_gap_frames}, {max_gap_frames}]")
        self.meta = LeRobotDatasetMetadata(
            repo_id, Path(root) if root is not None else None, revision, force_cache_sync=force_cache_sync
        )
        self.root = self.meta.root
        self.image_key = image_key
        self.num_frames = num_frames
        self.min_gap_sec = min_gap_sec
        self.max_gap_sec = max_gap_sec
        self.min_gap_frames = min_gap_frames
        self.max_gap_frames = max_gap_frames
        self.sample_by_frames = min_gap_frames is not None
        self.gap_sampling = gap_sampling
        self.seed = seed
        self.allow_short_history = allow_short_history
        self.video_backend = video_backend
        self.tolerance_s = tolerance_s
        if image_key not in self.meta.video_keys:
            raise ValueError(
                f"image_key={image_key!r} is not a video key. Available video keys: {self.meta.video_keys}"
            )
        self.fps = int(self.meta.fps)
        self.episodes: list[dict[str, Any]] = []
        for ep in self.meta.episodes:
            ep_idx = int(ep["episode_index"])
            if episode_indices is not None and ep_idx not in episode_indices:
                continue
            length = int(ep["length"])
            if length <= 0:
                continue
            duration = (length - 1) / float(self.fps)
            if allow_short_history:
                min_current_frame = 0
            elif self.sample_by_frames:
                min_current_frame = (num_frames - 1) * self.max_gap_frames
            else:
                required_history_sec = (num_frames - 1) * max_gap_sec
                min_current_frame = int(math.ceil(required_history_sec * self.fps))
            if min_current_frame >= length:
                history_requirement = (
                    f"{(num_frames - 1) * self.max_gap_frames} frames"
                    if self.sample_by_frames
                    else f"{(num_frames - 1) * max_gap_sec:.3f}s"
                )
                logger.warning(
                    "Skipping episode %s: length=%d duration=%.3fs is too short for "
                    "K=%d required_history=%s. Use --allow-short-history to clamp.",
                    ep_idx,
                    length,
                    duration,
                    num_frames,
                    history_requirement,
                )
                continue
            self.episodes.append(
                {
                    "episode_index": ep_idx,
                    "length": length,
                    "duration": duration,
                    "min_current_frame": min_current_frame,
                    "video_path": self.root / self.meta.get_video_file_path(ep_idx, image_key),
                    "video_from_timestamp": float(ep[f"videos/{image_key}/from_timestamp"]),
                }
            )
        if not self.episodes:
            raise RuntimeError(
                "No episodes have enough temporal context. Try --allow-short-history or reduce "
                "the maximum gap / --num-frames."
            )
        self.samples: list[tuple[int, int]] = []
        for local_ep_idx, ep in enumerate(self.episodes):
            for frame_idx in range(ep["min_current_frame"], ep["length"]):
                self.samples.append((local_ep_idx, frame_idx))
        self.num_samples = num_samples
        self.start_sample_index = start_sample_index

    def __len__(self) -> int:
        return self.num_samples

    def _sample_single_gap(self, rng: random.Random) -> float | int:
        if self.sample_by_frames:
            if self.max_gap_frames == self.min_gap_frames:
                return self.min_gap_frames
            if self.gap_sampling == "uniform_single":
                return rng.randint(self.min_gap_frames, self.max_gap_frames)
            if self.gap_sampling == "log_uniform_single":
                lo = max(self.min_gap_frames, 1)
                sampled = round(math.exp(rng.uniform(math.log(lo), math.log(self.max_gap_frames))))
                return max(self.min_gap_frames, min(self.max_gap_frames, sampled))
            raise ValueError(f"Unsupported single-gap mode: {self.gap_sampling}")
        if self.max_gap_sec == self.min_gap_sec:
            return self.min_gap_sec
        if self.gap_sampling == "uniform_single":
            return rng.uniform(self.min_gap_sec, self.max_gap_sec)
        if self.gap_sampling == "log_uniform_single":
            lo = max(self.min_gap_sec, 1e-6)
            hi = max(self.max_gap_sec, lo)
            return math.exp(rng.uniform(math.log(lo), math.log(hi)))
        raise ValueError(f"Unsupported single-gap mode: {self.gap_sampling}")

    def _sample_offsets(self, rng: random.Random) -> list[float | int]:
        if self.num_frames == 1:
            return [0.0]
        if self.gap_sampling in ("uniform_single", "log_uniform_single"):
            gap = self._sample_single_gap(rng)
            return [-(self.num_frames - 1 - i) * gap for i in range(self.num_frames)]
        if self.gap_sampling == "uniform_each":
            if self.sample_by_frames:
                gaps = [
                    rng.randint(self.min_gap_frames, self.max_gap_frames) for _ in range(self.num_frames - 1)
                ]
            else:
                gaps = [rng.uniform(self.min_gap_sec, self.max_gap_sec) for _ in range(self.num_frames - 1)]
            offsets = [0.0]
            acc = 0.0
            for g in reversed(gaps):
                acc -= g
                offsets.append(acc)
            return list(reversed(offsets))
        raise ValueError(f"Unsupported --gap-sampling {self.gap_sampling!r}")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        logical_idx = self.start_sample_index + idx
        rng = random if self.seed is None else random.Random(self.seed + logical_idx)
        local_ep_idx, frame_idx = rng.choice(self.samples)
        ep = self.episodes[local_ep_idx]
        current_t = frame_idx / float(self.fps)
        offsets = self._sample_offsets(rng)
        if self.sample_by_frames:
            query_frames = [frame_idx + int(off) for off in offsets]
            if self.allow_short_history:
                query_frames = [max(0, min(ep["length"] - 1, f)) for f in query_frames]
            elif min(query_frames) < 0:
                raise RuntimeError(f"Negative query frame without --allow-short-history: {query_frames}")
            query_ts = [f / float(self.fps) for f in query_frames]
        else:
            query_ts = [current_t + off for off in offsets]
            if self.allow_short_history:
                query_ts = [max(0.0, min(ep["duration"], t)) for t in query_ts]
            elif min(query_ts) < -1e-6:
                raise RuntimeError(f"Negative query time without --allow-short-history: {query_ts}")
        shifted_ts = [ep["video_from_timestamp"] + t for t in query_ts]
        frames = decode_video_frames(
            ep["video_path"],
            shifted_ts,
            self.tolerance_s,
            backend=self.video_backend,
            return_uint8=False,
            is_depth=False,
        )
        return {
            "frames": frames,  # [K,C,H,W], float32 [0,1]
            "episode_index": ep["episode_index"],
            "frame_index": frame_idx,
            "current_time": current_t,
            "offsets": torch.tensor(offsets, dtype=torch.float32),
            "query_times": torch.tensor(query_ts, dtype=torch.float32),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained-path", type=str, required=True)
    p.add_argument("--dataset-repo-id", type=str, required=True)
    p.add_argument("--dataset-root", type=str, default=None)
    p.add_argument("--dataset-revision", type=str, default=None)
    p.add_argument(
        "--test-dataset-repo-id",
        type=str,
        default=None,
        help="Optional separate LeRobot dataset repo id for the held-out test split.",
    )
    p.add_argument(
        "--test-dataset-root",
        type=str,
        default=None,
        help="Optional separate LeRobot dataset root containing a test split. Train/val still use --dataset-root.",
    )
    p.add_argument(
        "--test-dataset-revision",
        type=str,
        default=None,
        help="Revision for --test-dataset-repo-id. Defaults to --dataset-revision.",
    )
    p.add_argument("--image-key", type=str, default="observation.images.wrist")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help=(
            "Resume a complete training state from a checkpoint file or an output directory "
            "containing mem_vit_distill_latest.pt. --max-steps remains the absolute target step."
        ),
    )
    p.add_argument("--num-frames", type=int, default=4)
    p.add_argument("--min-gap-sec", type=float, default=0.2)
    p.add_argument("--max-gap-sec", type=float, default=5.0)
    p.add_argument(
        "--min-gap-frames",
        type=int,
        default=None,
        help="Sample exact integer frame gaps; must be used with --max-gap-frames.",
    )
    p.add_argument(
        "--max-gap-frames",
        type=int,
        default=None,
        help="Sample exact integer frame gaps; overrides the --*-gap-sec values.",
    )
    p.add_argument(
        "--gap-sampling",
        choices=["uniform_single", "uniform_each", "log_uniform_single"],
        default="uniform_single",
    )
    p.add_argument("--allow-short-history", action="store_true")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of micro-batches to accumulate before each optimizer step.",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        required=True,
        help="Absolute target optimizer step, including steps restored from a checkpoint.",
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument(
        "--test-eval-batches",
        type=int,
        default=None,
        help="Final held-out test batches. Defaults to --eval-batches when a test split exists.",
    )
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument(
        "--amp-dtype",
        choices=["bfloat16", "float32"],
        default="bfloat16",
        help="Forward-pass precision. Student weights and optimizer states remain float32.",
    )
    p.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trade extra student forward compute for lower activation memory.",
    )
    p.add_argument("--cosine-weight", type=float, default=0.0)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument(
        "--save-final",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a checkpoint after the final step (disable for memory-only smoke tests).",
    )
    p.add_argument(
        "--keep-last-checkpoints",
        type=int,
        default=2,
        help="Keep this many numbered checkpoints; 0 keeps all. latest.pt is a hard link.",
    )
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--force-cache-sync", action="store_true")
    p.add_argument(
        "--video-backend", type=str, default=None, choices=[None, "pyav", "torchcodec", "video_reader"]
    )
    p.add_argument("--video-tolerance-sec", type=float, default=None, help="Default: 0.55 / dataset_fps")
    p.add_argument("--train-projector", action="store_true")
    p.add_argument("--save-projector", action="store_true")
    p.add_argument("--wandb-enable", action="store_true")
    p.add_argument("--wandb-project", type=str, default="lerobot")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-notes", type=str, default=None)
    p.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
        help="Use 'offline' to save a local W&B run without uploading it immediately.",
    )
    return p.parse_args()


def setup_distributed(device_arg: str) -> tuple[torch.device, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size == 1:
        return torch.device(device_arg), rank, world_size, False
    if not torch.cuda.is_available():
        raise RuntimeError("Multi-GPU training requires CUDA and the NCCL backend")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return torch.device("cuda", local_rank), rank, world_size, True


def reduce_metrics(metrics: dict[str, float], device: torch.device, world_size: int) -> dict[str, float]:
    reduced = dict(metrics)
    if world_size > 1:
        keys = tuple(metrics)
        values = torch.tensor([metrics[key] for key in keys], dtype=torch.float64, device=device)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= world_size
        reduced = dict(zip(keys, values.tolist(), strict=True))
    reduced["nmse"] = reduced["mse"] / max(reduced["teacher_energy"], torch.finfo(torch.float64).eps)
    reduced["teacher_rms"] = math.sqrt(max(reduced["teacher_energy"], 0.0))
    reduced["student_rms"] = math.sqrt(max(reduced["student_energy"], 0.0))
    return reduced


def eligible_episode_indices(
    episodes: list[dict[str, Any]],
    *,
    num_frames: int,
    max_gap_sec: float,
    max_gap_frames: int | None,
    fps: int,
    allow_short_history: bool,
) -> set[int]:
    eligible: list[int] = []
    for episode in episodes:
        length = int(episode["length"])
        if length <= 0:
            continue
        if allow_short_history:
            min_current_frame = 0
        elif max_gap_frames is not None:
            min_current_frame = (num_frames - 1) * max_gap_frames
        else:
            min_current_frame = math.ceil((num_frames - 1) * max_gap_sec * fps)
        if min_current_frame < length:
            eligible.append(int(episode["episode_index"]))
    return set(eligible)


def parse_split_range(spec: str, total_episodes: int) -> set[int]:
    start_s, end_s = spec.split(":", 1)
    start = int(start_s) if start_s else 0
    end = int(end_s) if end_s else total_episodes
    if start < 0 or end < start or end > total_episodes:
        raise ValueError(f"Invalid split range {spec!r} for total_episodes={total_episodes}")
    return set(range(start, end))


def load_dataset_episode_splits(
    meta: LeRobotDatasetMetadata,
    *,
    num_frames: int,
    max_gap_sec: float,
    max_gap_frames: int | None,
    fps: int,
    allow_short_history: bool,
) -> tuple[set[int], set[int], set[int] | None]:
    splits = getattr(meta.info, "splits", None) or {}
    if "train" not in splits or "val" not in splits or "test" not in splits:
        raise ValueError("Dataset metadata must contain train, val, and test splits")
    eligible = eligible_episode_indices(
        list(meta.episodes),
        num_frames=num_frames,
        max_gap_sec=max_gap_sec,
        max_gap_frames=max_gap_frames,
        fps=fps,
        allow_short_history=allow_short_history,
    )
    train = parse_split_range(splits["train"], meta.total_episodes) & eligible
    val = parse_split_range(splits["val"], meta.total_episodes) & eligible
    test = parse_split_range(splits["test"], meta.total_episodes) & eligible
    if not train:
        raise RuntimeError("Dataset train split has no eligible episodes")
    if not val:
        raise RuntimeError("Dataset val split has no eligible episodes")
    if not test:
        raise RuntimeError("Dataset test split has no eligible episodes")
    return train, val, test


def load_dataset_episode_split(
    meta: LeRobotDatasetMetadata,
    split_name: str,
    *,
    num_frames: int,
    max_gap_sec: float,
    max_gap_frames: int | None,
    fps: int,
    allow_short_history: bool,
) -> set[int]:
    splits = getattr(meta.info, "splits", None) or {}
    if split_name not in splits:
        raise ValueError(f"Dataset metadata must contain a {split_name!r} split")
    eligible = eligible_episode_indices(
        list(meta.episodes),
        num_frames=num_frames,
        max_gap_sec=max_gap_sec,
        max_gap_frames=max_gap_frames,
        fps=fps,
        allow_short_history=allow_short_history,
    )
    split = parse_split_range(splits[split_name], meta.total_episodes) & eligible
    if not split:
        raise RuntimeError(f"Dataset {split_name} split has no eligible episodes")
    return split


def resolve_resume_checkpoint(path_or_dir: str | None) -> Path | None:
    if path_or_dir is None:
        return None
    path = Path(path_or_dir).expanduser()
    if path.is_dir():
        path = path / "mem_vit_distill_latest.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    return path.resolve()


def load_resume_checkpoint(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Invalid resume checkpoint (expected a dictionary): {path}")
    required = {
        "checkpoint_version",
        "step",
        "student_vision_tower",
        "optimizer_state_dict",
        "rng_states",
        "world_size",
        "args",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise RuntimeError(
            f"Checkpoint {path} predates complete resume support or is incomplete; missing keys: {missing}"
        )
    if checkpoint["checkpoint_version"] != 2:
        raise RuntimeError(
            f"Unsupported checkpoint version {checkpoint['checkpoint_version']!r} in {path}; expected 2"
        )
    return checkpoint


def validate_resume_compatibility(
    args: argparse.Namespace, checkpoint: dict[str, Any], *, world_size: int
) -> None:
    saved_args = checkpoint["args"]
    immutable_args = (
        "pretrained_path",
        "dataset_repo_id",
        "dataset_root",
        "dataset_revision",
        "image_key",
        "num_frames",
        "min_gap_sec",
        "max_gap_sec",
        "min_gap_frames",
        "max_gap_frames",
        "gap_sampling",
        "allow_short_history",
        "batch_size",
        "gradient_accumulation_steps",
        "seed",
        "eval_batches",
        "test_eval_batches",
        "lr",
        "weight_decay",
        "grad_clip_norm",
        "amp_dtype",
        "gradient_checkpointing",
        "cosine_weight",
        "train_projector",
    )
    mismatches = []
    for name in immutable_args:
        saved = saved_args.get(name, 1 if name == "gradient_accumulation_steps" else None)
        current = getattr(args, name)
        if saved != current:
            mismatches.append(f"{name}: checkpoint={saved!r}, current={current!r}")
    if int(checkpoint["world_size"]) != world_size:
        mismatches.append(f"world_size: checkpoint={checkpoint['world_size']!r}, current={world_size!r}")
    if mismatches:
        raise ValueError(
            "Resume configuration is incompatible with the checkpoint:\n  " + "\n  ".join(mismatches)
        )
    step = int(checkpoint["step"])
    if step >= args.max_steps:
        raise ValueError(
            f"Checkpoint step {step} has already reached --max-steps={args.max_steps}; "
            "set a larger absolute target."
        )


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        if "torch_cuda" not in state:
            raise RuntimeError("Resume checkpoint has no CUDA RNG state")
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def init_wandb(
    args: argparse.Namespace, output_dir: Path, resume_checkpoint: dict[str, Any] | None
) -> Any | None:
    if not args.wandb_enable:
        return None

    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - depends on installed extras
        raise RuntimeError(
            "W&B logging requires the training dependencies. Run `uv sync --extra training`."
        ) from exc

    resume_id = None if resume_checkpoint is None else resume_checkpoint.get("wandb_run_id")
    if resume_checkpoint is not None and resume_id is None:
        logger.warning("Checkpoint has no W&B run ID; starting a new W&B run for resumed training")
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=None if resume_id is not None else (args.wandb_run_name or output_dir.name),
        notes=args.wandb_notes,
        dir=output_dir,
        config=vars(args),
        job_type="train",
        mode=args.wandb_mode,
        save_code=False,
        id=resume_id,
        resume="must" if resume_id is not None and args.wandb_mode == "online" else None,
    )
    run.define_metric("train/global_step")
    run.define_metric("train/*", step_metric="train/global_step")
    run.define_metric("val/*", step_metric="train/global_step")
    run.define_metric("data/*", step_metric="train/global_step")
    logger.info("W&B logging enabled: %s", run.url or f"mode={args.wandb_mode}")
    return run


def get_vision_model(vision_tower: nn.Module) -> nn.Module:
    return vision_tower.vision_model if hasattr(vision_tower, "vision_model") else vision_tower


def set_mem_vit_runtime(
    image_path: Pi05ImageFeaturePath, *, num_frames: int, use_original_for_k1: bool
) -> int:
    encoder = get_vision_model(image_path.vision_tower).encoder
    patched = 0
    for layer in encoder.layers:
        attn = layer.self_attn
        if not hasattr(attn, "_original_forward"):
            continue
        attn.mem_num_frames = num_frames
        attn.mem_use_original_for_k1 = use_original_for_k1
        patched += 1
    return patched


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    module.train()
    for p in module.parameters():
        p.requires_grad = True


def resolve_checkpoint_file(pretrained_path: str, *, local_files_only: bool) -> Path:
    from transformers.utils import cached_file

    resolved = cached_file(
        pretrained_path,
        "model.safetensors",
        local_files_only=local_files_only,
    )
    if resolved is None:
        raise FileNotFoundError(f"model.safetensors was not found in {pretrained_path}")
    return Path(resolved)


def load_prefixed_module_state(module: nn.Module, checkpoint: Path, prefix: str) -> None:
    from safetensors import safe_open

    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as f:
        for key in f.keys():  # noqa: SIM118 - safe_open is not iterable
            if key.startswith(prefix):
                state_dict[key.removeprefix(prefix)] = f.get_tensor(key)
    if not state_dict:
        raise RuntimeError(f"No tensors with prefix {prefix!r} found in {checkpoint}")
    module.load_state_dict(state_dict, strict=True, assign=True)


def load_pi05_image_path(
    pretrained_path: str,
    *,
    device: torch.device,
    local_files_only: bool,
    dtype: torch.dtype,
) -> tuple[Pi05ImageFeaturePath, tuple[int, int], list[str]]:
    """Load only the Pi0.5 vision tower and projector, never the language/action models."""
    from transformers import SiglipVisionModel
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.paligemma.modeling_paligemma import PaliGemmaMultiModalProjector

    config = PreTrainedConfig.from_pretrained(pretrained_path, local_files_only=local_files_only)
    checkpoint = resolve_checkpoint_file(pretrained_path, local_files_only=local_files_only)

    paligemma_config = CONFIG_MAPPING["paligemma"]()
    paligemma_config.vision_config.image_size = config.image_resolution[0]
    paligemma_config.vision_config.intermediate_size = 4304
    paligemma_config.vision_config.projection_dim = 2048
    paligemma_config.vision_config.projector_hidden_act = "gelu_fast"

    # Build only the vision modules on CPU. This avoids all language/action allocations and
    # keeps Transformers' non-persistent buffers on a real device (they are absent from the checkpoint).
    vision_tower = SiglipVisionModel(paligemma_config.vision_config)
    projector = PaliGemmaMultiModalProjector(paligemma_config)
    load_prefixed_module_state(
        vision_tower,
        checkpoint,
        "paligemma_with_expert.paligemma.model.vision_tower.",
    )
    load_prefixed_module_state(
        projector,
        checkpoint,
        "paligemma_with_expert.paligemma.model.multi_modal_projector.",
    )
    _patch_siglip_vision_tower_for_mem_vit(vision_tower)
    image_path = Pi05ImageFeaturePath(vision_tower, projector).to(device=device, dtype=dtype)
    return image_path, tuple(config.image_resolution), list(config.image_features)


def preprocess_pi05_image(
    img: torch.Tensor, *, image_resolution: tuple[int, int], device: torch.device
) -> torch.Tensor:
    img = img.to(device)
    if img.dtype != torch.float32:
        img = img.to(torch.float32)
    if img.ndim != 4:
        raise ValueError(f"Expected 4D image tensor, got {tuple(img.shape)}")
    is_channels_first = img.shape[1] == 3
    if is_channels_first:
        img = img.permute(0, 2, 3, 1)
    if tuple(img.shape[1:3]) != tuple(image_resolution):
        img = resize_with_pad_torch(img, *image_resolution)
    img = img * 2.0 - 1.0
    if is_channels_first:
        img = img.permute(0, 3, 1, 2)
    return img.contiguous()


def distill_loss(
    student_features: torch.Tensor, teacher_features: torch.Tensor, *, cosine_weight: float
) -> tuple[torch.Tensor, dict[str, float]]:
    mse = F.mse_loss(student_features, teacher_features)
    teacher_energy = teacher_features.square().mean()
    student_energy = student_features.square().mean()
    s = F.normalize(student_features.flatten(1), dim=-1)
    t = F.normalize(teacher_features.flatten(1), dim=-1)
    cosine_similarity = (s * t).sum(dim=-1).mean().clamp(-1.0, 1.0)
    cosine_distance = 1.0 - cosine_similarity
    loss = mse + cosine_weight * cosine_distance
    return loss, {
        "loss": float(loss.detach().cpu()),
        "mse": float(mse.detach().cpu()),
        "teacher_energy": float(teacher_energy.detach().cpu()),
        "student_energy": float(student_energy.detach().cpu()),
        "cosine_similarity": float(cosine_similarity.detach().cpu()),
        "cosine_distance": float(cosine_distance.detach().cpu()),
    }


def evaluate_distillation(
    student: Pi05ImageFeaturePath,
    teacher: Pi05ImageFeaturePath,
    dataloader: DataLoader,
    *,
    image_resolution: tuple[int, int],
    num_frames: int,
    device: torch.device,
    use_amp: bool,
    cosine_weight: float,
    world_size: int,
) -> dict[str, float]:
    was_training = student.training
    student.eval()
    totals: dict[str, float] = {}
    num_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            frames = batch["frames"]
            bsz, k = frames.shape[:2]
            if k != num_frames:
                raise RuntimeError(f"Expected validation K={num_frames}, got K={k}")
            flat_frames = frames.reshape(bsz * k, *frames.shape[2:]).contiguous()
            current_frames = frames[:, -1].contiguous()
            student_pixels = preprocess_pi05_image(
                flat_frames, image_resolution=image_resolution, device=device
            )
            teacher_pixels = preprocess_pi05_image(
                current_frames, image_resolution=image_resolution, device=device
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                teacher_features = teacher(teacher_pixels)
                student_features_all = student(student_pixels)
            student_features_all = student_features_all.view(bsz, k, *student_features_all.shape[1:])
            _, metrics = distill_loss(
                student_features_all[:, -1].float(),
                teacher_features.float(),
                cosine_weight=cosine_weight,
            )
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            num_batches += 1
    if was_training:
        student.train()
    averaged = {key: value / num_batches for key, value in totals.items()}
    return reduce_metrics(averaged, device, world_size)


def report_validation(
    metrics: dict[str, float],
    *,
    split_name: str,
    step: int,
    wandb_run: Any | None,
    is_main_process: bool,
) -> None:
    if not is_main_process:
        return
    logger.info(
        "%s step=%d loss=%.6f mse=%.6f nmse=%.6f cos_sim=%.6f teacher_rms=%.6f student_rms=%.6f",
        split_name,
        step,
        metrics["loss"],
        metrics["mse"],
        metrics["nmse"],
        metrics["cosine_similarity"],
        metrics["teacher_rms"],
        metrics["student_rms"],
    )
    if wandb_run is not None:
        wandb_run.log(
            {
                "train/global_step": step,
                **{f"{split_name}/{key}": value for key, value in metrics.items()},
            }
        )


def save_checkpoint(
    output_dir: Path,
    *,
    step: int,
    student: Pi05ImageFeaturePath,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    metrics: dict[str, float] | None,
    last_eval_step: int,
    rng_device: torch.device,
    rank: int,
    world_size: int,
    wandb_run_id: str | None,
) -> Path | None:
    if isinstance(optimizer, ZeroRedundancyOptimizer):
        optimizer.consolidate_state_dict(to=0)

    local_rng_state = capture_rng_state(rng_device)
    if world_size > 1:
        rng_states: list[dict[str, Any] | None] | None = [None] * world_size if rank == 0 else None
        dist.gather_object(local_rng_state, rng_states, dst=0)
    else:
        rng_states = [local_rng_state]

    if rank != 0:
        return None
    if rng_states is None or any(state is None for state in rng_states):
        raise RuntimeError("Failed to gather RNG states from every distributed rank")

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "checkpoint_version": 2,
        "step": step,
        "last_eval_step": last_eval_step,
        "world_size": world_size,
        "wandb_run_id": wandb_run_id,
        "image_key": args.image_key,
        "num_frames": args.num_frames,
        "min_gap_sec": args.min_gap_sec,
        "max_gap_sec": args.max_gap_sec,
        "min_gap_frames": args.min_gap_frames,
        "max_gap_frames": args.max_gap_frames,
        "gap_sampling": args.gap_sampling,
        "pretrained_path": args.pretrained_path,
        "student_vision_tower": student.vision_tower.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_states": rng_states,
        "metrics": metrics or {},
        "args": vars(args),
    }
    if args.save_projector or args.train_projector:
        ckpt["student_multi_modal_projector"] = student.multi_modal_projector.state_dict()
    if args.keep_last_checkpoints > 0:
        prune_step_checkpoints(output_dir, max(args.keep_last_checkpoints - 1, 1))
    path = output_dir / f"mem_vit_distill_step_{step}.pt"
    torch_save_atomic(ckpt, path)
    update_latest_checkpoint(path, output_dir / "mem_vit_distill_latest.pt")
    prune_step_checkpoints(output_dir, args.keep_last_checkpoints)
    return path


def main() -> None:
    args = parse_args()
    if (args.min_gap_frames is None) != (args.max_gap_frames is None):
        raise ValueError("--min-gap-frames and --max-gap-frames must be provided together")
    if args.max_steps < 1:
        raise ValueError(f"--max-steps must be positive, got {args.max_steps}")
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            f"--gradient-accumulation-steps must be positive, got {args.gradient_accumulation_steps}"
        )
    if args.log_every < 1:
        raise ValueError(f"--log-every must be positive, got {args.log_every}")
    if args.save_every < 0:
        raise ValueError(f"--save-every must be non-negative, got {args.save_every}")
    if args.keep_last_checkpoints < 0:
        raise ValueError(f"--keep-last-checkpoints must be non-negative, got {args.keep_last_checkpoints}")
    if args.num_workers < 0:
        raise ValueError(f"--num-workers must be non-negative, got {args.num_workers}")
    if args.eval_every < 1:
        raise ValueError(f"--eval-every must be positive, got {args.eval_every}")
    if args.eval_batches < 1:
        raise ValueError(f"--eval-batches must be positive, got {args.eval_batches}")
    if args.test_eval_batches is not None and args.test_eval_batches < 1:
        raise ValueError(f"--test-eval-batches must be positive, got {args.test_eval_batches}")
    resume_path = resolve_resume_checkpoint(args.resume_from_checkpoint)
    resume_checkpoint = load_resume_checkpoint(resume_path)
    device, rank, world_size, distributed = setup_distributed(args.device)
    is_main_process = rank == 0
    logging.basicConfig(
        level=logging.INFO if is_main_process else logging.WARNING,
        format=f"[%(asctime)s] rank={rank} %(levelname)s: %(message)s",
        force=True,
    )
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed + rank)
    if resume_checkpoint is not None:
        validate_resume_compatibility(args, resume_checkpoint, world_size=world_size)
    initial_step = 0 if resume_checkpoint is None else int(resume_checkpoint["step"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb(args, output_dir, resume_checkpoint) if is_main_process else None
    logger.info(
        "Distributed training: enabled=%s rank=%d world_size=%d device=%s global_batch_size=%d",
        distributed,
        rank,
        world_size,
        device,
        args.batch_size * world_size * args.gradient_accumulation_steps,
    )
    logger.info("Loading student image path from %s", args.pretrained_path)
    student, image_resolution, image_features = load_pi05_image_path(
        args.pretrained_path,
        device=device,
        local_files_only=args.local_files_only,
        dtype=torch.float32,
    )
    logger.info("Loading teacher image path from %s", args.pretrained_path)
    teacher, teacher_image_resolution, _ = load_pi05_image_path(
        args.pretrained_path,
        device=device,
        local_files_only=args.local_files_only,
        dtype=torch.bfloat16 if args.amp_dtype == "bfloat16" and device.type == "cuda" else torch.float32,
    )
    if teacher_image_resolution != image_resolution:
        raise RuntimeError(
            f"Teacher/student image resolution mismatch: {teacher_image_resolution} vs {image_resolution}"
        )
    if args.image_key not in image_features:
        logger.warning(
            "image_key=%s not found in Pi0.5 config image_features=%s; continuing.",
            args.image_key,
            image_features,
        )
    student_patched = set_mem_vit_runtime(student, num_frames=args.num_frames, use_original_for_k1=False)
    teacher_patched = set_mem_vit_runtime(teacher, num_frames=1, use_original_for_k1=True)
    if student_patched == 0 or teacher_patched == 0:
        raise RuntimeError(
            "No MEM-ViT patched attention layers found. Apply modeling_pi05.py MEM-ViT patch first."
        )
    if args.gradient_checkpointing:
        student.vision_tower.gradient_checkpointing_enable()
        logger.info("Student vision gradient checkpointing enabled")
    freeze_module(teacher)
    freeze_module(student)
    unfreeze_module(student.vision_tower)
    if args.train_projector:
        unfreeze_module(student.multi_modal_projector)
    else:
        freeze_module(student.multi_modal_projector)
    if resume_checkpoint is not None:
        student.vision_tower.load_state_dict(resume_checkpoint["student_vision_tower"], strict=True)
        projector_state = resume_checkpoint.get("student_multi_modal_projector")
        if args.train_projector and projector_state is None:
            raise RuntimeError("Resume checkpoint is missing the trainable projector state")
        if projector_state is not None:
            student.multi_modal_projector.load_state_dict(projector_state, strict=True)
        logger.info("Loaded student state from %s at step %d", resume_path, resume_checkpoint["step"])
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    logger.info("Trainable parameters: %.3f M", sum(p.numel() for p in trainable_params) / 1e6)
    train_student: nn.Module = student
    if distributed:
        train_student = DDP(
            student,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
    train_val_meta = LeRobotDatasetMetadata(
        args.dataset_repo_id,
        Path(args.dataset_root) if args.dataset_root is not None else None,
        args.dataset_revision,
        force_cache_sync=args.force_cache_sync,
    )
    dataset_fps = int(train_val_meta.fps)
    train_episode_indices = load_dataset_episode_split(
        train_val_meta,
        "train",
        num_frames=args.num_frames,
        max_gap_sec=args.max_gap_sec,
        max_gap_frames=args.max_gap_frames,
        fps=dataset_fps,
        allow_short_history=args.allow_short_history,
    )
    val_episode_indices = load_dataset_episode_split(
        train_val_meta,
        "val",
        num_frames=args.num_frames,
        max_gap_sec=args.max_gap_sec,
        max_gap_frames=args.max_gap_frames,
        fps=dataset_fps,
        allow_short_history=args.allow_short_history,
    )
    test_dataset_repo_id = args.test_dataset_repo_id or args.dataset_repo_id
    test_dataset_root = args.test_dataset_root or args.dataset_root
    test_dataset_revision = (
        args.test_dataset_revision if args.test_dataset_revision is not None else args.dataset_revision
    )
    if args.test_dataset_root is not None or args.test_dataset_repo_id is not None:
        test_meta = LeRobotDatasetMetadata(
            test_dataset_repo_id,
            Path(test_dataset_root) if test_dataset_root is not None else None,
            test_dataset_revision,
            force_cache_sync=args.force_cache_sync,
        )
    else:
        test_meta = train_val_meta
    test_dataset_fps = int(test_meta.fps)
    test_episode_indices = load_dataset_episode_split(
        test_meta,
        "test",
        num_frames=args.num_frames,
        max_gap_sec=args.max_gap_sec,
        max_gap_frames=args.max_gap_frames,
        fps=test_dataset_fps,
        allow_short_history=args.allow_short_history,
    )
    if test_meta is not train_val_meta:
        del test_meta
    del train_val_meta
    video_tolerance = (
        args.video_tolerance_sec if args.video_tolerance_sec is not None else 0.55 / float(dataset_fps)
    )
    test_video_tolerance = (
        args.video_tolerance_sec
        if args.video_tolerance_sec is not None
        else 0.55 / float(test_dataset_fps)
    )
    train_ds = RandomSparseHistoryImageDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        revision=args.dataset_revision,
        image_key=args.image_key,
        num_frames=args.num_frames,
        min_gap_sec=args.min_gap_sec,
        max_gap_sec=args.max_gap_sec,
        min_gap_frames=args.min_gap_frames,
        max_gap_frames=args.max_gap_frames,
        gap_sampling=args.gap_sampling,
        num_samples=((args.max_steps - initial_step) * args.batch_size * args.gradient_accumulation_steps),
        start_sample_index=(initial_step * args.batch_size * args.gradient_accumulation_steps),
        episode_indices=train_episode_indices,
        seed=args.seed + rank * 1_000_000_000,
        allow_short_history=args.allow_short_history,
        force_cache_sync=args.force_cache_sync,
        video_backend=args.video_backend,
        tolerance_s=video_tolerance,
    )
    val_ds = RandomSparseHistoryImageDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        revision=args.dataset_revision,
        image_key=args.image_key,
        num_frames=args.num_frames,
        min_gap_sec=args.min_gap_sec,
        max_gap_sec=args.max_gap_sec,
        min_gap_frames=args.min_gap_frames,
        max_gap_frames=args.max_gap_frames,
        gap_sampling=args.gap_sampling,
        num_samples=args.eval_batches * args.batch_size,
        start_sample_index=0,
        episode_indices=val_episode_indices,
        seed=args.seed + 1_000_000 + rank * 1_000_000_000,
        allow_short_history=args.allow_short_history,
        force_cache_sync=args.force_cache_sync,
        video_backend=args.video_backend,
        tolerance_s=video_tolerance,
    )
    test_ds = RandomSparseHistoryImageDataset(
        repo_id=test_dataset_repo_id,
        root=test_dataset_root,
        revision=test_dataset_revision,
        image_key=args.image_key,
        num_frames=args.num_frames,
        min_gap_sec=args.min_gap_sec,
        max_gap_sec=args.max_gap_sec,
        min_gap_frames=args.min_gap_frames,
        max_gap_frames=args.max_gap_frames,
        gap_sampling=args.gap_sampling,
        num_samples=(args.test_eval_batches or args.eval_batches) * args.batch_size,
        start_sample_index=0,
        episode_indices=test_episode_indices,
        seed=args.seed + 2_000_000 + rank * 1_000_000_000,
        allow_short_history=args.allow_short_history,
        force_cache_sync=args.force_cache_sync,
        video_backend=args.video_backend,
        tolerance_s=test_video_tolerance,
    )
    logger.info(
        "Sparse history train/val dataset: episodes=%d valid_samples=%d per_rank_training_samples=%d "
        "fps=%d tolerance=%.6fs",
        len(train_ds.episodes),
        len(train_ds.samples),
        len(train_ds),
        train_ds.fps,
        video_tolerance,
    )
    logger.info(
        "Sparse history test dataset: episodes=%d valid_samples=%d per_rank_test_samples=%d "
        "fps=%d tolerance=%.6fs root=%s",
        len(test_ds.episodes),
        len(test_ds.samples),
        len(test_ds),
        test_ds.fps,
        test_video_tolerance,
        test_dataset_root,
    )
    logger.info(
        "Episode split: train=%d validation=%d test=%d validation_samples_per_rank=%d "
        "test_samples_per_rank=%d",
        len(train_episode_indices),
        len(val_episode_indices),
        len(test_episode_indices),
        len(val_ds),
        len(test_ds),
    )
    dataloader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed + rank),
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        generator=torch.Generator().manual_seed(args.seed + 1_000_000 + rank),
    )
    test_dataloader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        generator=torch.Generator().manual_seed(args.seed + 2_000_000 + rank),
    )
    if wandb_run is not None:
        wandb_run.config.update(
            {
                "resolved_image_resolution": list(image_resolution),
                "dataset_fps": dataset_fps,
                "test_dataset_fps": test_dataset_fps,
                "resolved_test_dataset_repo_id": test_dataset_repo_id,
                "resolved_test_dataset_root": test_dataset_root,
                "resolved_test_dataset_revision": test_dataset_revision,
                "resolved_video_tolerance_sec": video_tolerance,
                "resolved_test_video_tolerance_sec": test_video_tolerance,
                "student_patched_layers": student_patched,
                "teacher_patched_layers": teacher_patched,
                "num_valid_samples": len(train_ds.samples),
                "num_episodes": len(train_ds.episodes),
                "num_training_samples": len(train_ds) * world_size,
                "per_rank_training_samples": len(train_ds),
                "world_size": world_size,
                "per_device_batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "global_batch_size": (args.batch_size * world_size * args.gradient_accumulation_steps),
                "num_train_episodes": len(train_episode_indices),
                "num_val_episodes": len(val_episode_indices),
                "num_test_episodes": len(test_episode_indices),
                "validation_samples": len(val_ds) * world_size,
                "test_samples": len(test_ds) * world_size,
            },
            allow_val_change=True,
        )
    optimizer: torch.optim.Optimizer
    if distributed:
        optimizer = ZeroRedundancyOptimizer(
            trainable_params,
            optimizer_class=torch.optim.AdamW,
            lr=args.lr,
            weight_decay=args.weight_decay,
            foreach=False,
        )
        logger.info("AdamW optimizer states are sharded across %d ranks", world_size)
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            foreach=False,
        )
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        rng_states = resume_checkpoint["rng_states"]
        if len(rng_states) != world_size:
            raise RuntimeError(
                f"Checkpoint contains {len(rng_states)} RNG states for world_size={world_size}"
            )
        restore_rng_state(rng_states[rank], device)
        logger.info("Restored optimizer and RNG state at step %d", initial_step)
    if is_main_process:
        with (output_dir / "dataset_split.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "seed": args.seed,
                    "dataset_repo_id": args.dataset_repo_id,
                    "dataset_root": args.dataset_root,
                    "dataset_revision": args.dataset_revision,
                    "test_dataset_repo_id": test_dataset_repo_id,
                    "test_dataset_root": test_dataset_root,
                    "test_dataset_revision": test_dataset_revision,
                    "train_episode_indices": sorted(train_episode_indices),
                    "val_episode_indices": sorted(val_episode_indices),
                    "test_episode_indices": sorted(test_episode_indices),
                },
                f,
                indent=2,
            )
        with (output_dir / "train_mem_vit_distill_config.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    **vars(args),
                    "resolved_image_resolution": list(image_resolution),
                    "dataset_fps": dataset_fps,
                    "test_dataset_fps": test_dataset_fps,
                    "resolved_test_dataset_repo_id": test_dataset_repo_id,
                    "resolved_test_dataset_root": test_dataset_root,
                    "resolved_test_dataset_revision": test_dataset_revision,
                    "resolved_video_tolerance_sec": video_tolerance,
                    "resolved_test_video_tolerance_sec": test_video_tolerance,
                    "student_patched_layers": student_patched,
                    "teacher_patched_layers": teacher_patched,
                    "num_valid_samples": len(train_ds.samples),
                    "num_episodes": len(train_ds.episodes),
                    "num_training_samples": len(train_ds) * world_size,
                    "per_rank_training_samples": len(train_ds),
                    "world_size": world_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "global_batch_size": (args.batch_size * world_size * args.gradient_accumulation_steps),
                    "num_train_episodes": len(train_episode_indices),
                    "num_val_episodes": len(val_episode_indices),
                    "num_test_episodes": len(test_episode_indices),
                    "validation_samples": len(val_ds) * world_size,
                    "test_samples": len(test_ds) * world_size,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    global_step = initial_step
    checkpoint_metrics = {} if resume_checkpoint is None else resume_checkpoint.get("metrics", {})
    last_metrics = None
    use_amp = args.amp_dtype == "bfloat16" and device.type == "cuda"
    saved_val_metrics = {
        key.removeprefix("val_"): value for key, value in checkpoint_metrics.items() if key.startswith("val_")
    }
    saved_test_metrics = {
        key.removeprefix("test_"): value
        for key, value in checkpoint_metrics.items()
        if key.startswith("test_")
    }
    if resume_checkpoint is None or not saved_val_metrics:
        last_val_metrics = evaluate_distillation(
            student,
            teacher,
            val_dataloader,
            image_resolution=image_resolution,
            num_frames=args.num_frames,
            device=device,
            use_amp=use_amp,
            cosine_weight=args.cosine_weight,
            world_size=world_size,
        )
        last_eval_step = global_step
        report_validation(
            last_val_metrics,
            split_name="val",
            step=global_step,
            wandb_run=wandb_run,
            is_main_process=is_main_process,
        )
        last_test_metrics = evaluate_distillation(
            student,
            teacher,
            test_dataloader,
            image_resolution=image_resolution,
            num_frames=args.num_frames,
            device=device,
            use_amp=use_amp,
            cosine_weight=args.cosine_weight,
            world_size=world_size,
        )
        report_validation(
            last_test_metrics,
            split_name="test",
            step=global_step,
            wandb_run=wandb_run,
            is_main_process=is_main_process,
        )
    else:
        last_val_metrics = saved_val_metrics
        last_test_metrics = saved_test_metrics or None
        last_eval_step = int(resume_checkpoint.get("last_eval_step", global_step))
        logger.info("Reusing validation metrics from checkpoint step %d", last_eval_step)
    train_batches = iter(dataloader)
    step_iter = range(initial_step, args.max_steps)
    if tqdm is not None and is_main_process:
        step_iter = tqdm(step_iter, total=args.max_steps, initial=initial_step, desc="steps")
    for _ in step_iter:
        optimizer.zero_grad(set_to_none=True)
        accumulated_metrics: dict[str, float] = {}
        accumulated_offset_min = math.inf
        accumulated_offset_max = -math.inf
        for _accumulation_index in range(args.gradient_accumulation_steps):
            batch = next(train_batches)
            frames = batch["frames"]
            if frames.ndim != 5:
                raise RuntimeError(f"Expected frames [B,K,C,H,W], got {tuple(frames.shape)}")
            bsz, k = frames.shape[:2]
            if k != args.num_frames:
                raise RuntimeError(f"Expected K={args.num_frames}, got K={k}")
            flat_frames = frames.reshape(bsz * k, *frames.shape[2:]).contiguous()
            current_frames = frames[:, -1].contiguous()
            student_pixels = preprocess_pi05_image(
                flat_frames, image_resolution=image_resolution, device=device
            )
            teacher_pixels = preprocess_pi05_image(
                current_frames, image_resolution=image_resolution, device=device
            )
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp),
            ):
                teacher_features = teacher(teacher_pixels)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                student_features_all = train_student(student_pixels)
                student_features_all = student_features_all.view(bsz, k, *student_features_all.shape[1:])
                student_features = student_features_all[:, -1]
                loss, micro_metrics = distill_loss(
                    student_features.float(),
                    teacher_features.detach().float(),
                    cosine_weight=args.cosine_weight,
                )
                (loss / args.gradient_accumulation_steps).backward()
            for key, value in micro_metrics.items():
                accumulated_metrics[key] = accumulated_metrics.get(key, 0.0) + value
            offsets = batch["offsets"]
            accumulated_offset_min = min(accumulated_offset_min, float(offsets.min().item()))
            accumulated_offset_max = max(accumulated_offset_max, float(offsets.max().item()))
        if args.grad_clip_norm is not None and args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip_norm)
        optimizer.step()
        global_step += 1
        metrics = {
            key: value / args.gradient_accumulation_steps for key, value in accumulated_metrics.items()
        }
        metrics = reduce_metrics(metrics, device, world_size)
        last_metrics = metrics
        offset_unit = "frames" if train_ds.sample_by_frames else "seconds"
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/global_step": global_step,
                    "train/loss": metrics["loss"],
                    "train/mse": metrics["mse"],
                    "train/nmse": metrics["nmse"],
                    "train/cosine_similarity": metrics["cosine_similarity"],
                    "train/cosine_distance": metrics["cosine_distance"],
                    "train/teacher_rms": metrics["teacher_rms"],
                    "train/student_rms": metrics["student_rms"],
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/progress": global_step / args.max_steps,
                    "train/samples_seen": (
                        global_step * args.batch_size * world_size * args.gradient_accumulation_steps
                    ),
                    f"data/min_offset_{offset_unit}": accumulated_offset_min,
                    f"data/max_offset_{offset_unit}": accumulated_offset_max,
                }
            )
        if is_main_process and global_step % args.log_every == 0:
            offset_unit = "frames" if train_ds.sample_by_frames else "s"
            logger.info(
                "step=%d/%d loss=%.6f mse=%.6f nmse=%.6f cos_sim=%.6f "
                "teacher_rms=%.6f student_rms=%.6f offsets=[%.2f, %.2f]%s",
                global_step,
                args.max_steps,
                metrics["loss"],
                metrics["mse"],
                metrics["nmse"],
                metrics["cosine_similarity"],
                metrics["teacher_rms"],
                metrics["student_rms"],
                accumulated_offset_min,
                accumulated_offset_max,
                offset_unit,
            )
        if global_step % args.eval_every == 0:
            last_val_metrics = evaluate_distillation(
                student,
                teacher,
                val_dataloader,
                image_resolution=image_resolution,
                num_frames=args.num_frames,
                device=device,
                use_amp=use_amp,
                cosine_weight=args.cosine_weight,
                world_size=world_size,
            )
            last_eval_step = global_step
            report_validation(
                last_val_metrics,
                split_name="val",
                step=global_step,
                wandb_run=wandb_run,
                is_main_process=is_main_process,
            )
            last_test_metrics = evaluate_distillation(
                student,
                teacher,
                test_dataloader,
                image_resolution=image_resolution,
                num_frames=args.num_frames,
                device=device,
                use_amp=use_amp,
                cosine_weight=args.cosine_weight,
                world_size=world_size,
            )
            report_validation(
                last_test_metrics,
                split_name="test",
                step=global_step,
                wandb_run=wandb_run,
                is_main_process=is_main_process,
            )
        if args.save_every > 0 and global_step % args.save_every == 0:
            saved_metrics = {
                **metrics,
                **{f"val_{key}": value for key, value in last_val_metrics.items()},
                **(
                    {f"test_{key}": value for key, value in last_test_metrics.items()}
                ),
            }
            path = save_checkpoint(
                output_dir,
                step=global_step,
                student=student,
                optimizer=optimizer,
                args=args,
                metrics=saved_metrics,
                last_eval_step=last_eval_step,
                rng_device=device,
                rank=rank,
                world_size=world_size,
                wandb_run_id=wandb_run.id if wandb_run is not None else None,
            )
            if is_main_process:
                logger.info("Saved checkpoint: %s", path)
            if distributed:
                dist.barrier()
    if last_eval_step != global_step:
        last_val_metrics = evaluate_distillation(
            student,
            teacher,
            val_dataloader,
            image_resolution=image_resolution,
            num_frames=args.num_frames,
            device=device,
            use_amp=use_amp,
            cosine_weight=args.cosine_weight,
            world_size=world_size,
        )
        report_validation(
            last_val_metrics,
            split_name="val",
            step=global_step,
            wandb_run=wandb_run,
            is_main_process=is_main_process,
        )
        last_test_metrics = evaluate_distillation(
            student,
            teacher,
            test_dataloader,
            image_resolution=image_resolution,
            num_frames=args.num_frames,
            device=device,
            use_amp=use_amp,
            cosine_weight=args.cosine_weight,
            world_size=world_size,
        )
        report_validation(
            last_test_metrics,
            split_name="test",
            step=global_step,
            wandb_run=wandb_run,
            is_main_process=is_main_process,
        )
        last_eval_step = global_step
    checkpoint_metrics = {
        **(last_metrics or {}),
        **{f"val_{key}": value for key, value in last_val_metrics.items()},
        **{f"test_{key}": value for key, value in last_test_metrics.items()},
    }
    final_path = None
    if args.save_final:
        final_path = save_checkpoint(
            output_dir,
            step=global_step,
            student=student,
            optimizer=optimizer,
            args=args,
            metrics=checkpoint_metrics,
            last_eval_step=last_eval_step,
            rng_device=device,
            rank=rank,
            world_size=world_size,
            wandb_run_id=wandb_run.id if wandb_run is not None else None,
        )
    if is_main_process:
        if final_path is not None:
            logger.info("Training finished. Final checkpoint: %s", final_path)
        else:
            logger.info("Training finished without a final checkpoint (--no-save-final)")
        if wandb_run is not None:
            if final_path is not None:
                wandb_run.summary["final_checkpoint"] = str(final_path)
            wandb_run.summary["final_step"] = global_step
            if last_metrics is not None:
                wandb_run.summary.update({f"final_{key}": value for key, value in last_metrics.items()})
            wandb_run.summary.update({f"final_val_{key}": value for key, value in last_val_metrics.items()})
            wandb_run.summary.update(
                {f"final_test_{key}": value for key, value in last_test_metrics.items()}
            )
            wandb_run.finish()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

# HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1,2 uv run torchrun \
#   --standalone --nproc-per-node=3 -m lerobot.scripts.train_mem_vit_distill \
#   --pretrained-path /data/checkpoints/lerobot_pi05_base \
#   --local-files-only \
#   --dataset-repo-id local/vlnce_smooth_memvit \
#   --dataset-root /data/VLNCE_smooth_lerobot_final \
#   --image-key observation.images.rgb \
#   --output-dir outputs/mem_vit_distill_smoke \
#   --num-frames 6 \
#   --min-gap-frames 1 \
#   --max-gap-frames 5 \
#   --gap-sampling uniform_single \
#   --batch-size 1 \
#   --max-steps 100 \
#   --num-workers 0 \
#   --eval-every 100 \
#   --eval-batches 20 \
#   --test-eval-batches 20 \
#   --log-every 1 \
#   --save-every 10 \
#   --wandb-enable \
#   --wandb-project mem-vit-distill \
#   --wandb-run-name vlnce-smoke
