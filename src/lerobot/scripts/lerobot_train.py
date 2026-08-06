#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Train a policy.

Requires: pip install 'lerobot[training]'  (includes dataset + accelerate + wandb extras)
"""

import dataclasses
import hashlib
import json
import logging
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from accelerate import Accelerator

import numpy as np
import torch
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.common.train_utils import (
    gather_fsdp_state_dicts,
    get_step_checkpoint_dir,
    get_step_identifier,
    load_fsdp_optimizer_state,
    load_training_batch_size,
    load_training_num_processes,
    load_training_state,
    push_checkpoint_to_hub,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs import JobConfig, parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import EpisodeAwareSampler, compute_sampler_state
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.jobs import submit_to_hf
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
from lerobot.rewards import make_reward_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    cycle,
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)

from .lerobot_eval import eval_policy_all


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: "Accelerator",
    lr_scheduler=None,
    lock=None,
    sample_weighter=None,
    loss_scale: float = 1.0,
    optimizer_step: bool = True,
) -> tuple[MetricsTracker, dict | None]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.
        sample_weighter: Optional SampleWeighter instance for per-sample loss weighting.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Compute sample weights if a weighter is provided
    sample_weights = None
    weight_stats = None
    if sample_weighter is not None:
        sample_weights, weight_stats = sample_weighter.compute_batch_weights(batch)

    sync_context = nullcontext() if optimizer_step else accelerator.no_sync(policy)
    with sync_context:
        # Let accelerator handle mixed precision
        with accelerator.autocast():
            if sample_weights is not None:
                # Use per-sample loss for weighted training
                # Note: Policies supporting sample weighting must implement forward(batch, reduction="none")
                per_sample_loss, output_dict = policy.forward(batch, reduction="none")

                # Weighted loss: each sample's contribution is scaled by its weight.
                # We divide by weight sum (not batch size) so that if some weights are zero,
                # the remaining samples contribute proportionally more, preserving gradient scale.
                # Weights are pre-normalized to sum to batch_size for stable training dynamics.
                epsilon = 1e-6
                loss = (per_sample_loss * sample_weights).sum() / (sample_weights.sum() + epsilon)

                # Log weighting statistics
                if output_dict is None:
                    output_dict = {}
                for key, value in weight_stats.items():
                    output_dict[f"sample_weight_{key}"] = value
            else:
                loss, output_dict = policy.forward(batch)

            # TODO(rcadene): policy.unnormalize_outputs(out_dict)

        # Use accelerator's backward method. Scale the loss so accumulated gradients
        # match the mean-gradient semantics of a single larger batch.
        accelerator.backward(loss / loss_scale)

    if optimizer_step:
        # Clip gradients if specified
        if grad_clip_norm > 0:
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), float("inf"), error_if_nonfinite=False
            )

        # Optimizer step
        with lock if lock is not None else nullcontext():
            optimizer.step()

        optimizer.zero_grad()

        # Step through pytorch scheduler at every optimizer update instead of every micro-batch
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Update internal buffers if policy has update method
        if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
            accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()
        train_metrics.grad_norm = grad_norm.item()

    train_metrics.loss = loss.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    if torch.cuda.is_available():
        train_metrics.gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    return train_metrics, output_dict


def _resolve_task_variants_path(dataset_root: Path, configured_path: str | None) -> Path | None:
    if configured_path:
        path = Path(configured_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Task variants file does not exist: {path}")
        return path
    default_path = dataset_root / "meta" / "task_variants.json"
    return default_path if default_path.exists() else None


def load_task_variants(dataset_root: Path, configured_path: str | None) -> dict[int, list[str]]:
    """Load optional episode-level task rephrasings.

    Expected format:
        {
          "0": ["instruction A", "instruction B"],
          "1": ["instruction C", "instruction D"]
        }

    For convenience, values may also be {"tasks": [...]} or {"variants": [...]}.
    """
    path = _resolve_task_variants_path(dataset_root, configured_path)
    if path is None:
        return {}

    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and "episodes" in raw and isinstance(raw["episodes"], dict):
        raw = raw["episodes"]
    if not isinstance(raw, dict):
        raise ValueError(f"Task variants file must contain a JSON object: {path}")

    variants: dict[int, list[str]] = {}
    for ep_key, value in raw.items():
        if isinstance(value, str):
            candidates = [value]
        elif isinstance(value, dict):
            nested = value.get("tasks", value.get("variants"))
            candidates = nested if isinstance(nested, list) else []
        elif isinstance(value, list):
            candidates = value
        else:
            candidates = []
        candidates = [str(item).strip() for item in candidates if isinstance(item, str) and item.strip()]
        if candidates:
            variants[int(ep_key)] = candidates
    if not variants:
        raise ValueError(f"Task variants file contains no usable episode variants: {path}")

    logging.info("Loaded task variants from %s for %d episode(s)", path, len(variants))
    return variants


def _to_int_list(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.detach().cpu().view(-1).tolist()]
    return [int(item) for item in value]


def _batch_tasks_as_list(task_value: Any, batch_size: int) -> list[str]:
    if isinstance(task_value, str):
        return [task_value] * batch_size
    if isinstance(task_value, (list, tuple)):
        return [str(item) for item in task_value]
    return [""] * batch_size


def _select_task_variant(
    candidates: list[str],
    *,
    seed: int,
    step: int,
    sample_index: int,
    episode_index: int,
    randomize: bool,
) -> str:
    if not randomize or len(candidates) == 1:
        return candidates[0]
    digest = hashlib.blake2b(
        f"{seed}:{step}:{sample_index}:{episode_index}".encode(),
        digest_size=8,
    ).digest()
    variant_index = int.from_bytes(digest, "big") % len(candidates)
    return candidates[variant_index]


def apply_task_variants_to_batch(
    batch: dict[str, Any],
    task_variants: dict[int, list[str]],
    *,
    step: int,
    seed: int | None,
    randomize: bool,
) -> int:
    if not task_variants or "episode_index" not in batch:
        return 0
    episode_indices = _to_int_list(batch["episode_index"])
    sample_indices = _to_int_list(batch["index"]) if "index" in batch else list(range(len(episode_indices)))
    tasks = _batch_tasks_as_list(batch.get("task"), len(episode_indices))

    applied = 0
    for i, episode_index in enumerate(episode_indices):
        candidates = task_variants.get(episode_index)
        if not candidates:
            continue
        tasks[i] = _select_task_variant(
            candidates,
            seed=seed if seed is not None else 0,
            step=step,
            sample_index=sample_indices[i],
            episode_index=episode_index,
            randomize=randomize,
        )
        applied += 1
    if applied:
        batch["task"] = tasks
    return applied


def resolve_task_complete_sampling(dataset, policy_cfg) -> tuple[list[int], dict[int, int]] | None:
    """Cap post-completion chunk starts while retaining the full episode as label context."""
    from lerobot.policies.pi05.b2_action_transform import DATASET_ACTION_NAMES, TASK_COMPLETE_NAME

    action_names = dataset.meta.features.get(ACTION, {}).get("names") or []
    if tuple(action_names) != DATASET_ACTION_NAMES:
        return None
    tail_seconds = getattr(policy_cfg, "task_complete_sample_tail_seconds", 2.0)
    tail_frames = None if tail_seconds is None else int(np.ceil(float(tail_seconds) * dataset.meta.fps))
    selected_episodes = set(dataset.episodes or range(dataset.num_episodes))
    first_complete: dict[int, int] = {}
    seen_incomplete_after_complete: set[int] = set()
    dim = action_names.index(TASK_COMPLETE_NAME)
    columns = dataset.hf_dataset.select_columns([ACTION, "episode_index", "frame_index"]).with_format("numpy")
    for batch in columns.iter(batch_size=65_536):
        actions = np.asarray(batch[ACTION])
        episode_indices = np.asarray(batch["episode_index"], dtype=np.int64)
        frame_indices = np.asarray(batch["frame_index"], dtype=np.int64)
        complete = actions[:, dim] > 0
        for episode_index, frame_index, is_complete in zip(
            episode_indices, frame_indices, complete, strict=True
        ):
            episode_index = int(episode_index)
            if episode_index not in selected_episodes:
                continue
            if is_complete:
                first_complete.setdefault(episode_index, int(frame_index))
            elif episode_index in first_complete:
                seen_incomplete_after_complete.add(episode_index)
    if seen_incomplete_after_complete:
        raise ValueError(
            "task_complete must be monotonic within every episode; false follows true in episodes "
            f"{sorted(seen_incomplete_after_complete)[:10]}"
        )
    missing = sorted(selected_episodes - first_complete.keys())
    if missing:
        raise ValueError(
            f"Every selected episode must contain an explicit task_complete tail; missing {missing[:10]}"
        )

    from_indices = [int(value) for value in dataset.meta.episodes["dataset_from_index"]]
    original_to = [int(value) for value in dataset.meta.episodes["dataset_to_index"]]
    capped_to = list(original_to)
    start_counts: dict[int, int] = {}
    for episode_index in selected_episodes:
        episode_length = original_to[episode_index] - from_indices[episode_index]
        count = (
            episode_length
            if tail_frames is None
            else min(episode_length, first_complete[episode_index] + 1 + tail_frames)
        )
        start_counts[episode_index] = count
        capped_to[episode_index] = from_indices[episode_index] + count
    return capped_to, start_counts


def configure_action_bool_balance(
    cfg: TrainPipelineConfig, dataset, start_counts: dict[int, int] | None = None
) -> dict[str, dict[str, int | float]] | None:
    """Resolve fixed class priors for every enabled boolean action from the train split."""
    policy_cfg = cfg.trainable_config
    action_names = dataset.meta.features.get(ACTION, {}).get("names") or []
    from lerobot.policies.pi05.b2_action_transform import DATASET_ACTION_NAMES

    if tuple(action_names) != DATASET_ACTION_NAMES:
        return None

    enabled = {
        "arm_teleop_inactive": bool(getattr(policy_cfg, "action_predict_arm_teleop_inactive", False)),
        "arm_reset": bool(getattr(policy_cfg, "action_predict_arm_reset", False)),
        "gripper_target": bool(getattr(policy_cfg, "action_predict_gripper", False)),
        "task_complete": bool(getattr(policy_cfg, "action_predict_task_complete", False)),
    }
    enabled_names = [name for name, is_enabled in enabled.items() if is_enabled]
    if not enabled_names:
        return None

    # Keep the generic training entry point free of eager PI0.5 imports.
    from lerobot.policies.pi05.b2_action_transform import (
        action_label_multiplicity,
        action_sample_offsets,
    )

    episode_indices = dataset.episodes if dataset.episodes is not None else list(range(dataset.num_episodes))
    episodes_meta = dataset.meta.episodes
    lengths = [
        int(episodes_meta[episode_index]["dataset_to_index"])
        - int(episodes_meta[episode_index]["dataset_from_index"])
        for episode_index in episode_indices
    ]
    control_frequency_hz = float(getattr(policy_cfg, "control_frequency_hz", None) or dataset.meta.fps)
    offsets = action_sample_offsets(
        policy_cfg.chunk_size,
        float(dataset.meta.fps),
        control_frequency_hz,
    )
    episode_multiplicities = {
        episode_index: action_label_multiplicity(
            length,
            offsets,
            num_start_frames=None if start_counts is None else start_counts[episode_index],
        ).numpy()
        for episode_index, length in zip(episode_indices, lengths, strict=True)
    }
    counts: dict[str, list[int]] = {name: [0, 0] for name in enabled_names}
    if counts:
        missing_names = [name for name in counts if name not in action_names]
        if missing_names:
            raise ValueError(f"Enabled boolean actions are absent from the dataset: {missing_names}")
        action_indices = {name: action_names.index(name) for name in counts}
        completion_dim = action_names.index("task_complete")
        columns = dataset.hf_dataset.select_columns([ACTION, "episode_index", "frame_index"]).with_format(
            "numpy"
        )
        for batch in columns.iter(batch_size=65_536):
            actions = np.asarray(batch[ACTION])
            batch_episode_indices = np.asarray(batch["episode_index"], dtype=np.int64)
            frame_indices = np.asarray(batch["frame_index"], dtype=np.int64)
            label_multiplicity = np.fromiter(
                (
                    episode_multiplicities[int(episode_index)][int(frame_index)]
                    for episode_index, frame_index in zip(batch_episode_indices, frame_indices, strict=True)
                ),
                dtype=np.int64,
                count=len(frame_indices),
            )
            for name, dim in action_indices.items():
                applicable = np.ones(len(actions), dtype=bool)
                if name != "task_complete":
                    applicable = actions[:, completion_dim] <= 0
                if (
                    name == "gripper_target"
                    and getattr(policy_cfg, "action_gripper_target_true_side", "negative") == "negative"
                ):
                    target_true = actions[:, dim] < 0
                else:
                    target_true = actions[:, dim] > 0
                applicable_multiplicity = label_multiplicity[applicable]
                positive = int(label_multiplicity[target_true & applicable].sum())
                counts[name][0] += positive
                counts[name][1] += int(applicable_multiplicity.sum()) - positive

    stats: dict[str, dict[str, int | float]] = {}
    true_fractions: dict[str, float] = {}
    bool_weight = float(policy_cfg.action_bool_loss_weight)
    for name, (positive, negative) in counts.items():
        if positive == 0 or negative == 0:
            raise ValueError(
                f"Cannot class-balance {name}: the train split must contain both classes, "
                f"got positive={positive}, negative={negative}."
            )
        known = positive + negative
        true_fraction = positive / known
        true_fractions[name] = true_fraction
        stats[name] = {
            "positive_labels": positive,
            "negative_labels": negative,
            "known_labels": known,
            "true_fraction": true_fraction,
            "true_weight": bool_weight * 0.5 / true_fraction,
            "false_weight": bool_weight * 0.5 / (1.0 - true_fraction),
        }

    saved_fractions = dict(getattr(policy_cfg, "action_bool_true_fractions", {}))
    if cfg.resume and saved_fractions and saved_fractions != true_fractions:
        raise ValueError(
            "Resume train-split boolean priors disagree with the checkpoint: "
            f"checkpoint={saved_fractions}, current={true_fractions}"
        )
    policy_cfg.action_bool_true_fractions = true_fractions
    return stats


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: "Accelerator | None" = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    if cfg.job.is_remote:
        return submit_to_hf(cfg)

    from lerobot.utils.import_utils import require_package

    require_package("accelerate", extra="training")
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, DistributedType

    cfg.validate()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # Accelerate auto-detects the device based on the available hardware and ignores the policy.device setting.
        # Force the device to be CPU when the active config's device is set to CPU (works for both policy and reward model training).
        force_cpu = cfg.trainable_config.device == "cpu"
        # Drive Accelerate's autocast from policy.dtype (bf16/fp16 activate it; float32/absent -> launcher default).
        policy_dtype = getattr(cfg.trainable_config, "dtype", None)
        mixed_precision = {"bfloat16": "bf16", "float16": "fp16", "float32": "no"}.get(policy_dtype)
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            mixed_precision=mixed_precision,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: the global main process downloads once to the shared
    # dataset root, then a barrier lets every other rank read the already-populated copy.
    # LeRobotDataset skips its snapshot_download when try_load() succeeds, so no rank re-downloads.
    if is_main_process:
        logging.info("Creating dataset")
        dataset, eval_dataset = make_train_eval_datasets(cfg)

    accelerator.wait_for_everyone()

    # Other ranks read from the shared copy populated by the main process.
    if not is_main_process:
        dataset, eval_dataset = make_train_eval_datasets(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.env_eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    task_variants = load_task_variants(dataset.root, cfg.dataset.task_variants_path)
    completion_sampling = resolve_task_complete_sampling(dataset, cfg.trainable_config)
    capped_train_to = completion_sampling[0] if completion_sampling is not None else None
    train_start_counts = completion_sampling[1] if completion_sampling is not None else None
    if is_main_process and train_start_counts is not None:
        retained = sum(train_start_counts.values())
        logging.info(
            "Explicit task-completion sampling: retained %d chunk starts across %d train episodes "
            "(post-completion input tail cap=%s seconds)",
            retained,
            len(train_start_counts),
            cfg.trainable_config.task_complete_sample_tail_seconds,
        )
    action_bool_balance = configure_action_bool_balance(cfg, dataset, train_start_counts)
    if is_main_process and action_bool_balance is not None:
        for name, balance in action_bool_balance.items():
            logging.info(
                "%s train-split balance: positive=%d, negative=%d, true_fraction=%.6f, "
                "effective_true_weight=%.4f, effective_false_weight=%.4f",
                name,
                balance["positive_labels"],
                balance["negative_labels"],
                balance["true_fraction"],
                balance["true_weight"],
                balance["false_weight"],
            )

    if cfg.is_reward_model_training:
        if is_main_process:
            logging.info("Creating reward model")
        from lerobot.rewards import make_reward_model

        policy = make_reward_model(
            cfg=cfg.reward_model,
            dataset_stats=dataset.meta.stats,
            dataset_meta=dataset.meta,
        )
        if not policy.is_trainable:
            raise ValueError(
                f"Reward model '{policy.name}' is zero-shot and cannot be trained via lerobot-train. "
                "Use it directly for inference via compute_reward() (e.g. offline precompute)."
            )
    else:
        if is_main_process:
            logging.info("Creating policy")
        policy = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
            rename_map=cfg.rename_map,
        )
        deployment_metadata = getattr(policy.config, "deployment_metadata", None)
        if is_main_process and callable(deployment_metadata):
            resolved_deployment_metadata = deployment_metadata()
            logging.info("PI0.5 deployment metadata: %s", resolved_deployment_metadata)
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
            (cfg.output_dir / "pi05_deployment_metadata.json").write_text(
                json.dumps(resolved_deployment_metadata, indent=2), encoding="utf-8"
            )

        if is_main_process:
            action_chunk_size = int(policy.config.chunk_size)
            action_steps_to_execute = int(policy.config.n_action_steps)
            control_frequency_hz = float(policy.config.control_frequency_hz)
            action_dt = 1.0 / control_frequency_hz
            logging.info(
                "Policy temporal semantics: dataset_fps=%g, control_frequency_hz=%g, action_dt=%.6fs, "
                "chunk_size=%d (%.3fs horizon), n_action_steps=%d (%.3fs before replanning)",
                float(dataset.meta.fps),
                control_frequency_hz,
                action_dt,
                action_chunk_size,
                action_chunk_size * action_dt,
                action_steps_to_execute,
                action_steps_to_execute * action_dt,
            )
            if getattr(policy.config, "b2_action_representation", None) == "local_trajectory":
                logging.info(
                    "B2 local-trajectory source: %s",
                    "global observation.state pose transform"
                    if getattr(policy.config, "b2_global_pose_state_indices", None) is not None
                    else "SE(2) integration of commanded body twist",
                )

    if is_main_process and wandb_logger:
        wandb_logger.update_config(
            {
                "runtime/dataset_num_frames": int(dataset.num_frames),
                "runtime/dataset_num_episodes": int(dataset.num_episodes),
                "runtime/dataset_fps": float(dataset.meta.fps),
                "runtime/dataset_camera_keys": list(dataset.meta.camera_keys),
                "runtime/dataset_features": dataset.meta.features,
                "runtime/task_variants_enabled": bool(task_variants),
                "runtime/task_variant_episodes": len(task_variants),
                "runtime/random_task_variant": cfg.dataset.random_task_variant,
                "runtime/eval_task_variant": cfg.dataset.eval_task_variant,
                "runtime/policy_input_features": {
                    key: {"type": value.type.value, "shape": list(value.shape)}
                    for key, value in policy.config.input_features.items()
                }
                if not cfg.is_reward_model_training
                else None,
                "runtime/policy_output_features": {
                    key: {"type": value.type.value, "shape": list(value.shape)}
                    for key, value in policy.config.output_features.items()
                }
                if not cfg.is_reward_model_training
                else None,
                "runtime/pi05_deployment_metadata": (
                    policy.config.deployment_metadata()
                    if not cfg.is_reward_model_training
                    and callable(getattr(policy.config, "deployment_metadata", None))
                    else None
                ),
                "runtime/per_device_batch_size": cfg.batch_size,
                "runtime/gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                "runtime/effective_batch_size": (
                    cfg.batch_size * cfg.gradient_accumulation_steps * accelerator.num_processes
                ),
                "runtime/num_processes": accelerator.num_processes,
                "runtime/action_bool_balance": action_bool_balance,
                "runtime/action_dt_seconds": (
                    1.0 / float(policy.config.control_frequency_hz)
                    if not cfg.is_reward_model_training
                    else None
                ),
                "runtime/action_chunk_size": (
                    int(policy.config.chunk_size) if not cfg.is_reward_model_training else None
                ),
                "runtime/action_horizon_seconds": (
                    int(policy.config.chunk_size) / float(policy.config.control_frequency_hz)
                    if not cfg.is_reward_model_training
                    else None
                ),
                "runtime/action_steps_to_execute": (
                    int(policy.config.n_action_steps) if not cfg.is_reward_model_training else None
                ),
                "runtime/replan_interval_seconds": (
                    int(policy.config.n_action_steps) / float(policy.config.control_frequency_hz)
                    if not cfg.is_reward_model_training
                    else None
                ),
                "runtime/b2_trajectory_source": (
                    (
                        "global_pose_transform"
                        if getattr(policy.config, "b2_global_pose_state_indices", None) is not None
                        else "twist_integration"
                    )
                    if not cfg.is_reward_model_training
                    and getattr(policy.config, "b2_action_representation", None) == "local_trajectory"
                    else None
                ),
            }
        )

    if cfg.peft is not None:
        if cfg.is_reward_model_training:
            raise ValueError("PEFT is only supported for policy training. ")
        from peft import PeftModel

        if isinstance(policy, PeftModel):
            logging.info("PEFT adapter already loaded from checkpoint, skipping wrap_with_peft.")
        else:
            logging.info("Using PEFT! Wrapping model.")
            peft_cli_overrides = dataclasses.asdict(cfg.peft)
            policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    # Wait for all processes to finish model creation before continuing
    accelerator.wait_for_everyone()

    active_cfg = cfg.trainable_config
    processor_pretrained_path = active_cfg.pretrained_path

    processor_kwargs = {}
    if (processor_pretrained_path and not cfg.resume) or not processor_pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats
    # The configured PI0.5 I/O schema derives selected state/action statistics
    # from the unchanged dataset stats. This is also required on resume because
    # the generic checkpoint overrides below are built from raw stats.
    if getattr(active_cfg, "io_schema_resolved", False):
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.is_reward_model_training:
        processor_kwargs["dataset_meta"] = dataset.meta

    if not cfg.is_reward_model_training and processor_pretrained_path is not None:
        preprocessor_overrides = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        postprocessor_overrides = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }
        if getattr(active_cfg, "use_relative_actions", False):
            preprocessor_overrides["relative_actions_processor"] = {
                "enabled": True,
                "exclude_joints": getattr(active_cfg, "relative_exclude_joints", []),
                "action_names": getattr(active_cfg, "action_feature_names", None),
            }
            postprocessor_overrides["absolute_actions_processor"] = {"enabled": True}
        processor_kwargs["preprocessor_overrides"] = preprocessor_overrides
        processor_kwargs["postprocessor_overrides"] = postprocessor_overrides

    if cfg.is_reward_model_training:
        preprocessor, postprocessor = make_reward_pre_post_processors(
            cfg.reward_model,
            **processor_kwargs,
        )
    else:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=processor_pretrained_path,
            pretrained_revision=getattr(cfg.policy, "pretrained_revision", None),
            **processor_kwargs,
        )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Create sample weighter if configured (e.g., for RA-BC training)
    sample_weighter = None
    if cfg.sample_weighting is not None:
        from lerobot.utils.sample_weighting import make_sample_weighter

        if is_main_process:
            logging.info(f"Creating sample weighter: {cfg.sample_weighting.type}")
        sample_weighter = make_sample_weighter(
            cfg.sample_weighting,
            policy,
            device,
            dataset_root=cfg.dataset.root,
            dataset_repo_id=cfg.dataset.repo_id,
        )

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        # Under FSDP the optimizer state is sharded and must be loaded after `accelerator.prepare()`
        # (see load_fsdp_optimizer_state below), so skip the optimizer here and load it then.
        is_fsdp = accelerator.distributed_type == DistributedType.FSDP
        step, optimizer, lr_scheduler = load_training_state(
            cfg.checkpoint_path, optimizer, lr_scheduler, load_optimizer=not is_fsdp
        )

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * cfg.gradient_accumulation_steps * num_processes
        logging.info(
            "Effective batch size: "
            f"{cfg.batch_size} x grad_accum {cfg.gradient_accumulation_steps} x {num_processes} = "
            f"{effective_bs}"
        )
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if not cfg.dataset.streaming:
        # All non-streaming (map-style) datasets use EpisodeAwareSampler.
        # The order is a pure function of (seed, epoch), so every rank independently produces the
        # same permutation. accelerate then shards it disjointly across ranks via BatchSamplerShard
        # without needing a `generator` attribute to synchronize an RNG, and resume is sample-exact.
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            capped_train_to or dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=getattr(active_cfg, "drop_n_last_frames", 0),
            shuffle=True,
            seed=cfg.seed if cfg.seed is not None else 0,
            absolute_to_relative_idx=dataset.absolute_to_relative_idx,
        )
        if cfg.resume and step > 0:
            # The resume offset depends on the (num_processes, batch_size) that produced `step`, so
            # use the values recorded in the checkpoint (falling back to the current ones for older
            # ckpts that did not store them).
            saved_num_processes = load_training_num_processes(cfg.checkpoint_path)
            saved_batch_size = load_training_batch_size(cfg.checkpoint_path)
            ckpt_num_processes = saved_num_processes or accelerator.num_processes
            current_sampler_batch_size = cfg.batch_size * cfg.gradient_accumulation_steps
            ckpt_batch_size = saved_batch_size or current_sampler_batch_size
            if is_main_process and saved_num_processes not in (None, accelerator.num_processes):
                logging.warning(
                    f"Resuming with num_processes={accelerator.num_processes} but the checkpoint was "
                    f"written with num_processes={saved_num_processes}. The data order resumes at the "
                    "right epoch/offset, but per-rank sample-exactness requires the same world size."
                )
            if is_main_process and saved_batch_size not in (None, current_sampler_batch_size):
                logging.warning(
                    f"Resuming with per-step sampler batch size={current_sampler_batch_size} "
                    f"(batch_size={cfg.batch_size}, "
                    f"gradient_accumulation_steps={cfg.gradient_accumulation_steps}) but the checkpoint "
                    f"was written with batch_size={saved_batch_size}. The data order resumes at the "
                    "right epoch/offset, but per-rank sample-exactness requires the same per-step "
                    "sampler batch size."
                )
            sampler_state = compute_sampler_state(step, len(sampler), ckpt_batch_size, ckpt_num_processes)
            sampler.load_state_dict(sampler_state)
            if is_main_process:
                logging.info(
                    f"Resuming data order at epoch {sampler_state['epoch']}, "
                    f"sample {sampler_state['start_index']}"
                )
    else:
        shuffle = True
        sampler = None

    # Only swap in the language-aware collate when the dataset actually
    # declares language columns; otherwise stay on PyTorch's default
    # collate so non-language training runs are unaffected.
    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )

    # Build eval dataloader if a held-out split exists
    eval_dataloader = None
    if eval_dataset is not None:
        eval_ds = eval_dataset
        eval_completion_sampling = resolve_task_complete_sampling(eval_dataset, cfg.trainable_config)
        eligible_eval_indices = None
        if eval_completion_sampling is not None:
            eligible_eval_indices = EpisodeAwareSampler(
                eval_dataset.meta.episodes["dataset_from_index"],
                eval_completion_sampling[0],
                episode_indices_to_use=eval_dataset.episodes,
                absolute_to_relative_idx=eval_dataset.absolute_to_relative_idx,
            ).indices
        if cfg.max_eval_samples > 0 and hasattr(eval_dataset, "hf_dataset"):
            task_arr = eval_dataset.hf_dataset.data.column("task_index").to_numpy()
            unique_tasks = sorted(set(task_arr.tolist()))
            per_task = max(1, cfg.max_eval_samples // len(unique_tasks))
            selected: list[int] = []
            for t in unique_tasks:
                candidates = (
                    np.asarray(eligible_eval_indices, dtype=np.int64)
                    if eligible_eval_indices is not None
                    else np.arange(len(task_arr))
                )
                frames = candidates[task_arr[candidates] == t][:per_task]
                selected.extend(frames.tolist())
            eval_ds = torch.utils.data.Subset(eval_dataset, selected)
        elif eligible_eval_indices is not None:
            eval_ds = torch.utils.data.Subset(eval_dataset, eligible_eval_indices)

        eval_collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
        eval_dataloader = torch.utils.data.DataLoader(
            eval_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            collate_fn=eval_collate_fn,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        )

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    if eval_dataloader is not None:
        policy, optimizer, dataloader, lr_scheduler, eval_dataloader = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler, eval_dataloader
        )
    else:
        policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler
        )

    # FSDP optimizer state is sharded across ranks, so it can only be loaded once the optimizer and
    # model are FSDP-wrapped (i.e. after `prepare`). Collective: every rank must participate.
    if cfg.resume and accelerator.distributed_type == DistributedType.FSDP:
        load_fsdp_optimizer_state(policy, optimizer, cfg.checkpoint_path)

    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        # Per-rank loss reflects only one shard of the global batch; mean recovers the loss DDP
        # is actually optimizing. grad_norm and lr are already identical on every rank (post
        # gradient sync / deterministic scheduler) so reducing them would be a no-op collective.
        "loss": AverageMeter("loss", ":.3f", reduction="mean"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        # Report the slowest rank for bottleneck-style timings so multi-GPU runs surface the
        # true straggler instead of rank 0's view.
        "update_s": AverageMeter("updt_s", ":.3f", reduction="max"),
        "dataloading_s": AverageMeter("data_s", ":.3f", reduction="max"),
        # Derived from the post-reduce max step time; set once per log window on the main rank.
        "samples_per_s": AverageMeter("smp/s", ":.0f"),
        "task_variant_applied": AverageMeter("task_var", ":.0f", reduction="sum"),
    }
    if torch.cuda.is_available():
        # max() because headroom is gated by the worst-case rank.
        train_metrics["gpu_mem_gb"] = AverageMeter("mem_gb", ":.2f", reduction="max")

    # Keep global batch size for logging; MetricsTracker handles world size internally.
    effective_batch_size = cfg.batch_size * cfg.gradient_accumulation_steps * accelerator.num_processes
    train_tracker = MetricsTracker(
        cfg.batch_size * cfg.gradient_accumulation_steps,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        progbar = tqdm(
            total=cfg.steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        output_dict = None
        optimizer.zero_grad()
        for accum_idx in range(cfg.gradient_accumulation_steps):
            batch = next(dl_iter)
            for cam_key in dataset.meta.camera_keys:
                if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                    batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
            train_tracker.task_variant_applied = apply_task_variants_to_batch(
                batch,
                task_variants,
                step=step * cfg.gradient_accumulation_steps + accum_idx,
                seed=cfg.seed,
                randomize=cfg.dataset.random_task_variant,
            )
            batch = preprocessor(batch)
            train_tracker.dataloading_s = time.perf_counter() - start_time

            train_tracker, output_dict = update_policy(
                train_tracker,
                policy,
                batch,
                optimizer,
                cfg.optimizer.grad_clip_norm,
                accelerator=accelerator,
                lr_scheduler=lr_scheduler,
                sample_weighter=sample_weighter,
                loss_scale=float(cfg.gradient_accumulation_steps),
                optimizer_step=accum_idx == cfg.gradient_accumulation_steps - 1,
            )
            start_time = time.perf_counter()

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        if is_main_process:
            progbar.update(1)
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_env_eval_step = cfg.env_eval_freq > 0 and step % cfg.env_eval_freq == 0
        is_eval_step = cfg.eval_steps > 0 and eval_dataloader is not None and step % cfg.eval_steps == 0

        if is_log_step:
            # Collective reduce must run on every rank, before the main-process gate below.
            train_tracker.reduce_across_ranks()
            if is_main_process:
                # Cluster-wide throughput, derived from the already-reduced (max) step time so it
                # reflects the slowest rank — which is what actually gates the next iteration.
                step_time = (
                    train_tracker.update_s.avg + train_tracker.dataloading_s.avg
                ) * cfg.gradient_accumulation_steps
                if step_time > 0:
                    train_tracker.samples_per_s = effective_batch_size / step_time
                logging.info(train_tracker)
                if wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    if output_dict:
                        wandb_log_dict.update(output_dict)
                    # Log sample weighting statistics if enabled
                    if sample_weighter is not None:
                        weighter_stats = sample_weighter.get_stats()
                        wandb_log_dict.update({f"sample_weighting/{k}": v for k, v in weighter_stats.items()})
                    wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if is_eval_step:
            policy.eval()
            eval_loss_sum = 0.0
            n_eval_batches = 0
            with torch.no_grad(), accelerator.autocast():
                for eval_batch in eval_dataloader:
                    for cam_key in dataset.meta.camera_keys:
                        if cam_key in eval_batch and eval_batch[cam_key].dtype == torch.uint8:
                            eval_batch[cam_key] = eval_batch[cam_key].to(dtype=torch.float32) / 255.0
                    if cfg.dataset.eval_task_variant:
                        apply_task_variants_to_batch(
                            eval_batch,
                            task_variants,
                            step=step,
                            seed=cfg.seed,
                            randomize=False,
                        )
                    eval_batch = preprocessor(eval_batch)
                    loss, _ = policy.forward(eval_batch)
                    eval_loss_sum += loss.item()
                    n_eval_batches += 1
            eval_loss = eval_loss_sum / max(n_eval_batches, 1)
            eval_loss = torch.tensor(eval_loss, device=device)
            eval_loss = accelerator.reduce(eval_loss, reduction="mean").item()
            policy.train()

            if is_main_process:
                logging.info(f"step {step}: eval_loss={eval_loss:.4f}")
                if wandb_logger:
                    wandb_logger.log_dict(
                        {"eval_loss": eval_loss, "eval_batches": n_eval_batches},
                        step=step,
                        mode="eval",
                    )

        if cfg.save_checkpoint and is_saving_step:
            # Under FSDP, gathering the full model + optimizer state dicts is a cross-rank collective,
            # so all ranks must participate; rank 0 then writes the materialized dicts. For DDP /
            # single-GPU the state dicts are saved the normal way inside save_checkpoint.
            is_fsdp = accelerator.distributed_type == DistributedType.FSDP
            if is_fsdp:
                model_state_dict, optim_state_dict = gather_fsdp_state_dicts(policy, optimizer)
            else:
                model_state_dict, optim_state_dict = None, None
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    num_processes=accelerator.num_processes,
                    batch_size=cfg.batch_size * cfg.gradient_accumulation_steps,
                    model_state_dict=model_state_dict,
                    optim_state_dict=optim_state_dict,
                )
                update_last_checkpoint(checkpoint_dir)
                if cfg.save_checkpoint_to_hub:
                    push_checkpoint_to_hub(
                        checkpoint_dir,
                        cfg.policy.repo_id,
                        private=cfg.policy.private,
                    )
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_env_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if is_main_process:
        progbar.close()

    if eval_env:
        close_envs(eval_env)

    is_fsdp = accelerator.distributed_type == DistributedType.FSDP
    model_state_dict = accelerator.get_state_dict(policy) if is_fsdp else None
    if is_main_process:
        logging.info("End of training")

        if getattr(active_cfg, "push_to_hub", False):
            unwrapped_model = accelerator.unwrap_model(policy)
            # PEFT only applies when training a policy — reward models use the plain path.
            if not cfg.is_reward_model_training and cfg.policy.use_peft:
                unwrapped_model.push_model_to_hub(cfg, peft_model=unwrapped_model, dataset_meta=dataset.meta)
            else:
                unwrapped_model.push_model_to_hub(cfg, state_dict=model_state_dict, dataset_meta=dataset.meta)
            preprocessor.push_to_hub(active_cfg.repo_id)
            postprocessor.push_to_hub(active_cfg.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def _remote_target_in_argv() -> bool:
    """True when the CLI requests a remote HF Jobs run (--job.target=<non-local>)."""
    target = None
    args = sys.argv[1:]
    for i, tok in enumerate(args):
        if tok == "--job.target" and i + 1 < len(args):
            target = args[i + 1]
        elif tok.startswith("--job.target="):
            target = tok.split("=", 1)[1]
    return JobConfig.is_remote_target(target)


def main():
    register_third_party_plugins()
    if _remote_target_in_argv():
        # The policy device is resolved on the remote pod, not here, so silence the
        # client-side "Device '...' is not available" warning PreTrainedConfig emits
        # while parsing the config (it fires before train() can dispatch remotely).
        logging.getLogger("lerobot.configs.policies").setLevel(logging.ERROR)
    train()


if __name__ == "__main__":
    main()
