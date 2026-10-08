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
import logging
import os
import re
from glob import glob
from pathlib import Path

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from termcolor import colored

from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.constants import PRETRAINED_MODEL_DIR


def cfg_to_group(
    cfg: TrainPipelineConfig, return_list: bool = False, truncate_tags: bool = False, max_tag_length: int = 64
) -> list[str] | str:
    """Return a group name for logging. Optionally returns group name as list."""

    def _maybe_truncate(tag: str) -> str:
        """Truncate tag to max_tag_length characters if required.

        wandb rejects tags longer than 64 characters.
        See: https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py
        """
        if len(tag) <= max_tag_length:
            return tag
        return tag[:max_tag_length]

    if cfg.is_reward_model_training:
        trainable_tag = f"reward_model:{cfg.reward_model.type}"
    else:
        trainable_tag = f"policy:{cfg.policy.type}"
    lst = [
        trainable_tag,
        f"seed:{cfg.seed}",
    ]
    if cfg.dataset is not None:
        lst.append(f"dataset:{cfg.dataset.repo_id}")
    if cfg.env is not None:
        lst.append(f"env:{cfg.env.type}")
    if truncate_tags:
        lst = [_maybe_truncate(tag) for tag in lst]
    return lst if return_list else "-".join(lst)


_SEMANTIC_PHASE_STAGE_TAGS = {
    "approach": "stage1",
    "handle_press": "stage2",
    "traversal": "stage3",
}


def cfg_to_wandb_tags(cfg: TrainPipelineConfig, max_tag_length: int = 64) -> list[str]:
    """Build stable tags for filtering runs with genuinely comparable semantics."""
    tags = list(cfg_to_group(cfg, return_list=True))
    if cfg.dataset is not None:
        phases = list(cfg.dataset.semantic_phases)
        stages_present = {
            _SEMANTIC_PHASE_STAGE_TAGS[phase]
            for phase in phases
            if phase in _SEMANTIC_PHASE_STAGE_TAGS
        }
        stages = [stage for stage in ("stage1", "stage2", "stage3") if stage in stages_present]
        tags.extend(f"phase:{phase}" for phase in phases)
        tags.append(f"phase-set:{'+'.join(sorted(phases))}" if phases else "phase-set:unspecified")
        tags.extend(f"stage:{stage}" for stage in stages)
        tags.append(f"stage-set:{'+'.join(stages)}" if stages else "stage-set:unspecified")
        tags.append(f"image-source:{cfg.dataset.image_source}")
        if cfg.dataset.semantic_state_instruction_matrix:
            tags.append("semantic-conditioning:state-instruction-matrix")
        elif cfg.dataset.semantic_view_kind_weights:
            tags.append("semantic-conditioning:legacy-view-kinds")
        else:
            tags.append("semantic-conditioning:none")

    if not cfg.is_reward_model_training:
        policy = cfg.policy
        tags.extend(
            [
                f"memory:{'mem-vit' if getattr(policy, 'mem_vit_enabled', False) else 'standard'}",
                f"state-encoding:{getattr(policy, 'state_action_encoding', 'none')}",
                f"b2-output:{getattr(policy, 'b2_action_representation', 'unspecified')}",
                f"z1-output:{getattr(policy, 'z1_action_representation', 'unspecified')}",
                f"arm-mode-output:{str(bool(getattr(policy, 'action_predict_arm_teleop_inactive', False) and getattr(policy, 'action_predict_arm_reset', False))).lower()}",
                f"task-complete-output:{str(bool(getattr(policy, 'action_predict_task_complete', False))).lower()}",
                f"task-blocked-output:{str(bool(getattr(policy, 'action_predict_task_blocked', False))).lower()}",
            ]
        )
    tags.extend(cfg.wandb.tags)

    deduplicated = []
    seen = set()
    for tag in tags:
        tag = tag[:max_tag_length]
        if tag not in seen:
            seen.add(tag)
            deduplicated.append(tag)
    return deduplicated


def get_wandb_run_id_from_filesystem(log_dir: Path) -> str:
    # Get the WandB run ID.
    paths = glob(str(log_dir / "wandb/latest-run/run-*"))
    if len(paths) != 1:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    match = re.search(r"run-([^\.]+).wandb", paths[0].split("/")[-1])
    if match is None:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    wandb_run_id = match.groups(0)[0]
    return wandb_run_id


def get_safe_wandb_artifact_name(name: str):
    """WandB artifacts don't accept ":" or "/" in their name."""
    return name.replace(":", "_").replace("/", "_")


class WandBLogger:
    """A helper class to log object using wandb."""

    def __init__(self, cfg: TrainPipelineConfig):
        self.cfg = cfg.wandb
        self.log_dir = cfg.output_dir
        self.job_name = cfg.job_name
        self.env_fps = cfg.env.fps if cfg.env else None
        self._group = cfg_to_group(cfg)

        # Set up WandB.
        os.environ["WANDB_SILENT"] = "True"
        import wandb

        resume_training_run = cfg.resume and cfg.wandb.resume_training_run
        wandb_run_id = (
            cfg.wandb.run_id
            if resume_training_run and cfg.wandb.run_id
            else get_wandb_run_id_from_filesystem(self.log_dir)
            if resume_training_run
            else None
        )
        automatic_tags = cfg_to_wandb_tags(cfg) if self.cfg.add_tags else None
        wandb.init(
            id=wandb_run_id,
            project=self.cfg.project,
            entity=self.cfg.entity,
            name=self.job_name,
            notes=self.cfg.notes,
            tags=automatic_tags,
            dir=self.log_dir,
            config=cfg.to_dict(),
            # TODO(rcadene): try set to True
            save_code=False,
            # TODO(rcadene): split train and eval, and run async eval with job_type="eval"
            job_type="train_eval",
            resume="must" if resume_training_run else None,
            mode=self.cfg.mode if self.cfg.mode in ["online", "offline", "disabled"] else "online",
            settings=wandb.Settings(
                console="off",
                disable_code=True,
                disable_git=True,
                disable_job_creation=True,
                save_code=False,
                x_save_requirements=False,
            ),
        )
        if automatic_tags is not None:
            # W&B may retain the old tag set when resuming an existing run.
            # Preserve tags edited in the UI while ensuring current semantic
            # comparability tags are present on the resumed run as well.
            wandb.run.tags = tuple(dict.fromkeys([*(wandb.run.tags or ()), *automatic_tags]))
        run_id = wandb.run.id
        # NOTE: We will override the cfg.wandb.run_id with the wandb run id.
        # This is because we want to be able to resume the run from the wandb run id.
        cfg.wandb.run_id = run_id
        # Handle custom step key for rl asynchronous training.
        self._wandb_custom_step_key: set[str] | None = None
        self._wandb = wandb
        self._define_default_metrics()
        logging.info(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
        logging.info("W&B tags: %s", list(wandb.run.tags or ()))
        logging.info(f"Track this run --> {colored(wandb.run.get_url(), 'yellow', attrs=['bold'])}")

    def _define_default_metrics(self) -> None:
        """Use explicit train/eval step axes so charts do not depend on wandb's internal _step."""
        self._wandb.define_metric("train/step", hidden=True)
        self._wandb.define_metric("train/*", step_metric="train/step")
        self._wandb.define_metric("eval/step", hidden=True)
        self._wandb.define_metric("eval/*", step_metric="eval/step")
        self._wandb.define_metric("optimizer_step", hidden=True)
        for group in (
            "overview",
            "training_progress",
            "optimization",
            "performance",
            "continuous_action",
            "discrete_action",
            "action_dimensions",
            "memory",
            "data",
            "sample_weighting",
            "policy_diagnostics",
        ):
            self._wandb.define_metric(f"{group}/*", step_metric="optimizer_step")
        self._wandb.define_metric("overview/train_loss", step_metric="optimizer_step", summary="min")
        self._wandb.define_metric("overview/val_loss", step_metric="optimizer_step", summary="min")

    def update_config(self, d: dict, *, allow_val_change: bool = True) -> None:
        """Add runtime-derived metadata that is not known when wandb.init receives the CLI config."""
        self._wandb.config.update(d, allow_val_change=allow_val_change)

    def log_policy(self, checkpoint_dir: Path):
        """Checkpoints the policy to wandb."""
        if self.cfg.disable_artifact:
            return

        step_id = checkpoint_dir.name
        artifact_name = f"{self._group}-{step_id}"
        artifact_name = get_safe_wandb_artifact_name(artifact_name)
        artifact = self._wandb.Artifact(artifact_name, type="model")
        pretrained_model_dir = checkpoint_dir / PRETRAINED_MODEL_DIR

        # Check if this is a PEFT model (has adapter files instead of model.safetensors)
        adapter_model_file = pretrained_model_dir / "adapter_model.safetensors"
        standard_model_file = pretrained_model_dir / SAFETENSORS_SINGLE_FILE

        if adapter_model_file.exists():
            # PEFT model: add adapter files and configs
            artifact.add_file(adapter_model_file)
            adapter_config_file = pretrained_model_dir / "adapter_config.json"
            if adapter_config_file.exists():
                artifact.add_file(adapter_config_file)
            # Also add the policy config which is needed for loading
            config_file = pretrained_model_dir / "config.json"
            if config_file.exists():
                artifact.add_file(config_file)
        elif standard_model_file.exists():
            # Standard model: add the single safetensors file
            artifact.add_file(standard_model_file)
        else:
            logging.warning(
                f"No {SAFETENSORS_SINGLE_FILE} or adapter_model.safetensors found in {pretrained_model_dir}. "
                "Skipping model artifact upload to WandB."
            )
            return

        self._wandb.log_artifact(artifact)

    def log_dict(
        self, d: dict, step: int | None = None, mode: str = "train", custom_step_key: str | None = None
    ):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided.")

        # NOTE: This is not simple. Wandb step must always monotonically increase and it
        # increases with each wandb.log call, but in the case of asynchronous RL for example,
        # multiple time steps is possible. For example, the interaction step with the environment,
        # the training step, the evaluation step, etc. So we need to define a custom step key
        # to log the correct step for each metric.
        if custom_step_key is not None:
            if self._wandb_custom_step_key is None:
                self._wandb_custom_step_key = set()
            new_custom_key = f"{mode}/{custom_step_key}"
            if new_custom_key not in self._wandb_custom_step_key:
                self._wandb_custom_step_key.add(new_custom_key)
                self._wandb.define_metric(new_custom_key, hidden=True)

        batch_data = {}
        for k, v in d.items():
            # Skip the custom step key here, it's added to the batch below.
            if custom_step_key is not None and k == custom_step_key:
                continue

            if not isinstance(v, (int | float | str)):
                if isinstance(v, (list | tuple)) and all(isinstance(item, (int | float)) for item in v):
                    for idx, item in enumerate(v):
                        batch_data[f"{mode}/{k}/{idx}"] = item
                    continue
                logging.warning(
                    f'WandB logging of key "{k}" was ignored as its type "{type(v)}" is not handled by this wrapper.'
                )
                continue

            batch_data[f"{mode}/{k}"] = v

        if batch_data:
            if custom_step_key is not None:
                batch_data[f"{mode}/{custom_step_key}"] = d[custom_step_key]
                self._wandb.log(batch_data)
            else:
                batch_data[f"{mode}/step"] = step
                self._wandb.log(data=batch_data, step=step)

    def log_grouped_dict(self, d: dict[str, int | float], step: int) -> None:
        """Log pre-grouped metric paths on a shared optimizer-step axis."""
        batch_data = {key: value for key, value in d.items() if isinstance(value, int | float)}
        if batch_data:
            batch_data["optimizer_step"] = step
            self._wandb.log(data=batch_data, step=step)

    def log_video(self, video_path: str, step: int, mode: str = "train"):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        wandb_video = self._wandb.Video(video_path, fps=self.env_fps, format="mp4")
        self._wandb.log({f"{mode}/video": wandb_video}, step=step)
