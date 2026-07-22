#!/usr/bin/env bash

set -euo pipefail

# Always resolve relative paths from the repository root.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

# =========================
# User configuration
# =========================

enable_mem="false"

dataset_repo_id="local/b2_z1_vla"
dataset_root="/data/b2_z1_vla_lerobot_0721"
base_policy="/data/checkpoints/lerobot_pi05_base_local_tokenizer"

steps="10000"
batch_size="2"
gradient_accumulation_steps="4"
num_workers="4"

# With 29 episodes and a single task, eval_split=0.1 holds out ceil(29*0.1)=3
# episodes for validation and leaves 26 episodes for training.
eval_split="0.1"
eval_steps="500"
max_eval_samples="512"

log_freq="10"
save_freq="2000"
wandb_project="b2-z1-vla"

output_root="/data/b2_z1_vla_pi05_outputs"

# MEM-only configuration. Used only when enable_mem="true".
mem_vit_checkpoint="/data/mem_vit_distill_outputs/mem_vit_distill_20260716_142702/mem_vit_distill_latest.pt"

# Choose one MEM window mode when enable_mem="true".
mem_fixed_num_frames="6"
mem_random_min_num_frames=""
mem_random_max_num_frames=""

# =========================
# Launch
# =========================

timestamp="$(date +%Y%m%d_%H%M%S)"
log_dir="$repo_root/logs"

job_prefix="pi05_b2_z1_vla"
if [[ "$enable_mem" == "true" ]]; then
  job_prefix="mem_pi05_b2_z1_vla"
  output_root="/data/b2_z1_vla_mem_outputs"
  wandb_project="b2-z1-mem-vla"
fi

output_dir="$output_root/${job_prefix}_${timestamp}"
log_file="$log_dir/${job_prefix}_${timestamp}.log"
pid_file="$log_dir/${job_prefix}_${timestamp}.pid"

policy_mem_args=()
if [[ "$enable_mem" == "true" ]]; then
  if [[ ! -f "$mem_vit_checkpoint" ]]; then
    echo "MEM ViT checkpoint not found: $mem_vit_checkpoint" >&2
    exit 1
  fi
  policy_mem_args+=(--policy.mem_vit_checkpoint="$mem_vit_checkpoint")
  if [[ -n "$mem_random_min_num_frames" || -n "$mem_random_max_num_frames" ]]; then
    if [[ -z "$mem_random_min_num_frames" || -z "$mem_random_max_num_frames" ]]; then
      echo "mem_random_min_num_frames and mem_random_max_num_frames must be set together." >&2
      exit 1
    fi
    policy_mem_args+=(--policy.mem_vit_min_num_frames="$mem_random_min_num_frames")
    policy_mem_args+=(--policy.mem_vit_max_num_frames="$mem_random_max_num_frames")
  else
    policy_mem_args+=(--policy.mem_vit_num_frames="$mem_fixed_num_frames")
  fi
fi

mkdir -p "$log_dir" "$output_root"

setsid uv run python -u -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="$dataset_repo_id" \
  --dataset.root="$dataset_root" \
  --dataset.eval_split="$eval_split" \
  --policy.path="$base_policy" \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.max_state_dim=40 \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.train_expert_only=true \
  "${policy_mem_args[@]}" \
  --policy.push_to_hub=false \
  --output_dir="$output_dir" \
  --job_name="${job_prefix}_${timestamp}" \
  --steps="$steps" \
  --batch_size="$batch_size" \
  --gradient_accumulation_steps="$gradient_accumulation_steps" \
  --num_workers="$num_workers" \
  --log_freq="$log_freq" \
  --eval_steps="$eval_steps" \
  --max_eval_samples="$max_eval_samples" \
  --env_eval_freq=0 \
  --save_checkpoint=true \
  --save_freq="$save_freq" \
  --wandb.enable=true \
  --wandb.project="$wandb_project" \
  >"$log_file" 2>&1 </dev/null &

pid=$!
echo "$pid" >"$pid_file"

echo "VLA training started"
echo "PID:              $pid"
echo "MEM:              $enable_mem"
echo "Dataset:          $dataset_root"
echo "Train/eval split: 26 train episodes / 3 eval episodes (eval_split=$eval_split)"
echo "Steps:            $steps optimizer updates"
echo "Batch:            $batch_size x $gradient_accumulation_steps accumulation"
echo "Log:              $log_file"
echo "Output:           $output_dir"
echo "Watch:            tail -f '$log_file'"
