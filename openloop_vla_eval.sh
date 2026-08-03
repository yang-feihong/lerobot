#!/usr/bin/env bash

set -euo pipefail

# Always resolve relative paths from the repository root.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

# =========================
# User configuration
# =========================

# Can be:
#   1) a full training run dir containing checkpoints/
#   2) a checkpoint step dir, e.g. .../checkpoints/012000
#   3) a pretrained_model dir, e.g. .../checkpoints/012000/pretrained_model
# If this is a run dir, the latest numeric checkpoint is used automatically.
policy_path="/data/b2_z1_vla_pi05_outputs/pi05_b2_z1_vla_20260731_152045/checkpoints/025000"

dataset_repo_id="local/b2_z1_vla"
dataset_root="/data/b2_z1_vla_lerobot"

eval_split="0.1"

# Both sets are evaluated sequentially. Explicit episode selections are run in
# full and replace their corresponding local output directories.
train_episodes="0-3"
eval_episodes="629-632"

# Number of complete episodes to evaluate. Set 0 for all selected episodes.
max_episodes="4"
# Set 0 to evaluate every frame through the end of each selected episode.
max_frames_per_episode="0"
# Set 0 to save one independent 50-step action-chunk plot for every inference frame.
max_chunk_plots_per_episode="0"

# Use a single idle GPU if available. Use "cpu" only for tiny sanity checks.
device="cuda"
gpu_id="0"

batch_size="2"
num_workers="2"
plot_workers="8"

output_root="/data/b2_z1_vla_openloop_eval"

# =========================
# Launch
# =========================

run_openloop() {
  local split="$1"
  local episodes="$2"
  local output_dir="$output_root/new_action_025000_${split}"

  # Each directory represents one complete run. Remove stale partial plots and
  # metrics before writing the replacement.
  rm -rf "$output_dir"

  local -a cmd=(
    uv run python -m lerobot.scripts.openloop_vla_eval
    --policy-path "$policy_path"
    --dataset-repo-id "$dataset_repo_id"
    --dataset-root "$dataset_root"
    --output-dir "$output_dir"
    --split "$split"
    --eval-split "$eval_split"
    --episodes "$episodes"
    --max-episodes "$max_episodes"
    --max-frames-per-episode "$max_frames_per_episode"
    --max-chunk-plots-per-episode "$max_chunk_plots_per_episode"
    --batch-size "$batch_size"
    --num-workers "$num_workers"
    --device "$device"
    --task-variant first
    --plot-workers "$plot_workers"
  )

  if [[ "$device" == cuda* ]]; then
    CUDA_VISIBLE_DEVICES="$gpu_id" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi

  echo "Open-loop $split output: $output_dir"
}

run_openloop train "$train_episodes"
run_openloop eval "$eval_episodes"
