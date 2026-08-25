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
policy_path="/data/b2_z1_vla_pi05_outputs/pi05_b2_z1_vla_20260806_131138/checkpoints/004500"

dataset_repo_id="local/b2_z1_vla"
dataset_root="/data/b2_z1_vla_lerobot"

eval_split="0.1"

# This checkpoint was trained with the first 327 episodes: 0-293 train and
# 294-326 validation. Evaluate fixed representatives from those exact splits,
# even if more episodes are appended to dataset_root later.
train_episodes="0-3"
eval_episodes="294-297"

# Number of complete episodes to evaluate. Set 0 for all selected episodes.
max_episodes="4"
# Set 0 to evaluate every frame through the end of each selected episode.
max_frames_per_episode="0"
# Run one inference every N source frames.
frame_stride="10"
# Set 0 for no maximum; chunk_plot_stride below still controls sampling.
max_chunk_plots_per_episode="0"
# Save one independent 50-step plot every N inferences (5 × 10 = 50 source frames).
chunk_plot_stride="5"

# Use a single idle GPU if available. Use "cpu" only for tiny sanity checks.
device="cuda"
gpu_id="0"

batch_size="2"
num_workers="2"
plot_workers="8"
include_onset_windows="true"

output_root="/data/b2_z1_vla_openloop_eval/pi05_b2_z1_vla_20260806_131138_004500"

while (( $# > 0 )); do
  case "$1" in
    --policy-path=*) policy_path="${1#*=}" ;;
    --dataset-repo-id=*) dataset_repo_id="${1#*=}" ;;
    --dataset-root=*) dataset_root="${1#*=}" ;;
    --output-root=*) output_root="${1#*=}" ;;
    --gpu-id=*) gpu_id="${1#*=}" ;;
    --device=*) device="${1#*=}" ;;
    --train-episodes=*) train_episodes="${1#*=}" ;;
    --eval-episodes=*) eval_episodes="${1#*=}" ;;
    --max-episodes=*) max_episodes="${1#*=}" ;;
    --max-frames-per-episode=*) max_frames_per_episode="${1#*=}" ;;
    --frame-stride=*) frame_stride="${1#*=}" ;;
    --include-onset-windows=*) include_onset_windows="${1#*=}" ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

# =========================
# Launch
# =========================

run_openloop() {
  local split="$1"
  local episodes="$2"
  local output_dir="$output_root/$split"

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
    --frame-stride "$frame_stride"
    --max-chunk-plots-per-episode "$max_chunk_plots_per_episode"
    --chunk-plot-stride "$chunk_plot_stride"
    --batch-size "$batch_size"
    --num-workers "$num_workers"
    --device "$device"
    --task-variant first
    --plot-workers "$plot_workers"
  )
  if [[ "$include_onset_windows" == "false" ]]; then
    cmd+=(--no-include-onset-windows)
  elif [[ "$include_onset_windows" != "true" ]]; then
    echo "--include-onset-windows must be true or false" >&2
    exit 2
  fi

  if [[ "$device" == cuda* ]]; then
    CUDA_VISIBLE_DEVICES="$gpu_id" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi

  echo "Open-loop $split output: $output_dir"
}

run_openloop train "$train_episodes"
run_openloop eval "$eval_episodes"
