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

# full | approach | handle_press | door_traversal | custom
eval_stage="full"
dataset_repo_id=""
dataset_root=""
dataset_repo_id_overridden="false"
dataset_root_overridden="false"

eval_split="0.1"

# Empty means that the evaluator selects representative episodes from the
# metadata-defined train/eval split. Explicit episode ranges still override it.
train_episodes=""
eval_episodes=""

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
eval_splits="both"
print_config="false"

output_root=""
output_root_overridden="false"

while (( $# > 0 )); do
  case "$1" in
    --policy-path=*) policy_path="${1#*=}" ;;
    --stage=*) eval_stage="${1#*=}" ;;
    --dataset-repo-id=*) dataset_repo_id="${1#*=}"; dataset_repo_id_overridden="true" ;;
    --dataset-root=*) dataset_root="${1#*=}"; dataset_root_overridden="true" ;;
    --output-root=*) output_root="${1#*=}"; output_root_overridden="true" ;;
    --gpu-id=*) gpu_id="${1#*=}" ;;
    --device=*) device="${1#*=}" ;;
    --train-episodes=*) train_episodes="${1#*=}" ;;
    --eval-episodes=*) eval_episodes="${1#*=}" ;;
    --max-episodes=*) max_episodes="${1#*=}" ;;
    --max-frames-per-episode=*) max_frames_per_episode="${1#*=}" ;;
    --frame-stride=*) frame_stride="${1#*=}" ;;
    --include-onset-windows=*) include_onset_windows="${1#*=}" ;;
    --splits=*) eval_splits="${1#*=}" ;;
    --print-config) print_config="true" ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

case "$eval_stage" in
  full)
    stage_dataset_repo_id="local/b2_z1_vla_staff1_command_clean"
    stage_dataset_root="/data/b2_z1_vla_lerobot_staff1_command_clean"
    ;;
  approach)
    stage_dataset_repo_id="local/b2_z1_vla_staff1_command_clean_b2_approach"
    stage_dataset_root="/data/b2_z1_vla_lerobot_staff1_command_clean_b2_approach"
    ;;
  handle_press)
    stage_dataset_repo_id="local/b2_z1_vla_stage2_handle_press_with_randomized_pose"
    stage_dataset_root="/data/b2_z1_vla_lerobot_stage2_handle_press_with_randomized_pose"
    ;;
  door_traversal)
    stage_dataset_repo_id="local/b2_z1_vla_staff1_command_clean_door_traversal"
    stage_dataset_root="/data/b2_z1_vla_lerobot_staff1_command_clean_door_traversal"
    ;;
  custom)
    stage_dataset_repo_id=""
    stage_dataset_root=""
    ;;
  *)
    echo "--stage must be full, approach, handle_press, door_traversal, or custom" >&2
    exit 2
    ;;
esac

if [[ "$dataset_repo_id_overridden" == false ]]; then
  dataset_repo_id="$stage_dataset_repo_id"
fi
if [[ "$dataset_root_overridden" == false ]]; then
  dataset_root="$stage_dataset_root"
fi
if [[ -z "$dataset_repo_id" || -z "$dataset_root" ]]; then
  echo "The custom stage requires --dataset-repo-id and --dataset-root." >&2
  exit 2
fi
if [[ "$output_root_overridden" == false ]]; then
  policy_label="$(basename "${policy_path%/pretrained_model}")"
  output_root="/data/b2_z1_vla_openloop_eval/${eval_stage}_${policy_label}"
fi

case "$eval_splits" in
  both|train|eval) ;;
  *) echo "--splits must be both, train, or eval" >&2; exit 2 ;;
esac

if [[ "$print_config" == true ]]; then
  printf 'stage=%s\npolicy_path=%s\ndataset_repo_id=%s\ndataset_root=%s\noutput_root=%s\nsplits=%s\n' \
    "$eval_stage" "$policy_path" "$dataset_repo_id" "$dataset_root" "$output_root" "$eval_splits"
  exit 0
fi

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

if [[ "$eval_splits" == both || "$eval_splits" == train ]]; then
  run_openloop train "$train_episodes"
fi
if [[ "$eval_splits" == both || "$eval_splits" == eval ]]; then
  run_openloop eval "$eval_episodes"
fi
