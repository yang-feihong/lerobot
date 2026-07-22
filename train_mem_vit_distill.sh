#!/usr/bin/env bash

set -euo pipefail

# Always resolve relative paths from the repository root.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

timestamp="$(date +%Y%m%d_%H%M%S)"
gpu_devices="0,1,2"
batch_size="4"
gradient_accumulation_steps="4"
learning_rate="1e-6"
max_steps="25000"
IFS=',' read -r -a gpu_array <<< "$gpu_devices"
nproc_per_node="${#gpu_array[@]}"

log_dir="$repo_root/logs"
output_root="/data/mem_vit_distill_outputs"
log_file="$log_dir/train_mem_vit_distill_${timestamp}.log"
pid_file="$log_dir/train_mem_vit_distill_${timestamp}.pid"
resume_from_checkpoint="${1:-}"
resume_args=()
wandb_name_args=(--wandb-run-name "vlnce-${timestamp}")

if [[ -n "$resume_from_checkpoint" ]]; then
  if [[ -d "$resume_from_checkpoint" ]]; then
    resume_from_checkpoint="$resume_from_checkpoint/mem_vit_distill_latest.pt"
  fi
  if [[ ! -f "$resume_from_checkpoint" ]]; then
    echo "Resume checkpoint not found: $resume_from_checkpoint" >&2
    exit 1
  fi
  resume_from_checkpoint="$(cd "$(dirname "$resume_from_checkpoint")" && pwd)/$(basename "$resume_from_checkpoint")"
  output_dir="$(dirname "$resume_from_checkpoint")"
  resume_args=(--resume-from-checkpoint "$resume_from_checkpoint")
  wandb_name_args=()
else
  output_dir="$output_root/mem_vit_distill_${timestamp}"
fi

mkdir -p "$log_dir" "$output_dir"

setsid env \
  PYTHONUNBUFFERED=1 \
  HF_HUB_OFFLINE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="$gpu_devices" \
  uv run torchrun \
    --standalone \
    --nproc-per-node="$nproc_per_node" \
    -m lerobot.scripts.train_mem_vit_distill \
    --pretrained-path /data/checkpoints/lerobot_pi05_base \
    --local-files-only \
    --dataset-repo-id local/vlnce_smooth_memvit \
    --dataset-root /data/VLNCE_smooth_lerobot_final \
    --test-dataset-repo-id local/vlnce_smooth_memvit \
    --test-dataset-root /data/mem_vit_drone_test \
    --image-key observation.images.rgb \
    --output-dir "$output_dir" \
    "${resume_args[@]}" \
    --num-frames 6 \
    --min-gap-frames 1 \
    --max-gap-frames 5 \
    --gap-sampling uniform_single \
    --batch-size "$batch_size" \
    --gradient-accumulation-steps "$gradient_accumulation_steps" \
    --max-steps "$max_steps" \
    --num-workers 0 \
    --lr "$learning_rate" \
    --eval-every 50 \
    --eval-batches 50 \
    --test-eval-batches 50 \
    --log-every 10 \
    --save-every 500 \
    --keep-last-checkpoints 2 \
    --wandb-enable \
    --wandb-project mem-vit-distill \
    "${wandb_name_args[@]}" \
  >"$log_file" 2>&1 </dev/null &

pid=$!
echo "$pid" >"$pid_file"

echo "Training started"
echo "PID:    $pid"
echo "GPUs:   $gpu_devices"
echo "Batch:  $batch_size x $nproc_per_node GPUs x $gradient_accumulation_steps accumulation"
echo "LR:     $learning_rate"
echo "Log:    $log_file"
echo "Output: $output_dir"
if [[ -n "$resume_from_checkpoint" ]]; then
  echo "Resume: $resume_from_checkpoint"
fi
echo "Watch:  tail -f '$log_file'"
