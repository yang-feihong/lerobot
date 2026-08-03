#!/usr/bin/env bash

set -euo pipefail

# Always resolve relative paths from the repository root.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

# =========================
# User configuration
# =========================

enable_mem="false"

# Learn a chunk-local SE(2) base trajectory instead of discontinuous
# vx/vy/omega commands. The loader also derives task_complete from each episode
# boundary; the checkpoint postprocessor converts predictions back to velocity.
b2_local_trajectory="true"

# GPUs are selected here. Examples:
#   gpu_ids="0"      -> single GPU
#   gpu_ids="0,1,2"  -> 3-GPU DDP via accelerate
gpu_ids="0,1,2"

# Choose exactly one fine-tuning mode:
#   expert = freeze PaliGemma/VLM and full fine-tune only the action expert/projections.
#            Measured peak, batch_size=2/grad_accum=4:
#              non-MEM ≈ 11.0GB; MEM(K=6, full MEM-ViT) ≈ 17.2GB.
#   lora   = full fine-tune the action expert/projections, plus LoRA on the
#            PaliGemma/VLM backbone. Recommended on this RTX 4090 machine.
#            Measured peak, batch_size=2/grad_accum=4:
#              non-MEM ≈ 13.4GB; MEM(K=6, full MEM-ViT) ≈ 19.0GB.
#   full   = full VLA fine-tuning without LoRA.
#            Does not fit on this RTX 4090 with AdamW: batch_size=1 OOM at ≈23.5GB
#            during optimizer-state initialization. Plan for at least 32GB, preferably
#            40GB/48GB or multi-GPU/ZeRO/FSDP/8-bit optimizer.
finetune_mode="lora"

dataset_repo_id="local/b2_z1_vla"
dataset_root="/data/b2_z1_vla_lerobot"
base_policy="/data/checkpoints/lerobot_pi05_base_local_tokenizer"
max_state_dim="46"

steps="25000"
# Training batch semantics are independent of the selected GPU count:
#   global_batch_size = batch_size_per_gpu * number_of_gpus * computed_grad_accum
# Keeping global_batch_size / batch_size_per_gpu a multiple of 24 supports the
# common 1/2/3/4/6/8/12/24-GPU counts without changing the effective batch.
batch_size_per_gpu="2"
global_batch_size="48"
num_workers="4"

# Fraction of episodes held out for periodic validation.
eval_split="0.1"
eval_steps="500"
max_eval_samples="512"

log_freq="10"
save_freq="2000"
wandb_project="b2-z1-vla"

output_root="/data/b2_z1_vla_pi05_outputs"

# Used only when finetune_mode="lora". Action expert/projections are full fine-tuned;
# PaliGemma/VLM backbone uses LoRA. In non-MEM mode, ViT also uses LoRA.
# In MEM mode, MEM-ViT is full fine-tuned instead of using LoRA adapters.
lora_rank="16"
lora_alpha="32"

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

peft_args=()
case "$finetune_mode" in
  lora)
    train_expert_only="false"
    peft_args+=(--peft.method_type=LORA)
    peft_args+=(--peft.r="$lora_rank")
    peft_args+=(--peft.lora_alpha="$lora_alpha")
    ;;
  expert)
    train_expert_only="true"
    ;;
  full)
    train_expert_only="false"
    ;;
  *)
    echo "Unknown finetune_mode=$finetune_mode. Expected one of: lora, expert, full." >&2
    exit 1
    ;;
esac

mkdir -p "$log_dir" "$output_root"

if [[ -z "$gpu_ids" ]]; then
  echo "gpu_ids must not be empty." >&2
  exit 1
fi

IFS=',' read -r -a gpu_id_array <<<"$gpu_ids"
num_gpus="${#gpu_id_array[@]}"
for gpu_id in "${gpu_id_array[@]}"; do
  if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
    echo "Invalid gpu_ids=$gpu_ids. Use a comma-separated list like 0,1,2." >&2
    exit 1
  fi
done

if [[ ! "$batch_size_per_gpu" =~ ^[1-9][0-9]*$ ]]; then
  echo "batch_size_per_gpu must be a positive integer, got $batch_size_per_gpu." >&2
  exit 1
fi
if [[ ! "$global_batch_size" =~ ^[1-9][0-9]*$ ]]; then
  echo "global_batch_size must be a positive integer, got $global_batch_size." >&2
  exit 1
fi

batch_per_micro_step=$((batch_size_per_gpu * num_gpus))
if (( global_batch_size % batch_per_micro_step != 0 )); then
  echo "global_batch_size=$global_batch_size must be divisible by " \
    "batch_size_per_gpu=$batch_size_per_gpu x num_gpus=$num_gpus " \
    "(micro-step global batch=$batch_per_micro_step)." >&2
  echo "Choose a global_batch_size / batch_size_per_gpu ratio divisible by $num_gpus; " \
    "a multiple of 24 supports common GPU counts." >&2
  exit 1
fi
gradient_accumulation_steps=$((global_batch_size / batch_per_micro_step))

train_args=(
  --dataset.repo_id="$dataset_repo_id" \
  --dataset.root="$dataset_root" \
  --dataset.eval_split="$eval_split" \
  --policy.path="$base_policy" \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.max_state_dim="$max_state_dim" \
  --policy.b2_local_trajectory_enabled="$b2_local_trajectory" \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.train_expert_only="$train_expert_only" \
  "${policy_mem_args[@]}" \
  --policy.push_to_hub=false \
  "${peft_args[@]}" \
  --output_dir="$output_dir" \
  --job_name="${job_prefix}_${timestamp}" \
  --steps="$steps" \
  --batch_size="$batch_size_per_gpu" \
  --gradient_accumulation_steps="$gradient_accumulation_steps" \
  --num_workers="$num_workers" \
  --log_freq="$log_freq" \
  --eval_steps="$eval_steps" \
  --max_eval_samples="$max_eval_samples" \
  --env_eval_freq=0 \
  --save_checkpoint=true \
  --save_freq="$save_freq" \
  --wandb.enable=true \
  --wandb.project="$wandb_project"
)

if (( num_gpus > 1 )); then
  setsid env CUDA_VISIBLE_DEVICES="$gpu_ids" uv run accelerate launch \
    --multi_gpu \
    --num_processes "$num_gpus" \
    --num_machines 1 \
    --gpu_ids "$gpu_ids" \
    --mixed_precision bf16 \
    --dynamo_backend no \
    -m lerobot.scripts.lerobot_train \
    "${train_args[@]}" \
    >"$log_file" 2>&1 </dev/null &
else
  setsid env CUDA_VISIBLE_DEVICES="$gpu_ids" uv run python -u -m lerobot.scripts.lerobot_train \
    "${train_args[@]}" \
    >"$log_file" 2>&1 </dev/null &
fi

pid=$!
echo "$pid" >"$pid_file"

echo "VLA training started"
echo "PID:              $pid"
echo "MEM:              $enable_mem"
echo "B2 trajectory:    $b2_local_trajectory"
echo "Finetune mode:    $finetune_mode"
echo "Train expert only: $train_expert_only"
echo "GPUs:             $gpu_ids ($num_gpus process(es))"
echo "Dataset:          $dataset_root"
echo "Train/eval split: eval_split=$eval_split"
echo "Steps:            $steps optimizer updates"
echo "Batch per GPU:    $batch_size_per_gpu"
echo "Gradient accum:   $gradient_accumulation_steps (computed)"
echo "Global batch:     $batch_size_per_gpu x $num_gpus GPU(s) x $gradient_accumulation_steps = $global_batch_size"
echo "Log:              $log_file"
echo "Output:           $output_dir"
echo "Watch:            tail -f '$log_file'"
