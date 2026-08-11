#!/usr/bin/env bash

set -euo pipefail

# Always resolve relative paths from the repository root.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

# =========================
# User configuration
# =========================

enable_mem="false"

# Physical I/O schema. These values are persisted in every checkpoint and are
# automatically restored by resume and open-loop evaluation.
state_use_arm_joint_positions="true"
state_use_arm_joint_velocities="false"
state_use_arm_gripper_feedback="true"
state_use_b2_joint_positions="false"
state_use_b2_joint_velocities="false"
state_use_b2_trunk_pose="true"
state_use_b2_linear_velocity="false"
state_use_b2_angular_velocity="false"
b2_action_representation="local_trajectory" # "velocity" or "local_trajectory"
z1_action_representation="ee_pose" # "ee_pose" or "ee_delta"
predict_arm_teleop_inactive="true"
predict_arm_reset="true"
predict_ee_pose="true"
predict_gripper="true"
predict_task_complete="true"
task_complete_sample_tail_seconds="2.0"

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
max_state_dim="32"

steps="20000"
# Temporal semantics for the planned 50 Hz dataset. A chunk covers one second,
# while deployment requests a fresh observation/chunk after executing 0.5 s.
# A dataset at another FPS is timestamp-resampled to this model frequency;
# lower-frequency data is allowed with a strong warning because frames repeat.
action_chunk_size="50"
action_steps_to_execute="25"
control_frequency_hz="50"
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
save_freq="500"
wandb_project="b2-z1-vla"

output_root="/data/b2_z1_vla_pi05_outputs"

# Leave empty for a new run. To resume in place, point this at a complete
# numeric checkpoint directory or its `last` symlink. The checkpoint's saved
# optimizer, scheduler, RNG, data-order and W&B run state are restored.
resume_checkpoint=""

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
# MEM image sampling interval in seconds.
mem_frame_interval_seconds="0.5"
# "text": current state in the prompt; "continuous": linear state-history tokens.
state_action_encoding="continuous"
# With 50 Hz data, 13 samples at 0.04 s intervals cover 0.48 s.
state_num_frames="13"
state_history_frame_interval_seconds="0.04"
# Uses the state-history clock and excludes the current action.
action_history_enabled="true"

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

resume_args=()
policy_source_args=(--policy.path="$base_policy")
if [[ -n "$resume_checkpoint" ]]; then
  resume_checkpoint="$(readlink -f "$resume_checkpoint")"
  resume_config="$resume_checkpoint/pretrained_model/train_config.json"
  if [[ ! -f "$resume_config" ]]; then
    echo "Resume train config not found: $resume_config" >&2
    exit 1
  fi
  output_dir="$(dirname "$(dirname "$resume_checkpoint")")"
  job_name="$(basename "$output_dir")"
  resume_args+=(--config_path="$resume_config" --resume=true)
  policy_source_args=()
  log_file="$log_dir/${job_name}_resume_${timestamp}.log"
  pid_file="$log_dir/${job_name}_resume_${timestamp}.pid"
else
  output_dir="$output_root/${job_prefix}_${timestamp}"
  job_name="${job_prefix}_${timestamp}"
  log_file="$log_dir/${job_name}.log"
  pid_file="$log_dir/${job_name}.pid"
fi

policy_io_args=()
if [[ -z "$resume_checkpoint" ]]; then
  policy_io_args+=(
    --policy.max_state_dim="$max_state_dim"
    --policy.state_use_arm_joint_positions="$state_use_arm_joint_positions"
    --policy.state_use_arm_joint_velocities="$state_use_arm_joint_velocities"
    --policy.state_use_arm_gripper_feedback="$state_use_arm_gripper_feedback"
    --policy.state_use_b2_joint_positions="$state_use_b2_joint_positions"
    --policy.state_use_b2_joint_velocities="$state_use_b2_joint_velocities"
    --policy.state_use_b2_trunk_pose="$state_use_b2_trunk_pose"
    --policy.state_use_b2_linear_velocity="$state_use_b2_linear_velocity"
    --policy.state_use_b2_angular_velocity="$state_use_b2_angular_velocity"
    --policy.b2_action_representation="$b2_action_representation"
    --policy.z1_action_representation="$z1_action_representation"
    --policy.action_predict_arm_teleop_inactive="$predict_arm_teleop_inactive"
    --policy.action_predict_arm_reset="$predict_arm_reset"
    --policy.action_predict_ee_pose="$predict_ee_pose"
    --policy.action_predict_gripper="$predict_gripper"
    --policy.action_predict_task_complete="$predict_task_complete"
    --policy.task_complete_sample_tail_seconds="$task_complete_sample_tail_seconds"
    --policy.chunk_size="$action_chunk_size"
    --policy.n_action_steps="$action_steps_to_execute"
    --policy.control_frequency_hz="$control_frequency_hz"
  )
fi

policy_mem_args=()
if [[ "$enable_mem" == "true" && -z "$resume_checkpoint" ]]; then
  if [[ ! -f "$mem_vit_checkpoint" ]]; then
    echo "MEM ViT checkpoint not found: $mem_vit_checkpoint" >&2
    exit 1
  fi
  policy_mem_args+=(--policy.mem_vit_checkpoint="$mem_vit_checkpoint")
  policy_mem_args+=(--policy.mem_vit_frame_interval_seconds="$mem_frame_interval_seconds")
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

policy_history_args=()
if [[ -z "$resume_checkpoint" ]]; then
  policy_history_args+=(
    --policy.state_action_encoding="$state_action_encoding"
    --policy.state_num_frames="$state_num_frames"
    --policy.state_history_frame_interval_seconds="$state_history_frame_interval_seconds"
    --policy.action_history_enabled="$action_history_enabled"
  )
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
  "${resume_args[@]}" \
  --dataset.repo_id="$dataset_repo_id" \
  --dataset.root="$dataset_root" \
  --dataset.eval_split="$eval_split" \
  "${policy_source_args[@]}" \
  --policy.input_features=null \
  --policy.output_features=null \
  "${policy_io_args[@]}" \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.train_expert_only="$train_expert_only" \
  "${policy_mem_args[@]}" \
  "${policy_history_args[@]}" \
  --policy.push_to_hub=false \
  "${peft_args[@]}" \
  --output_dir="$output_dir" \
  --job_name="$job_name" \
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
  --wandb.project="$wandb_project" \
  --wandb.disable_artifact=true
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
echo "Resume checkpoint: ${resume_checkpoint:-none}"
echo "MEM:              $enable_mem"
if [[ -n "$resume_checkpoint" ]]; then
  echo "Deployment metadata: restored from $resume_checkpoint/pretrained_model/pi05_deployment_metadata.json"
else
  echo "B2 action:        $b2_action_representation"
  echo "State arm q/qd/gripper: $state_use_arm_joint_positions/$state_use_arm_joint_velocities/$state_use_arm_gripper_feedback"
  echo "State B2 q/qd/trunk/v/w: $state_use_b2_joint_positions/$state_use_b2_joint_velocities/$state_use_b2_trunk_pose/$state_use_b2_linear_velocity/$state_use_b2_angular_velocity"
fi
echo "Finetune mode:    $finetune_mode"
echo "Train expert only: $train_expert_only"
echo "GPUs:             $gpu_ids ($num_gpus process(es))"
echo "Dataset:          $dataset_root"
if [[ -z "$resume_checkpoint" ]]; then
  echo "Action timing:    ${control_frequency_hz}Hz, chunk=$action_chunk_size, execute=$action_steps_to_execute (dt derived automatically)"
else
  echo "Action timing:    restored from checkpoint deployment metadata"
fi
echo "Train/eval split: eval_split=$eval_split"
echo "Steps:            $steps optimizer updates"
echo "Batch per GPU:    $batch_size_per_gpu"
echo "Gradient accum:   $gradient_accumulation_steps (computed)"
echo "Global batch:     $batch_size_per_gpu x $num_gpus GPU(s) x $gradient_accumulation_steps = $global_batch_size"
echo "Log:              $log_file"
echo "Output:           $output_dir"
echo "Watch:            tail -f '$log_file'"
