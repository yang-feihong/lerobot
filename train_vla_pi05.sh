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
b2_action_representation="velocity" # "velocity" or "pose_delta"
z1_action_representation="ee_delta" # "ee_delta" (adjacent target) or "ee_state_delta" (inference-time state anchor)
ee_delta_rotation_representation="rotvec" # "rotvec" for new EE-delta checkpoints
action_semantics_profile="joint_control_ee_v1"
predict_arm_teleop_inactive="false"
predict_arm_reset="false"
predict_ee_pose="true"
predict_gripper="true"
predict_task_complete="false"
discrete_action_training_mode="continuous_flow" # "continuous_flow" or "structured_temporal"
ee_target_dataset_semantics="joint_control_inactive_interpolated"
ee_supervision_source="control_action"
ee_delta_supervision_mode="all" # "active_only" or "all"
gripper_target_representation="continuous_position"
action_loss_schema="uniform_valid"
task_complete_sample_tail_seconds="2.0"
new_module_optimizer_lr_multiplier="40.0"
structured_action_crf_initial_stay_bias="4.0"

# GPUs are selected here. Examples:
#   gpu_ids="0"      -> single GPU
#   gpu_ids="0,1,2"  -> 3-GPU DDP via accelerate
gpu_ids="0,1,2"
main_process_port="29500"

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
seed="1000"
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
motion_balanced_sampling="true"
motion_priority_fraction="0.5"
motion_ee_translation_threshold_m="0.05"
motion_ee_rotation_threshold_rad="0.17453292519943295"
motion_gripper_change_threshold="0.5"

# Fraction of episodes held out for periodic validation.
eval_split="0.1"
eval_steps="500"
max_eval_samples="512"

log_freq="10"
save_freq="500"
wandb_project="b2-z1-vla"
wandb_enable="true"

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
state_action_encoding="text"
# With 50 Hz data, 13 samples at 0.04 s intervals cover 0.48 s.
state_num_frames="13"
state_history_frame_interval_seconds="0.04"
# Uses the state-history clock and excludes the current action.
action_history_enabled="false"

# Optional named overrides used by checked-in experiment launchers. Ordinary
# single-run training can keep editing the user configuration above.
job_suffix=""
output_root_override=""
wandb_project_override=""
dry_run="false"
dataset_episodes=""
for argument in "$@"; do
  if [[ "$argument" == --action-semantics-profile=* ]]; then
    action_semantics_profile="${argument#*=}"
  fi
done
case "$action_semantics_profile" in
  joint_control_ee_v1)
    predict_arm_teleop_inactive="false"
    predict_arm_reset="false"
    predict_task_complete="false"
    discrete_action_training_mode="continuous_flow"
    ee_target_dataset_semantics="joint_control_inactive_interpolated"
    ee_supervision_source="control_action"
    ee_delta_supervision_mode="all"
    gripper_target_representation="continuous_position"
    action_loss_schema="uniform_valid"
    ;;
  custom) ;;
  *)
    echo "Unknown action_semantics_profile=$action_semantics_profile" >&2
    exit 2
    ;;
esac
while (( $# > 0 )); do
  case "$1" in
    --gpu-id=*) gpu_ids="${1#*=}" ;;
    --main-process-port=*) main_process_port="${1#*=}" ;;
    --enable-mem=*) enable_mem="${1#*=}" ;;
    --state-action-encoding=*) state_action_encoding="${1#*=}" ;;
    --action-history-enabled=*) action_history_enabled="${1#*=}" ;;
    --b2-action-representation=*) b2_action_representation="${1#*=}" ;;
    --z1-action-representation=*) z1_action_representation="${1#*=}" ;;
    --action-semantics-profile=*) action_semantics_profile="${1#*=}" ;;
    --discrete-action-training-mode=*) discrete_action_training_mode="${1#*=}" ;;
    --ee-delta-supervision-mode=*) ee_delta_supervision_mode="${1#*=}" ;;
    --gripper-target-representation=*) gripper_target_representation="${1#*=}" ;;
    --action-loss-schema=*) action_loss_schema="${1#*=}" ;;
    --predict-arm-teleop-inactive=*) predict_arm_teleop_inactive="${1#*=}" ;;
    --predict-arm-reset=*) predict_arm_reset="${1#*=}" ;;
    --predict-task-complete=*) predict_task_complete="${1#*=}" ;;
    --new-module-optimizer-lr-multiplier=*) new_module_optimizer_lr_multiplier="${1#*=}" ;;
    --structured-action-crf-initial-stay-bias=*) structured_action_crf_initial_stay_bias="${1#*=}" ;;
    --batch-size-per-gpu=*) batch_size_per_gpu="${1#*=}" ;;
    --global-batch-size=*) global_batch_size="${1#*=}" ;;
    --num-workers=*) num_workers="${1#*=}" ;;
    --motion-balanced-sampling=*) motion_balanced_sampling="${1#*=}" ;;
    --motion-priority-fraction=*) motion_priority_fraction="${1#*=}" ;;
    --motion-ee-translation-threshold-m=*) motion_ee_translation_threshold_m="${1#*=}" ;;
    --motion-ee-rotation-threshold-rad=*) motion_ee_rotation_threshold_rad="${1#*=}" ;;
    --motion-gripper-change-threshold=*) motion_gripper_change_threshold="${1#*=}" ;;
    --finetune-mode=*) finetune_mode="${1#*=}" ;;
    --dataset-repo-id=*) dataset_repo_id="${1#*=}" ;;
    --dataset-root=*) dataset_root="${1#*=}" ;;
    --dataset-episodes=*) dataset_episodes="${1#*=}" ;;
    --gpu-ids=*) gpu_ids="${1#*=}" ;;
    --steps=*) steps="${1#*=}" ;;
    --log-freq=*) log_freq="${1#*=}" ;;
    --eval-steps=*) eval_steps="${1#*=}" ;;
    --max-eval-samples=*) max_eval_samples="${1#*=}" ;;
    --save-freq=*) save_freq="${1#*=}" ;;
    --seed=*) seed="${1#*=}" ;;
    --job-suffix=*) job_suffix="${1#*=}" ;;
    --output-root=*) output_root_override="${1#*=}" ;;
    --wandb-project=*) wandb_project_override="${1#*=}" ;;
    --wandb-enable=*) wandb_enable="${1#*=}" ;;
    --resume-checkpoint=*) resume_checkpoint="${1#*=}" ;;
    --dry-run=*) dry_run="${1#*=}" ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "$b2_action_representation" != "velocity" && "$b2_action_representation" != "pose_delta" ]]; then
  echo "B2 representation must be velocity or pose_delta." >&2
  exit 2
fi
if [[ "$z1_action_representation" != "ee_delta" && "$z1_action_representation" != "ee_state_delta" ]]; then
  echo "Z1 representation must be ee_delta or ee_state_delta." >&2
  exit 2
fi

case "$action_semantics_profile" in
  joint_control_ee_v1)
    expected_semantics=(false false false continuous_flow joint_control_inactive_interpolated control_action all continuous_position uniform_valid)
    ;;
  custom)
    expected_semantics=()
    ;;
  *)
    echo "Unknown action_semantics_profile=$action_semantics_profile" >&2
    exit 2
    ;;
esac
if (( ${#expected_semantics[@]} )); then
  actual_semantics=(
    "$predict_arm_teleop_inactive" "$predict_arm_reset" "$predict_task_complete"
    "$discrete_action_training_mode" "$ee_target_dataset_semantics" "$ee_supervision_source"
    "$ee_delta_supervision_mode"
    "$gripper_target_representation" "$action_loss_schema"
  )
  if [[ "${actual_semantics[*]}" != "${expected_semantics[*]}" ]]; then
    echo "Profile $action_semantics_profile was mixed with incompatible overrides." >&2
    echo "Use --action-semantics-profile=custom for an ablation." >&2
    exit 2
  fi
fi

# =========================
# Launch
# =========================

timestamp="$(date +%Y%m%d_%H%M%S)"
log_dir="$repo_root/logs"

job_prefix="pi05_b2_z1_vla"
if [[ "$enable_mem" == "true" ]]; then
  job_prefix="mem_pi05_b2_z1_vla"
  output_root="/data/b2_z1_vla_mem_outputs"
fi
if [[ -n "$job_suffix" ]]; then
  if [[ ! "$job_suffix" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "Invalid job suffix: $job_suffix" >&2
    exit 2
  fi
  job_prefix="${job_prefix}_${job_suffix}"
fi
if [[ -n "$output_root_override" ]]; then
  output_root="$output_root_override"
fi
if [[ -n "$wandb_project_override" ]]; then
  wandb_project="$wandb_project_override"
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
  output_dir="$output_root/${timestamp}_${job_prefix}"
  job_name="${timestamp}_${job_prefix}"
  log_file="$log_dir/${job_name}.log"
  pid_file="$log_dir/${job_name}.pid"
fi

policy_io_args=()
if [[ -z "$resume_checkpoint" ]]; then
  policy_io_args+=(
    --policy.input_features=null
    --policy.output_features=null
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
    --policy.ee_delta_rotation_representation="$ee_delta_rotation_representation"
    --policy.action_predict_arm_teleop_inactive="$predict_arm_teleop_inactive"
    --policy.action_predict_arm_reset="$predict_arm_reset"
    --policy.action_predict_ee_pose="$predict_ee_pose"
    --policy.action_predict_gripper="$predict_gripper"
    --policy.action_predict_task_complete="$predict_task_complete"
    --policy.discrete_action_training_mode="$discrete_action_training_mode"
    --policy.ee_target_dataset_semantics="$ee_target_dataset_semantics"
    --policy.ee_supervision_source="$ee_supervision_source"
    --policy.ee_delta_supervision_mode="$ee_delta_supervision_mode"
    --policy.gripper_target_representation="$gripper_target_representation"
    --policy.action_loss_schema="$action_loss_schema"
    --policy.task_complete_sample_tail_seconds="$task_complete_sample_tail_seconds"
    --policy.new_module_optimizer_lr_multiplier="$new_module_optimizer_lr_multiplier"
    --policy.structured_action_crf_initial_stay_bias="$structured_action_crf_initial_stay_bias"
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
policy_runtime_args=()
train_expert_only="restored"
if [[ -z "$resume_checkpoint" ]]; then
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
  policy_runtime_args+=(
    --policy.device=cuda
    --policy.dtype=bfloat16
    --policy.gradient_checkpointing=true
    --policy.train_expert_only="$train_expert_only"
  )
fi

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
if [[ ! "$main_process_port" =~ ^[0-9]+$ ]] || (( main_process_port < 1024 || main_process_port > 65535 )); then
  echo "main_process_port must be an integer in [1024, 65535], got $main_process_port." >&2
  exit 1
fi

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

runtime_bin="${LEROBOT_RUNTIME_BIN:-}"
if [[ -n "$runtime_bin" ]]; then
  [[ -x "$runtime_bin/accelerate" ]] || { echo "Missing $runtime_bin/accelerate" >&2; exit 1; }
  [[ -x "$runtime_bin/python" ]] || { echo "Missing $runtime_bin/python" >&2; exit 1; }
fi

train_args=(
  "${resume_args[@]}" \
  --dataset.repo_id="$dataset_repo_id" \
  --dataset.root="$dataset_root" \
  --dataset.eval_split="$eval_split" \
  "${policy_source_args[@]}" \
  "${policy_io_args[@]}" \
  "${policy_runtime_args[@]}" \
  "${policy_mem_args[@]}" \
  "${policy_history_args[@]}" \
  --policy.push_to_hub=false \
  "${peft_args[@]}" \
  --output_dir="$output_dir" \
  --job_name="$job_name" \
  --steps="$steps" \
  --seed="$seed" \
  --batch_size="$batch_size_per_gpu" \
  --gradient_accumulation_steps="$gradient_accumulation_steps" \
  --num_workers="$num_workers" \
  --motion_balanced_sampling.enabled="$motion_balanced_sampling" \
  --motion_balanced_sampling.priority_fraction="$motion_priority_fraction" \
  --motion_balanced_sampling.ee_translation_threshold_m="$motion_ee_translation_threshold_m" \
  --motion_balanced_sampling.ee_rotation_threshold_rad="$motion_ee_rotation_threshold_rad" \
  --motion_balanced_sampling.gripper_change_threshold="$motion_gripper_change_threshold" \
  --log_freq="$log_freq" \
  --eval_steps="$eval_steps" \
  --max_eval_samples="$max_eval_samples" \
  --env_eval_freq=0 \
  --save_checkpoint=true \
  --save_freq="$save_freq" \
  --wandb.enable="$wandb_enable" \
  --wandb.project="$wandb_project" \
  --wandb.disable_artifact=true
)

if [[ -n "$dataset_episodes" ]]; then
  train_args+=(--dataset.episodes="$dataset_episodes")
fi

if [[ "$dry_run" == "true" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q lerobot_train' "$gpu_ids"
  printf ' %q' "${train_args[@]}"
  printf '\n'
  exit 0
elif [[ "$dry_run" != "false" ]]; then
  echo "--dry-run must be true or false, got $dry_run" >&2
  exit 2
fi

if (( num_gpus > 1 )); then
  if [[ -n "$runtime_bin" ]]; then
    launch_command=("$runtime_bin/accelerate")
  else
    launch_command=(uv run --no-sync accelerate)
  fi
  setsid env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="$gpu_ids" "${launch_command[@]}" launch \
    --multi_gpu \
    --num_processes "$num_gpus" \
    --num_machines 1 \
    --main_process_port "$main_process_port" \
    --gpu_ids "$gpu_ids" \
    --mixed_precision bf16 \
    --dynamo_backend no \
    -m lerobot.scripts.lerobot_train \
    "${train_args[@]}" \
    >"$log_file" 2>&1 </dev/null &
else
  if [[ -n "$runtime_bin" ]]; then
    launch_command=("$runtime_bin/python")
  else
    launch_command=(uv run --no-sync python)
  fi
  setsid env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="$gpu_ids" "${launch_command[@]}" -u -m lerobot.scripts.lerobot_train \
    "${train_args[@]}" \
    >"$log_file" 2>&1 </dev/null &
fi

pid=$!
echo "$pid" >"$pid_file"
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo "Training process exited during startup. See: $log_file" >&2
  wait "$pid"
fi

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
if (( num_gpus > 1 )); then
  echo "DDP port:         $main_process_port"
fi
echo "Dataset:          $dataset_root"
echo "Motion sampling:  enabled=$motion_balanced_sampling, priority=$motion_priority_fraction, translation=${motion_ee_translation_threshold_m}m, rotation=${motion_ee_rotation_threshold_rad}rad, gripper=$motion_gripper_change_threshold"
if [[ -z "$resume_checkpoint" ]]; then
  echo "Action timing:    ${control_frequency_hz}Hz, chunk=$action_chunk_size, execute=$action_steps_to_execute (dt derived automatically)"
else
  echo "Action timing:    restored from checkpoint deployment metadata"
fi
echo "Train/eval split: eval_split=$eval_split"
echo "Steps:            $steps optimizer updates"
echo "Action semantics: $action_semantics_profile ($ee_target_dataset_semantics, EE source=$ee_supervision_source, mask=$ee_delta_supervision_mode, gripper=$gripper_target_representation, loss=$action_loss_schema)"
echo "Seed:             $seed"
echo "Batch per GPU:    $batch_size_per_gpu"
echo "Gradient accum:   $gradient_accumulation_steps (computed)"
echo "Global batch:     $batch_size_per_gpu x $num_gpus GPU(s) x $gradient_accumulation_steps = $global_batch_size"
echo "Log:              $log_file"
echo "Output:           $output_dir"
echo "Watch:            tail -f '$log_file'"
