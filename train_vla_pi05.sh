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
action_semantics_profile="joint_control_arm_mode_v2"
predict_arm_teleop_inactive="true"
predict_arm_reset="true"
predict_ee_pose="true"
predict_gripper="true"
predict_task_complete="false"
discrete_action_training_mode="continuous_flow" # "continuous_flow" or "structured_temporal"
ee_target_dataset_semantics="joint_control_inactive_interpolated"
ee_supervision_source="control_action"
ee_delta_supervision_mode="all" # "active_only" or "all"
gripper_target_representation="continuous_position"
action_loss_schema="auto"
task_complete_sample_tail_seconds="2.0"
new_module_optimizer_lr_multiplier="40.0"
structured_action_crf_initial_stay_bias="4.0"

# GPUs are selected here. Examples:
#   gpu_ids="0"      -> single GPU
#   gpu_ids="0,1,2"  -> 3-GPU DDP via accelerate
gpu_ids="0,1,2"
main_process_port="29500"

# Choose exactly one fine-tuning mode:
#   expert = freeze PaliGemma/VLM and full fine-tune the action expert/projections.
#            MEM-ViT training is controlled separately below (defaults to full).
#   lora   = full fine-tune the action expert/projections, plus LoRA on the
#            PaliGemma/VLM backbone and, by default, MEM-ViT when enabled.
#            Full MEM-ViT training has OOMed with DDP at batch_size=2 on 24GB.
#            MEM LoRA/frozen modes still require measuring the actual workload.
#   full   = full VLA fine-tuning without LoRA.
#            Does not fit on this RTX 4090 with AdamW: batch_size=1 OOM at ≈23.5GB
#            during optimizer-state initialization. Plan for at least 32GB, preferably
#            40GB/48GB or multi-GPU/ZeRO/FSDP/8-bit optimizer.
finetune_mode="lora"

dataset_repo_id="local/b2_z1_vla"
dataset_root="/data/b2_z1_vla_lerobot"
# PyAV is the deployment image's supported and reproducible video decoder.
# Override explicitly only when validating another installed backend.
video_backend="pyav"
# CLI: --image-source=real|sim|mixed. Simulated images require the physically
# paired manifest and rollout root for the selected dataset.
image_source="real" # "real", "sim", or "mixed"
sim_image_manifest=""
sim_image_root=""
mixed_sim_probability="0.5"
base_policy="/data/checkpoints/lerobot_pi05_base_local_tokenizer"
max_state_dim="32"

steps="20000"
seed="1000"
optimizer_lr="2.5e-5"
lr_scheduler_type="constant_with_warmup" # warmup, then hold optimizer_lr
scheduler_warmup_steps="1000"
scheduler_decay_steps="30000" # used only by cosine_decay_with_warmup
scheduler_decay_lr="2.5e-6" # used only by cosine_decay_with_warmup
# Temporal semantics for the planned 50 Hz dataset. A chunk covers one second,
# while deployment requests a fresh observation/chunk after executing 0.5 s.
# A dataset at another FPS is timestamp-resampled to this model frequency;
# lower-frequency data is allowed with a strong warning because frames repeat.
action_chunk_size="50"
action_steps_to_execute="25"
control_frequency_hz="50"
# Optional training-time RTC. 5 samples delays 0..4 (exclusive upper bound).
# Runtime inference RTC remains a separate deployment choice.
training_rtc="false"
training_rtc_simulated_delay="16"
training_rtc_delay_distribution="uniform"
# Training batch semantics are independent of the selected GPU count:
#   global_batch_size = batch_size_per_gpu * number_of_gpus * computed_grad_accum
# Keeping global_batch_size / batch_size_per_gpu a multiple of 24 supports the
# common 1/2/3/4/6/8/12/24-GPU counts without changing the effective batch.
batch_size_per_gpu="2"
global_batch_size="48"
num_workers="4"
motion_balanced_sampling="false" # legacy-only; new runs use static-horizon sampling below
motion_priority_fraction="0.5"
motion_ee_translation_threshold_m="0.05"
motion_ee_rotation_threshold_rad="0.17453292519943295"
motion_gripper_change_threshold="0.5"
static_horizon_sampling="true"
interior_static_weight="0.0"
terminal_static_weight="1.0" # increase above 1.0 to strengthen terminal holds
dataset_mixture_sampling="false"
dataset_mixture_manifest=""
# A mixture manifest describes source membership in a losslessly merged dataset:
# {"sources":[
#   {"name":"stage1_stage2","weight":0.8,"episode_range":[0,1085]},
#   {"name":"standalone_stage2","weight":0.2,"episode_range":[1085,2225]}
# ]}

# Fraction of episodes held out for periodic validation.
eval_split="0.1"
eval_steps="500"
max_eval_samples="512"

log_freq="10"
save_freq="500"
keep_last_checkpoints="2"
keep_checkpoint_every_n_steps="10000"
wandb_project="b2-z1-vla"
wandb_enable="true"

output_root="/data/b2_z1_vla_pi05_outputs"

# Leave empty for a new run. To resume in place, point this at a complete
# numeric checkpoint directory or its `last` symlink. The checkpoint's saved
# optimizer, scheduler, RNG, data-order and W&B run state are restored.
resume_checkpoint=""
resume_with_updated_dataset="false"
# Set a basename to fork the saved model/optimizer/scheduler state into a new
# output directory and a new W&B run. Empty keeps the historical in-place
# resume behavior.
resume_new_run_name=""

# Used only when finetune_mode="lora". Action expert/projections are full fine-tuned;
# PaliGemma/VLM backbone uses LoRA. In non-MEM mode, unfrozen ViT also uses LoRA.
lora_rank="16"
lora_alpha="32"
# Freezes the entire visual encoder, including MEM when enabled. This takes
# precedence over mem_vit_finetune_mode; the language model keeps its chosen mode.
freeze_vision_encoder="false"

# MEM-only configuration. Used only when enable_mem="true".
mem_vit_checkpoint="/data/mem_vit_distill_outputs/mem_vit_distill_20260716_142702/mem_vit_distill_latest.pt"
# auto selects lora with finetune_mode=lora, otherwise full (legacy behavior).
# Explicit CLI modes are full, lora, frozen. Resume restores the saved strategy.
mem_vit_finetune_mode="auto"

# Choose one MEM window mode when enable_mem="true".
mem_fixed_num_frames="6"
mem_random_min_num_frames="1"
mem_random_max_num_frames="6"
# MEM image sampling interval in seconds.
mem_frame_interval_seconds="0.5"
# Training samples one shared interval bias per history window, then samples
# every adjacent frame gap independently around it.
mem_random_interval_sampling="true"
mem_global_interval_std_seconds="0.15"
mem_local_interval_std_seconds="0.05"
mem_min_interval_seconds="0.25"
mem_max_interval_seconds="1.0"
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
dataset_repo_id_explicit="false"
dataset_root_explicit="false"
dataset_episodes_explicit="false"
image_source_explicit="false"
sim_image_manifest_explicit="false"
sim_image_root_explicit="false"
mixed_sim_probability_explicit="false"
motion_balanced_sampling_explicit="false"
motion_priority_fraction_explicit="false"
motion_ee_translation_threshold_m_explicit="false"
motion_ee_rotation_threshold_rad_explicit="false"
motion_gripper_change_threshold_explicit="false"
static_horizon_sampling_explicit="false"
interior_static_weight_explicit="false"
terminal_static_weight_explicit="false"
dataset_mixture_sampling_explicit="false"
dataset_mixture_manifest_explicit="false"
training_rtc_explicit="false"
training_rtc_simulated_delay_explicit="false"
training_rtc_delay_distribution_explicit="false"
for argument in "$@"; do
  if [[ "$argument" == --action-semantics-profile=* ]]; then
    action_semantics_profile="${argument#*=}"
  fi
done
case "$action_semantics_profile" in
  joint_control_ee_v1)
    predict_arm_teleop_inactive="false"
    predict_arm_reset="false"
    predict_ee_pose="true"
    predict_gripper="true"
    predict_task_complete="false"
    discrete_action_training_mode="continuous_flow"
    ee_target_dataset_semantics="joint_control_inactive_interpolated"
    ee_supervision_source="control_action"
    ee_delta_supervision_mode="all"
    gripper_target_representation="continuous_position"
    action_loss_schema="uniform_valid"
    ;;
  joint_control_arm_mode_v2)
    predict_arm_teleop_inactive="true"
    predict_arm_reset="true"
    predict_ee_pose="true"
    predict_gripper="true"
    predict_task_complete="false"
    discrete_action_training_mode="continuous_flow"
    ee_target_dataset_semantics="joint_control_inactive_interpolated"
    ee_supervision_source="control_action"
    ee_delta_supervision_mode="all"
    gripper_target_representation="continuous_position"
    action_loss_schema="auto"
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
    --mem-vit-finetune-mode=*)
      mem_vit_finetune_mode="${1#*=}"
      case "$mem_vit_finetune_mode" in
        full|lora|frozen) ;;
        *) echo "--mem-vit-finetune-mode must be full, lora or frozen." >&2; exit 2 ;;
      esac
      ;;
    --freeze-vision-encoder=*) freeze_vision_encoder="${1#*=}" ;;
    --mem-vit-checkpoint=*) mem_vit_checkpoint="${1#*=}" ;;
    --mem-random-min-num-frames=*) mem_random_min_num_frames="${1#*=}" ;;
    --mem-random-max-num-frames=*) mem_random_max_num_frames="${1#*=}" ;;
    --mem-frame-interval-seconds=*) mem_frame_interval_seconds="${1#*=}" ;;
    --mem-random-interval-sampling=*) mem_random_interval_sampling="${1#*=}" ;;
    --mem-global-interval-std-seconds=*) mem_global_interval_std_seconds="${1#*=}" ;;
    --mem-local-interval-std-seconds=*) mem_local_interval_std_seconds="${1#*=}" ;;
    --mem-min-interval-seconds=*) mem_min_interval_seconds="${1#*=}" ;;
    --mem-max-interval-seconds=*) mem_max_interval_seconds="${1#*=}" ;;
    --lora-rank=*) lora_rank="${1#*=}" ;;
    --lora-alpha=*) lora_alpha="${1#*=}" ;;
    --base-policy=*) base_policy="${1#*=}" ;;
    --state-action-encoding=*) state_action_encoding="${1#*=}" ;;
    --action-history-enabled=*) action_history_enabled="${1#*=}" ;;
    --training-rtc=*) training_rtc="${1#*=}"; training_rtc_explicit="true" ;;
    --training-rtc-simulated-delay=*) training_rtc_simulated_delay="${1#*=}"; training_rtc_simulated_delay_explicit="true" ;;
    --training-rtc-delay-distribution=*) training_rtc_delay_distribution="${1#*=}"; training_rtc_delay_distribution_explicit="true" ;;
    --b2-action-representation=*) b2_action_representation="${1#*=}" ;;
    --z1-action-representation=*) z1_action_representation="${1#*=}" ;;
    --action-semantics-profile=*) action_semantics_profile="${1#*=}" ;;
    --discrete-action-training-mode=*) discrete_action_training_mode="${1#*=}" ;;
    --ee-delta-supervision-mode=*) ee_delta_supervision_mode="${1#*=}" ;;
    --gripper-target-representation=*) gripper_target_representation="${1#*=}" ;;
    --action-loss-schema=*) action_loss_schema="${1#*=}" ;;
    --predict-arm-teleop-inactive=*) predict_arm_teleop_inactive="${1#*=}" ;;
    --predict-arm-reset=*) predict_arm_reset="${1#*=}" ;;
    --predict-ee-pose=*) predict_ee_pose="${1#*=}" ;;
    --predict-gripper=*) predict_gripper="${1#*=}" ;;
    --predict-task-complete=*) predict_task_complete="${1#*=}" ;;
    --new-module-optimizer-lr-multiplier=*) new_module_optimizer_lr_multiplier="${1#*=}" ;;
    --structured-action-crf-initial-stay-bias=*) structured_action_crf_initial_stay_bias="${1#*=}" ;;
    --batch-size-per-gpu=*) batch_size_per_gpu="${1#*=}" ;;
    --global-batch-size=*) global_batch_size="${1#*=}" ;;
    --num-workers=*) num_workers="${1#*=}" ;;
    --motion-balanced-sampling=*) motion_balanced_sampling="${1#*=}"; motion_balanced_sampling_explicit="true" ;;
    --motion-priority-fraction=*) motion_priority_fraction="${1#*=}"; motion_priority_fraction_explicit="true" ;;
    --motion-ee-translation-threshold-m=*) motion_ee_translation_threshold_m="${1#*=}"; motion_ee_translation_threshold_m_explicit="true" ;;
    --motion-ee-rotation-threshold-rad=*) motion_ee_rotation_threshold_rad="${1#*=}"; motion_ee_rotation_threshold_rad_explicit="true" ;;
    --motion-gripper-change-threshold=*) motion_gripper_change_threshold="${1#*=}"; motion_gripper_change_threshold_explicit="true" ;;
    --static-horizon-sampling=*) static_horizon_sampling="${1#*=}"; static_horizon_sampling_explicit="true" ;;
    --interior-static-weight=*) interior_static_weight="${1#*=}"; interior_static_weight_explicit="true" ;;
    --terminal-static-weight=*) terminal_static_weight="${1#*=}"; terminal_static_weight_explicit="true" ;;
    --dataset-mixture-sampling=*) dataset_mixture_sampling="${1#*=}"; dataset_mixture_sampling_explicit="true" ;;
    --dataset-mixture-manifest=*) dataset_mixture_manifest="${1#*=}"; dataset_mixture_manifest_explicit="true" ;;
    --finetune-mode=*) finetune_mode="${1#*=}" ;;
    --dataset-repo-id=*) dataset_repo_id="${1#*=}"; dataset_repo_id_explicit="true" ;;
    --dataset-root=*) dataset_root="${1#*=}"; dataset_root_explicit="true" ;;
    --video-backend=*)
      video_backend="${1#*=}"
      case "$video_backend" in
        ""|pyav|torchcodec|video_reader) ;;
        *) echo "--video-backend must be empty, pyav, torchcodec or video_reader." >&2; exit 2 ;;
      esac
      ;;
    --dataset-episodes=*) dataset_episodes="${1#*=}"; dataset_episodes_explicit="true" ;;
    --image-source=*) image_source="${1#*=}"; image_source_explicit="true" ;;
    --sim-image-manifest=*) sim_image_manifest="${1#*=}"; sim_image_manifest_explicit="true" ;;
    --sim-image-root=*) sim_image_root="${1#*=}"; sim_image_root_explicit="true" ;;
    --mixed-sim-probability=*) mixed_sim_probability="${1#*=}"; mixed_sim_probability_explicit="true" ;;
    --gpu-ids=*) gpu_ids="${1#*=}" ;;
    --steps=*) steps="${1#*=}" ;;
    --optimizer-lr=*) optimizer_lr="${1#*=}" ;;
    --lr-scheduler-type=*) lr_scheduler_type="${1#*=}" ;;
    --scheduler-warmup-steps=*) scheduler_warmup_steps="${1#*=}" ;;
    --scheduler-decay-steps=*) scheduler_decay_steps="${1#*=}" ;;
    --scheduler-decay-lr=*) scheduler_decay_lr="${1#*=}" ;;
    --log-freq=*) log_freq="${1#*=}" ;;
    --eval-steps=*) eval_steps="${1#*=}" ;;
    --max-eval-samples=*) max_eval_samples="${1#*=}" ;;
    --save-freq=*) save_freq="${1#*=}" ;;
    --keep-last-checkpoints=*) keep_last_checkpoints="${1#*=}" ;;
    --keep-checkpoint-every-n-steps=*) keep_checkpoint_every_n_steps="${1#*=}" ;;
    --seed=*) seed="${1#*=}" ;;
    --job-suffix=*) job_suffix="${1#*=}" ;;
    --output-root=*) output_root_override="${1#*=}" ;;
    --wandb-project=*) wandb_project_override="${1#*=}" ;;
    --wandb-enable=*) wandb_enable="${1#*=}" ;;
    --resume-checkpoint=*) resume_checkpoint="${1#*=}" ;;
    --resume-with-updated-dataset=*) resume_with_updated_dataset="${1#*=}" ;;
    --resume-new-run-name=*) resume_new_run_name="${1#*=}" ;;
    --dry-run=*) dry_run="${1#*=}" ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "$training_rtc" != "true" && "$training_rtc" != "false" ]]; then
  echo "--training-rtc must be true or false." >&2
  exit 2
fi
if [[ ! "$training_rtc_simulated_delay" =~ ^[1-9][0-9]*$ ]]; then
  echo "--training-rtc-simulated-delay must be a positive integer (exclusive upper bound)." >&2
  exit 2
fi
if [[ "$training_rtc_delay_distribution" != "exponential" && "$training_rtc_delay_distribution" != "uniform" ]]; then
  echo "--training-rtc-delay-distribution must be exponential or uniform." >&2
  exit 2
fi
if [[ -z "$resume_checkpoint" && "$training_rtc" == "true" ]]; then
  if (( training_rtc_simulated_delay > action_chunk_size )); then
    echo "--training-rtc-simulated-delay must not exceed action chunk size ($action_chunk_size)." >&2
    exit 2
  fi
  if [[ "$discrete_action_training_mode" != "continuous_flow" ]]; then
    echo "Training RTC requires --discrete-action-training-mode=continuous_flow." >&2
    exit 2
  fi
fi

if [[ -z "$resume_checkpoint" ]]; then
  case "$finetune_mode" in
    lora|expert|full) ;;
    *) echo "Unknown finetune_mode=$finetune_mode. Expected one of: lora, expert, full." >&2; exit 2 ;;
  esac
  if [[ "$freeze_vision_encoder" != "true" && "$freeze_vision_encoder" != "false" ]]; then
    echo "--freeze-vision-encoder must be true or false." >&2
    exit 2
  fi
  if [[ "$finetune_mode" == "lora" ]]; then
    if [[ ! "$lora_rank" =~ ^[1-9][0-9]*$ || ! "$lora_alpha" =~ ^[1-9][0-9]*$ ]]; then
      echo "--lora-rank and --lora-alpha must be positive integers." >&2
      exit 2
    fi
  fi
  if [[ "$enable_mem" == "true" ]]; then
    if [[ "$freeze_vision_encoder" == "true" || "$mem_vit_finetune_mode" == "frozen" ]]; then
      mem_vit_finetune_mode="frozen"
      freeze_vision_encoder="true"
    elif [[ "$mem_vit_finetune_mode" == "auto" ]]; then
      if [[ "$finetune_mode" == "lora" ]]; then
        mem_vit_finetune_mode="lora"
      else
        mem_vit_finetune_mode="full"
      fi
    fi
    if [[ "$mem_vit_finetune_mode" == "lora" && "$finetune_mode" != "lora" ]]; then
      echo "MEM LoRA requires --finetune-mode=lora." >&2
      exit 2
    fi
  fi
fi

if [[ "$b2_action_representation" != "velocity" && "$b2_action_representation" != "pose_delta" ]]; then
  echo "B2 representation must be velocity or pose_delta." >&2
  exit 2
fi
if [[ "$z1_action_representation" != "ee_delta" && "$z1_action_representation" != "ee_state_delta" ]]; then
  echo "Z1 representation must be ee_delta or ee_state_delta." >&2
  exit 2
fi
if [[ "$image_source" != "real" && "$image_source" != "sim" && "$image_source" != "mixed" ]]; then
  echo "Image source must be real, sim or mixed." >&2
  exit 2
fi
if [[ "$image_source" != "real" && ( -z "$sim_image_manifest" || -z "$sim_image_root" ) ]]; then
  echo "Sim and mixed image sources require --sim-image-manifest and --sim-image-root." >&2
  exit 2
fi
if [[ -n "$video_backend" && "$video_backend" != "pyav" && "$video_backend" != "torchcodec" && "$video_backend" != "video_reader" ]]; then
  echo "Video backend must be pyav, torchcodec or video_reader." >&2
  exit 2
fi

case "$action_semantics_profile" in
  joint_control_ee_v1)
    expected_semantics=(false false true true continuous_flow joint_control_inactive_interpolated control_action all continuous_position uniform_valid)
    ;;
  joint_control_arm_mode_v2)
    expected_semantics=(true true true true continuous_flow joint_control_inactive_interpolated control_action all continuous_position auto)
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
    "$predict_arm_teleop_inactive" "$predict_arm_reset" "$predict_ee_pose" "$predict_gripper"
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
if [[ "$image_source" != "real" ]]; then
  job_prefix="${job_prefix}_${image_source}_images"
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
  if [[ "$resume_with_updated_dataset" != "true" && "$resume_with_updated_dataset" != "false" ]]; then
    echo "--resume-with-updated-dataset must be true or false, got $resume_with_updated_dataset" >&2
    exit 2
  fi
  resume_args+=(
    --config_path="$resume_config"
    --resume=true
    --resume_with_updated_dataset="$resume_with_updated_dataset"
  )
  if [[ -n "$resume_new_run_name" ]]; then
    if [[ "$resume_new_run_name" == */* || ! "$resume_new_run_name" =~ ^[a-zA-Z0-9._-]+$ ]]; then
      echo "--resume-new-run-name must be a safe directory basename, got $resume_new_run_name" >&2
      exit 2
    fi
    output_dir="$output_root/$resume_new_run_name"
    job_name="$resume_new_run_name"
    if [[ -e "$output_dir" ]]; then
      echo "Forked resume output already exists: $output_dir" >&2
      exit 1
    fi
    resume_args+=(--wandb.resume_training_run=false)
  fi
  policy_source_args=()
  log_file="$log_dir/${job_name}_resume_${timestamp}.log"
  pid_file="$log_dir/${job_name}_resume_${timestamp}.pid"
else
  output_dir="$output_root/${timestamp}_${job_prefix}"
  job_name="${timestamp}_${job_prefix}"
  log_file="$log_dir/${job_name}.log"
  pid_file="$log_dir/${job_name}.pid"
fi

policy_training_rtc_args=()
if [[ -z "$resume_checkpoint" ]]; then
  policy_training_rtc_args+=(
    --policy.training_rtc_config.enabled="$training_rtc"
    --policy.training_rtc_config.simulated_delay="$training_rtc_simulated_delay"
    --policy.training_rtc_config.delay_distribution="$training_rtc_delay_distribution"
  )
elif [[ "$training_rtc_explicit" == "true" || "$training_rtc_simulated_delay_explicit" == "true" || "$training_rtc_delay_distribution_explicit" == "true" ]]; then
  # Resume must preserve the training objective and RNG/data-order semantics.
  # Explicit matching values are accepted as assertions, never as overrides.
  rtc_config_python="${LEROBOT_RUNTIME_BIN:+$LEROBOT_RUNTIME_BIN/python}"
  rtc_config_python="${rtc_config_python:-python3}"
  "$rtc_config_python" - "$resume_config" \
    "$training_rtc_explicit" "$training_rtc" \
    "$training_rtc_simulated_delay_explicit" "$training_rtc_simulated_delay" \
    "$training_rtc_delay_distribution_explicit" "$training_rtc_delay_distribution" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    saved = json.load(stream).get("policy", {}).get("training_rtc_config") or {}
checks = (
    ("enabled", False, sys.argv[2], sys.argv[3] == "true"),
    ("simulated_delay", 5, sys.argv[4], int(sys.argv[5])),
    ("delay_distribution", "exponential", sys.argv[6], sys.argv[7]),
)
for field, default, explicit, requested in checks:
    if explicit == "true" and requested != saved.get(field, default):
        print(
            f"Resume preserves checkpoint training_rtc_config.{field}={saved.get(field, default)!r}; "
            f"requested {requested!r}. Start a new fine-tuning run with --base-policy=<checkpoint>/pretrained_model "
            "to change the training RTC objective.",
            file=sys.stderr,
        )
        sys.exit(2)
PY
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
  policy_mem_args+=(--policy.mem_vit_finetune_mode="$mem_vit_finetune_mode")
  policy_mem_args+=(--policy.mem_vit_frame_interval_seconds="$mem_frame_interval_seconds")
  policy_mem_args+=(--policy.mem_vit_random_interval_sampling="$mem_random_interval_sampling")
  policy_mem_args+=(--policy.mem_vit_global_interval_std_seconds="$mem_global_interval_std_seconds")
  policy_mem_args+=(--policy.mem_vit_local_interval_std_seconds="$mem_local_interval_std_seconds")
  policy_mem_args+=(--policy.mem_vit_min_interval_seconds="$mem_min_interval_seconds")
  policy_mem_args+=(--policy.mem_vit_max_interval_seconds="$mem_max_interval_seconds")
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

dataset_args=()
if [[ -z "$resume_checkpoint" ]]; then
  dataset_args+=(
    --dataset.repo_id="$dataset_repo_id"
    --dataset.root="$dataset_root"
    --dataset.eval_split="$eval_split"
    --dataset.image_source="$image_source"
    --dataset.mixed_sim_probability="$mixed_sim_probability"
    --dataset.image_source_seed="$seed"
  )
  if [[ "$image_source" != "real" ]]; then
    dataset_args+=(
      --dataset.sim_image_manifest="$sim_image_manifest"
      --dataset.sim_image_root="$sim_image_root"
    )
  fi
else
  [[ "$dataset_repo_id_explicit" == "false" ]] || dataset_args+=(--dataset.repo_id="$dataset_repo_id")
  [[ "$dataset_root_explicit" == "false" ]] || dataset_args+=(--dataset.root="$dataset_root")
  [[ "$image_source_explicit" == "false" ]] || dataset_args+=(--dataset.image_source="$image_source")
  [[ "$sim_image_manifest_explicit" == "false" ]] || dataset_args+=(--dataset.sim_image_manifest="$sim_image_manifest")
  [[ "$sim_image_root_explicit" == "false" ]] || dataset_args+=(--dataset.sim_image_root="$sim_image_root")
  [[ "$mixed_sim_probability_explicit" == "false" ]] || dataset_args+=(--dataset.mixed_sim_probability="$mixed_sim_probability")
fi
if [[ -n "$video_backend" ]]; then
  dataset_args+=(--dataset.video_backend="$video_backend")
fi

sampling_args=()
if [[ -z "$resume_checkpoint" ]]; then
  sampling_args+=(
    --motion_balanced_sampling.enabled="$motion_balanced_sampling"
    --static_horizon_sampling.enabled="$static_horizon_sampling"
    --static_horizon_sampling.interior_weight="$interior_static_weight"
    --static_horizon_sampling.terminal_weight="$terminal_static_weight"
    --dataset_mixture_sampling.enabled="$dataset_mixture_sampling"
  )
  if [[ "$motion_balanced_sampling" == "true" ]]; then
    sampling_args+=(
      --motion_balanced_sampling.priority_fraction="$motion_priority_fraction"
      --motion_balanced_sampling.ee_translation_threshold_m="$motion_ee_translation_threshold_m"
      --motion_balanced_sampling.ee_rotation_threshold_rad="$motion_ee_rotation_threshold_rad"
      --motion_balanced_sampling.gripper_change_threshold="$motion_gripper_change_threshold"
    )
  fi
  if [[ "$dataset_mixture_sampling" == "true" ]]; then
    [[ -n "$dataset_mixture_manifest" ]] || { echo "--dataset-mixture-manifest is required" >&2; exit 2; }
    sampling_args+=(--dataset_mixture_sampling.manifest_path="$dataset_mixture_manifest")
  fi
else
  [[ "$motion_balanced_sampling_explicit" == "false" ]] || sampling_args+=(--motion_balanced_sampling.enabled="$motion_balanced_sampling")
  [[ "$motion_priority_fraction_explicit" == "false" ]] || sampling_args+=(--motion_balanced_sampling.priority_fraction="$motion_priority_fraction")
  [[ "$motion_ee_translation_threshold_m_explicit" == "false" ]] || sampling_args+=(--motion_balanced_sampling.ee_translation_threshold_m="$motion_ee_translation_threshold_m")
  [[ "$motion_ee_rotation_threshold_rad_explicit" == "false" ]] || sampling_args+=(--motion_balanced_sampling.ee_rotation_threshold_rad="$motion_ee_rotation_threshold_rad")
  [[ "$motion_gripper_change_threshold_explicit" == "false" ]] || sampling_args+=(--motion_balanced_sampling.gripper_change_threshold="$motion_gripper_change_threshold")
  [[ "$static_horizon_sampling_explicit" == "false" ]] || sampling_args+=(--static_horizon_sampling.enabled="$static_horizon_sampling")
  [[ "$interior_static_weight_explicit" == "false" ]] || sampling_args+=(--static_horizon_sampling.interior_weight="$interior_static_weight")
  [[ "$terminal_static_weight_explicit" == "false" ]] || sampling_args+=(--static_horizon_sampling.terminal_weight="$terminal_static_weight")
  [[ "$dataset_mixture_sampling_explicit" == "false" ]] || sampling_args+=(--dataset_mixture_sampling.enabled="$dataset_mixture_sampling")
  [[ "$dataset_mixture_manifest_explicit" == "false" ]] || sampling_args+=(--dataset_mixture_sampling.manifest_path="$dataset_mixture_manifest")
fi
if [[ "$dataset_episodes_explicit" == "true" ]]; then
  dataset_args+=(--dataset.episodes="$dataset_episodes")
fi

peft_args=()
policy_runtime_args=()
train_expert_only="restored"
if [[ -z "$resume_checkpoint" ]]; then
  case "$finetune_mode" in
    lora)
      train_expert_only="false"
      policy_runtime_args+=(--policy.peft_train_active_modules_only=true)
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
    --policy.freeze_vision_encoder="$freeze_vision_encoder"
    --policy.train_expert_only="$train_expert_only"
    --policy.optimizer_lr="$optimizer_lr"
    --policy.lr_scheduler_type="$lr_scheduler_type"
    --policy.scheduler_warmup_steps="$scheduler_warmup_steps"
    --policy.scheduler_decay_steps="$scheduler_decay_steps"
    --policy.scheduler_decay_lr="$scheduler_decay_lr"
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
  "${dataset_args[@]}" \
  "${policy_source_args[@]}" \
  "${policy_io_args[@]}" \
  "${policy_runtime_args[@]}" \
  "${policy_mem_args[@]}" \
  "${policy_history_args[@]}" \
  "${policy_training_rtc_args[@]}" \
  --policy.push_to_hub=false \
  "${peft_args[@]}" \
  --output_dir="$output_dir" \
  --job_name="$job_name" \
  --steps="$steps" \
  --seed="$seed" \
  --batch_size="$batch_size_per_gpu" \
  --gradient_accumulation_steps="$gradient_accumulation_steps" \
  --num_workers="$num_workers" \
  "${sampling_args[@]}" \
  --log_freq="$log_freq" \
  --eval_steps="$eval_steps" \
  --max_eval_samples="$max_eval_samples" \
  --env_eval_freq=0 \
  --save_checkpoint=true \
  --save_freq="$save_freq" \
  --keep_last_checkpoints="$keep_last_checkpoints" \
  --keep_checkpoint_every_n_steps="$keep_checkpoint_every_n_steps" \
  --wandb.enable="$wandb_enable" \
  --wandb.project="$wandb_project" \
  --wandb.disable_artifact=true
)

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
if [[ -z "$resume_checkpoint" ]]; then
  echo "Training RTC:     enabled=$training_rtc, simulated_delay=$training_rtc_simulated_delay (exclusive), distribution=$training_rtc_delay_distribution"
else
  echo "Training RTC:     restored from checkpoint"
fi
if [[ -n "$resume_checkpoint" ]]; then
  echo "Deployment metadata: restored from $resume_checkpoint/pretrained_model/pi05_deployment_metadata.json"
else
  echo "B2 action:        $b2_action_representation"
  echo "State arm q/qd/gripper: $state_use_arm_joint_positions/$state_use_arm_joint_velocities/$state_use_arm_gripper_feedback"
  echo "State B2 q/qd/trunk/v/w: $state_use_b2_joint_positions/$state_use_b2_joint_velocities/$state_use_b2_trunk_pose/$state_use_b2_linear_velocity/$state_use_b2_angular_velocity"
fi
if [[ -z "$resume_checkpoint" ]]; then
  echo "Finetune mode:    $finetune_mode"
  if [[ "$enable_mem" == "true" ]]; then
    echo "Vision strategy:  MEM $mem_vit_finetune_mode (freeze_vision_encoder=$freeze_vision_encoder)"
  elif [[ "$freeze_vision_encoder" == "true" || "$finetune_mode" == "expert" ]]; then
    echo "Vision strategy:  SigLIP frozen"
  else
    echo "Vision strategy:  SigLIP $finetune_mode"
  fi
  if [[ "$finetune_mode" == "lora" ]]; then
    echo "LoRA rank/alpha:  $lora_rank/$lora_alpha"
  fi
else
  echo "Finetune mode:    restored from checkpoint"
  echo "Vision strategy:  restored from checkpoint"
fi
echo "Train expert only: $train_expert_only"
echo "GPUs:             $gpu_ids ($num_gpus process(es))"
if (( num_gpus > 1 )); then
  echo "DDP port:         $main_process_port"
fi
if [[ -z "$resume_checkpoint" ]]; then
  echo "Dataset:          $dataset_root"
  echo "Image source:     $image_source (sim probability=$mixed_sim_probability)"
else
  echo "Dataset:          restored from checkpoint unless explicitly overridden"
  echo "Image source:     restored from checkpoint unless explicitly overridden"
fi
if [[ -z "$resume_checkpoint" ]]; then
  echo "Motion sampling:  legacy enabled=$motion_balanced_sampling"
  echo "Static horizons:  enabled=$static_horizon_sampling, interior_weight=$interior_static_weight, terminal_weight=$terminal_static_weight"
  echo "Dataset mixture:  enabled=$dataset_mixture_sampling, manifest=${dataset_mixture_manifest:-none}"
else
  echo "Sampling:         restored from checkpoint unless explicitly overridden"
fi
if [[ -z "$resume_checkpoint" ]]; then
  echo "Action timing:    ${control_frequency_hz}Hz, chunk=$action_chunk_size, execute=$action_steps_to_execute (dt derived automatically)"
else
  echo "Action timing:    restored from checkpoint deployment metadata"
fi
echo "Train/eval split: eval_split=$eval_split"
echo "Steps:            $steps optimizer updates"
echo "Checkpoints:      every $save_freq steps; keep latest $keep_last_checkpoints and every ${keep_checkpoint_every_n_steps}-step milestone"
if [[ -z "$resume_checkpoint" ]]; then
  echo "Action semantics: $action_semantics_profile ($ee_target_dataset_semantics, EE source=$ee_supervision_source, mask=$ee_delta_supervision_mode, gripper=$gripper_target_representation, loss=$action_loss_schema)"
else
  echo "Action semantics: restored from checkpoint"
fi
echo "Seed:             $seed"
echo "Batch per GPU:    $batch_size_per_gpu"
echo "Gradient accum:   $gradient_accumulation_steps (computed)"
echo "Global batch:     $batch_size_per_gpu x $num_gpus GPU(s) x $gradient_accumulation_steps = $global_batch_size"
echo "Log:              $log_file"
echo "Output:           $output_dir"
echo "Watch:            tail -f '$log_file'"
