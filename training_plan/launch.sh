#!/usr/bin/env bash
set -euo pipefail

plan_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${plan_dir}/.." && pwd)"
env_file="${ENV_FILE:-${plan_dir}/.env}"
[[ -f "${env_file}" ]] || { echo "Missing ${env_file}" >&2; exit 1; }
set -a
# shellcheck disable=SC1090
source "${env_file}"
set +a

required=(
  VLA_STORAGE_ROOT STAFF1_DATASET_ROOT ALL_SCENES_DATASET_ROOT BASE_POLICY
  MEM_VIT_CHECKPOINT OUTPUT_ROOT WANDB_PROJECT TRAIN_STEPS GLOBAL_BATCH_SIZE
  BATCH_SIZE_PER_GPU EVAL_STEPS PHYSICAL_EVAL_SAMPLES SAVE_FREQ
  KEEP_LAST_CHECKPOINTS KEEP_CHECKPOINT_EVERY_N_STEPS
  B2_ACTION_REPRESENTATION Z1_ACTION_REPRESENTATION ACTION_SEMANTICS_PROFILE
  ACTION_LOSS_SCHEMA PREDICT_TASK_COMPLETE PREDICT_TASK_BLOCKED
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "Missing ${name} in ${env_file}" >&2; exit 2; }
done
[[ "${ACTION_SEMANTICS_PROFILE}" == "joint_control_arm_mode_v2" ]] || {
  echo "Plan 16 requires joint_control_arm_mode_v2 (mechanical-arm three-state output)." >&2
  exit 2
}
[[ "${PREDICT_TASK_COMPLETE}" == "true" ]] || {
  echo "Plan 16 requires PREDICT_TASK_COMPLETE=true." >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage:
  training_plan/launch.sh describe TASK_ID
  training_plan/launch.sh start TASK_ID GPU_ID [PORT]
  training_plan/launch.sh dry-run TASK_ID GPU_ID [PORT]

TASK_ID values:
  staff1_stage1_standard       staff1_stage2_standard
  staff1_stage3_standard       staff1_stage1_mem_vit
  staff1_stage2_mem_vit        staff1_stage3_mem_vit
  staff1_stage1_full_mem       staff1_stage2_full_mem
  staff1_stage3_full_mem       staff1_joint_standard
  staff1_joint_mem_vit         staff1_joint_full_mem
  all_scenes_stage1_standard   all_scenes_stage2_standard
  all_scenes_stage3_standard   all_scenes_joint_standard
EOF
}

task_id="${2:-}"
dataset_family=""
phase=""
memory_mode=""
case "${task_id}" in
  staff1_stage1_standard) dataset_family=staff1; phase=approach; memory_mode=standard ;;
  staff1_stage2_standard) dataset_family=staff1; phase=handle_press; memory_mode=standard ;;
  staff1_stage3_standard) dataset_family=staff1; phase=traversal; memory_mode=standard ;;
  staff1_stage1_mem_vit) dataset_family=staff1; phase=approach; memory_mode=mem_vit ;;
  staff1_stage2_mem_vit) dataset_family=staff1; phase=handle_press; memory_mode=mem_vit ;;
  staff1_stage3_mem_vit) dataset_family=staff1; phase=traversal; memory_mode=mem_vit ;;
  staff1_stage1_full_mem) dataset_family=staff1; phase=approach; memory_mode=full_mem ;;
  staff1_stage2_full_mem) dataset_family=staff1; phase=handle_press; memory_mode=full_mem ;;
  staff1_stage3_full_mem) dataset_family=staff1; phase=traversal; memory_mode=full_mem ;;
  staff1_joint_standard) dataset_family=staff1; phase=joint; memory_mode=standard ;;
  staff1_joint_mem_vit) dataset_family=staff1; phase=joint; memory_mode=mem_vit ;;
  staff1_joint_full_mem) dataset_family=staff1; phase=joint; memory_mode=full_mem ;;
  all_scenes_stage1_standard) dataset_family=all_scenes; phase=approach; memory_mode=standard ;;
  all_scenes_stage2_standard) dataset_family=all_scenes; phase=handle_press; memory_mode=standard ;;
  all_scenes_stage3_standard) dataset_family=all_scenes; phase=traversal; memory_mode=standard ;;
  all_scenes_joint_standard) dataset_family=all_scenes; phase=joint; memory_mode=standard ;;
  *) usage >&2; exit 2 ;;
esac

if [[ "${dataset_family}" == staff1 ]]; then
  dataset_root="${STAFF1_DATASET_ROOT}"
else
  dataset_root="${ALL_SCENES_DATASET_ROOT}"
fi
semantic_views_path="${dataset_root}/meta/semantic_views.json"
if [[ "${phase}" == joint ]]; then
  semantic_phases='["approach","handle_press","traversal"]'
  semantic_phase_weights='{"approach":1.0,"handle_press":1.0,"traversal":1.0}'
else
  semantic_phases="[\"${phase}\"]"
  semantic_phase_weights='{}'
fi

memory_args=()
case "${memory_mode}" in
  standard)
    memory_args+=(--enable-mem=false --state-action-encoding=text)
    ;;
  mem_vit)
    memory_args+=(
      --enable-mem=true --state-action-encoding=text
      --mem-random-min-num-frames=1 --mem-random-max-num-frames=1
      --mem-random-interval-sampling=false
    )
    ;;
  full_mem)
    memory_args+=(
      --enable-mem=true --state-action-encoding=continuous
      --mem-random-min-num-frames="${MEM_RANDOM_MIN_NUM_FRAMES}"
      --mem-random-max-num-frames="${MEM_RANDOM_MAX_NUM_FRAMES}"
      --mem-frame-interval-seconds="${MEM_FRAME_INTERVAL_SECONDS}"
      --mem-random-interval-sampling="${MEM_RANDOM_INTERVAL_SAMPLING}"
      --mem-global-interval-std-seconds="${MEM_GLOBAL_INTERVAL_STD_SECONDS}"
      --mem-local-interval-std-seconds="${MEM_LOCAL_INTERVAL_STD_SECONDS}"
      --mem-min-interval-seconds="${MEM_MIN_INTERVAL_SECONDS}"
      --mem-max-interval-seconds="${MEM_MAX_INTERVAL_SECONDS}"
    )
    ;;
esac

gpu_id="${3:-0}"
port="${4:-$((29500 + gpu_id))}"
common_args=(
  --gpu-id="${gpu_id}" --main-process-port="${port}"
  --job-suffix="plan16_${task_id}_arm_mode_complete"
  --dataset-root="${dataset_root}"
  --dataset-repo-id="local/b2_z1_vla_${dataset_family}_training"
  --semantic-views-path="${semantic_views_path}"
  --semantic-view-kind-weights='{"primary":1.0}'
  --semantic-phases="${semantic_phases}"
  --semantic-phase-weights="${semantic_phase_weights}"
  --random-semantic-view=true
  --base-policy="${BASE_POLICY}"
  --mem-vit-checkpoint="${MEM_VIT_CHECKPOINT}"
  --steps="${TRAIN_STEPS}"
  --batch-size-per-gpu="${BATCH_SIZE_PER_GPU}"
  --global-batch-size="${GLOBAL_BATCH_SIZE}"
  --eval-steps="${EVAL_STEPS}"
  --physical-eval-samples="${PHYSICAL_EVAL_SAMPLES}"
  --save-freq="${SAVE_FREQ}"
  --keep-last-checkpoints="${KEEP_LAST_CHECKPOINTS}"
  --keep-checkpoint-every-n-steps="${KEEP_CHECKPOINT_EVERY_N_STEPS}"
  --b2-action-representation="${B2_ACTION_REPRESENTATION}"
  --z1-action-representation="${Z1_ACTION_REPRESENTATION}"
  --action-semantics-profile="${ACTION_SEMANTICS_PROFILE}"
  --action-loss-schema="${ACTION_LOSS_SCHEMA}"
  --predict-task-complete="${PREDICT_TASK_COMPLETE}"
  --predict-task-blocked="${PREDICT_TASK_BLOCKED}"
  --motion-balanced-sampling=false
  --static-horizon-sampling=true
  --interior-static-weight="${INTERIOR_STATIC_WEIGHT}"
  --terminal-static-weight="${TERMINAL_STATIC_WEIGHT}"
  --wandb-project="${WANDB_PROJECT}"
  --output-root="${OUTPUT_ROOT}"
  "${memory_args[@]}"
)

describe() {
  printf 'task_id=%s\ndataset=%s\nphase=%s\nmemory=%s\narm_mode=true\ntask_complete=%s\ntask_blocked=%s\ngpu=%s\n' \
    "${task_id}" "${dataset_root}" "${phase}" "${memory_mode}" \
    "${PREDICT_TASK_COMPLETE}" "${PREDICT_TASK_BLOCKED}" "${gpu_id}"
}

case "${1:-}" in
  describe) describe ;;
  dry-run)
    describe
    exec bash "${repo_root}/train_vla_pi05.sh" "${common_args[@]}" --dry-run=true
    ;;
  start)
    [[ -d "${dataset_root}" ]] || { echo "Missing dataset: ${dataset_root}" >&2; exit 1; }
    [[ -f "${semantic_views_path}" ]] || { echo "Missing semantic views: ${semantic_views_path}" >&2; exit 1; }
    [[ -d "${BASE_POLICY}" ]] || { echo "Missing base policy: ${BASE_POLICY}" >&2; exit 1; }
    if [[ "${memory_mode}" != standard ]]; then
      [[ -f "${MEM_VIT_CHECKPOINT}" ]] || { echo "Missing MEM ViT: ${MEM_VIT_CHECKPOINT}" >&2; exit 1; }
    fi
    describe
    exec bash "${repo_root}/train_vla_pi05.sh" "${common_args[@]}"
    ;;
  *) usage >&2; exit 2 ;;
esac
