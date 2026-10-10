#!/usr/bin/env bash
set -euo pipefail

plan_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${plan_dir}/.." && pwd)"
env_file="${ENV_FILE:-${plan_dir}/.env}"
semantic_presets_file="${SEMANTIC_TRAINING_PRESETS:-${plan_dir}/semantic_training_presets.json}"
training_jobs_file="${TRAINING_JOBS_FILE:-${plan_dir}/training_jobs.json}"
artifacts_lock="${ARTIFACTS_LOCK_FILE:-${plan_dir}/artifacts.lock}"
[[ -f "${env_file}" ]] || { echo "Missing ${env_file}" >&2; exit 1; }
[[ -f "${semantic_presets_file}" ]] || { echo "Missing ${semantic_presets_file}" >&2; exit 1; }
[[ -f "${training_jobs_file}" ]] || { echo "Missing ${training_jobs_file}" >&2; exit 1; }
[[ -f "${artifacts_lock}" ]] || { echo "Missing ${artifacts_lock}" >&2; exit 1; }
set -a
# shellcheck disable=SC1090
source "${env_file}"
# shellcheck disable=SC1090
source "${artifacts_lock}"
set +a
export LEROBOT_ARTIFACT_REGISTRY="${LEROBOT_ARTIFACT_REGISTRY:-${plan_dir}/base_artifact_registry.json}"

required=(
  VLA_STORAGE_ROOT BASE_POLICY
  MEM_VIT_CHECKPOINT OUTPUT_ROOT WANDB_PROJECT TRAIN_STEPS GLOBAL_BATCH_SIZE
  BATCH_SIZE_PER_GPU EVAL_STEPS PHYSICAL_EVAL_SAMPLES SAVE_FREQ
  KEEP_LAST_CHECKPOINTS KEEP_CHECKPOINT_EVERY_N_STEPS
  B2_ACTION_REPRESENTATION Z1_ACTION_REPRESENTATION ACTION_SEMANTICS_PROFILE
  ACTION_LOSS_SCHEMA ACTION_CONTINUOUS_GROUP_WEIGHT ACTION_DISCRETE_GROUP_WEIGHT
  PREDICT_TASK_COMPLETE PREDICT_TASK_BLOCKED
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "Missing ${name} in ${env_file}" >&2; exit 2; }
done
[[ "${ACTION_SEMANTICS_PROFILE}" == "joint_control_arm_mode_v2" ]] || {
  echo "This training plan requires joint_control_arm_mode_v2 (mechanical-arm three-state output)." >&2
  exit 2
}
[[ "${PREDICT_TASK_COMPLETE}" == "true" ]] || {
  echo "This training plan requires PREDICT_TASK_COMPLETE=true." >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage:
  training_plan/launch.sh describe TASK_ID
  training_plan/launch.sh start TASK_ID GPU_ID [PORT]
  training_plan/launch.sh dry-run TASK_ID GPU_ID [PORT]
  training_plan/launch.sh resume TASK_ID GPU_ID CHECKPOINT [PORT]
  training_plan/launch.sh resume-dry-run TASK_ID GPU_ID CHECKPOINT [PORT]

TASK_ID values are defined in training_plan/training_jobs.json:
EOF
  python3 "${plan_dir}/semantic_plan.py" \
    --presets "${semantic_presets_file}" --jobs "${training_jobs_file}" list | sed 's/^/  /'
}

task_id="${2:-}"
[[ -n "${task_id}" ]] || { usage >&2; exit 2; }
mapfile -t semantic_task_spec < <(
  python3 "${plan_dir}/semantic_plan.py" \
    --presets "${semantic_presets_file}" --jobs "${training_jobs_file}" resolve "${task_id}"
)
[[ "${#semantic_task_spec[@]}" -eq 8 ]] || {
  echo "Failed to resolve semantic task ${task_id} from ${training_jobs_file}" >&2
  exit 2
}
dataset_family="${semantic_task_spec[0]}"
memory_mode="${semantic_task_spec[1]}"
semantic_phases="${semantic_task_spec[2]}"
semantic_phase_weights="${semantic_task_spec[3]}"
semantic_state_instruction_matrix="${semantic_task_spec[4]}"
configured_node="${semantic_task_spec[5]}"
configured_gpu_ids="${semantic_task_spec[6]}"
configured_port="${semantic_task_spec[7]}"
semantic_source_weights='{}'

if [[ "${dataset_family}" == staff1 ]]; then
  [[ -n "${STAFF1_DATASET_ROOT:-}" ]] || {
    echo "Task ${task_id} requires STAFF1_DATASET_ROOT in ${env_file}." >&2
    exit 2
  }
  dataset_root="${STAFF1_DATASET_ROOT}"
else
  [[ -n "${ALL_SCENES_DATASET_ROOT:-}" ]] || {
    echo "Task ${task_id} requires ALL_SCENES_DATASET_ROOT in ${env_file}." >&2
    exit 2
  }
  dataset_root="${ALL_SCENES_DATASET_ROOT}"
fi
semantic_views_path="${dataset_root}/meta/semantic_views.json"

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

command="${1:-}"
gpu_id="${3:-${configured_gpu_ids}}"
if [[ "${command}" == resume || "${command}" == resume-dry-run ]]; then
  port="${5:-${configured_port}}"
else
  port="${4:-${configured_port}}"
fi
[[ "${gpu_id}" == "${configured_gpu_ids}" ]] || {
  echo "Task ${task_id} requires GPUs ${configured_gpu_ids}, got ${gpu_id}." >&2
  exit 2
}
[[ "${port}" == "${configured_port}" ]] || {
  echo "Task ${task_id} requires DDP port ${configured_port}, got ${port}." >&2
  exit 2
}
common_args=(
  --gpu-id="${gpu_id}" --main-process-port="${port}"
  --job-suffix="training_job_${task_id}_arm_mode_complete"
  --dataset-root="${dataset_root}"
  --dataset-repo-id="local/b2_z1_vla_${dataset_family}_training"
  --semantic-views-path="${semantic_views_path}"
  --semantic-state-instruction-matrix="${semantic_state_instruction_matrix}"
  --semantic-phases="${semantic_phases}"
  --semantic-phase-weights="${semantic_phase_weights}"
  --semantic-source-weights="${semantic_source_weights}"
  --random-semantic-view=true
  --base-policy="${BASE_POLICY}"
  --mem-vit-checkpoint="${MEM_VIT_CHECKPOINT}"
  --policy-base-artifact-id="${BASE_POLICY_ARTIFACT_ID}"
  --policy-base-sha256="${BASE_POLICY_SHA256}"
  --mem-vit-artifact-id="${MEM_VIT_ARTIFACT_ID}"
  --mem-vit-sha256="${MEM_VIT_SHA256}"
  --policy-tokenizer-artifact-id="${POLICY_TOKENIZER_ARTIFACT_ID}"
  --policy-tokenizer-sha256="${POLICY_TOKENIZER_SHA256}"
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
  --action-continuous-group-weight="${ACTION_CONTINUOUS_GROUP_WEIGHT}"
  --action-discrete-group-weight="${ACTION_DISCRETE_GROUP_WEIGHT}"
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
  printf 'task_id=%s\ndataset=%s\nmemory=%s\nnode=%s\narm_mode=true\ntask_complete=%s\ntask_blocked=%s\ngpus=%s\n' \
    "${task_id}" "${dataset_root}" "${memory_mode}" \
    "${configured_node}" "${PREDICT_TASK_COMPLETE}" "${PREDICT_TASK_BLOCKED}" "${gpu_id}"
  semantic_describe_args=(
    --presets "${semantic_presets_file}" --jobs "${training_jobs_file}" describe "${task_id}"
  )
  if [[ -f "${semantic_views_path}" ]]; then
    semantic_describe_args+=(--sidecar "${semantic_views_path}")
  fi
  python3 "${plan_dir}/semantic_plan.py" "${semantic_describe_args[@]}"
}

validate_resume_checkpoint() {
  local checkpoint="$1" relative
  local required=(
    pretrained_model/train_config.json
    pretrained_model/config.json
    training_state/optimizer_state.safetensors
    training_state/optimizer_param_groups.json
    training_state/rng_state.safetensors
    training_state/scheduler_state.json
    training_state/training_step.json
  )
  for relative in "${required[@]}"; do
    [[ -s "${checkpoint}/${relative}" ]] || {
      echo "Incomplete resume checkpoint (missing or empty ${checkpoint}/${relative})." >&2
      return 1
    }
  done
  if [[ ! -s "${checkpoint}/pretrained_model/model.safetensors" && ! -s "${checkpoint}/pretrained_model/adapter_model.safetensors" ]]; then
    echo "Incomplete resume checkpoint (no policy weight file under ${checkpoint}/pretrained_model)." >&2
    return 1
  fi
}

case "${command}" in
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
  resume|resume-dry-run)
    checkpoint="${4:-}"
    [[ -n "${checkpoint}" ]] || { echo "Missing checkpoint path." >&2; exit 2; }
    checkpoint="$(readlink -f "${checkpoint}")"
    validate_resume_checkpoint "${checkpoint}"
    resume_config="${checkpoint}/pretrained_model/train_config.json"
    if ! python3 - "${resume_config}" "${dataset_root}" "${semantic_phases}" "${semantic_phase_weights}" \
      "${semantic_state_instruction_matrix}" "${memory_mode}" \
      "${B2_ACTION_REPRESENTATION}" "${Z1_ACTION_REPRESENTATION}" \
      "${PREDICT_TASK_COMPLETE}" "${PREDICT_TASK_BLOCKED}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    config = json.load(stream)
run_id = config.get("wandb", {}).get("run_id")
dataset = config.get("dataset", {})
policy = config.get("policy", {})
expected_phases = json.loads(sys.argv[3])
expected_phase_weights = json.loads(sys.argv[4])
expected_state_instruction_matrix = json.loads(sys.argv[5])
expected_encoding = "continuous" if sys.argv[6] == "full_mem" else "text"
expected = {
    "wandb.run_id": isinstance(run_id, str) and bool(run_id.strip()),
    "dataset.root": dataset.get("root") == sys.argv[2],
    "dataset.semantic_phases": dataset.get("semantic_phases") == expected_phases,
    "dataset.semantic_phase_weights": dataset.get("semantic_phase_weights") == expected_phase_weights,
    "dataset.semantic_state_instruction_matrix": dataset.get("semantic_state_instruction_matrix") == expected_state_instruction_matrix,
    "policy.state_action_encoding": policy.get("state_action_encoding") == expected_encoding,
    "policy.b2_action_representation": policy.get("b2_action_representation") == sys.argv[7],
    "policy.z1_action_representation": policy.get("z1_action_representation") == sys.argv[8],
    "policy.action_predict_arm_teleop_inactive": policy.get("action_predict_arm_teleop_inactive") is True,
    "policy.action_predict_arm_reset": policy.get("action_predict_arm_reset") is True,
    "policy.action_predict_task_complete": policy.get("action_predict_task_complete") is (sys.argv[9] == "true"),
    "policy.action_predict_task_blocked": policy.get("action_predict_task_blocked") is (sys.argv[10] == "true"),
}
failed = [name for name, valid in expected.items() if not valid]
if failed:
    print("checkpoint configuration mismatch: " + ", ".join(failed), file=sys.stderr)
    raise SystemExit(1)
PY
    then
      echo "Resume checkpoint does not match task ${task_id}: ${resume_config}" >&2
      exit 1
    fi
    describe
    resume_args=(
      --gpu-id="${gpu_id}"
      --main-process-port="${port}"
      --resume-checkpoint="${checkpoint}"
      --resume-with-updated-dataset=false
    )
    if [[ "${command}" == resume-dry-run ]]; then
      resume_args+=(--dry-run=true)
    fi
    # No --resume-new-run-name is passed: train_vla_pi05.sh therefore restores
    # wandb.run_id and uses resume="must" for the original W&B run.
    exec bash "${repo_root}/train_vla_pi05.sh" "${resume_args[@]}"
    ;;
  *) usage >&2; exit 2 ;;
esac
