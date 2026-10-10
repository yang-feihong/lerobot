#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

# Portable LoRA checkpoints resolve their immutable base model/tokenizer through
# the same registry used by the regular open-loop entrypoint.  Keep the
# standard benchmark self-contained so moving a checkpoint between machines
# does not silently change (or prevent) evaluation.
export LEROBOT_ARTIFACT_REGISTRY="${LEROBOT_ARTIFACT_REGISTRY:-${repo_root}/deployment/base_artifact_registry.json}"

protocol_path="$repo_root/evaluation_protocols/b2_z1_staff1_stage1_full_horizon_v1.json"
policy_path=""
dataset_root="${B2_Z1_STAGE1_BENCHMARK_DATASET_ROOT:-}"
output_dir=""
device="cuda"
gpu_id="0"
gpu_id_explicit="false"
inherited_cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-}"
batch_size=""
num_workers="4"
plot_workers="2"
print_config="false"

while (( $# > 0 )); do
  case "$1" in
    --policy-path=*) policy_path="${1#*=}" ;;
    --dataset-root=*) dataset_root="${1#*=}" ;;
    --output-dir=*) output_dir="${1#*=}" ;;
    --gpu-id=*) gpu_id="${1#*=}"; gpu_id_explicit="true" ;;
    --device=*) device="${1#*=}" ;;
    --batch-size=*) batch_size="${1#*=}" ;;
    --num-workers=*) num_workers="${1#*=}" ;;
    --plot-workers=*) plot_workers="${1#*=}" ;;
    --protocol=*) protocol_path="${1#*=}" ;;
    --print-config) print_config="true" ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if [[ "$device" == cuda* && "$gpu_id_explicit" == "true" && -n "$inherited_cuda_visible_devices" \
      && "$inherited_cuda_visible_devices" != "$gpu_id" ]]; then
  echo "Conflicting GPU selection: inherited CUDA_VISIBLE_DEVICES=$inherited_cuda_visible_devices, --gpu-id=$gpu_id" >&2
  echo "Specify the physical GPU only once, or make both values identical." >&2
  exit 2
fi

[[ -n "$policy_path" ]] || { echo "--policy-path is required" >&2; exit 2; }
[[ -n "$dataset_root" ]] || {
  echo "--dataset-root or B2_Z1_STAGE1_BENCHMARK_DATASET_ROOT is required" >&2
  exit 2
}
[[ -n "$output_dir" ]] || { echo "--output-dir is required" >&2; exit 2; }
[[ -f "$protocol_path" ]] || { echo "Protocol not found: $protocol_path" >&2; exit 2; }

mapfile -t protocol < <(python3 - "$protocol_path" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
for key in (
    "protocol_id", "dataset_repo_id", "episodes", "max_episodes",
    "max_frames_per_episode", "frame_stride", "max_chunk_plots_per_episode",
    "chunk_plot_stride", "task", "task_variant", "seed",
    "inference_batch_size",
):
    print(d[key])
print("true" if d["include_onset_windows"] else "false")
print(d["dataset_info_sha256"])
print(d["dataset_total_episodes"])
print(d["dataset_total_frames"])
PY
)

protocol_id="${protocol[0]}"
dataset_repo_id="${protocol[1]}"
episodes="${protocol[2]}"
max_episodes="${protocol[3]}"
max_frames="${protocol[4]}"
frame_stride="${protocol[5]}"
max_chunk_plots="${protocol[6]}"
chunk_plot_stride="${protocol[7]}"
task="${protocol[8]}"
task_variant="${protocol[9]}"
seed="${protocol[10]}"
protocol_batch_size="${protocol[11]}"
include_onset_windows="${protocol[12]}"
expected_dataset_info_sha256="${protocol[13]}"
expected_dataset_total_episodes="${protocol[14]}"
expected_dataset_total_frames="${protocol[15]}"

if [[ -z "$batch_size" ]]; then
  batch_size="$protocol_batch_size"
elif [[ "$batch_size" != "$protocol_batch_size" ]]; then
  echo "Protocol $protocol_id requires --batch-size=$protocol_batch_size, got $batch_size" >&2
  exit 2
fi

dataset_info_path="$dataset_root/meta/info.json"
[[ -f "$dataset_info_path" ]] || {
  echo "Dataset info not found: $dataset_info_path" >&2
  exit 2
}
actual_dataset_info_sha256="$(sha256sum "$dataset_info_path" | awk '{print $1}')"
[[ "$actual_dataset_info_sha256" == "$expected_dataset_info_sha256" ]] || {
  echo "Dataset identity mismatch for protocol $protocol_id" >&2
  echo "expected meta/info.json sha256: $expected_dataset_info_sha256" >&2
  echo "actual   meta/info.json sha256: $actual_dataset_info_sha256" >&2
  exit 2
}
python3 - "$dataset_info_path" "$expected_dataset_total_episodes" "$expected_dataset_total_frames" <<'PY'
import json, sys
info = json.load(open(sys.argv[1], encoding="utf-8"))
expected_episodes, expected_frames = map(int, sys.argv[2:])
actual = (int(info["total_episodes"]), int(info["total_frames"]))
expected = (expected_episodes, expected_frames)
if actual != expected:
    raise SystemExit(f"Dataset size mismatch: expected episodes/frames={expected}, actual={actual}")
PY

if [[ "$print_config" == "true" ]]; then
  printf 'protocol_id=%s\npolicy_path=%s\ndataset_repo_id=%s\ndataset_root=%s\noutput_dir=%s\nepisodes=%s\nframe_stride=%s\ntask=%s\n' \
    "$protocol_id" "$policy_path" "$dataset_repo_id" "$dataset_root" "$output_dir" \
    "$episodes" "$frame_stride" "$task"
  exit 0
fi

[[ ! -e "$output_dir" ]] || {
  echo "Refusing to overwrite existing standard-evaluation output: $output_dir" >&2
  exit 2
}
if [[ -n "${LEROBOT_EVAL_PYTHON:-}" ]]; then
  [[ -x "$LEROBOT_EVAL_PYTHON" ]] || {
    echo "LEROBOT_EVAL_PYTHON is not executable: $LEROBOT_EVAL_PYTHON" >&2
    exit 2
  }
  python_runner=("$LEROBOT_EVAL_PYTHON")
else
  python_runner=(uv run python)
fi

cmd=(
  "${python_runner[@]}" -m lerobot.scripts.openloop_vla_eval
  --policy-path "$policy_path"
  --dataset-repo-id "$dataset_repo_id"
  --dataset-root "$dataset_root"
  --output-dir "$output_dir"
  --split all
  --episodes "$episodes"
  --max-episodes "$max_episodes"
  --max-frames-per-episode "$max_frames"
  --frame-stride "$frame_stride"
  --max-chunk-plots-per-episode "$max_chunk_plots"
  --chunk-plot-stride "$chunk_plot_stride"
  --batch-size "$batch_size"
  --num-workers "$num_workers"
  --device "$device"
  --task-variant "$task_variant"
  --task-override "$task"
  --seed "$seed"
  --deterministic-flow-noise
  --no-write-plots
  --plot-workers "$plot_workers"
)
if [[ "$include_onset_windows" == "false" ]]; then
  cmd+=(--no-include-onset-windows)
fi

if [[ "$device" == cuda* ]]; then
  CUDA_VISIBLE_DEVICES="$gpu_id" "${cmd[@]}"
else
  "${cmd[@]}"
fi

python3 - "$protocol_path" "$output_dir/standard_full_horizon_metrics.json" <<'PY'
import json, sys
protocol = json.load(open(sys.argv[1], encoding="utf-8"))
result = json.load(open(sys.argv[2], encoding="utf-8"))
if result["horizon_steps"] != protocol["required_horizon_steps"]:
    raise SystemExit(
        f"Unexpected horizon: {result['horizon_steps']} != {protocol['required_horizon_steps']}"
    )
expected_anchors = protocol["expected_selected_anchor_count"]
expected_steps = protocol["expected_valid_action_step_count"]
contract = result["comparison_contract"]
if contract["evaluated_anchor_count"] != expected_anchors:
    raise SystemExit(
        f"Unexpected anchor count: {contract['evaluated_anchor_count']} != {expected_anchors}"
    )
if contract["evaluated_action_step_count"] != expected_steps:
    raise SystemExit(
        "Unexpected valid action-step count: "
        f"{contract['evaluated_action_step_count']} != {expected_steps}"
    )
result["protocol_id"] = protocol["protocol_id"]
open(sys.argv[2], "w", encoding="utf-8").write(json.dumps(result, indent=2) + "\n")
PY

echo "Standard benchmark complete: $output_dir/standard_full_horizon_metrics.json"
