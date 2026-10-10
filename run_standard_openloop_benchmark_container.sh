#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image="${LEROBOT_EVAL_IMAGE:-huggingface/lerobot-gpu:latest}"
host_data_root="${LEROBOT_EVAL_DATA_ROOT:-}"
gpu_id="0"
container_name=""
detach="false"

usage() {
  cat <<'EOF'
Usage:
  LEROBOT_EVAL_DATA_ROOT=/path/to/data \
    ./run_standard_openloop_benchmark_container.sh \
      [--gpu-id=N] [--container-name=NAME] [--detach] -- \
      --policy-path=/data/... --dataset-root=/data/... --output-dir=/data/... \
      [run_standard_openloop_benchmark.sh arguments]

The wrapper fixes the container runtime contract used by the standard open-loop
benchmark. Paths passed after `--` are container paths: repository at
/workspace/lerobot and the host data root at /data.
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --gpu-id=*) gpu_id="${1#*=}" ;;
    --container-name=*) container_name="${1#*=}" ;;
    --detach) detach="true" ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown wrapper argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[[ -n "$host_data_root" ]] || {
  echo "LEROBOT_EVAL_DATA_ROOT is required" >&2
  exit 2
}
[[ -d "$host_data_root" ]] || {
  echo "Data root does not exist: $host_data_root" >&2
  exit 2
}
(( $# > 0 )) || {
  echo "Arguments for run_standard_openloop_benchmark.sh are required after --" >&2
  exit 2
}
[[ "$gpu_id" =~ ^[0-9]+$ ]] || {
  echo "--gpu-id must be a non-negative integer, got: $gpu_id" >&2
  exit 2
}

docker_args=(
  run
  --user 0:0
  --gpus "device=${gpu_id}"
  --shm-size=16g
  --volume "${host_data_root}:/data"
  --volume "${repo_root}:/workspace/lerobot:ro"
  --workdir /workspace/lerobot
  --env HOME=/data/lerobot/home
  --env HF_HOME=/data/lerobot/cache/huggingface
  --env PYTHONPATH=/workspace/lerobot/src
  --env LEROBOT_EVAL_PYTHON=/lerobot/.venv/bin/python
)

if [[ -n "$container_name" ]]; then
  docker_args+=(--name "$container_name")
fi
if [[ "$detach" == "true" ]]; then
  docker_args+=(-d)
else
  docker_args+=(--rm)
fi

# Only the selected physical GPU is visible inside the container, hence the
# benchmark always addresses container-local GPU 0.
exec docker "${docker_args[@]}" "$image" \
  bash run_standard_openloop_benchmark.sh --gpu-id=0 "$@"
