#!/usr/bin/env bash
set -euo pipefail

plan_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${plan_dir}/.." && pwd)"
env_file="${ENV_FILE:-${plan_dir}/.env}"
lock_file="${plan_dir}/artifacts.lock"
[[ -f "${env_file}" ]] || { echo "Missing ${env_file}" >&2; exit 1; }
[[ -f "${lock_file}" ]] || { echo "Missing ${lock_file}" >&2; exit 1; }
set -a
# shellcheck disable=SC1090
source "${env_file}"
# shellcheck disable=SC1090
source "${lock_file}"
set +a

required=(
  CLUSTER_SSH_CONFIG CLUSTER_SSH_CONTROL_PATH CLUSTER_CONTAINER_NAME CLUSTER_IMAGE_BWLIMIT_KB
  CLUSTER_LOCAL_CHECKPOINT_ROOT CLUSTER_F_MIN_FREE_GIB CLUSTER_F_TARGET_FREE_GIB
  CLUSTER_CU124_IMAGE CLUSTER_CU124_IMAGE_ID
  CLUSTER_CU124_IMAGE_ARCHIVE CLUSTER_F_IMAGE CLUSTER_F_IMAGE_ID
  CLUSTER_F_IMAGE_ARCHIVE BASE_POLICY_SHA256 MEM_VIT_SHA256
  STAFF1_EPISODES STAFF1_VIDEOS STAFF1_INFO_SHA256 STAFF1_SEMANTIC_SHA256
  STAFF1_BOUNDARIES_SHA256
  CLUSTER_F_SSH_HOST CLUSTER_F_DATA_ROOT CLUSTER_F_SOURCE_ROOT CLUSTER_F_GPU_COUNT CLUSTER_F_CONTAINER_USER
  CLUSTER_WB2_SSH_HOST CLUSTER_WB2_DATA_ROOT CLUSTER_WB2_SOURCE_ROOT CLUSTER_WB2_GPU_COUNT CLUSTER_WB2_CONTAINER_USER
  CLUSTER_WB3_SSH_HOST CLUSTER_WB3_DATA_ROOT CLUSTER_WB3_SOURCE_ROOT CLUSTER_WB3_GPU_COUNT CLUSTER_WB3_CONTAINER_USER
  CLUSTER_WB4_SSH_HOST CLUSTER_WB4_DATA_ROOT CLUSTER_WB4_SOURCE_ROOT CLUSTER_WB4_GPU_COUNT CLUSTER_WB4_CONTAINER_USER
  CLUSTER_WB5_SSH_HOST CLUSTER_WB5_DATA_ROOT CLUSTER_WB5_SOURCE_ROOT CLUSTER_WB5_GPU_COUNT CLUSTER_WB5_CONTAINER_USER
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "Missing ${name} in ${env_file}" >&2; exit 2; }
done
[[ -f "${CLUSTER_SSH_CONFIG}" ]] || { echo "Missing SSH config: ${CLUSTER_SSH_CONFIG}" >&2; exit 1; }

nodes=(f wb2 wb3 wb4 wb5)
semantic_plan_args=(
  python3 "${plan_dir}/semantic_plan.py"
  --presets "${plan_dir}/semantic_training_presets.json"
  --jobs "${plan_dir}/training_jobs.json"
)

node_profile() {
  local node="$1"
  case "${node}" in
    f)
      ssh_host="${CLUSTER_F_SSH_HOST}"
      host_data="${CLUSTER_F_DATA_ROOT}"
      host_source="${CLUSTER_F_SOURCE_ROOT}"
      image="${CLUSTER_F_IMAGE}"
      image_id="${CLUSTER_F_IMAGE_ID}"
      image_archive="${CLUSTER_F_IMAGE_ARCHIVE}"
      gpu_count="${CLUSTER_F_GPU_COUNT}"
      container_user="${CLUSTER_F_CONTAINER_USER}"
      ;;
    wb2|wb3|wb4|wb5)
      local prefix="CLUSTER_${node^^}"
      local ssh_var="${prefix}_SSH_HOST" data_var="${prefix}_DATA_ROOT"
      local source_var="${prefix}_SOURCE_ROOT" gpu_var="${prefix}_GPU_COUNT"
      local user_var="${prefix}_CONTAINER_USER"
      ssh_host="${!ssh_var}"
      host_data="${!data_var}"
      host_source="${!source_var}"
      image="${CLUSTER_CU124_IMAGE}"
      image_id="${CLUSTER_CU124_IMAGE_ID}"
      image_archive="${CLUSTER_CU124_IMAGE_ARCHIVE}"
      gpu_count="${!gpu_var}"
      container_user="${!user_var}"
      ;;
    *) echo "Unknown node: ${node}" >&2; exit 2 ;;
  esac
}

node_tasks() {
  "${semantic_plan_args[@]}" assignments | awk -F '\t' -v node="$1" '$2 == node {print $1}'
}

plan_nodes() {
  "${semantic_plan_args[@]}" assignments | awk -F '\t' '!seen[$2]++ {print $2}'
}

task_resources() {
  local wanted="$1"
  "${semantic_plan_args[@]}" assignments | \
    awk -F '\t' -v task="${wanted}" '$1 == task {print $3, $4; found=1} END {exit !found}'
}

validate_plan_resources() {
  local task node gpu_csv port gpu
  local -A occupied=() ports=()
  while IFS=$'\t' read -r task node gpu_csv port; do
    node_profile "${node}"
    IFS=',' read -r -a gpu_ids <<<"${gpu_csv}"
    [[ "${#gpu_ids[@]}" -eq 4 ]] || {
      echo "${task}: expected exactly four GPUs, got ${gpu_csv}" >&2
      return 1
    }
    for gpu in "${gpu_ids[@]}"; do
      (( gpu < gpu_count )) || {
        echo "${task}: GPU ${gpu} is outside ${node}'s ${gpu_count}-GPU range" >&2
        return 1
      }
      [[ -z "${occupied[${node}:${gpu}]:-}" ]] || {
        echo "${task}: GPU ${node}:${gpu} overlaps ${occupied[${node}:${gpu}]}" >&2
        return 1
      }
      occupied["${node}:${gpu}"]="${task}"
    done
    [[ -z "${ports[${node}:${port}]:-}" ]] || {
      echo "${task}: DDP port ${node}:${port} overlaps ${ports[${node}:${port}]}" >&2
      return 1
    }
    ports["${node}:${port}"]="${task}"
  done < <("${semantic_plan_args[@]}" assignments)
}

ssh_node() {
  local node="$1"; shift
  node_profile "${node}"
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    ssh -n -F "${CLUSTER_SSH_CONFIG}" \
      -o ControlMaster=auto -o ControlPersist=120 -o "ControlPath=${CLUSTER_SSH_CONTROL_PATH}" \
      "${ssh_host}" "$@"
}

snapshot_file_list() {
  {
    git -C "${repo_root}" ls-files -z
    printf '%s\0' training_plan/cluster.sh training_plan/storage_guard.sh \
      training_plan/artifacts.lock training_plan/base_artifact_registry.json training_plan/README_zh.md \
      training_plan/semantic_plan.py training_plan/semantic_training_presets.json \
      training_plan/training_jobs.json
  } | sort -zu
}

code_commit() {
  local head content
  head="$(git -C "${repo_root}" rev-parse --short=8 HEAD)"
  content="$(cd "${repo_root}" && snapshot_file_list | xargs -0 sha256sum | sha256sum | cut -c1-8)"
  printf '%s-%s\n' "${head}" "${content}"
}

remote_code_dir() {
  local node="$1"
  node_profile "${node}"
  printf '%s/lerobot_main_%s\n' "${host_source}" "$(code_commit)"
}

container_code_dir() {
  printf '/workspace/source/lerobot_main_%s\n' "$(code_commit)"
}

usage() {
  cat <<'EOF'
Usage:
  training_plan/cluster.sh describe
  training_plan/cluster.sh check [all|f|wb2|wb3|wb4|wb5]
  training_plan/cluster.sh sync-code [all|f|wb2|wb3|wb4|wb5]
  training_plan/cluster.sh prepare [all|f|wb2|wb3|wb4|wb5]
  training_plan/cluster.sh dry-run-plan
  training_plan/cluster.sh resume-dry-run-plan
  training_plan/cluster.sh start-plan
  training_plan/cluster.sh resume-plan
  training_plan/cluster.sh status
  training_plan/cluster.sh archive-f-checkpoints

prepare is idempotent. It installs the exact image only when missing, uploads the
tracked Git snapshot to a versioned directory, validates data/model integrity,
and runs a disposable CUDA probe. Training uses one container per task, with the
trainer as PID 1; it never relies on docker exec (unsupported by Mizar runtime).
dry-run-plan and resume-dry-run-plan use disposable CPU-only containers. They
exercise the same image, mounts, code snapshot, task mapping, and launcher as a
real run without starting training or allocating a GPU.
EOF
}

select_nodes() {
  local selected="${1:-all}"
  if [[ "${selected}" == all ]]; then
    printf '%s\n' "${nodes[@]}"
  else
    node_profile "${selected}" >/dev/null
    printf '%s\n' "${selected}"
  fi
}

describe() {
  local node gpu port task
  local -a task_list
  printf 'commit=%s container=%s\n' "$(code_commit)" "${CLUSTER_CONTAINER_NAME}"
  for node in "${nodes[@]}"; do
    node_profile "${node}"
    printf '%s host=%s gpus=%s image=%s image_id=%s data=%s source=%s\n' \
      "${node}" "${ssh_host}" "${gpu_count}" "${image}" "${image_id}" \
      "${host_data}" "${host_source}"
    mapfile -t task_list < <(node_tasks "${node}")
    for task in "${task_list[@]}"; do
      read -r gpu port < <(task_resources "${task}")
      printf '  gpus=%s port=%s task=%s\n' "${gpu}" "${port}" "${task}"
    done
  done
}

sync_code() {
  local node="$1" commit target tmp
  commit="$(code_commit)"
  node_profile "${node}"
  target="${host_source}/lerobot_main_${commit}"
  if ssh_node "${node}" "test -f '${target}/.source_commit' && test \"\$(cat '${target}/.source_commit')\" = '${commit}' && cd '${target}' && sha256sum -c .source_manifest.sha256 >/dev/null 2>&1"; then
    ssh_node "${node}" "chown -R '${container_user}' '${target}'"
    return
  fi
  tmp="${target}.partial.$$"
  ssh_node "${node}" "rm -rf '${tmp}'; mkdir -p '${tmp}'"
  (cd "${repo_root}" && tar --null -T <(snapshot_file_list) -cf -) | \
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
      ssh -F "${CLUSTER_SSH_CONFIG}" \
        -o ControlMaster=auto -o ControlPersist=120 -o "ControlPath=${CLUSTER_SSH_CONTROL_PATH}" \
        "${ssh_host}" "tar -xf - -C '${tmp}'"
  ssh_node "${node}" "cd '${tmp}'; find . -type f ! -name '.source_commit' ! -name '.source_manifest.sha256' -print0 | sort -z | xargs -0 sha256sum > .source_manifest.sha256; printf '%s\\n' '${commit}' > .source_commit; cd /; rm -rf '${target}'; mv '${tmp}' '${target}'"
  ssh_node "${node}" "chown -R '${container_user}' '${target}'"
}

ensure_image() {
  local node="$1" actual remote_archive
  node_profile "${node}"
  actual="$(ssh_node "${node}" "docker image inspect '${image}' --format '{{.Id}}' 2>/dev/null || true")"
  [[ "${actual}" == "${image_id}" ]] && return
  [[ -f "${image_archive}" ]] || { echo "Missing local image archive: ${image_archive}" >&2; exit 1; }
  remote_archive="${host_data}/lerobot/images/$(basename "${image_archive}")"
  ssh_node "${node}" "mkdir -p '$(dirname "${remote_archive}")'"
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    rsync -a --partial --bwlimit="${CLUSTER_IMAGE_BWLIMIT_KB}" \
      -e "ssh -F ${CLUSTER_SSH_CONFIG} -o ControlMaster=auto -o ControlPersist=120 -o ControlPath=${CLUSTER_SSH_CONTROL_PATH}" \
      "${image_archive}" "${ssh_host}:${remote_archive}"
  ssh_node "${node}" "gzip -dc '${remote_archive}' | docker load >/dev/null; rm -f '${remote_archive}'"
  actual="$(ssh_node "${node}" "docker image inspect '${image}' --format '{{.Id}}'")"
  [[ "${actual}" == "${image_id}" ]] || {
    echo "Image ID mismatch on ${node}: ${actual}, expected ${image_id}" >&2
    exit 1
  }
}

verify_large_file() {
  local node="$1" path="$2" expected="$3" key marker log fingerprint attempt
  node_profile "${node}"
  key="$(basename "${path}").${expected}"
  marker="${host_data}/lerobot/integrity/${key}.ok"
  log="${host_data}/lerobot/integrity/${key}.log"
  fingerprint="$(ssh_node "${node}" "stat -c '%s:%Y' '${path}'")"
  if ssh_node "${node}" "test -f '${marker}' && test \"\$(cat '${marker}')\" = '${fingerprint}'"; then
    return
  fi
  ssh_node "${node}" "mkdir -p '${host_data}/lerobot/integrity'; if ! pgrep -f '[s]ha256sum ${path}' >/dev/null; then nohup sh -c 'before=\$(stat -c \"%s:%Y\" \"${path}\"); actual=\$(sha256sum \"${path}\" | cut -d\" \" -f1); after=\$(stat -c \"%s:%Y\" \"${path}\"); test \"\$before\" = \"\$after\" && test \"\$actual\" = \"${expected}\" && printf \"%s\\n\" \"\$after\" >\"${marker}\"' >'${log}' 2>&1 </dev/null & fi"
  # The base policy is roughly 14 GiB. Slow or busy data disks can need more
  # than one minute for a cold SHA256 pass, so allow five minutes before
  # reporting a verification timeout. Subsequent checks use the size/mtime
  # fingerprint marker and return immediately.
  for attempt in {1..60}; do
    if ssh_node "${node}" "test -f '${marker}' && test \"\$(cat '${marker}')\" = '${fingerprint}'"; then
      return
    fi
    sleep 5
  done
  echo "${node}: checksum verification did not finish for ${path}; inspect ${log}" >&2
  return 1
}

probe_gpu_container() {
  local node="$1" probe_name output
  node_profile "${node}"
  probe_name="${CLUSTER_CONTAINER_NAME}-gpu-probe"
  output="$(ssh_node "${node}" "docker rm -f '${probe_name}' >/dev/null 2>&1 || true; docker run -d --name '${probe_name}' --user '${container_user}' --gpus all -e NVIDIA_DISABLE_REQUIRE=1 -e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64 '${image}' /lerobot/.venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.version.cuda, torch.cuda.get_device_name(0))' >/dev/null; for i in 1 2 3 4 5; do state=\$(docker inspect '${probe_name}' --format '{{.State.Status}}'); test \"\$state\" != running && break; sleep 1; done; docker inspect '${probe_name}' --format '{{.State.ExitCode}}'; docker logs '${probe_name}' 2>&1; docker rm '${probe_name}' >/dev/null")"
  [[ "${output}" == 0$'\n'* ]] || { echo "${node}: CUDA probe failed: ${output}" >&2; return 1; }
  echo "${node}: CUDA probe ${output//$'\n'/ }"
}

check_node() {
  local node="$1" commit code_dir actual_image actual_gpus
  local staff1_episodes staff1_videos
  node_profile "${node}"
  commit="$(code_commit)"
  code_dir="${host_source}/lerobot_main_${commit}"
  actual_image="$(ssh_node "${node}" "docker image inspect '${image}' --format '{{.Id}}' 2>/dev/null || true")"
  [[ "${actual_image}" == "${image_id}" ]] || { echo "${node}: image mismatch" >&2; return 1; }
  actual_gpus="$(ssh_node "${node}" "nvidia-smi --query-gpu=index --format=csv,noheader | wc -l")"
  (( actual_gpus >= gpu_count )) || { echo "${node}: only ${actual_gpus}/${gpu_count} GPUs" >&2; return 1; }
  ssh_node "${node}" "test -f '${code_dir}/.source_commit' && test \"\$(cat '${code_dir}/.source_commit')\" = '${commit}' && cd '${code_dir}' && sha256sum -c .source_manifest.sha256 >/dev/null 2>&1" || {
    echo "${node}: code snapshot mismatch" >&2; return 1;
  }
  ssh_node "${node}" "test -f '${host_data}/datasets/b2_z1/training/staff1/meta/info.json' && test -f '${host_data}/checkpoints/lerobot_pi05_base_local_tokenizer/model.safetensors' && test -f '${host_data}/checkpoints/mem_vit_distill_20260716_142702/mem_vit_distill_latest.pt'" || {
    echo "${node}: dataset or model missing" >&2; return 1;
  }
  verify_large_file "${node}" "${host_data}/checkpoints/lerobot_pi05_base_local_tokenizer/model.safetensors" "${BASE_POLICY_SHA256}"
  verify_large_file "${node}" "${host_data}/checkpoints/mem_vit_distill_20260716_142702/mem_vit_distill_latest.pt" "${MEM_VIT_SHA256}"
  read -r staff1_episodes staff1_videos < <(ssh_node "${node}" "printf '%s %s\\n' \"\$(find '${host_data}/datasets/b2_z1/training/staff1/data' -type f -name '*.parquet' | wc -l)\" \"\$(find '${host_data}/datasets/b2_z1/training/staff1/videos' -type f -name '*.mp4' | wc -l)\"")
  [[ "${staff1_episodes}" == "${STAFF1_EPISODES}" && "${staff1_videos}" == "${STAFF1_VIDEOS}" ]] || {
    echo "${node}: dataset file-count mismatch" >&2; return 1;
  }
  ssh_node "${node}" "{ echo '${STAFF1_INFO_SHA256}  ${host_data}/datasets/b2_z1/training/staff1/meta/info.json'; echo '${STAFF1_SEMANTIC_SHA256}  ${host_data}/datasets/b2_z1/training/staff1/meta/semantic_views.json'; echo '${STAFF1_BOUNDARIES_SHA256}  ${host_data}/datasets/b2_z1/training/staff1/meta/stage_boundaries.json'; } | sha256sum -c - >/dev/null" || {
    echo "${node}: dataset metadata checksum mismatch" >&2; return 1;
  }
  echo "${node}: OK image=${actual_image} gpus=${actual_gpus} code=${commit}"
}

prepare_node() {
  local node="$1"
  node_profile "${node}"
  ensure_runtime_home "${node}"
  sync_code "${node}"
  ensure_image "${node}"
  check_node "${node}"
  probe_gpu_container "${node}"
}

ensure_runtime_home() {
  local node="$1"
  node_profile "${node}"
  ssh_node "${node}" "mkdir -p '${host_source}' '${host_data}/lerobot/cache/huggingface' '${host_data}/lerobot/home'; if ! test -s '${host_data}/lerobot/home/.netrc' && test -s /root/.netrc; then install -m 600 /root/.netrc '${host_data}/lerobot/home/.netrc'; fi; test -s '${host_data}/lerobot/home/.netrc' || { echo 'Missing persistent W&B credential: ${host_data}/lerobot/home/.netrc' >&2; exit 1; }; chmod 600 '${host_data}/lerobot/home/.netrc'; chown -R '${container_user}' '${host_data}/lerobot/home' '${host_data}/lerobot/cache'"
}

task_container_name() {
  printf '%s-%s\n' "${CLUSTER_CONTAINER_NAME}" "$2"
}

run_task_container() {
  local node="$1" task="$2" mode="$3" gpu="$4" port="$5" checkpoint="${6:-}"
  local name code command_q existing
  node_profile "${node}"
  ensure_runtime_home "${node}"
  name="$(task_container_name "${node}" "${task}")"
  code="$(container_code_dir)"
  existing="$(ssh_node "${node}" "docker inspect '${name}' --format '{{.State.Status}}' 2>/dev/null || true")"
  if [[ "${existing}" == running ]]; then
    echo "${node}/${task}: already running in ${name}"
    return
  fi
  [[ -z "${existing}" ]] || {
    echo "${node}/${task}: existing stopped container ${name}; use resume-plan after preserving its logs" >&2
    return 1
  }
  if [[ "${mode}" == start ]]; then
    printf -v command_q '%q ' bash "${code}/training_plan/launch.sh" start "${task}" "${gpu}" "${port}"
  else
    printf -v command_q '%q ' bash "${code}/training_plan/launch.sh" resume "${task}" "${gpu}" "${checkpoint}" "${port}"
  fi
  ssh_node "${node}" "docker run -d --name '${name}' --user '${container_user}' --gpus all --ipc=host --shm-size=64g --stop-timeout=120 -v '${host_data}:/data:rw' -v '${host_source}:/workspace/source:rw' -w '${code}' -e NVIDIA_DISABLE_REQUIRE=1 -e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64 -e PYTHONPATH='${code}/src' -e UV_PROJECT_ENVIRONMENT=/lerobot/.venv -e VIRTUAL_ENV=/lerobot/.venv -e LEROBOT_RUNTIME_BIN=/lerobot/.venv/bin -e LEROBOT_TRAIN_FOREGROUND=true -e PATH=/lerobot/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin -e HF_HOME=/data/lerobot/cache/huggingface -e HOME=/data/lerobot/home -e TZ=Asia/Shanghai '${image}' ${command_q} >/dev/null"
  echo "${node}/${task}: started ${name} (${mode})"
}

run_task_dry_run() {
  local node="$1" task="$2" mode="$3" gpu="$4" port="$5" checkpoint="${6:-}"
  local code command_q output dry_name
  node_profile "${node}"
  code="$(container_code_dir)"
  dry_name="${CLUSTER_CONTAINER_NAME}-dry-run-${task}"
  if [[ "${mode}" == start ]]; then
    printf -v command_q '%q ' bash "${code}/training_plan/launch.sh" dry-run "${task}" "${gpu}" "${port}"
  else
    printf -v command_q '%q ' bash "${code}/training_plan/launch.sh" resume-dry-run "${task}" "${gpu}" "${checkpoint}" "${port}"
  fi
  # Mizar's frontend does not relay stdout from foreground docker run. Launch a
  # CPU-only short-lived container and read its logs after it exits instead.
  output="$(ssh_node "${node}" "set -e
    docker rm -f '${dry_name}' >/dev/null 2>&1 || true
    docker run -d --name '${dry_name}' --entrypoint bash --user '${container_user}' -v '${host_data}:/data:ro' -v '${host_source}:/workspace/source:rw' -w '${code}' -e PYTHONPATH='${code}/src' -e UV_PROJECT_ENVIRONMENT=/lerobot/.venv -e VIRTUAL_ENV=/lerobot/.venv -e LEROBOT_RUNTIME_BIN=/lerobot/.venv/bin -e PATH=/lerobot/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin -e HF_HOME=/data/lerobot/cache/huggingface -e HOME=/tmp -e TZ=Asia/Shanghai '${image}' -lc ${command_q@Q} >/dev/null
    for i in 1 2 3 4 5 6 7 8 9 10; do
      state=\$(docker inspect '${dry_name}' --format '{{.State.Status}}')
      test \"\${state}\" != running && break
      sleep 1
    done
    state=\$(docker inspect '${dry_name}' --format '{{.State.Status}}')
    test \"\${state}\" != running || { docker rm -f '${dry_name}' >/dev/null; echo 'dry-run container timed out' >&2; exit 1; }
    exit_code=\$(docker inspect '${dry_name}' --format '{{.State.ExitCode}}')
    docker logs '${dry_name}' 2>&1
    docker rm '${dry_name}' >/dev/null
    test \"\${exit_code}\" = 0")"
  if [[ "${mode}" == start ]]; then
    [[ "${output}" == *"--policy.action_predict_task_complete=true"* ]] || {
      echo "${node}/${task}: dry-run omitted task-complete output" >&2; return 1;
    }
    [[ "${output}" == *"--policy.action_predict_task_blocked=false"* ]] || {
      echo "${node}/${task}: dry-run changed task-blocked output" >&2; return 1;
    }
    [[ "${output}" == *"--steps=${TRAIN_STEPS}"* ]] || {
      echo "${node}/${task}: start dry-run has wrong training horizon" >&2; return 1;
    }
  else
    [[ "${output}" == *"--resume=true"* && "${output}" == *"--wandb.resume_training_run=true"* ]] || {
      echo "${node}/${task}: resume dry-run did not preserve the original run" >&2; return 1;
    }
    [[ "${output}" != *"--steps="* ]] || {
      echo "${node}/${task}: resume dry-run unexpectedly overrides saved steps" >&2; return 1;
    }
  fi
  echo "${node}/${task}: ${mode} dry-run OK"
}

find_running_task() {
  local node="$1" task="$2"
  ssh_node "${node}" "for c in \$(docker ps --format '{{.Names}}'); do docker top \"\$c\" -eo pid,args 2>/dev/null; done | grep -F -- 'training_job_${task}_arm_mode_complete' | grep -v grep | head -1" || true
}

latest_run_dir() {
  local node="$1" task="$2"
  node_profile "${node}"
  ssh_node "${node}" "find '${host_data}/models/trained/pi05' -mindepth 1 -maxdepth 1 -type d -name '*training_job_${task}_arm_mode_complete' -printf '%T@ %p\\n' | sort -nr | head -1 | cut -d' ' -f2-"
}

latest_complete_checkpoint() {
  local node="$1" run_dir="$2"
  ssh_node "${node}" "find '${run_dir}/checkpoints' -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' -printf '%f\\n' 2>/dev/null | sort -n | while read s; do c='${run_dir}/checkpoints/'\"\$s\"; test -s \"\$c/pretrained_model/train_config.json\" && test -s \"\$c/pretrained_model/config.json\" && { test -s \"\$c/pretrained_model/model.safetensors\" || test -s \"\$c/pretrained_model/adapter_model.safetensors\"; } && test -s \"\$c/training_state/optimizer_state.safetensors\" && test -s \"\$c/training_state/optimizer_param_groups.json\" && test -s \"\$c/training_state/rng_state.safetensors\" && test -s \"\$c/training_state/scheduler_state.json\" && test -s \"\$c/training_state/training_step.json\" && echo \"\$s\"; done | tail -1"
}

remove_incomplete_checkpoint_dirs() {
  local node="$1" run_dir="$2" complete_step="$3"
  ssh_node "${node}" "find '${run_dir}/checkpoints' -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' -printf '%f\n' 2>/dev/null | sort -n | while read s; do
    test \"\${s#0}\" -gt \"$((10#${complete_step}))\" 2>/dev/null || continue
    c='${run_dir}/checkpoints/'\"\$s\"
    if test -s \"\$c/pretrained_model/train_config.json\" && test -s \"\$c/pretrained_model/config.json\" && { test -s \"\$c/pretrained_model/model.safetensors\" || test -s \"\$c/pretrained_model/adapter_model.safetensors\"; } && test -s \"\$c/training_state/optimizer_state.safetensors\" && test -s \"\$c/training_state/optimizer_param_groups.json\" && test -s \"\$c/training_state/rng_state.safetensors\" && test -s \"\$c/training_state/scheduler_state.json\" && test -s \"\$c/training_state/training_step.json\"; then
      echo \"Refusing to remove complete checkpoint newer than resume point: \$c\" >&2
      exit 1
    fi
    rm -rf -- \"\$c\"
    echo \"${node}: removed incomplete checkpoint \$c\"
  done"
}

start_task() {
  local node="$1" task="$2" gpu port
  read -r gpu port < <(task_resources "${task}")
  if [[ -n "$(find_running_task "${node}" "${task}")" ]]; then
    echo "${node}/${task}: already running"
    return
  fi
  run_task_container "${node}" "${task}" start "${gpu}" "${port}"
}

resume_task() {
  local node="$1" task="$2" gpu port run_dir step checkpoint name
  read -r gpu port < <(task_resources "${task}")
  if [[ -n "$(find_running_task "${node}" "${task}")" ]]; then
    echo "${node}/${task}: already running"
    return
  fi
  run_dir="$(latest_run_dir "${node}" "${task}")"
  [[ -n "${run_dir}" ]] || { echo "${node}/${task}: no run directory" >&2; return 1; }
  step="$(latest_complete_checkpoint "${node}" "${run_dir}")"
  [[ -n "${step}" ]] || { echo "${node}/${task}: no complete checkpoint" >&2; return 1; }
  if (( 10#${step} >= TRAIN_STEPS )); then
    echo "${node}/${task}: complete at ${step}"
    return
  fi
  checkpoint="/data/models/trained/pi05/$(basename "${run_dir}")/checkpoints/${step}"
  remove_incomplete_checkpoint_dirs "${node}" "${run_dir}" "${step}"
  name="$(task_container_name "${node}" "${task}")"
  ssh_node "${node}" "mkdir -p '${host_data}/lerobot/logs'; if test -n \"\$(docker inspect '${name}' --format '{{.State.Status}}' 2>/dev/null)\"; then docker logs '${name}' >'${host_data}/lerobot/logs/${name}.container.log' 2>&1 || true; docker rm '${name}' >/dev/null; fi"
  echo "${node}/${task}: resuming ${checkpoint} in the original W&B run"
  run_task_container "${node}" "${task}" resume "${gpu}" "${port}" "${checkpoint}"
}

node_needs_resume() {
  local node="$1" task
  local -a task_list
  mapfile -t task_list < <(node_tasks "${node}")
  for task in "${task_list[@]}"; do
    if [[ -z "$(find_running_task "${node}" "${task}")" ]]; then
      return 0
    fi
  done
  return 1
}

plan_action() {
  local action="$1" node task
  local -a task_list
  if [[ "${action}" == start ]]; then
    while IFS= read -r node; do prepare_node "${node}"; done < <(plan_nodes)
  else
    # A resume happens while sibling tasks may still occupy every GPU. Reuse the
    # environment proven by the initial prepare; never inject a CUDA probe into
    # those running jobs.
    while IFS= read -r node; do
      if node_needs_resume "${node}"; then
        sync_code "${node}"
        check_node "${node}"
      fi
    done < <(plan_nodes)
  fi
  while IFS= read -r node; do
    mapfile -t task_list < <(node_tasks "${node}")
    for task in "${task_list[@]}"; do
      "${action}_task" "${node}" "${task}"
    done
  done < <(plan_nodes)
}

dry_run_plan() {
  local mode="$1" node task gpu port run_dir step checkpoint
  local -a task_list
  while IFS= read -r node; do
    mapfile -t task_list < <(node_tasks "${node}")
    for task in "${task_list[@]}"; do
      read -r gpu port < <(task_resources "${task}")
      checkpoint=""
      if [[ "${mode}" == resume ]]; then
        run_dir="$(latest_run_dir "${node}" "${task}")"
        [[ -n "${run_dir}" ]] || { echo "${node}/${task}: no run directory" >&2; return 1; }
        step="$(latest_complete_checkpoint "${node}" "${run_dir}")"
        [[ -n "${step}" ]] || { echo "${node}/${task}: no complete checkpoint" >&2; return 1; }
        checkpoint="/data/models/trained/pi05/$(basename "${run_dir}")/checkpoints/${step}"
      fi
      run_task_dry_run "${node}" "${task}" "${mode}" "${gpu}" "${port}" "${checkpoint}"
    done
  done < <(plan_nodes)
}

status() {
  local node task running run_dir step logged_step free errors name container_log active_log
  local -a task_list
  while IFS= read -r node; do
    node_profile "${node}"
    free="$(ssh_node "${node}" "df -BG '${host_data}' | tail -1 | awk '{print \$4}'")"
    mapfile -t task_list < <(node_tasks "${node}")
    for task in "${task_list[@]}"; do
      running=no
      [[ -n "$(find_running_task "${node}" "${task}")" ]] && running=yes
      run_dir="$(latest_run_dir "${node}" "${task}")"
      step=none
      logged_step=none
      errors=0
      [[ -z "${run_dir}" ]] || step="$(latest_complete_checkpoint "${node}" "${run_dir}")"
      if [[ -n "${run_dir}" ]]; then
        name="$(task_container_name "${node}" "${task}")"
        container_log="$(ssh_node "${node}" "docker logs '${name}' 2>/dev/null | sed -n 's/^Log:[[:space:]]*//p' | tail -1" || true)"
        if [[ "${container_log}" == /workspace/source/* ]]; then
          active_log="${host_source}${container_log#/workspace/source}"
        else
          active_log="$(ssh_node "${node}" "run=\$(basename '${run_dir}'); find '${host_source}' -path '*/logs/'\"\$run\"'*.log' -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-")"
        fi
        if [[ -n "${active_log}" ]]; then
          read -r logged_step errors < <(ssh_node "${node}" "if test -f '${active_log}'; then step=\$(grep -o 'step:[0-9K.]*' '${active_log}' | tail -1 | cut -d: -f2); err=\$(grep -Eci 'Traceback|No space left|SafetensorError|CUDA out of memory|NCCL.*error' '${active_log}' || true); echo \"\${step:-none} \${err:-0}\"; else echo 'none 0'; fi")
        fi
      fi
      printf '%s|%s|running=%s|step=%s|checkpoint=%s|errors=%s|free=%s\n' "${node}" "${task}" "${running}" "${logged_step}" "${step:-none}" "${errors}" "${free}"
    done
  done < <(plan_nodes)
}

validate_plan_resources

case "${1:-}" in
  describe) describe ;;
  check)
    while IFS= read -r node; do check_node "${node}"; done < <(select_nodes "${2:-all}")
    ;;
  sync-code)
    while IFS= read -r node; do sync_code "${node}"; done < <(select_nodes "${2:-all}")
    ;;
  prepare)
    while IFS= read -r node; do prepare_node "${node}"; done < <(select_nodes "${2:-all}")
    ;;
  dry-run-plan) dry_run_plan start ;;
  resume-dry-run-plan) dry_run_plan resume ;;
  start-plan) plan_action start ;;
  resume-plan) plan_action resume ;;
  status) status ;;
  archive-f-checkpoints) exec bash "${plan_dir}/storage_guard.sh" archive ;;
  *) usage >&2; exit 2 ;;
esac
