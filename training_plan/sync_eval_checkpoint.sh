#!/usr/bin/env bash
set -euo pipefail

plan_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="${ENV_FILE:-${plan_dir}/.env}"
set -a
# shellcheck disable=SC1090
source "${env_file}"
set +a

if (( $# != 2 )); then
  echo "Usage: $0 SSH_HOST REMOTE_CHECKPOINT" >&2
  exit 2
fi

ssh_host="$1"
remote_checkpoint="${2%/}"
case "${remote_checkpoint}" in
  */models/trained/pi05/*/checkpoints/[0-9]*) ;;
  *) echo "Refusing unexpected checkpoint path: ${remote_checkpoint}" >&2; exit 1 ;;
esac

relative="${remote_checkpoint#*/models/trained/pi05/}"
if [[ "${relative}" == "${remote_checkpoint}" ]]; then
  echo "Checkpoint is outside a models/trained/pi05 tree: ${remote_checkpoint}" >&2
  exit 1
fi
remote_data_root="${remote_checkpoint%%/models/trained/pi05/*}"
remote_snapshot="${remote_data_root}/.checkpoint_eval_snapshots/${relative}"
remote_snapshot_parent="$(dirname "${remote_snapshot}")"
local_checkpoint="${CLUSTER_LOCAL_CHECKPOINT_ROOT}/${relative}"
local_model="${local_checkpoint}/pretrained_model"
mkdir -p "${local_model}"
parallel_parts="${SYNC_PARALLEL_PARTS:-1}"
if ! [[ "${parallel_parts}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SYNC_PARALLEL_PARTS must be a positive integer, got: ${parallel_parts}" >&2
  exit 2
fi

ssh_base=(
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY
  ssh -F "${CLUSTER_SSH_CONFIG}"
  -o ControlMaster=auto -o ControlPersist=120 -o "ControlPath=${CLUSTER_SSH_CONTROL_PATH}"
  "${ssh_host}"
)

"${ssh_base[@]}" "set -e
  if test ! -d '${remote_snapshot}'; then
    test -d '${remote_checkpoint}'
    test -s '${remote_checkpoint}/pretrained_model/config.json'
    test -s '${remote_checkpoint}/pretrained_model/train_config.json'
    test -s '${remote_checkpoint}/pretrained_model/adapter_model.safetensors' -o -s '${remote_checkpoint}/pretrained_model/model.safetensors'
    mkdir -p '${remote_snapshot_parent}'
    snapshot_tmp='${remote_snapshot}.building.'\$\$
    trap 'rm -rf -- \"\${snapshot_tmp}\"' EXIT
    cp -al -- '${remote_checkpoint}' \"\${snapshot_tmp}\"
    mv -- \"\${snapshot_tmp}\" '${remote_snapshot}'
    trap - EXIT
  fi"

largest_entry="$("${ssh_base[@]}" "cd '${remote_snapshot}/pretrained_model' && find . -type f -name '*.safetensors' -printf '%s %P\n' | sort -nr | head -n 1")"
largest_size="${largest_entry%% *}"
largest_rel="${largest_entry#* }"

if (( parallel_parts == 1 )) || [[ -z "${largest_entry}" ]] || (( largest_size < 268435456 )); then
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    rsync -a --partial --info=progress2 \
    -e "ssh -F ${CLUSTER_SSH_CONFIG} -o ControlMaster=auto -o ControlPersist=120 -o ControlPath=${CLUSTER_SSH_CONTROL_PATH}" \
    "${ssh_host}:${remote_snapshot}/pretrained_model/" "${local_model}/"
else
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    rsync -a --partial --info=progress2 --exclude="/${largest_rel}" \
    -e "ssh -F ${CLUSTER_SSH_CONFIG} -o ControlMaster=auto -o ControlPersist=120 -o ControlPath=${CLUSTER_SSH_CONTROL_PATH}" \
    "${ssh_host}:${remote_snapshot}/pretrained_model/" "${local_model}/"

  block_size=1048576
  total_blocks=$(( (largest_size + block_size - 1) / block_size ))
  blocks_per_part=$(( (total_blocks + parallel_parts - 1) / parallel_parts ))
  local_large="${local_model}/${largest_rel}"
  local_tmp="${local_large}.parallel-partial"
  mkdir -p "$(dirname "${local_large}")"
  rm -f -- "${local_tmp}"
  truncate -s "${largest_size}" "${local_tmp}"
  transfer_pids=()
  for (( part = 0; part < parallel_parts; part++ )); do
    skip_blocks=$(( part * blocks_per_part ))
    if (( skip_blocks >= total_blocks )); then
      break
    fi
    count_blocks="${blocks_per_part}"
    if (( skip_blocks + count_blocks > total_blocks )); then
      count_blocks=$(( total_blocks - skip_blocks ))
    fi
    (
      set -o pipefail
      env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
        ssh -F "${CLUSTER_SSH_CONFIG}" -o ControlMaster=no "${ssh_host}" \
        "dd if='${remote_snapshot}/pretrained_model/${largest_rel}' bs=${block_size} skip=${skip_blocks} count=${count_blocks} status=none" \
        | dd of="${local_tmp}" bs="${block_size}" seek="${skip_blocks}" conv=notrunc status=none
    ) &
    transfer_pids+=("$!")
  done
  transfer_failed=false
  for transfer_pid in "${transfer_pids[@]}"; do
    if ! wait "${transfer_pid}"; then
      transfer_failed=true
    fi
  done
  if [[ "${transfer_failed}" == true ]]; then
    rm -f -- "${local_tmp}"
    echo "Parallel transfer failed; evaluation snapshot retained: ${remote_snapshot}" >&2
    exit 1
  fi
  truncate -s "${largest_size}" "${local_tmp}"
  mv -f -- "${local_tmp}" "${local_large}"
fi

remote_manifest="$(mktemp)"
local_manifest="$(mktemp)"
trap 'rm -f "${remote_manifest}" "${local_manifest}"' EXIT
"${ssh_base[@]}" "cd '${remote_snapshot}/pretrained_model' && find . -type f -print0 | sort -z | xargs -0 sha256sum" \
  >"${remote_manifest}"
(
  cd "${local_model}"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) >"${local_manifest}"
if ! cmp -s "${remote_manifest}" "${local_manifest}"; then
  echo "Checksum mismatch; evaluation snapshot retained: ${remote_snapshot}" >&2
  exit 1
fi

"${ssh_base[@]}" "rm -rf -- '${remote_snapshot}'"
printf 'Synced and verified evaluation model: %s/%s\n' "${ssh_host}" "${relative}"
