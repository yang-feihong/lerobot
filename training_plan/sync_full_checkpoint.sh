#!/usr/bin/env bash
set -euo pipefail

plan_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="${ENV_FILE:-${plan_dir}/.env}"
# shellcheck disable=SC1090
source "${env_file}"

if [[ "$#" -ne 4 ]]; then
  echo "Usage: $0 SOURCE_SSH_HOST SOURCE_CHECKPOINT DEST_SSH_HOST DEST_CHECKPOINT" >&2
  exit 2
fi

source_host="$1"
source_checkpoint="$2"
destination_host="$3"
destination_checkpoint="$4"
ssh_config="${CLUSTER_SSH_CONFIG}"

for path in "${source_checkpoint}" "${destination_checkpoint}"; do
  [[ "${path}" =~ ^/[a-zA-Z0-9_./-]+/models/trained/pi05/[a-zA-Z0-9_./-]+/checkpoints/[0-9]+$ \
    && "/${path}/" != */../* && "/${path}/" != */./* ]] || {
    echo "Refusing unexpected checkpoint path: ${path}" >&2
    exit 2
  }
done
[[ "${source_checkpoint##*/}" == "${destination_checkpoint##*/}" ]] || {
  echo "Source and destination checkpoint steps differ." >&2
  exit 2
}

required=(
  pretrained_model/train_config.json
  pretrained_model/config.json
  training_state/optimizer_state.safetensors
  training_state/optimizer_param_groups.json
  training_state/rng_state.safetensors
  training_state/scheduler_state.json
  training_state/training_step.json
)

ssh_cmd=(ssh -F "${ssh_config}" -o ServerAliveInterval=30 -o ServerAliveCountMax=10)
source_manifest="$(mktemp)"
destination_manifest="$(mktemp)"
trap 'rm -f -- "${source_manifest}" "${destination_manifest}"' EXIT

printf -v source_q '%q' "${source_checkpoint}"
printf -v destination_q '%q' "${destination_checkpoint}"
staging="${destination_checkpoint}.sync-partial"
printf -v staging_q '%q' "${staging}"
printf -v destination_parent_q '%q' "$(dirname "${destination_checkpoint}")"

if [[ "${source_host}" == local ]]; then
  [[ "${destination_host}" != local ]] || {
    echo "At least one side of a checkpoint synchronization must be remote." >&2
    exit 2
  }
  for relative in "${required[@]}"; do
    [[ -s "${source_checkpoint}/${relative}" ]] || {
      echo "Incomplete local source checkpoint: ${source_checkpoint}/${relative}" >&2
      exit 1
    }
  done
  [[ -s "${source_checkpoint}/pretrained_model/model.safetensors" \
    || -s "${source_checkpoint}/pretrained_model/adapter_model.safetensors" ]] || {
    echo "Local source checkpoint has no policy weight file." >&2
    exit 1
  }
  (cd "${source_checkpoint}" && find . -type f -print0 | sort -z | xargs -0 sha256sum) \
    >"${source_manifest}"
  if [[ "${RESUME_STAGING:-0}" == 1 ]]; then
    staging_check="test -d ${staging_q}"
  else
    staging_check="test ! -e ${staging_q}; mkdir -p ${staging_q}"
  fi
  "${ssh_cmd[@]}" "${destination_host}" \
    "test ! -e ${destination_q}; mkdir -p ${destination_parent_q}; ${staging_check}"
  ssh_transport="ssh -F ${ssh_config@Q} -o ServerAliveInterval=30 -o ServerAliveCountMax=10"
  rsync -a --partial --append-verify -e "${ssh_transport}" \
    "${source_checkpoint}/" "${destination_host}:${staging}/"
else
  for relative in "${required[@]}"; do
    printf -v relative_q '%q' "${relative}"
    "${ssh_cmd[@]}" "${source_host}" "test -s ${source_q}/${relative_q}"
  done
  "${ssh_cmd[@]}" "${source_host}" \
    "test -s ${source_q}/pretrained_model/model.safetensors || test -s ${source_q}/pretrained_model/adapter_model.safetensors"
  "${ssh_cmd[@]}" "${source_host}" \
    "cd ${source_q} && find . -type f -print0 | sort -z | xargs -0 sha256sum" \
    >"${source_manifest}"
  if [[ "${destination_host}" == local ]]; then
    [[ ! -e "${destination_checkpoint}" && ! -e "${staging}" ]] || {
      echo "Local destination or staging path already exists." >&2
      exit 1
    }
    mkdir -p "$(dirname "${destination_checkpoint}")" "${staging}"
    ssh_transport="ssh -F ${ssh_config@Q} -o ServerAliveInterval=30 -o ServerAliveCountMax=10"
    rsync -a --partial -e "${ssh_transport}" \
      "${source_host}:${source_checkpoint}/" "${staging}/"
  else
    "${ssh_cmd[@]}" "${destination_host}" \
      "test ! -e ${destination_q}; test ! -e ${staging_q}; mkdir -p ${destination_parent_q} ${staging_q}"
    "${ssh_cmd[@]}" "${source_host}" "tar -C ${source_q} -cf - ." | \
      "${ssh_cmd[@]}" "${destination_host}" "tar -C ${staging_q} -xf -"
  fi
fi

if [[ "${destination_host}" == local ]]; then
  (cd "${staging}" && find . -type f -print0 | sort -z | xargs -0 sha256sum) \
    >"${destination_manifest}"
  cmp "${source_manifest}" "${destination_manifest}"
  mv -- "${staging}" "${destination_checkpoint}"
else
  "${ssh_cmd[@]}" "${destination_host}" \
    "cd ${staging_q} && find . -type f -print0 | sort -z | xargs -0 sha256sum" \
    >"${destination_manifest}"
  cmp "${source_manifest}" "${destination_manifest}"
  "${ssh_cmd[@]}" "${destination_host}" "mv -- ${staging_q} ${destination_q}"
fi

echo "FULL_CHECKPOINT_SYNCED source=${source_host}:${source_checkpoint} destination=${destination_host}:${destination_checkpoint}"
