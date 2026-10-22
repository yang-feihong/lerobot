#!/usr/bin/env bash
set -euo pipefail

plan_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "${ENV_FILE:-${plan_dir}/.env}"

[[ "$#" -eq 4 ]] || {
  echo "Usage: $0 pull HOST REMOTE_CHECKPOINT LOCAL_CHECKPOINT | push LOCAL_CHECKPOINT HOST REMOTE_CHECKPOINT" >&2
  exit 2
}

mode="${1:?mode must be pull or push}"
first="${2:?source is required}"
second="${3:?source checkpoint is required}"
third="${4:?destination is required}"
streams="${CHECKPOINT_SYNC_STREAMS:-8}"
[[ "${mode}" == pull || "${mode}" == push ]] || { echo "mode must be pull or push" >&2; exit 2; }
[[ "${streams}" =~ ^[1-9][0-9]*$ && "${streams}" -le 32 ]] || {
  echo "CHECKPOINT_SYNC_STREAMS must be an integer in [1, 32]" >&2; exit 2;
}
if [[ "${mode}" == pull ]]; then
  checkpoint_paths=("${second}" "${third}")
else
  checkpoint_paths=("${first}" "${third}")
fi
for checkpoint_path in "${checkpoint_paths[@]}"; do
  [[ "${checkpoint_path}" =~ ^/[a-zA-Z0-9_./-]+/models/trained/pi05/[a-zA-Z0-9_./-]+/checkpoints/[0-9]+$ \
    && "/${checkpoint_path}/" != */../* && "/${checkpoint_path}/" != */./* ]] || {
    echo "Refusing unexpected checkpoint path: ${checkpoint_path}" >&2; exit 2;
  }
done
[[ "${checkpoint_paths[0]##*/}" == "${checkpoint_paths[1]##*/}" ]] || {
  echo "Source and destination checkpoint steps differ" >&2; exit 2;
}
block_bytes=1048576
chunk_blocks=64
ssh_config="${CLUSTER_SSH_CONFIG}"
ssh_base=(ssh -F "${ssh_config}" -o Compression=no -o ServerAliveInterval=30 -o ServerAliveCountMax=10)
large_files=(
  pretrained_model/semantic_views.json
  pretrained_model/adapter_model.safetensors
  training_state/optimizer_state.safetensors
)

hash_local_tree() {
  (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum)
}

hash_remote_tree() {
  "${ssh_base[@]}" "$1" "cd '$2' && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum"
}

copy_large_pull() {
  local host="$1" remote_root="$2" local_root="$3" relative="$4"
  local remote_file="${remote_root}/${relative}" local_file="${local_root}/${relative}"
  local size blocks file_chunk_blocks parts_dir active start count expected part tmp remote_hash local_hash
  size="$("${ssh_base[@]}" "${host}" "stat -c %s '${remote_file}'")"
  file_chunk_blocks=${chunk_blocks}
  ((size >= 128 * block_bytes)) || file_chunk_blocks=1
  blocks=$(((size + block_bytes - 1) / block_bytes))
  parts_dir="${local_file}.parts"
  mkdir -p "${parts_dir}" "$(dirname "${local_file}")"
  find "${parts_dir}" -maxdepth 1 -type f -name '*.tmp.*' -delete
  active=0
  for ((start=0; start<blocks; start+=file_chunk_blocks)); do
    count=${file_chunk_blocks}
    ((start + count <= blocks)) || count=$((blocks - start))
    expected=$((count * block_bytes))
    ((start + count != blocks)) || expected=$((size - start * block_bytes))
    printf -v part '%s/part.%09d' "${parts_dir}" "${start}"
    if [[ -f "${part}" && "$(stat -c %s "${part}")" -eq "${expected}" ]]; then
      continue
    fi
    (
      tmp="${part}.tmp.$$"
      for attempt in 1 2 3 4 5; do
        rm -f -- "${tmp}"
        if "${ssh_base[@]}" -c aes128-gcm@openssh.com "${host}" \
          "dd if='${remote_file}' bs=${block_bytes} skip=${start} count=${count} status=none" \
          >"${tmp}" && [[ "$(stat -c %s "${tmp}")" -eq "${expected}" ]]; then
          mv -- "${tmp}" "${part}"
          exit 0
        fi
      done
      exit 1
    ) &
    active=$((active + 1))
    if ((active >= streams)); then wait -n; active=$((active - 1)); fi
  done
  wait
  truncate -s "${size}" "${local_file}"
  for ((start=0; start<blocks; start+=file_chunk_blocks)); do
    printf -v part '%s/part.%09d' "${parts_dir}" "${start}"
    dd if="${part}" of="${local_file}" bs=${block_bytes} seek=${start} conv=notrunc status=none
  done
  remote_hash="$("${ssh_base[@]}" "${host}" "sha256sum '${remote_file}'" | awk '{print $1}')"
  local_hash="$(sha256sum "${local_file}" | awk '{print $1}')"
  [[ "${remote_hash}" == "${local_hash}" ]]
  rm -rf -- "${parts_dir}"
}

copy_large_push() {
  local local_root="$1" host="$2" remote_root="$3" relative="$4"
  local local_file="${local_root}/${relative}" remote_file="${remote_root}/${relative}"
  local size blocks file_chunk_blocks parts_dir active start count expected existing offset_bytes remaining part remote_hash local_hash
  size="$(stat -c %s "${local_file}")"
  file_chunk_blocks=${chunk_blocks}
  ((size >= 128 * block_bytes)) || file_chunk_blocks=1
  blocks=$(((size + block_bytes - 1) / block_bytes))
  parts_dir="${remote_file}.parts"
  "${ssh_base[@]}" "${host}" \
    "mkdir -p '${parts_dir}' '$(dirname "${remote_file}")'; find '${parts_dir}' -maxdepth 1 -type f -name '*.tmp.*' -delete"
  active=0
  for ((start=0; start<blocks; start+=file_chunk_blocks)); do
    count=${file_chunk_blocks}
    ((start + count <= blocks)) || count=$((blocks - start))
    expected=$((count * block_bytes))
    ((start + count != blocks)) || expected=$((size - start * block_bytes))
    printf -v part '%s/part.%09d' "${parts_dir}" "${start}"
    existing="$("${ssh_base[@]}" "${host}" "stat -c %s '${part}' 2>/dev/null || echo 0")"
    if [[ "${existing}" -eq "${expected}" ]]; then
      continue
    fi
    if ((existing > expected)); then
      "${ssh_base[@]}" "${host}" "rm -f -- '${part}'"
      existing=0
    fi
    offset_bytes=$((start * block_bytes + existing))
    remaining=$((expected - existing))
    (
      dd if="${local_file}" bs=1M iflag=skip_bytes,count_bytes \
        skip=${offset_bytes} count=${remaining} status=none | \
        "${ssh_base[@]}" -c aes128-gcm@openssh.com "${host}" \
          "touch '${part}'; test \"\$(stat -c %s '${part}')\" -eq '${existing}'; dd of='${part}' bs=1M oflag=append conv=notrunc status=none; test \"\$(stat -c %s '${part}')\" -eq '${expected}'"
    ) &
    active=$((active + 1))
    if ((active >= streams)); then wait -n; active=$((active - 1)); fi
  done
  wait
  local_hash="$(sha256sum "${local_file}" | awk '{print $1}')"
  "${ssh_base[@]}" "${host}" \
    "find '${parts_dir}' -maxdepth 1 -type f -regextype posix-extended -regex '.*/part\\.[0-9]{9}' -print0 | sort -z | xargs -0 cat >'${remote_file}.assemble'; test \"\$(stat -c %s '${remote_file}.assemble')\" -eq '${size}'; test \"\$(sha256sum '${remote_file}.assemble' | awk '{print \$1}')\" = '${local_hash}'; mv -- '${remote_file}.assemble' '${remote_file}'; rm -rf -- '${parts_dir}'"
  remote_hash="$("${ssh_base[@]}" "${host}" "sha256sum '${remote_file}'" | awk '{print $1}')"
  [[ "${remote_hash}" == "${local_hash}" ]]
}

case "${mode}" in
  pull)
    remote_host="${first}"
    remote_checkpoint="${second}"
    local_checkpoint="${third}"
    staging="${local_checkpoint}.sync-partial"
    [[ ! -e "${local_checkpoint}" ]] || { echo "Destination exists: ${local_checkpoint}" >&2; exit 1; }
    mkdir -p "${staging}"
    rsync -a --partial \
      --exclude='/pretrained_model/adapter_model.safetensors' \
      --exclude='/pretrained_model/semantic_views.json' \
      --exclude='/training_state/optimizer_state.safetensors' \
      -e "ssh -F ${ssh_config} -o Compression=no" \
      "${remote_host}:${remote_checkpoint}/" "${staging}/"
    for relative in "${large_files[@]}"; do
      copy_large_pull "${remote_host}" "${remote_checkpoint}" "${staging}" "${relative}"
    done
    [[ "$(hash_remote_tree "${remote_host}" "${remote_checkpoint}")" == "$(hash_local_tree "${staging}")" ]]
    mv -- "${staging}" "${local_checkpoint}"
    ;;
  push)
    local_checkpoint="${first}"
    remote_host="${second}"
    remote_checkpoint="${third}"
    staging="${remote_checkpoint}.sync-partial"
    [[ -d "${local_checkpoint}" ]] || { echo "Missing source: ${local_checkpoint}" >&2; exit 1; }
    "${ssh_base[@]}" "${remote_host}" "test ! -e '${remote_checkpoint}'; mkdir -p '${staging}'"
    rsync -a --partial \
      --exclude='/pretrained_model/adapter_model.safetensors' \
      --exclude='/pretrained_model/semantic_views.json' \
      --exclude='/training_state/optimizer_state.safetensors' \
      -e "ssh -F ${ssh_config} -o Compression=no" \
      "${local_checkpoint}/" "${remote_host}:${staging}/"
    for relative in "${large_files[@]}"; do
      copy_large_push "${local_checkpoint}" "${remote_host}" "${staging}" "${relative}"
    done
    [[ "$(hash_local_tree "${local_checkpoint}")" == "$(hash_remote_tree "${remote_host}" "${staging}")" ]]
    "${ssh_base[@]}" "${remote_host}" "mv -- '${staging}' '${remote_checkpoint}'"
    ;;
  *) echo "mode must be pull or push" >&2; exit 2 ;;
esac

echo "FULL_CHECKPOINT_CHUNK_SYNCED mode=${mode} destination=${third}"
