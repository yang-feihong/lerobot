#!/usr/bin/env bash
set -euo pipefail

plan_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="${ENV_FILE:-${plan_dir}/.env}"
set -a
# shellcheck disable=SC1090
source "${env_file}"
set +a

ssh_base=(env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY ssh -F "${CLUSTER_SSH_CONFIG}" -o ControlMaster=auto -o ControlPersist=120 -o "ControlPath=${CLUSTER_SSH_CONTROL_PATH}" "${CLUSTER_F_SSH_HOST}")
remote_root="${CLUSTER_F_DATA_ROOT}/models/trained/pi05"
snapshot_root="${CLUSTER_F_DATA_ROOT}/.checkpoint_archive_staging"

free_gib() {
  "${ssh_base[@]}" "df -B1 '${CLUSTER_F_DATA_ROOT}' | tail -1 | awk '{print int(\$4/1024/1024/1024)}'"
}

eligible_checkpoint() {
  "${ssh_base[@]}" "python3 - <<'PY'
from pathlib import Path
root = Path('${remote_root}')

def complete(path: Path) -> bool:
    required = (
        'pretrained_model/train_config.json',
        'pretrained_model/config.json',
        'training_state/optimizer_state.safetensors',
        'training_state/optimizer_param_groups.json',
        'training_state/rng_state.safetensors',
        'training_state/scheduler_state.json',
        'training_state/training_step.json',
    )
    model = path / 'pretrained_model/model.safetensors'
    adapter = path / 'pretrained_model/adapter_model.safetensors'
    return all((path / name).is_file() and (path / name).stat().st_size > 0 for name in required) and (
        (model.is_file() and model.stat().st_size > 0)
        or (adapter.is_file() and adapter.stat().st_size > 0)
    )

candidates = []
for run in root.glob('*plan16*'):
    checkpoints = sorted(
        (p for p in (run / 'checkpoints').glob('[0-9]*') if p.is_dir() and complete(p)),
        key=lambda p: int(p.name),
    )
    protected = set(checkpoints[-2:])
    for checkpoint in checkpoints:
        step = int(checkpoint.name)
        if step % 10_000 == 0 and checkpoint not in protected:
            candidates.append((step, checkpoint))
if candidates:
    print(min(candidates, key=lambda item: (item[0], str(item[1])))[1])
PY"
}

eligible_rolling_checkpoint() {
  "${ssh_base[@]}" "python3 - <<'PY'
from pathlib import Path
root = Path('${remote_root}')

def complete(path: Path) -> bool:
    required = (
        'pretrained_model/train_config.json',
        'pretrained_model/config.json',
        'training_state/optimizer_state.safetensors',
        'training_state/optimizer_param_groups.json',
        'training_state/rng_state.safetensors',
        'training_state/scheduler_state.json',
        'training_state/training_step.json',
    )
    model = path / 'pretrained_model/model.safetensors'
    adapter = path / 'pretrained_model/adapter_model.safetensors'
    return all((path / name).is_file() and (path / name).stat().st_size > 0 for name in required) and (
        (model.is_file() and model.stat().st_size > 0)
        or (adapter.is_file() and adapter.stat().st_size > 0)
    )

candidates = []
for run in root.glob('*plan16*'):
    checkpoints = sorted(
        (p for p in (run / 'checkpoints').glob('[0-9]*') if p.is_dir() and complete(p)),
        key=lambda p: int(p.name),
    )
    for checkpoint in checkpoints[:-1]:
        candidates.append((int(checkpoint.name), checkpoint))
if candidates:
    print(min(candidates, key=lambda item: (item[0], str(item[1])))[1])
PY"
}

manifest_remote() {
  local path="$1"
  "${ssh_base[@]}" "cd '${path}' && find . -type f -print0 | sort -z | xargs -0 sha256sum"
}

manifest_local() {
  local path="$1"
  (cd "${path}" && find . -type f -print0 | sort -z | xargs -0 sha256sum)
}

archive_one() {
  local remote_path="$1" relative local_path snapshot_path snapshot_parent remote_manifest local_manifest
  local remote_bytes local_free required_free
  [[ "${remote_path}" == "${remote_root}"/*/checkpoints/[0-9]* ]] || {
    echo "Refusing unexpected checkpoint path: ${remote_path}" >&2; exit 1;
  }
  relative="${remote_path#${remote_root}/}"
  local_path="${CLUSTER_LOCAL_CHECKPOINT_ROOT}/${relative}"
  snapshot_path="${snapshot_root}/${relative}"
  snapshot_parent="$(dirname "${snapshot_path}")"
  mkdir -p "$(dirname "${local_path}")"
  "${ssh_base[@]}" "set -e
    if test ! -d '${snapshot_path}'; then
      test -d '${remote_path}'
      mkdir -p '${snapshot_parent}'
      snapshot_tmp='${snapshot_path}.building.'\$\$
      trap 'rm -rf -- \"\${snapshot_tmp}\"' EXIT
      cp -al -- '${remote_path}' \"\${snapshot_tmp}\"
      mv -- \"\${snapshot_tmp}\" '${snapshot_path}'
      trap - EXIT
    fi"
  remote_bytes="$("${ssh_base[@]}" "du -sb '${snapshot_path}' | awk '{print \$1}'")"
  local_free="$(df -B1 --output=avail "${CLUSTER_LOCAL_CHECKPOINT_ROOT}" | tail -1 | tr -d ' ')"
  required_free=$((remote_bytes + remote_bytes / 5))
  if (( local_free < required_free )); then
    echo "Insufficient local space for ${relative}: need ${required_free} bytes, have ${local_free}; remote retained." >&2
    return 1
  fi
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    rsync -a --partial --info=progress2 -e "ssh -F ${CLUSTER_SSH_CONFIG} -o ControlMaster=auto -o ControlPersist=120 -o ControlPath=${CLUSTER_SSH_CONTROL_PATH}" \
      "${CLUSTER_F_SSH_HOST}:${snapshot_path}/" "${local_path}/"
  remote_manifest="$(mktemp)"
  local_manifest="$(mktemp)"
  manifest_remote "${snapshot_path}" >"${remote_manifest}"
  manifest_local "${local_path}" >"${local_manifest}"
  if ! cmp -s "${remote_manifest}" "${local_manifest}"; then
    rm -f "${remote_manifest}" "${local_manifest}"
    echo "Checksum mismatch; remote checkpoint retained: ${remote_path}" >&2
    return 1
  fi
  rm -f "${remote_manifest}" "${local_manifest}"
  "${ssh_base[@]}" "if test -d '${remote_path}'; then rm -rf -- '${remote_path}'; fi; rm -rf -- '${snapshot_path}'"
  echo "Archived and removed remote checkpoint: ${relative}"
}

archive_until_target() {
  local free candidate
  free="$(free_gib)"
  echo "wholebody_F free=${free}GiB threshold=${CLUSTER_F_MIN_FREE_GIB}GiB target=${CLUSTER_F_TARGET_FREE_GIB}GiB"
  (( free >= CLUSTER_F_MIN_FREE_GIB )) && return
  while (( free < CLUSTER_F_TARGET_FREE_GIB )); do
    candidate="$(eligible_checkpoint)"
    [[ -n "${candidate}" ]] || candidate="$(eligible_rolling_checkpoint)"
    if [[ -z "${candidate}" ]]; then
      echo "No complete checkpoint older than a run's latest recovery point is eligible for archival." >&2
      return 1
    fi
    archive_one "${candidate}"
    free="$(free_gib)"
  done
}

archive_all_eligible() {
  local candidate count=0
  while candidate="$(eligible_checkpoint)" && [[ -n "${candidate}" ]]; do
    archive_one "${candidate}"
    count=$((count + 1))
  done
  echo "Archived ${count} eligible milestone checkpoint(s); wholebody_F free=$(free_gib)GiB"
}

case "${1:-}" in
  archive) archive_until_target ;;
  archive-one)
    [[ -n "${2:-}" ]] || { echo "archive-one requires an exact remote checkpoint path" >&2; exit 2; }
    archive_one "$2"
    ;;
  archive-all-eligible) archive_all_eligible ;;
  *) echo "Usage: $0 {archive|archive-one REMOTE_CHECKPOINT|archive-all-eligible}" >&2; exit 2 ;;
esac
