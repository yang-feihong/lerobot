"""Exercise the actual shell dry-run gate without SSH, Docker, or real waiting."""
import os
from pathlib import Path
import subprocess


PLAN = Path(__file__).resolve().parents[2] / "training_plan"
FUNCTION = (PLAN / "cluster.sh").read_text().split("run_task_dry_run() {", 1)[1].split(
    "\nfind_running_task()", 1
)[0]
STUBS = r'''
set -euo pipefail
CLUSTER_CONTAINER_NAME=probe
training_jobs_rel=training_plan/jobs.json
node_profile() {
    host_data=/data; host_model_data=/data; host_hf_cache=/cache
    host_source=/source; container_user=root; image=mock
}
ensure_runtime_home() { :; }
container_code_dir() { echo /code; }
task_target_steps() { echo 100000; }
ssh_node() { bash -c "$2"; }
docker() {
    case "$1" in
        run) echo mock-container ;;
        inspect)
            if [[ "$*" == *State.ExitCode* ]]; then echo 0
            elif (( SECONDS >= MOCK_FINISH_SECONDS )); then echo exited
            else echo running; fi ;;
        logs) echo 'mock-diagnostic --policy.action_predict_task_complete=true --policy.action_predict_task_blocked=false --steps=100000' ;;
        rm) : ;;
        *) return 99 ;;
    esac
}
sleep() { SECONDS=$((SECONDS + 1)); }
export -f docker sleep
'''


def gate(timeout=None, finish=65):
    env = {**os.environ, "MOCK_FINISH_SECONDS": str(finish)}
    env.pop("CLUSTER_DRY_RUN_TIMEOUT_SECONDS", None)
    if timeout is not None:
        env["CLUSTER_DRY_RUN_TIMEOUT_SECONDS"] = str(timeout)
    return subprocess.run(
        ["bash", "-c", STUBS + "\nrun_task_dry_run() {" + FUNCTION
         + '\nrun_task_dry_run wb6 task start 0,1,2,3 29518\n'],
        env=env, text=True, capture_output=True, timeout=15,
    )


def test_default_allows_preflight_longer_than_old_minute():
    result = gate()
    assert result.returncode == 0, result.stderr
    assert "start dry-run OK" in result.stdout


def test_remote_dry_run_imports_actual_trainer_before_launch():
    assert "import lerobot.scripts.lerobot_train" in FUNCTION
    assert "&& ${command_q}" in FUNCTION


def test_configured_timeout_reports_logs_and_does_not_pass():
    result = gate(2)
    assert result.returncode != 0
    assert "mock-diagnostic" in result.stderr
    assert "timed out after 2s" in result.stderr
    assert "dry-run OK" not in result.stdout


def test_timeout_must_be_positive_integer():
    for invalid in (0, -1, "not-a-number"):
        result = gate(invalid)
        assert result.returncode != 0
        assert "must be a positive integer" in result.stderr
