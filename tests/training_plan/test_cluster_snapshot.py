"""Host-side preflight must use the same source-domain contract as training."""

import subprocess
from pathlib import Path


def test_snapshot_contains_domain_contract_and_resolves_outside_container(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    source = (repo / "training_plan/cluster.sh").read_text()
    function = source.split("snapshot_file_list() {", 1)[1].split("\n}\n", 1)[0]
    result = subprocess.run(
        ["bash", "-c", 'repo_root="$1"; training_jobs_rel=training_plan/training_jobs.json; '
         "snapshot_file_list() {" + function + "\n}\nsnapshot_file_list", "test", str(repo)],
        check=True, capture_output=True,
    )
    paths = result.stdout.decode().split("\0")
    assert "src/lerobot/data_domains.py" in paths
    assert "src/lerobot/common/distributed_preprocessing.py" in paths
    for name in paths:
        if not name or not (name.startswith("training_plan/") or name == "src/lerobot/data_domains.py"):
            continue
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((repo / name).read_bytes())
    subprocess.run(
        ["python3", "-S", str(tmp_path / "training_plan/resolve_dataset.py"),
         "--task", "staff1_stage2_full_mem", "--request-only"],
        check=True, capture_output=True,
    )
