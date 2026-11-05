"""Release preflight must not depend on a training node's system Python."""

import subprocess
from pathlib import Path


def run_resolver(ssh_body):
    source = (Path(__file__).resolve().parents[2] / "training_plan/cluster.sh").read_text()
    body = source.split("resolve_node_release() {", 1)[1].split("\n}\n", 1)[0]
    script = """set -euo pipefail
CLUSTER_CONTAINER_NAME=training
training_jobs_rel=training_plan/fixture.json
node_profile() {
  container_user=1001:1001
  host_data=/host/data
  image=pinned/training:image
}
ssh_node() { SSH_BODY; }
resolve_node_release() { RESOLVER_BODY
}
resolve_node_release f /host/source/snapshot fixture_task
""".replace("SSH_BODY", ssh_body).replace("RESOLVER_BODY", body)
    return subprocess.run(["bash", "-c", script], text=True, capture_output=True)


def test_release_resolver_uses_pinned_image_readonly_and_preserves_host_paths():
    result = run_resolver('printf "%s\\n" "$2"')
    assert result.returncode == 0, result.stderr
    command = result.stdout
    assert "--entrypoint /lerobot/.venv/bin/python 'pinned/training:image'" in command
    assert "-v '/host/data:/host/data:ro'" in command
    assert "-v '/host/source/snapshot:/host/source/snapshot:ro'" in command
    assert "--datasets-root '/host/data/datasets/b2_z1'" in command
    assert "--jobs '/host/source/snapshot/training_plan/fixture.json'" in command
    assert "--network none" in command
    assert "docker wait" in command and "docker logs" in command
    assert 'test "$result" = 0' in command
    assert "trap" in command and "docker rm -f 'training-release-probe-f-" in command
    assert "python3 '" not in command


def test_release_resolver_propagates_transport_or_container_failure():
    assert run_resolver("return 23").returncode == 23
