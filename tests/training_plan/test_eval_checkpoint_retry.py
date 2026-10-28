"""Exercise the real parallel transfer worker after transient SSH failures."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("failures", [0, 1, 4, 5])
def test_parallel_checkpoint_worker_retries_without_accepting_failure(tmp_path, failures):
    script = (
        Path(__file__).resolve().parents[2] / "training_plan/sync_eval_checkpoint.sh"
    ).read_text()
    worker = script.split("      set -o pipefail\n", 1)[1].split("    ) &", 1)[0]
    commands = tmp_path / "commands"
    commands.mkdir()
    ssh = commands / "ssh"
    ssh.write_text(
        '#!/bin/bash\n'
        'n=$(<"$RETRY_COUNTER"); n=$((n + 1)); echo "$n" >"$RETRY_COUNTER"\n'
        'if ((n <= RETRY_FAILURES)); then printf bad; exit 255; fi\n'
        'printf good\n'
    )
    ssh.chmod(0o755)
    sleep = commands / "sleep"
    sleep.write_text("#!/bin/bash\nexit 0\n")
    sleep.chmod(0o755)
    counter = tmp_path / "counter"
    counter.write_text("0\n")
    output = tmp_path / "weights"
    output.write_bytes(b"keep----tail")
    shell = (
        'set -euo pipefail\n'
        'CLUSTER_SSH_CONFIG=unused; ssh_host=host; remote_snapshot=/remote; largest_rel=weights\n'
        'block_size=4; skip_blocks=1; count_blocks=1; local_tmp="$1"\n'
        'set -o pipefail\n' + worker
    )
    result = subprocess.run(
        ["bash", "-c", shell, "--", str(output)],
        env={
            **os.environ,
            "PATH": f"{commands}:{os.environ['PATH']}",
            "RETRY_COUNTER": str(counter),
            "RETRY_FAILURES": str(failures),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert int(counter.read_text()) == min(failures + 1, 5)
    assert (result.returncode == 0) == (failures < 5)
    if failures < 5:
        assert output.read_bytes() == b"keepgoodtail"
