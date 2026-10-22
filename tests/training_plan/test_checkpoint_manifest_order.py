"""Checkpoint manifests must not inherit either host's locale sort order."""

import os
import subprocess
from pathlib import Path


def test_chunk_manifest_sort_is_locale_independent(tmp_path):
    plan = Path(__file__).resolve().parents[2] / "training_plan"
    script = (plan / "sync_full_checkpoint_chunks.sh").read_text()
    functions = script[script.index("hash_local_tree() {") : script.index("copy_large_pull() {")]
    tree = tmp_path / "checkpoint"
    tree.mkdir()
    (tree / "README.md").write_text("metadata")
    (tree / "adapter.json").write_text("weights")
    commands = tmp_path / "commands"
    commands.mkdir()
    # Simulate a remote locale with a different order unless the caller pins C.
    sort = commands / "sort"
    sort.write_text(
        '#!/bin/bash\nif [[ "$LC_ALL" == C ]]; then exec /usr/bin/sort "$@"; fi\nexec /usr/bin/sort -r "$@"\n'
    )
    sort.chmod(0o755)
    shell = (
        functions
        + '\nssh_mock() { bash -c "$2"; }\nssh_base=(ssh_mock)\nhash_local_tree "$1"\nhash_remote_tree host "$1"\n'
    )
    result = subprocess.run(
        ["bash", "-eu", "-c", shell, "--", str(tree)],
        env={**os.environ, "PATH": f"{commands}:{os.environ['PATH']}"},
        check=True,
        capture_output=True,
        text=True,
    )
    lines = result.stdout.splitlines()
    assert len(lines) == 4
    assert lines[:2] == lines[2:]
    assert lines[0].endswith("./README.md")


def test_full_checkpoint_manifest_sorts_pin_c_locale():
    script = (Path(__file__).resolve().parents[2] / "training_plan/sync_full_checkpoint.sh").read_text()
    manifest_lines = [line for line in script.splitlines() if "sort -z" in line]
    assert len(manifest_lines) == 4
    assert all("LC_ALL=C sort -z" in line for line in manifest_lines)
