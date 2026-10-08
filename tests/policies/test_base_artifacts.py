import hashlib
import json
from pathlib import Path

import pytest

from lerobot.policies.base_artifacts import (
    BASE_ARTIFACTS_FILENAME,
    resolve_artifact_uri,
    resolve_checkpoint_bases,
    sha256_directory,
    write_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_registry(path: Path, policy_base: Path, mem_vit: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "format": "lerobot.base_artifact_registry",
                "version": 1,
                "artifacts": {
                    "policy-v1": {"path": str(policy_base)},
                    "mem-v1": {"path": str(mem_vit)},
                },
            }
        ),
        encoding="utf-8",
    )


def test_resolves_identical_artifacts_at_machine_local_paths(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    policy_base = tmp_path / "this-machine" / "policy"
    policy_base.mkdir(parents=True)
    policy_weights = policy_base / "model.safetensors"
    policy_weights.write_bytes(b"policy")
    mem_vit = tmp_path / "this-machine" / "mem.pt"
    mem_vit.write_bytes(b"mem")
    registry = tmp_path / "registry.json"
    _write_registry(registry, policy_base, mem_vit)
    write_manifest(
        checkpoint,
        policy_artifact_id="policy-v1",
        policy_sha256=_sha256(policy_weights),
        mem_vit_artifact_id="mem-v1",
        mem_vit_sha256=_sha256(mem_vit),
    )

    assert resolve_checkpoint_bases(checkpoint, registry_path=registry) == {
        "policy_base": policy_base,
        "mem_vit_base": mem_vit,
    }


def test_manifest_is_optional_for_legacy_checkpoints(tmp_path: Path) -> None:
    assert resolve_checkpoint_bases(tmp_path) == {}


def test_content_mismatch_fails_instead_of_loading_wrong_base(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    policy_base = tmp_path / "policy"
    policy_base.mkdir()
    (policy_base / "model.safetensors").write_bytes(b"wrong")
    mem_vit = tmp_path / "mem.pt"
    mem_vit.write_bytes(b"mem")
    registry = tmp_path / "registry.json"
    _write_registry(registry, policy_base, mem_vit)
    write_manifest(
        checkpoint,
        policy_artifact_id="policy-v1",
        policy_sha256=hashlib.sha256(b"expected").hexdigest(),
    )

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        resolve_checkpoint_bases(checkpoint, registry_path=registry)


def test_portable_checkpoint_requires_a_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEROBOT_ARTIFACT_REGISTRY", raising=False)
    checkpoint = tmp_path / "checkpoint"
    write_manifest(
        checkpoint,
        policy_artifact_id="policy-v1",
        policy_sha256="0" * 64,
    )

    with pytest.raises(FileNotFoundError, match="LEROBOT_ARTIFACT_REGISTRY"):
        resolve_checkpoint_bases(checkpoint)


def test_manifest_rejects_unknown_fields(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / BASE_ARTIFACTS_FILENAME).write_text(
        json.dumps(
            {
                "format": "lerobot.base_artifacts",
                "version": 1,
                "requirements": {
                    "policy_base": {
                        "artifact_id": "policy-v1",
                        "kind": "directory",
                        "fingerprint_file": "model.safetensors",
                        "sha256": "0" * 64,
                        "absolute_path": "/remote/path",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid policy_base requirement keys"):
        resolve_checkpoint_bases(checkpoint)


def test_artifact_uri_resolves_without_embedding_a_machine_path(tmp_path: Path) -> None:
    policy_base = tmp_path / "policy"
    policy_base.mkdir()
    mem_vit = tmp_path / "mem.pt"
    mem_vit.write_bytes(b"mem")
    registry = tmp_path / "registry.json"
    _write_registry(registry, policy_base, mem_vit)

    assert resolve_artifact_uri("artifact://policy-v1", registry_path=registry) == policy_base


def test_directory_tree_requirement_detects_auxiliary_file_changes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    policy_base = tmp_path / "policy"
    policy_base.mkdir()
    weights = policy_base / "model.safetensors"
    weights.write_bytes(b"policy")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_bytes(b"tokens")
    (tokenizer / "tokenizer_config.json").write_bytes(b"config")
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "format": "lerobot.base_artifact_registry",
                "version": 1,
                "artifacts": {"policy-v1": str(policy_base), "tokenizer-v1": str(tokenizer)},
            }
        ),
        encoding="utf-8",
    )
    write_manifest(
        checkpoint,
        policy_artifact_id="policy-v1",
        policy_sha256=_sha256(weights),
        tokenizer_artifact_id="tokenizer-v1",
        tokenizer_sha256=sha256_directory(tokenizer),
    )
    assert resolve_checkpoint_bases(checkpoint, registry_path=registry)["policy_tokenizer"] == tokenizer

    (tokenizer / "tokenizer_config.json").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        resolve_checkpoint_bases(checkpoint, registry_path=registry)
