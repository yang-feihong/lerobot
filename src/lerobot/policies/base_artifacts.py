#!/usr/bin/env python

"""Portable, content-addressed references to immutable policy base artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

BASE_ARTIFACTS_FILENAME = "base_artifacts.json"
MANIFEST_FORMAT = "lerobot.base_artifacts"
REGISTRY_FORMAT = "lerobot.base_artifact_registry"
VERSION = 1
ARTIFACT_URI_PREFIX = "artifact://"


def artifact_uri(artifact_id: str) -> str:
    return f"{ARTIFACT_URI_PREFIX}{artifact_id}"


def is_artifact_uri(value: str | os.PathLike[str] | None) -> bool:
    return value is not None and str(value).startswith(ARTIFACT_URI_PREFIX)


def artifact_id_from_uri(value: str | os.PathLike[str]) -> str:
    raw = str(value)
    if not is_artifact_uri(raw):
        raise ValueError(f"Not an artifact URI: {raw}")
    artifact_id = raw.removeprefix(ARTIFACT_URI_PREFIX)
    if not artifact_id:
        raise ValueError("Artifact URI has an empty id")
    return artifact_id


def resolve_artifact_uri(value: str | os.PathLike[str], *, registry_path: str | Path | None = None) -> Path:
    """Resolve an artifact URI for consumers whose hash was verified with the checkpoint manifest."""
    artifact_id = artifact_id_from_uri(value)
    registry = load_registry(registry_path)
    if artifact_id not in registry:
        raise FileNotFoundError(f"Artifact {artifact_id!r} is absent from {_registry_path(registry_path)}")
    path = registry[artifact_id]
    if not path.exists():
        raise FileNotFoundError(f"Registered artifact does not exist: {path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_manifest(checkpoint_dir: str | Path) -> dict[str, Any] | None:
    path = Path(checkpoint_dir) / BASE_ARTIFACTS_FILENAME
    if not path.is_file():
        return None
    manifest = _read_json(path)
    if manifest.get("format") != MANIFEST_FORMAT or manifest.get("version") != VERSION:
        raise ValueError(f"Unsupported base-artifact manifest: {path}")
    requirements = manifest.get("requirements")
    if not isinstance(requirements, dict) or "policy_base" not in requirements:
        raise ValueError(f"Base-artifact manifest has no policy_base requirement: {path}")
    for role, requirement in requirements.items():
        if not isinstance(requirement, dict):
            raise ValueError(f"Invalid {role} requirement in {path}")
        required = {"artifact_id", "kind", "sha256"}
        if set(requirement) - (required | {"fingerprint_file"}) or not required <= set(requirement):
            raise ValueError(f"Invalid {role} requirement keys in {path}")
        if requirement["kind"] not in {"file", "directory", "directory_tree"}:
            raise ValueError(f"Invalid {role} artifact kind in {path}")
        if requirement["kind"] == "directory" and not requirement.get("fingerprint_file"):
            raise ValueError(f"Directory artifact {role} needs fingerprint_file in {path}")
        if requirement["kind"] != "directory" and requirement.get("fingerprint_file"):
            raise ValueError(f"Only directory fingerprint artifacts accept fingerprint_file in {path}")
        digest = requirement["sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Invalid {role} SHA256 in {path}")
    return manifest


def _registry_path(explicit: str | Path | None = None) -> Path:
    value = explicit or os.environ.get("LEROBOT_ARTIFACT_REGISTRY")
    if not value:
        raise FileNotFoundError(
            "Checkpoint uses portable base-artifact references, but LEROBOT_ARTIFACT_REGISTRY is unset"
        )
    return Path(value).expanduser()


def load_registry(explicit: str | Path | None = None) -> dict[str, Path]:
    path = _registry_path(explicit)
    registry = _read_json(path)
    if registry.get("format") != REGISTRY_FORMAT or registry.get("version") != VERSION:
        raise ValueError(f"Unsupported base-artifact registry: {path}")
    artifacts = registry.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"Base-artifact registry has no artifacts object: {path}")
    resolved: dict[str, Path] = {}
    for artifact_id, entry in artifacts.items():
        raw_path = entry.get("path") if isinstance(entry, dict) else entry
        if not isinstance(artifact_id, str) or not artifact_id or not isinstance(raw_path, str):
            raise ValueError(f"Invalid artifact registry entry {artifact_id!r} in {path}")
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        resolved[artifact_id] = candidate
    return resolved


@lru_cache(maxsize=32)
def _sha256_for_unchanged_file(path: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    stat = path.stat()
    return _sha256_for_unchanged_file(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"Cannot fingerprint empty artifact directory: {path}")
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode()
        digest.update(relative)
        digest.update(b"\0")
        digest.update(str(candidate.stat().st_size).encode())
        digest.update(b"\0")
        with candidate.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def resolve_requirement(
    requirement: dict[str, Any],
    *,
    registry_path: str | Path | None = None,
) -> Path:
    registry = load_registry(registry_path)
    artifact_id = requirement["artifact_id"]
    if artifact_id not in registry:
        raise FileNotFoundError(
            f"Artifact {artifact_id!r} is required by the checkpoint but absent from "
            f"{_registry_path(registry_path)}"
        )
    artifact_path = registry[artifact_id]
    kind = requirement["kind"]
    if kind in {"directory", "directory_tree"}:
        if not artifact_path.is_dir():
            raise FileNotFoundError(f"Artifact directory does not exist: {artifact_path}")
        fingerprint_path = (
            artifact_path / requirement["fingerprint_file"] if kind == "directory" else artifact_path
        )
    else:
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Artifact file does not exist: {artifact_path}")
        fingerprint_path = artifact_path
    if kind != "directory_tree" and not fingerprint_path.is_file():
        raise FileNotFoundError(f"Artifact fingerprint file does not exist: {fingerprint_path}")
    actual = sha256_directory(fingerprint_path) if kind == "directory_tree" else sha256_file(fingerprint_path)
    expected = requirement["sha256"]
    if actual != expected:
        raise RuntimeError(
            f"Artifact {artifact_id!r} SHA256 mismatch: expected {expected}, got {actual} "
            f"from {fingerprint_path}"
        )
    return artifact_path


def resolve_checkpoint_bases(
    checkpoint_dir: str | Path,
    *,
    registry_path: str | Path | None = None,
) -> dict[str, Path]:
    manifest = load_manifest(checkpoint_dir)
    if manifest is None:
        return {}
    return {
        role: resolve_requirement(requirement, registry_path=registry_path)
        for role, requirement in manifest["requirements"].items()
    }


def write_manifest(
    checkpoint_dir: str | Path,
    *,
    policy_artifact_id: str,
    policy_sha256: str,
    mem_vit_artifact_id: str | None = None,
    mem_vit_sha256: str | None = None,
    tokenizer_artifact_id: str | None = None,
    tokenizer_sha256: str | None = None,
) -> Path:
    requirements: dict[str, dict[str, str]] = {
        "policy_base": {
            "artifact_id": policy_artifact_id,
            "kind": "directory",
            "fingerprint_file": "model.safetensors",
            "sha256": policy_sha256,
        }
    }
    if (mem_vit_artifact_id is None) != (mem_vit_sha256 is None):
        raise ValueError("MEM-ViT artifact id and SHA256 must be provided together")
    if mem_vit_artifact_id is not None and mem_vit_sha256 is not None:
        requirements["mem_vit_base"] = {
            "artifact_id": mem_vit_artifact_id,
            "kind": "file",
            "sha256": mem_vit_sha256,
        }
    if (tokenizer_artifact_id is None) != (tokenizer_sha256 is None):
        raise ValueError("Tokenizer artifact id and SHA256 must be provided together")
    if tokenizer_artifact_id is not None and tokenizer_sha256 is not None:
        requirements["policy_tokenizer"] = {
            "artifact_id": tokenizer_artifact_id,
            "kind": "directory_tree",
            "sha256": tokenizer_sha256,
        }
    path = Path(checkpoint_dir) / BASE_ARTIFACTS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {"format": MANIFEST_FORMAT, "version": VERSION, "requirements": requirements},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path
