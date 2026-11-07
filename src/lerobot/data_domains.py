"""Dependency-free recording origin, independent of augmentation/source groups.

The frozen conversion manifest is sufficient for historical releases. Never
consult raw mounts, directory names, tags, or current filesystem availability.
"""

import json
import math
from pathlib import Path


def effective_domain_weights(weights):
    """None is the new automatic 3:1 policy; {} preserves legacy natural mixing."""
    return {"simulation": 3.0, "real": 1.0} if weights is None else weights


def validate_domain_weights(weights):
    if weights is None:
        return None
    if not isinstance(weights, dict) or set(weights) - {"simulation", "real"}:
        raise ValueError("Domain weights require only simulation and real keys")
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
        for v in weights.values()
    ):
        raise ValueError("Domain weights must be finite non-negative numbers")
    if weights and (not any(weights.values()) or not math.isfinite(sum(weights.values()))):
        raise ValueError("Domain weights must enable at least one domain with finite total")
    return weights


def recording_domain(record, *, required=True):
    """Interpret explicit metadata, with an audited legacy rosbag fallback."""
    explicit = record.get("data_domain")
    source_type = record.get("source_type")
    source = str(record.get("bag_path", ""))
    inferred = None
    if source_type == "accepted_simulation_shard":
        inferred = "simulation"
    elif source_type in {"rosbag", "rosbag1"} or source.endswith((".bag", ".bag.active")):
        inferred = "real"
    if explicit is not None and (explicit not in {"real", "simulation"} or inferred not in {None, explicit}):
        raise ValueError("Conflicting or invalid recording data_domain")
    if explicit or inferred:
        return explicit or inferred
    if required:
        raise ValueError("Recording has no authoritative origin metadata")
    return None


def validate_instruction_domain_weights(overrides, default, matrix):
    """Instruction-specific restrictions never silently invent a missing domain."""
    validate_domain_weights(default)
    default = effective_domain_weights(default)
    if not isinstance(overrides, dict):
        raise ValueError("Instruction domain weights must be an object")
    if overrides and not default:
        raise ValueError("Instruction domain weights require explicit default domain weights")
    intents = {intent for row in matrix.values() for intent in row}
    if set(overrides) - intents:
        raise ValueError("Instruction domain weights reference an unconfigured instruction")
    for weights in overrides.values():
        validate_domain_weights(weights)
        if not weights:
            raise ValueError("Instruction domain weights cannot be empty")
        if any(w > 0 and default.get(d, 0) <= 0 for d, w in weights.items()):
            raise ValueError("Instruction domain weights cannot enable a globally excluded domain")
    return overrides


def load_episode_domains(sidecar_path, episodes):
    """Join sidecar identity to immutable manifest; fail on reindex/membership drift."""
    manifest_path = Path(sidecar_path).parent.parent / "diagnostics/conversion_manifest.json"
    records = {}
    if manifest_path.is_file():
        for record in json.loads(manifest_path.read_text())["bags"].values():
            if record.get("status") != "done":
                continue
            index = int(record["episode_index"])
            if index in records:
                raise ValueError(f"Duplicate origin manifest episode: {index}")
            records[index] = record
    domains = {}
    for key, episode in episodes.items():
        index = int(key)
        explicit = episode.get("data_domain")
        if explicit is not None and explicit not in {"real", "simulation"}:
            raise ValueError(f"Invalid data_domain for episode {index}")
        record = records.get(index)
        if record is not None:
            if episode.get("source_path") != record["bag_path"]:
                raise ValueError(f"Origin manifest source mismatch for episode {index}")
            domain = recording_domain(record, required=False)
            if explicit is not None and domain is not None and explicit != domain:
                raise ValueError(f"Sidecar origin disagrees with manifest for episode {index}")
            if explicit or domain:
                domains[index] = explicit or domain
        elif explicit in {"real", "simulation"}:
            domains[index] = explicit
        elif explicit is not None:
            raise ValueError(f"Invalid data_domain for episode {index}")
    return domains


def select_domain_episodes(catalog, selected, weights):
    """Filter before split/loading/derived-action statistics without changing hashes."""
    validate_domain_weights(weights)
    weights = effective_domain_weights(weights)
    if not weights:
        return list(selected)
    missing = set(selected) - set(catalog.episode_domains)
    if missing:
        raise ValueError(f"Missing recording domain for episodes: {sorted(missing)[:10]}")
    result = [ep for ep in selected if weights.get(catalog.episode_domains[ep], 0) > 0]
    if not result:
        raise ValueError("No episodes in requested recording domains")
    return result
