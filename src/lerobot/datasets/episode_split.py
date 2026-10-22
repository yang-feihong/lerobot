"""Recording-level, append-stable train/validation assignment.

Identity is immutable acquisition provenance, never the current dataset path,
episode number, instruction text, annotation status, task or list position.
"""

import hashlib
import json
import math
import posixpath
from pathlib import Path

SPLIT_POLICY = "recording_hash_v1"


def canonical_recording_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("recording identity must be a nonempty immutable string")
    value = value.replace("\\", "/")
    if ".." in value.split("/"):
        raise ValueError("recording identity must not contain parent traversal")
    return posixpath.normpath(value)


def load_recording_ids(dataset_root: Path, episode_indices: list[int], catalog=None) -> dict[int, str]:
    """Semantic provenance or an explicit manifest is required; never guess from indices."""
    if catalog is not None:
        ids = catalog.recording_ids
    else:
        path = dataset_root / "meta/episode_split_groups.json"
        payload = json.loads(path.read_text())
        if payload["format"] != "lerobot.episode_split_groups" or payload["version"] != 1:
            raise ValueError(f"unsupported recording identity manifest: {path}")
        ids = {int(key): value for key, value in payload["recording_ids"].items()}
    missing = set(episode_indices) - ids.keys()
    if missing:
        raise ValueError(
            f"Missing immutable recording identity for episodes {sorted(missing)[:10]}; "
            "provide recording_id/source_path in semantic views or meta/episode_split_groups.json"
        )
    return {episode: canonical_recording_id(ids[episode]) for episode in episode_indices}


def recording_is_validation(recording_id: str, eval_split: float) -> bool:
    if not math.isfinite(eval_split) or not 0 <= eval_split < 1:
        raise ValueError("eval_split must be finite and in [0, 1)")
    identity = canonical_recording_id(recording_id)
    digest = hashlib.sha256((SPLIT_POLICY + "\0" + identity).encode("utf-8")).digest()
    numerator, denominator = eval_split.as_integer_ratio()
    return int.from_bytes(digest, "big") * denominator < numerator * (1 << 256)


def split_episode_indices(
    *, episode_indices: list[int], recording_id_by_episode: dict[int, str], eval_split: float
) -> tuple[list[int], list[int]]:
    if not math.isfinite(eval_split) or not 0 <= eval_split < 1:
        raise ValueError("eval_split must be finite and in [0, 1)")
    if len(set(episode_indices)) != len(episode_indices):
        raise ValueError("duplicate episode indices")
    train, validation = [], []
    for episode in episode_indices:
        if episode not in recording_id_by_episode:
            raise ValueError(f"missing recording identity for episode {episode}")
        target = (
            validation if recording_is_validation(recording_id_by_episode[episode], eval_split) else train
        )
        target.append(episode)
    return train, validation


def split_manifest(recording_ids: dict[int, str], eval_split: float) -> dict:
    train, validation = split_episode_indices(
        episode_indices=list(recording_ids), recording_id_by_episode=recording_ids, eval_split=eval_split
    )
    held_out = set(validation)
    return {
        "format": "lerobot.dataset_split",
        "version": 1,
        "policy": SPLIT_POLICY,
        "eval_split": eval_split,
        "train_episodes": train,
        "eval_episodes": validation,
        "episodes": {
            str(ep): {"recording_id": identity, "split": "eval" if ep in held_out else "train"}
            for ep, identity in recording_ids.items()
        },
    }


def validate_resume_split(saved: dict, current: dict, *, updated_dataset: bool) -> None:
    """Changing old holdout membership requires an explicit new-dataset fork."""
    if updated_dataset:
        return
    if saved.get("policy") != SPLIT_POLICY or saved.get("eval_split") != current["eval_split"]:
        raise ValueError("Checkpoint split differs: explicitly fork with an updated dataset and a new run")
    old_groups = {row["recording_id"]: row["split"] for row in saved["episodes"].values()}
    new_groups = {row["recording_id"]: row["split"] for row in current["episodes"].values()}
    if old_groups != new_groups:
        raise ValueError("Checkpoint recording membership changed: use the updated-dataset fork")
