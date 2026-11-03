"""Normalize recording tool axes at the numeric reader boundary, never on disk."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from lerobot.utils.ee_frames import EE_FRAME_CONTRACT, convert_named_features, validate_ee_frame


class EpisodeEEFrameAdapter:
    def __init__(self, root, features, model_frame, episodes):
        root = Path(root)
        self.model_frame = validate_ee_frame(model_frame)
        metadata = json.loads((root / "meta/ee_frames.json").read_text())
        self.metadata_sha256 = hashlib.sha256((root / "meta/ee_frames.json").read_bytes()).hexdigest()
        if metadata["version"] != 1 or metadata["ee_frame_contract"] != EE_FRAME_CONTRACT:
            raise ValueError("Unsupported dataset EE frame contract")
        manifest = json.loads((root / "diagnostics/conversion_manifest.json").read_text())
        self.frames = {}
        self.names = {key: list(features[key]["names"]) for key in ("action", "observation.state")}
        for key, record in manifest["bags"].items():
            if record.get("status") != "done" or int(record["episode_index"]) not in episodes:
                continue
            frame = validate_ee_frame(metadata["recordings"][key])
            if "ee_frame" in record and record["ee_frame"] != frame:
                raise ValueError(f"Recording EE frame disagrees with sidecar: {key}")
            episode = int(record["episode_index"])
            if episode in self.frames and self.frames[episode] != frame:
                raise ValueError(f"Conflicting EE frames for episode {episode}")
            self.frames[episode] = frame
        if set(self.frames) != set(episodes):
            raise ValueError("EE metadata does not cover selected episodes")

    def feature(self, values, key, episode):
        if self.frames[int(episode)] == self.model_frame:
            return values
        return convert_named_features(values, self.names[key], self.frames[int(episode)], self.model_frame)

    def sample(self, item, episode):
        for key in self.names:
            if key in item:
                item[key] = self.feature(item[key], key, episode)
        return item


def configure_dataset_ee_frame(dataset, model_frame):
    """Use the same adapter for ordinary samples and statistics' direct HF reads."""
    from .compute_stats import aggregate_stats, get_feature_stats

    episodes = dataset.episodes if dataset.episodes is not None else list(range(dataset.meta.total_episodes))
    adapter = EpisodeEEFrameAdapter(dataset.root, dataset.meta.features, model_frame, set(episodes))
    dataset.reader.ee_frame_adapter = adapter
    if all(frame == model_frame for frame in adapter.frames.values()):
        return
    stats = []
    columns = ["episode_index", *adapter.names]
    table = dataset.hf_dataset.select_columns(columns).with_format("numpy")
    for batch in table.iter(batch_size=65536):
        numeric = {}
        episode_indices = np.asarray(batch["episode_index"]).reshape(-1)
        for key in adapter.names:
            values = np.asarray(batch[key], dtype=np.float32).copy()
            for episode in np.unique(episode_indices):
                selected = episode_indices == episode
                values[selected] = adapter.feature(values[selected], key, int(episode))
            numeric[key] = get_feature_stats(values, axis=0, keepdims=False)
        stats.append(numeric)
    measured = aggregate_stats(stats)
    dataset.meta.stats = copy.deepcopy(dataset.meta.stats)
    # Unchanged channels keep their original statistics (including old resumes).
    for key, names in adapter.names.items():
        indices = [i for i, name in enumerate(names) if "ee_" in name and "rot6d_col1_" in name]
        for metric, value in measured[key].items():
            if metric != "count" and metric in dataset.meta.stats[key]:
                dataset.meta.stats[key][metric][indices] = value[indices]
