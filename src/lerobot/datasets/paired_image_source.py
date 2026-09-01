"""Resolve physically validated simulator videos paired with a LeRobot dataset."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SIM_VIDEO_FIELDS = {
    "observation.images.base_0_rgb": "sim_base_video",
    "observation.images.left_wrist_0_rgb": "sim_wrist_video",
}


@dataclass(frozen=True)
class PairedEpisode:
    base_video: Path
    wrist_video: Path
    correspondence: Path
    video_fps: float


class PairedImageSource:
    def __init__(
        self,
        *,
        mode: str,
        manifest: str | Path,
        root: str | Path,
        mixed_sim_probability: float,
        seed: int,
        episodes: list[int],
    ) -> None:
        if mode not in {"real", "sim", "mixed"}:
            raise ValueError(f"Unsupported paired image mode: {mode!r}")
        if not 0.0 <= mixed_sim_probability <= 1.0:
            raise ValueError("mixed_sim_probability must be in [0, 1]")
        self.mode = mode
        self.root = Path(root)
        self.mixed_sim_probability = mixed_sim_probability
        self.seed = seed
        self._episodes = self._load_manifest(Path(manifest))
        missing = sorted(set(episodes) - self._episodes.keys())
        if missing:
            preview = ", ".join(str(value) for value in missing[:20])
            raise ValueError(
                f"Simulator image manifest does not cover {len(missing)} requested episode(s): {preview}"
            )
        self._correspondence_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _load_manifest(self, path: Path) -> dict[int, PairedEpisode]:
        if not path.is_file():
            raise FileNotFoundError(path)
        result: dict[int, PairedEpisode] = {}
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not value.get("success", False):
                    raise ValueError(f"Unsuccessful replay in {path}:{line_number}")
                episode_index = int(value["episode_index"])
                if episode_index in result:
                    raise ValueError(f"Duplicate simulator replay for episode {episode_index}")
                recorded_rollout_dir = Path(value["rollout_dir"])
                rollout_dir = (
                    self.root / recorded_rollout_dir.name
                    if recorded_rollout_dir.is_absolute()
                    else self.root / recorded_rollout_dir
                )
                episode = PairedEpisode(
                    base_video=rollout_dir / value["sim_base_video"],
                    wrist_video=rollout_dir / value["sim_wrist_video"],
                    correspondence=rollout_dir / value["frame_correspondence"],
                    video_fps=float(value["sim_video_fps"]),
                )
                for required_path in (
                    episode.base_video,
                    episode.wrist_video,
                    episode.correspondence,
                ):
                    if not required_path.is_file():
                        raise FileNotFoundError(required_path)
                result[episode_index] = episode
        return result

    def use_sim(self, episode_index: int, source_timestamps: list[float]) -> bool:
        if self.mode == "sim":
            return True
        if self.mode == "real":
            return False
        current_timestamp = max(source_timestamps)
        token = f"{self.seed}:{episode_index}:{current_timestamp:.6f}".encode()
        draw = int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "big") / 2**64
        return draw < self.mixed_sim_probability

    def resolve(
        self, episode_index: int, video_key: str, source_timestamps: list[float]
    ) -> tuple[Path, list[float]]:
        if video_key not in SIM_VIDEO_FIELDS:
            raise KeyError(f"No simulator pairing is defined for camera {video_key!r}")
        episode = self._episodes[episode_index]
        if episode_index not in self._correspondence_cache:
            with episode.correspondence.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            source = np.asarray([float(row["source_time_s"]) for row in rows])
            if len(source) == 0 or np.any(np.diff(source) <= 0.0):
                raise ValueError(f"Invalid frame correspondence: {episode.correspondence}")
            simulator = np.arange(len(source), dtype=np.float64) / episode.video_fps
            self._correspondence_cache[episode_index] = source, simulator
        source, simulator = self._correspondence_cache[episode_index]
        query = np.asarray(source_timestamps, dtype=np.float64)
        right = np.searchsorted(source, query, side="left").clip(0, len(source) - 1)
        left = np.maximum(right - 1, 0)
        nearest = np.where(abs(source[right] - query) < abs(source[left] - query), right, left)
        video = episode.base_video if video_key.endswith("base_0_rgb") else episode.wrist_video
        return video, simulator[nearest].tolist()
