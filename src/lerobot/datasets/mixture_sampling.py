#!/usr/bin/env python

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DatasetMixtureSource:
    name: str
    weight: float
    episode_start: int
    episode_stop: int

    @property
    def episode_indices(self) -> list[int]:
        return list(range(self.episode_start, self.episode_stop))


def load_dataset_mixture_manifest(path: str | Path, *, num_episodes: int) -> list[DatasetMixtureSource]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or len(raw_sources) < 2:
        raise ValueError(f"{manifest_path}: sources must contain at least two entries")

    sources: list[DatasetMixtureSource] = []
    names: set[str] = set()
    owner = np.full(num_episodes, -1, dtype=np.int64)
    for source_index, raw in enumerate(raw_sources):
        name = str(raw.get("name", "")).strip()
        if not name or name in names:
            raise ValueError(f"{manifest_path}: every source name must be non-empty and unique")
        names.add(name)
        weight = float(raw["weight"])
        episode_range = raw.get("episode_range")
        if (
            not isinstance(episode_range, list)
            or len(episode_range) != 2
            or not all(isinstance(value, int) for value in episode_range)
        ):
            raise ValueError(f"{manifest_path}: source {name!r} needs integer episode_range=[start, stop]")
        start, stop = episode_range
        if weight <= 0.0 or not 0 <= start < stop <= num_episodes:
            raise ValueError(
                f"{manifest_path}: invalid source {name!r}: weight={weight}, "
                f"episode_range={episode_range}, num_episodes={num_episodes}"
            )
        if np.any(owner[start:stop] >= 0):
            raise ValueError(f"{manifest_path}: source {name!r} overlaps another episode range")
        owner[start:stop] = source_index
        sources.append(DatasetMixtureSource(name, weight, start, stop))

    if np.any(owner < 0):
        missing = np.flatnonzero(owner < 0)
        raise ValueError(
            f"{manifest_path}: source ranges must cover every episode exactly once; "
            f"first uncovered episodes={missing[:10].tolist()}"
        )
    total_weight = sum(source.weight for source in sources)
    return [
        DatasetMixtureSource(
            source.name,
            source.weight / total_weight,
            source.episode_start,
            source.episode_stop,
        )
        for source in sources
    ]
