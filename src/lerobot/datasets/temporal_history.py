#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass
from random import Random


@dataclass(frozen=True)
class RandomHistorySamplingConfig:
    """Hierarchical sampling distribution for a shared multi-camera history clock."""

    keys: tuple[str, ...]
    num_frames: int
    fps: float
    nominal_interval_seconds: float
    global_interval_std_seconds: float
    local_interval_std_seconds: float
    min_interval_seconds: float
    max_interval_seconds: float
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.keys:
            raise ValueError("Random history sampling requires at least one feature key")
        if self.num_frames < 1:
            raise ValueError("Random history sampling requires num_frames >= 1")
        if self.fps <= 0:
            raise ValueError("Random history sampling requires fps > 0")
        if self.min_interval_seconds <= 0:
            raise ValueError("min_interval_seconds must be positive")
        if self.max_interval_seconds < self.min_interval_seconds:
            raise ValueError("max_interval_seconds must be >= min_interval_seconds")
        if not self.min_interval_seconds <= self.nominal_interval_seconds <= self.max_interval_seconds:
            raise ValueError("nominal_interval_seconds must lie within the configured interval bounds")
        if self.global_interval_std_seconds < 0 or self.local_interval_std_seconds < 0:
            raise ValueError("History interval standard deviations must be non-negative")
        if math.ceil(self.min_interval_seconds * self.fps) > math.floor(self.max_interval_seconds * self.fps):
            raise ValueError("The interval bounds contain no valid frame gap at the dataset FPS")

    @staticmethod
    def _sample_truncated_normal(
        rng: Random,
        mean: float,
        std: float,
        lower: float,
        upper: float,
    ) -> float:
        if std == 0:
            return min(upper, max(lower, mean))
        if lower == upper:
            return lower
        if mean < lower or mean > upper:
            # Exponential rejection avoids arbitrarily slow Gaussian rejection
            # when a valid history boundary lies deep in the distribution tail.
            sign = 1 if mean < lower else -1
            a, b = (
                ((lower - mean) / std, (upper - mean) / std)
                if sign == 1
                else ((mean - upper) / std, (mean - lower) / std)
            )
            rate = (a + math.sqrt(a * a + 4)) / 2
            mass = -math.expm1(-rate * (b - a))
            while True:
                sample = a - math.log1p(-rng.random() * mass) / rate
                if rng.random() <= math.exp(-0.5 * (sample - rate) ** 2):
                    return min(upper, max(lower, mean + sign * std * sample))
        while True:
            sample = rng.gauss(mean, std)
            if lower <= sample <= upper:
                return sample

    @property
    def minimum_span_frames(self) -> int:
        gap = math.ceil(self.min_interval_seconds * self.fps)
        if self.global_interval_std_seconds == 0 and self.local_interval_std_seconds == 0:
            gap = min(
                math.floor(self.max_interval_seconds * self.fps),
                max(gap, round(self.nominal_interval_seconds * self.fps)),
            )
        return (self.num_frames - 1) * gap

    def sample_delta_indices(self, rng: Random, *, max_history_frames: int | None = None) -> list[int]:
        """Sample chronological frame offsets ending at the current frame.

        One base interval is shared by the entire history window, representing
        sample-level timing bias. Each adjacent gap is then sampled around that
        base interval, representing local timing jitter.
        """
        if self.num_frames == 1:
            return [0]

        min_gap_frames = math.ceil(self.min_interval_seconds * self.fps)
        if max_history_frames is not None and max_history_frames < self.minimum_span_frames:
            raise ValueError("Valid history interval cannot contain the requested MEM frames")

        base_upper = self.max_interval_seconds
        if max_history_frames is not None and self.local_interval_std_seconds == 0:
            base_upper = min(base_upper, (max_history_frames // (self.num_frames - 1)) / self.fps)
        base_interval = self._sample_truncated_normal(
            rng,
            self.nominal_interval_seconds,
            self.global_interval_std_seconds,
            self.min_interval_seconds,
            base_upper,
        )
        reverse_offsets = [0]
        max_gap_frames = math.floor(self.max_interval_seconds * self.fps)
        for index in range(self.num_frames - 1):
            gap_limit = max_gap_frames
            if max_history_frames is not None:
                remaining = self.num_frames - 2 - index
                gap_limit = min(
                    gap_limit, max_history_frames + reverse_offsets[-1] - remaining * min_gap_frames
                )
            interval = self._sample_truncated_normal(
                rng,
                base_interval,
                self.local_interval_std_seconds,
                self.min_interval_seconds,
                min(self.max_interval_seconds, gap_limit / self.fps),
            )
            gap_frames = min(gap_limit, max(min_gap_frames, round(interval * self.fps)))
            reverse_offsets.append(reverse_offsets[-1] - gap_frames)
        return list(reversed(reverse_offsets))
