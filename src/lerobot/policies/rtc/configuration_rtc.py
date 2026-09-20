#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
Real Time Chunking (RTC) and Bidirectional Decoding (BID) configuration classes.

Based on:
- Real Time Chunking: https://www.physicalintelligence.company/research/real_time_chunking
"""

from dataclasses import dataclass

from lerobot.configs import RTCAttentionSchedule


@dataclass
class RTCConfig:
    """Configuration for Real Time Chunking (RTC) inference.

    RTC improves real-time inference by treating chunk generation as an inpainting problem,
    strategically handling overlapping timesteps between action chunks using prefix attention.
    """

    # Infrastructure
    enabled: bool = True

    # Core RTC settings
    # Todo change to exp
    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.LINEAR
    max_guidance_weight: float = 10.0
    execution_horizon: int = 10

    # Debug settings
    debug: bool = False
    debug_maxlen: int = 100

    # Existing guided inpainting remains the default. "training" selects hard
    # prefix conditioning learned with TrainingRTCConfig.
    mode: str = "inference"

    def __post_init__(self):
        """Validate RTC configuration parameters."""
        if self.mode not in {"inference", "training"}:
            raise ValueError(f"RTC mode must be 'inference' or 'training', got {self.mode!r}")
        if self.max_guidance_weight <= 0:
            raise ValueError(f"max_guidance_weight must be positive, got {self.max_guidance_weight}")
        if self.debug_maxlen <= 0:
            raise ValueError(f"debug_maxlen must be positive, got {self.debug_maxlen}")


@dataclass
class TrainingRTCConfig:
    """Optional clean-prefix conditioning during flow-matching training.

    ``simulated_delay`` is an exclusive upper bound, matching the reference
    implementations: a value of 5 trains delays 0, 1, 2, 3, and 4.
    """

    enabled: bool = False
    simulated_delay: int = 5
    delay_distribution: str = "exponential"

    def __post_init__(self):
        if type(self.simulated_delay) is not int or self.simulated_delay < 1:
            raise ValueError("training RTC simulated_delay must be a positive integer")
        if self.delay_distribution not in {"exponential", "uniform"}:
            raise ValueError("training RTC delay_distribution must be 'exponential' or 'uniform'")
