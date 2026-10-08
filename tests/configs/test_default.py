# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
import pytest

from lerobot.configs.default import DatasetConfig


def test_dataset_config_valid():
    DatasetConfig(repo_id="user/repo", episodes=[0, 1, 2])


def test_dataset_config_negative_episodes():
    with pytest.raises(ValueError, match="non-negative"):
        DatasetConfig(repo_id="user/repo", episodes=[0, -1, 2])


def test_dataset_config_duplicate_episodes():
    with pytest.raises(ValueError, match="duplicates"):
        DatasetConfig(repo_id="user/repo", episodes=[0, 1, 1, 2])


def test_dataset_config_none_episodes_ok():
    DatasetConfig(repo_id="user/repo", episodes=None)


def test_dataset_config_empty_episodes_ok():
    DatasetConfig(repo_id="user/repo", episodes=[])


def test_dataset_config_semantic_phase_weights_are_explicit_and_consistent():
    DatasetConfig(
        repo_id="user/repo",
        semantic_views_path="/data/semantic_views.json",
        semantic_phases=["approach", "handle_press"],
        semantic_phase_weights={"approach": 1.0, "handle_press": 1.0},
        semantic_source_weights={"full_episode": 1.0},
    )
    with pytest.raises(ValueError, match="exactly match"):
        DatasetConfig(
            repo_id="user/repo",
            semantic_views_path="/data/semantic_views.json",
            semantic_phases=["approach", "handle_press"],
            semantic_phase_weights={"approach": 1.0},
        )
    with pytest.raises(ValueError, match="requires dataset.semantic_phases"):
        DatasetConfig(
            repo_id="user/repo",
            semantic_views_path="/data/semantic_views.json",
            semantic_source_weights={"full_episode": 1.0},
        )


def test_dataset_config_semantic_view_kind_weights_are_an_explicit_allowlist():
    config = DatasetConfig(
        repo_id="user/repo",
        semantic_views_path="/data/semantic_views.json",
        semantic_view_kind_weights={
            "primary": 1.0,
            "compatible_goal": 0.0,
            "completed_counterfactual": 0.0,
            "blocked_counterfactual": 0.0,
        },
    )
    assert config.semantic_view_kind_weights["primary"] == 1.0
    with pytest.raises(ValueError, match="must enable at least one"):
        DatasetConfig(
            repo_id="user/repo",
            semantic_views_path="/data/semantic_views.json",
            semantic_view_kind_weights={"primary": 0.0},
        )


def test_dataset_config_accepts_exact_state_instruction_matrix():
    matrix = {
        "approach": {
            "goal_approach": {
                "status": "active",
                "action_supervision": "demonstrated",
                "completion_boundary": "state_end",
            }
        }
    }
    config = DatasetConfig(
        repo_id="user/repo",
        semantic_views_path="/data/semantic_views.json",
        semantic_phases=["approach"],
        semantic_phase_weights={"approach": 1.0},
        semantic_state_instruction_matrix=matrix,
    )
    assert config.semantic_state_instruction_matrix == matrix
    with pytest.raises(ValueError, match="mutually exclusive"):
        DatasetConfig(
            repo_id="user/repo",
            semantic_views_path="/data/semantic_views.json",
            semantic_phases=["approach"],
            semantic_state_instruction_matrix=matrix,
            semantic_view_kind_weights={"primary": 1.0},
        )
