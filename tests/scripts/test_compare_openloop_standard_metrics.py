from __future__ import annotations

import json

import pytest

from lerobot.scripts.compare_openloop_standard_metrics import (
    assert_comparable,
    default_label,
    load_standard_evaluation,
)


def _payload() -> dict:
    return {
        "schema_version": 2,
        "protocol_id": "staff1_stage1_v1",
        "metric_scope": "all_valid_steps_of_every_predicted_action_chunk",
        "horizon_steps": 50,
        "metrics_si": {
            "x_mae_m": 0.01,
            "y_mae_m": 0.02,
            "yaw_mae_rad": 0.03,
            "xy_ade_m": 0.04,
            "xy_fde_m": 0.05,
            "yaw_ade_rad": 0.03,
            "yaw_fde_rad": 0.06,
        },
        "comparison_contract": {
            "dataset_repo_id": "local/staff1_approach",
            "split": "all",
            "episodes": [0, 1],
            "frame_stride": 20,
            "max_frames_per_episode": 0,
            "include_onset_windows": False,
            "task_variant": "first",
            "task_override": "Approach the door.",
            "sample_fingerprint_sha256": "samples",
            "b2_target_fingerprint_sha256": "targets",
            "evaluated_anchor_count": 10,
            "evaluated_action_step_count": 500,
            "inference_batch_size": 4,
            "dataset_frequency_hz": 50.0,
            "model_control_frequency_hz": 50.0,
            "b2_representation": "pose_delta",
            "flow_noise": {
                "deterministic": True,
                "scheme": "blake2b_seed_per_episode_frame_v1",
                "base_seed": 1000,
            },
        },
    }


def test_rejects_legacy_or_first_step_only_directory(tmp_path) -> None:
    (tmp_path / "trajectory_metrics.csv").write_text("not,a,benchmark\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="first-step metric files are not accepted"):
        load_standard_evaluation(tmp_path)


def test_loads_complete_seven_metric_contract(tmp_path) -> None:
    path = tmp_path / "standard_full_horizon_metrics.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    assert load_standard_evaluation(tmp_path)["horizon_steps"] == 50


@pytest.mark.parametrize(
    ("key", "replacement"),
    [
        ("sample_fingerprint_sha256", "other-samples"),
        ("b2_target_fingerprint_sha256", "other-targets"),
        ("evaluated_action_step_count", 499),
        ("b2_representation", "velocity"),
        ("task_override", "Open the door."),
        ("flow_noise", {"deterministic": False}),
        ("inference_batch_size", 8),
    ],
)
def test_rejects_incompatible_comparisons(key, replacement) -> None:
    reference = _payload()
    candidate = _payload()
    candidate["comparison_contract"][key] = replacement
    with pytest.raises(ValueError, match=key):
        assert_comparable(reference, candidate)


def test_accepts_different_checkpoints_with_identical_protocol() -> None:
    reference = _payload()
    candidate = _payload()
    reference["provenance"] = {"policy_path": "/a"}
    candidate["provenance"] = {"policy_path": "/b"}
    assert_comparable(reference, candidate)


def test_rejects_different_protocol_ids() -> None:
    reference = _payload()
    candidate = _payload()
    candidate["protocol_id"] = "other"
    with pytest.raises(ValueError, match="protocol_id"):
        assert_comparable(reference, candidate)


def test_default_label_uses_result_directory_for_metric_file(tmp_path) -> None:
    path = tmp_path / "model_010000" / "standard_full_horizon_metrics.json"
    assert default_label(path) == "model_010000"
