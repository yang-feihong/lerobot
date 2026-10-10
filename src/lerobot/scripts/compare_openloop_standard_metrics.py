#!/usr/bin/env python
"""Compare open-loop runs using only the canonical full-horizon metric contract.

This command intentionally refuses legacy output directories and incompatible
evaluation samples.  It is the supported input path for quantitative reports;
the first-action diagnostic CSV is never consulted.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


_FILENAME = "standard_full_horizon_metrics.json"
_CONTRACT_KEYS = (
    "dataset_repo_id",
    "split",
    "episodes",
    "frame_stride",
    "max_frames_per_episode",
    "include_onset_windows",
    "task_variant",
    "task_override",
    "sample_fingerprint_sha256",
    "b2_target_fingerprint_sha256",
    "evaluated_anchor_count",
    "evaluated_action_step_count",
    "inference_batch_size",
    "dataset_frequency_hz",
    "model_control_frequency_hz",
    "b2_representation",
    "flow_noise",
)


def load_standard_evaluation(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser()
    if path.is_dir():
        path = path / _FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. Run the current open-loop evaluator first; "
            "legacy/first-step metric files are not accepted."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError(f"Unsupported standard evaluation schema in {path}")
    if not payload.get("protocol_id"):
        raise ValueError(
            f"Missing protocol_id in {path}; only results produced by "
            "run_standard_openloop_benchmark.sh are accepted."
        )
    if payload.get("metric_scope") != "all_valid_steps_of_every_predicted_action_chunk":
        raise ValueError(f"Non-standard metric scope in {path}: {payload.get('metric_scope')!r}")
    metrics = payload.get("metrics_si")
    required = {
        "x_mae_m",
        "y_mae_m",
        "yaw_mae_rad",
        "xy_ade_m",
        "xy_fde_m",
        "yaw_ade_rad",
        "yaw_fde_rad",
    }
    if not isinstance(metrics, dict) or set(metrics) != required:
        raise ValueError(f"Standard seven-metric set is incomplete in {path}")
    return payload


def assert_comparable(reference: dict[str, Any], candidate: dict[str, Any]) -> None:
    if reference.get("protocol_id") != candidate.get("protocol_id"):
        raise ValueError(
            f"protocol_id mismatch: {reference.get('protocol_id')!r} != "
            f"{candidate.get('protocol_id')!r}"
        )
    if reference.get("horizon_steps") != candidate.get("horizon_steps"):
        raise ValueError(
            f"horizon_steps mismatch: {reference.get('horizon_steps')} != "
            f"{candidate.get('horizon_steps')}"
        )
    left = reference["comparison_contract"]
    right = candidate["comparison_contract"]
    mismatches = [key for key in _CONTRACT_KEYS if left.get(key) != right.get(key)]
    if mismatches:
        details = ", ".join(f"{key}: {left.get(key)!r} != {right.get(key)!r}" for key in mismatches)
        raise ValueError(f"Evaluation contracts are not comparable ({details})")


def _row(label: str, payload: dict[str, Any]) -> list[str]:
    metrics = payload["metrics_si"]
    return [
        label,
        f"{100 * metrics['x_mae_m']:.3f}",
        f"{100 * metrics['y_mae_m']:.3f}",
        f"{math.degrees(metrics['yaw_mae_rad']):.3f}",
        f"{100 * metrics['xy_ade_m']:.3f}",
        f"{100 * metrics['xy_fde_m']:.3f}",
        f"{math.degrees(metrics['yaw_ade_rad']):.3f}",
        f"{math.degrees(metrics['yaw_fde_rad']):.3f}",
    ]


def default_label(path: str | Path) -> str:
    path = Path(path).expanduser()
    return path.parent.name if path.name == _FILENAME else path.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluations", nargs="+", help="Standard open-loop output directories")
    parser.add_argument("--labels", nargs="*", default=None)
    args = parser.parse_args()
    if args.labels is not None and args.labels and len(args.labels) != len(args.evaluations):
        raise ValueError("--labels must contain exactly one label per evaluation")

    payloads = [load_standard_evaluation(path) for path in args.evaluations]
    for candidate in payloads[1:]:
        assert_comparable(payloads[0], candidate)
    labels = args.labels or [default_label(path) for path in args.evaluations]

    print("| Model | X MAE (cm) | Y MAE (cm) | Yaw MAE (deg) | XY ADE (cm) | XY FDE (cm) | Yaw ADE (deg) | Yaw FDE (deg) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, payload in zip(labels, payloads, strict=True):
        print("| " + " | ".join(_row(label, payload)) + " |")


if __name__ == "__main__":
    main()
