from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .b2_action_transform import EE_DELTA_ROTVEC_NAMES


def _active(actions: np.ndarray, names: Sequence[str]) -> np.ndarray:
    indices = {name: index for index, name in enumerate(names)}
    active = np.ones(actions.shape[:-1], dtype=bool)
    if "arm_teleop_inactive" in indices:
        active &= actions[..., indices["arm_teleop_inactive"]] < 0.5
    if "arm_reset" in indices:
        active &= actions[..., indices["arm_reset"]] < 0.5
    if "task_complete" in indices:
        active &= actions[..., indices["task_complete"]] < 0.5
    return active


def _rot6d_to_matrix(value: np.ndarray) -> np.ndarray:
    first = value[:3] / max(np.linalg.norm(value[:3]), 1e-12)
    second = value[3:6] - first * np.dot(first, value[3:6])
    second /= max(np.linalg.norm(second), 1e-12)
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def _rotvec_to_matrix(value: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(value))
    skew = np.array(
        [[0.0, -value[2], value[1]], [value[2], 0.0, -value[0]], [-value[1], value[0], 0.0]],
        dtype=np.float64,
    )
    if theta < 1e-8:
        return np.eye(3) + skew + 0.5 * (skew @ skew)
    return np.eye(3) + np.sin(theta) / theta * skew + (1.0 - np.cos(theta)) / theta**2 * (skew @ skew)


def _matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    cos_theta = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    vee = (
        np.array(
            [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]],
            dtype=np.float64,
        )
        * 0.5
    )
    if theta < 1e-8:
        return vee
    return vee * (theta / max(np.sin(theta), 1e-12))


def _rotation_decoder(names: Sequence[str]):
    indices = {name: index for index, name in enumerate(names)}
    if set(EE_DELTA_ROTVEC_NAMES).issubset(indices):
        selected = [indices[name] for name in EE_DELTA_ROTVEC_NAMES]
        return lambda action: _rotvec_to_matrix(action[selected])
    rot6d_names = [
        "height_invariant_ee_delta_rot6d_col0_x",
        "height_invariant_ee_delta_rot6d_col0_y",
        "height_invariant_ee_delta_rot6d_col0_z",
        "height_invariant_ee_delta_rot6d_col1_x",
        "height_invariant_ee_delta_rot6d_col1_y",
        "height_invariant_ee_delta_rot6d_col1_z",
    ]
    if set(rot6d_names).issubset(indices):
        selected = [indices[name] for name in rot6d_names]
        return lambda action: _rot6d_to_matrix(action[selected])
    return None


def _cumulative_rotation(
    chunk: np.ndarray,
    active: np.ndarray,
    decode_rotation,
) -> np.ndarray:
    rotation = np.eye(3)
    for action, is_active in zip(chunk, active, strict=True):
        if is_active:
            rotation = decode_rotation(action) @ rotation
    return rotation


def compute_manipulation_onset_metrics(
    *,
    frame_indices: np.ndarray,
    expert_chunks: np.ndarray,
    predicted_chunks: np.ndarray,
    action_names: Sequence[str],
    control_frequency_hz: float,
    dataset_frequency_hz: float,
    dataset_group: str,
    onset_tolerance_seconds: float = 0.2,
    wrist_sign_threshold_rad: float = 0.2,
) -> dict[str, float | int | str | None]:
    if expert_chunks.shape != predicted_chunks.shape or expert_chunks.ndim != 3:
        raise ValueError(
            f"Expected matching (N, T, D) chunks, got {expert_chunks.shape} and {predicted_chunks.shape}"
        )
    if expert_chunks.shape[-1] != len(action_names) or len(frame_indices) != len(expert_chunks):
        raise ValueError("Frame indices, chunks, and action names do not share one schema")
    expert_first_active = _active(expert_chunks[:, 0], action_names)
    predicted_first_active = _active(predicted_chunks[:, 0], action_names)
    true_positive = int(np.count_nonzero(expert_first_active & predicted_first_active))
    predicted_positive = int(np.count_nonzero(predicted_first_active))
    actual_positive = int(np.count_nonzero(expert_first_active))

    previous_expert = np.r_[bool(frame_indices[0] > 0), expert_first_active[:-1]]
    previous_predicted = np.r_[bool(frame_indices[0] > 0), predicted_first_active[:-1]]
    discontinuity = np.r_[False, np.diff(frame_indices) != 1]
    previous_expert[discontinuity] = True
    previous_predicted[discontinuity] = True
    expert_onsets = np.flatnonzero(expert_first_active & ~previous_expert)
    predicted_onsets = np.flatnonzero(predicted_first_active & ~previous_predicted)
    tolerance_frames = int(round(onset_tolerance_seconds * dataset_frequency_hz))
    unmatched_predicted = set(predicted_onsets.tolist())
    matched_onsets: list[int] = []
    for onset in expert_onsets:
        candidates = [
            index
            for index in unmatched_predicted
            if abs(int(frame_indices[index]) - int(frame_indices[onset])) <= tolerance_frames
        ]
        if candidates:
            best = min(
                candidates,
                key=lambda index: abs(int(frame_indices[index]) - int(frame_indices[onset])),
            )
            unmatched_predicted.remove(best)
            matched_onsets.append(int(onset))

    indices = {name: index for index, name in enumerate(action_names)}
    position_names = [
        "height_invariant_ee_delta_x",
        "height_invariant_ee_delta_y",
        "height_invariant_ee_delta_z",
    ]
    position_indices = (
        [indices[name] for name in position_names] if set(position_names).issubset(indices) else []
    )
    decode_rotation = _rotation_decoder(action_names)
    horizon = min(expert_chunks.shape[1], int(round(control_frequency_hz)))
    endpoint_errors: list[float] = []
    so3_errors: list[float] = []
    sign_correct: list[bool] = []
    expert_group_sign_correct: list[bool] = []
    expected_sign = 1 if "staff1" in dataset_group.lower() else -1 if "staff2" in dataset_group.lower() else 0
    predicted_active_onsets = 0
    for onset in expert_onsets:
        expert_chunk = expert_chunks[onset, :horizon]
        predicted_chunk = predicted_chunks[onset, :horizon]
        expert_active = _active(expert_chunk, action_names)
        predicted_active = _active(predicted_chunk, action_names)
        if not predicted_active.any():
            continue
        predicted_active_onsets += 1
        if position_indices:
            expert_endpoint = expert_chunk[expert_active][:, position_indices].sum(axis=0)
            predicted_endpoint = predicted_chunk[predicted_active][:, position_indices].sum(axis=0)
            endpoint_errors.append(float(np.linalg.norm(predicted_endpoint - expert_endpoint)))
        if decode_rotation is not None:
            expert_rotation = _cumulative_rotation(expert_chunk, expert_active, decode_rotation)
            predicted_rotation = _cumulative_rotation(predicted_chunk, predicted_active, decode_rotation)
            rotation_error = predicted_rotation @ expert_rotation.T
            so3_errors.append(float(np.arccos(np.clip((np.trace(rotation_error) - 1.0) * 0.5, -1.0, 1.0))))
            expert_twist = float(_matrix_to_rotvec(expert_rotation)[0])
            predicted_twist = float(_matrix_to_rotvec(predicted_rotation)[0])
            if abs(expert_twist) >= wrist_sign_threshold_rad:
                reference_sign = expected_sign or (1 if expert_twist > 0 else -1)
                sign_correct.append(predicted_twist * reference_sign > 0)
                if expected_sign:
                    expert_group_sign_correct.append(expert_twist * expected_sign > 0)

    return {
        "dataset_group": dataset_group,
        "manipulation_onset_count": int(len(expert_onsets)),
        "manipulation_onset_detected": int(len(matched_onsets)),
        "manipulation_onset_detection_rate": (
            float(len(matched_onsets) / len(expert_onsets)) if len(expert_onsets) else None
        ),
        "arm_mode_precision": float(true_positive / predicted_positive) if predicted_positive else None,
        "arm_mode_recall": float(true_positive / actual_positive) if actual_positive else None,
        "arm_mode_true_positive_frames": true_positive,
        "arm_mode_predicted_active_frames": predicted_positive,
        "arm_mode_expert_active_frames": actual_positive,
        "predicted_active_onset_count": predicted_active_onsets,
        "predicted_active_ee_translation_endpoint_error_m": (
            float(np.mean(endpoint_errors)) if endpoint_errors else None
        ),
        "wrist_twist_sign_sample_count": len(sign_correct),
        "wrist_twist_sign_accuracy": float(np.mean(sign_correct)) if sign_correct else None,
        "expert_staff_direction_consistency": (
            float(np.mean(expert_group_sign_correct)) if expert_group_sign_correct else None
        ),
        "cumulative_so3_error_1s_rad": float(np.mean(so3_errors)) if so3_errors else None,
        "cumulative_so3_error_sample_count": len(so3_errors),
    }


def aggregate_manipulation_metrics(
    rows: Sequence[dict[str, float | int | str | None]], dataset_group: str
) -> dict[str, float | int | str | None]:
    def total(key: str) -> int:
        return sum(int(row[key]) for row in rows)

    onset_count = total("manipulation_onset_count")
    onset_detected = total("manipulation_onset_detected")
    true_positive = total("arm_mode_true_positive_frames")
    predicted_positive = total("arm_mode_predicted_active_frames")
    actual_positive = total("arm_mode_expert_active_frames")
    predicted_active_onsets = total("predicted_active_onset_count")
    sign_count = total("wrist_twist_sign_sample_count")
    so3_count = total("cumulative_so3_error_sample_count")

    def weighted_mean(key: str, count_key: str) -> float | None:
        valid_rows = [row for row in rows if row[key] is not None]
        denominator = sum(int(row[count_key]) for row in valid_rows)
        if denominator == 0:
            return None
        return float(sum(float(row[key]) * int(row[count_key]) for row in valid_rows) / denominator)

    return {
        "dataset_group": dataset_group,
        "manipulation_onset_count": onset_count,
        "manipulation_onset_detected": onset_detected,
        "manipulation_onset_detection_rate": onset_detected / onset_count if onset_count else None,
        "arm_mode_precision": true_positive / predicted_positive if predicted_positive else None,
        "arm_mode_recall": true_positive / actual_positive if actual_positive else None,
        "arm_mode_true_positive_frames": true_positive,
        "arm_mode_predicted_active_frames": predicted_positive,
        "arm_mode_expert_active_frames": actual_positive,
        "predicted_active_onset_count": predicted_active_onsets,
        "predicted_active_ee_translation_endpoint_error_m": weighted_mean(
            "predicted_active_ee_translation_endpoint_error_m", "predicted_active_onset_count"
        ),
        "wrist_twist_sign_sample_count": sign_count,
        "wrist_twist_sign_accuracy": weighted_mean(
            "wrist_twist_sign_accuracy", "wrist_twist_sign_sample_count"
        ),
        "expert_staff_direction_consistency": weighted_mean(
            "expert_staff_direction_consistency", "wrist_twist_sign_sample_count"
        ),
        "cumulative_so3_error_1s_rad": weighted_mean(
            "cumulative_so3_error_1s_rad", "cumulative_so3_error_sample_count"
        ),
        "cumulative_so3_error_sample_count": so3_count,
    }
