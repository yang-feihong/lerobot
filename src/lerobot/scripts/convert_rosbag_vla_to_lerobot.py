#!/usr/bin/env python3
"""Convert teleoperation ROS1 bags to a LeRobot VLA dataset.

This converter is intentionally small and schema-first.  It does not require a
ROS installation; it reads ROS1 bags with the pure Python ``rosbags`` package.

Default semantics match the current B2+Z1 teleoperation logs:

* Pi0.5-native camera keys:
  - ``observation.images.base_0_rgb``: primary/front view camera
  - ``observation.images.left_wrist_0_rgb``: wrist/arm-mounted view camera
  - ``observation.images.right_wrist_0_rgb``: optional secondary/rear view camera
* state: arm q/qd/gripper + B2 joint pos/vel + trunk roll/pitch/height
* action: raw height-invariant EE target converted from [roll,pitch,yaw,x,y,z]
  to [6D rotation representation, x, y, z] + B2 [vx,vy,wz] + gripper target

Install-free invocation example:

    uv run --no-project --with rosbags --with "lerobot[all] @ ." \
      python src/lerobot/scripts/convert_rosbag_vla_to_lerobot.py ...

In a synced LeRobot environment with rosbags available:

    uv run python -m lerobot.scripts.convert_rosbag_vla_to_lerobot ...
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm

try:
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore
except ImportError as exc:  # pragma: no cover - depends on user environment
    raise SystemExit(
        "The pure Python `rosbags` package is required. See ROSBAG_READER_DEPS.md, "
        "or run with `uv run --no-project --with rosbags ...`."
    ) from exc

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset


LOG = logging.getLogger("convert_rosbag_vla_to_lerobot")

STATE_KEY = "observation.state"
ACTION_KEY = "action"
BASE_IMAGE_KEY = "observation.images.base_0_rgb"
WRIST_IMAGE_KEY = "observation.images.left_wrist_0_rgb"
RIGHT_WRIST_IMAGE_KEY = "observation.images.right_wrist_0_rgb"
DEFAULT_ARM_ACTION_TOPIC = "/height_invariant_raw_ee_target_pose"
DEFAULT_EPISODE_BOUNDARY_ARM_TOPIC = "/arm_target_pos"


@dataclass(frozen=True)
class TopicSeries:
    times: np.ndarray
    values: np.ndarray

    def nearest(self, timestamp_ns: int, *, tolerance_ns: int | None = None) -> np.ndarray:
        index = bisect.bisect_left(self.times, timestamp_ns)
        candidates = []
        if index < len(self.times):
            candidates.append(index)
        if index > 0:
            candidates.append(index - 1)
        if not candidates:
            raise RuntimeError("Cannot sample from an empty topic series")
        best = min(candidates, key=lambda i: abs(int(self.times[i]) - timestamp_ns))
        delta = abs(int(self.times[best]) - timestamp_ns)
        if tolerance_ns is not None and delta > tolerance_ns:
            raise RuntimeError(f"Nearest sample is outside tolerance: {delta / 1e9:.3f}s")
        return self.values[best]

    def ffill(self, timestamp_ns: int, *, default: np.ndarray | None = None) -> np.ndarray:
        index = bisect.bisect_right(self.times, timestamp_ns) - 1
        if index < 0:
            if default is None:
                raise RuntimeError("No previous sample for forward-fill")
            return default
        return self.values[index]


@dataclass(frozen=True)
class ResetSeries:
    times: np.ndarray
    reset: np.ndarray


@dataclass(frozen=True)
class BagContext:
    path: Path
    typestore: Any
    connections_by_topic: dict[str, list[Any]]
    start_time_ns: int
    end_time_ns: int

def optional_topic(value: str) -> str | None:
    """Parse a ROS topic argument, allowing None to disable the topic."""
    value = value.strip()
    if value.lower() in {"none", "null", ""}:
        return None
    return value

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag-dir", action="append", type=Path, required=True, help="Directory containing ROS1 bag files; repeatable.")
    parser.add_argument("--bag-glob", default="*.bag", help="Glob used inside each --bag-dir. Default: *.bag")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/b2_z1_vla")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--task", default="b2 z1 teleoperation")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Append only bags not marked done in the sidecar manifest.")
    parser.add_argument(
        "--wrist-camera-topic",
        type=optional_topic,
        default="/camera/color/image_raw",
        help="Wrist/arm-mounted camera topic, or None to disable this camera.",
    )
    parser.add_argument(
        "--base-camera-topic",
        type=optional_topic,
        default="/camera0/usb_cam_node/image_raw",
        help="Primary/front camera topic, or None to disable this camera.",
    )
    parser.add_argument(
        "--right-wrist-camera-topic",
        type=optional_topic,
        default="/camera1/usb_cam_node/image_raw",
        help="Optional secondary/rear camera topic, or None to disable this camera.",
    )

    parser.add_argument("--arm-state-topic", default="/arm_current_state")
    parser.add_argument("--b2-joint-state-topic", default="/b2_joint_states")
    parser.add_argument("--trunk-state-topic", default="/b2_body_rp_height")
    parser.add_argument("--arm-action-topic", default=DEFAULT_ARM_ACTION_TOPIC)
    parser.add_argument("--base-action-topic", default="/b2_target_velocity")
    parser.add_argument("--gripper-action-topic", default="/gripper_target_pos")
    parser.add_argument(
        "--episode-boundary-arm-topic",
        default=DEFAULT_EPISODE_BOUNDARY_ARM_TOPIC,
        help=(
            "Z1ArmTarget topic used only to choose the valid episode interval. "
            "Start considers the first reset=false message; end considers the last message on this topic."
        ),
    )
    parser.add_argument(
        "--base-action-nonzero-eps",
        type=float,
        default=1e-6,
        help="Absolute threshold used to decide whether base velocity is non-zero for episode end selection.",
    )

    parser.add_argument(
        "--codec",
        default="h264",
        help="RGB video codec. Default h264 keeps logs quieter than the LeRobot default AV1/SVT encoder.",
    )
    parser.add_argument("--video-crf", type=int, default=23, help="Video CRF; lower means higher quality/larger files.")
    parser.add_argument("--video-preset", default="veryfast", help="Video encoder preset.")
    parser.add_argument("--video-gop", type=int, default=10, help="Video GOP/keyframe interval.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive")
    args.bag = resolve_bag_paths(args.bag_dir, args.bag_glob)
    return args


def resolve_bag_paths(bag_dirs: list[Path], bag_glob: str) -> list[Path]:
    paths: list[Path] = []
    for bag_dir in bag_dirs:
        paths.extend(sorted(bag_dir.expanduser().glob(bag_glob)))
    paths = [path.resolve() for path in paths]
    unique_paths = list(dict.fromkeys(paths))
    if not unique_paths:
        raise SystemExit(f"No bag files matched --bag-glob {bag_glob!r} under: {bag_dirs}")
    missing = [path for path in unique_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"Bag file(s) not found: {missing}")
    return unique_paths


def open_bag_context(path: Path) -> BagContext:
    reader = Reader(path)
    reader.open()
    typestore = get_typestore(Stores.ROS1_NOETIC)
    custom_types = {}
    for conn in reader.connections:
        if conn.msgtype not in typestore.types and conn.msgdef.data:
            custom_types.update(get_types_from_msg(conn.msgdef.data, conn.msgtype))
    if custom_types:
        typestore.register(custom_types)
    by_topic: dict[str, list[Any]] = {}
    for conn in reader.connections:
        by_topic.setdefault(conn.topic, []).append(conn)
    ctx = BagContext(
        path=path,
        typestore=typestore,
        connections_by_topic=by_topic,
        start_time_ns=int(reader.start_time),
        end_time_ns=int(reader.end_time),
    )
    reader.close()
    return ctx


def get_connections(ctx: BagContext, topic: str) -> list[Any]:
    conns = ctx.connections_by_topic.get(topic)
    if not conns:
        raise RuntimeError(f"Topic not found in {ctx.path}: {topic}")
    return conns


def deserialize(ctx: BagContext, raw: bytes, msgtype: str) -> Any:
    return ctx.typestore.deserialize_ros1(raw, msgtype)


def iter_topic_messages(ctx: BagContext, topic: str):
    with Reader(ctx.path) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            raise RuntimeError(f"Topic not found in {ctx.path}: {topic}")
        for conn, timestamp_ns, raw in reader.messages(connections=conns):
            yield int(timestamp_ns), deserialize(ctx, raw, conn.msgtype)


def read_series(ctx: BagContext, topic: str, vectorizer) -> TopicSeries:
    times: list[int] = []
    values: list[np.ndarray] = []
    for timestamp_ns, msg in iter_topic_messages(ctx, topic):
        value = np.asarray(vectorizer(msg), dtype=np.float32)
        times.append(timestamp_ns)
        values.append(value)
    if not times:
        raise RuntimeError(f"No messages in topic: {topic}")
    return TopicSeries(np.asarray(times, dtype=np.int64), np.stack(values).astype(np.float32))


def read_reset_series(ctx: BagContext, topic: str) -> ResetSeries:
    times: list[int] = []
    reset: list[bool] = []
    for timestamp_ns, msg in iter_topic_messages(ctx, topic):
        if not hasattr(msg, "reset"):
            raise RuntimeError(f"Topic {topic} messages do not have a reset field")
        times.append(timestamp_ns)
        reset.append(bool(msg.reset))
    if not times:
        raise RuntimeError(f"No messages in topic: {topic}")
    return ResetSeries(np.asarray(times, dtype=np.int64), np.asarray(reset, dtype=bool))


def rel_sec(ctx: BagContext, timestamp_ns: int) -> float:
    return (int(timestamp_ns) - ctx.start_time_ns) / 1e9


def series_rate_hz(series: TopicSeries) -> float:
    if len(series.times) < 2:
        return 0.0
    duration_s = (int(series.times[-1]) - int(series.times[0])) / 1e9
    return (len(series.times) - 1) / duration_s if duration_s > 0 else 0.0


def format_vector(values: np.ndarray, *, decimals: int = 4) -> str:
    return np.array2string(
        np.asarray(values),
        precision=decimals,
        suppress_small=True,
        separator=", ",
        max_line_width=10_000,
    )


def log_series_summary(ctx: BagContext, label: str, topic: str, series: TopicSeries) -> None:
    LOG.info(
        "%s topic=%s count=%d dim=%d time=[%.3fs, %.3fs] rate≈%.1fHz",
        label,
        topic,
        len(series.times),
        series.values.shape[1],
        rel_sec(ctx, int(series.times[0])),
        rel_sec(ctx, int(series.times[-1])),
        series_rate_hz(series),
    )
    LOG.info(
        "%s range min=%s max=%s std=%s",
        label,
        format_vector(series.values.min(axis=0)),
        format_vector(series.values.max(axis=0)),
        format_vector(series.values.std(axis=0)),
    )


def vector_arm_state(msg: Any) -> np.ndarray:
    data = np.asarray(msg.data, dtype=np.float32)
    if data.shape == (13,):
        return data
    if data.shape == (7,):
        q = data[:6]
        qd = np.zeros(6, dtype=np.float32)
        gripper = data[6:7]
        return np.concatenate([q, qd, gripper]).astype(np.float32)
    raise RuntimeError(f"Expected arm state dim 7 or 13, got {data.shape}")


def vector_b2_joint_state(msg: Any) -> np.ndarray:
    position = np.asarray(msg.position, dtype=np.float32)
    velocity = np.asarray(msg.velocity, dtype=np.float32)
    if position.shape != (12,) or velocity.shape != (12,):
        raise RuntimeError(f"Expected B2 joint position/velocity dim 12, got {position.shape}/{velocity.shape}")
    return np.concatenate([position, velocity]).astype(np.float32)


def vector_trunk_state(msg: Any) -> np.ndarray:
    return np.asarray([msg.vector.x, msg.vector.y, msg.vector.z], dtype=np.float32)


def vector_arm_action(msg: Any) -> np.ndarray:
    value = np.asarray(msg.joint_pos, dtype=np.float32)
    if value.shape != (6,):
        raise RuntimeError(f"Expected arm EE action dim 6, got {value.shape}")
    roll, pitch, yaw = value[:3]
    rotation_matrix = Rotation.from_euler("xyz", [float(roll), float(pitch), float(yaw)]).as_matrix()
    rotation_6d = rotation_matrix[:, :2].reshape(-1, order="F")
    return np.concatenate([rotation_6d, value[3:6]]).astype(np.float32)


def vector_base_action(msg: Any) -> np.ndarray:
    return np.asarray([msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z], dtype=np.float32)


def vector_gripper_action(msg: Any) -> np.ndarray:
    value = np.asarray(msg.data, dtype=np.float32)
    if value.size < 1:
        raise RuntimeError("Empty gripper action")
    return value[:1]


def image_metadata(ctx: BagContext, topic: str) -> tuple[int, int, str]:
    for _, msg in iter_topic_messages(ctx, topic):
        return int(msg.height), int(msg.width), str(msg.encoding)
    raise RuntimeError(f"No image messages in topic: {topic}")


def image_to_rgb_array(msg: Any) -> np.ndarray:
    height = int(msg.height)
    width = int(msg.width)
    encoding = str(msg.encoding).lower()
    data = np.asarray(msg.data, dtype=np.uint8)
    if encoding in ("rgb8", "bgr8"):
        image = data.reshape(height, int(msg.step))[:, : width * 3].reshape(height, width, 3)
        if encoding == "bgr8":
            image = image[..., ::-1]
        return np.ascontiguousarray(image)
    if encoding in ("mono8", "8uc1"):
        image = data.reshape(height, int(msg.step))[:, :width]
        return np.repeat(image[..., None], 3, axis=2)
    raise RuntimeError(f"Unsupported image encoding for RGB conversion: {msg.encoding!r}")


def sample_images(ctx: BagContext, topic: str, sample_times_ns: np.ndarray, *, label: str) -> list[np.ndarray]:
    samples: list[np.ndarray | None] = [None] * len(sample_times_ns)
    target_index = 0
    previous: tuple[int, Any] | None = None

    with tqdm(total=len(sample_times_ns), desc=f"sample {label}", unit="frame") as progress:
        for timestamp_ns, msg in iter_topic_messages(ctx, topic):
            while target_index < len(sample_times_ns) and timestamp_ns >= int(sample_times_ns[target_index]):
                target = int(sample_times_ns[target_index])
                if previous is None:
                    chosen = msg
                else:
                    prev_ts, prev_msg = previous
                    chosen = prev_msg if abs(prev_ts - target) <= abs(timestamp_ns - target) else msg
                samples[target_index] = image_to_rgb_array(chosen)
                target_index += 1
                progress.update(1)
            previous = (timestamp_ns, msg)

        if previous is not None:
            while target_index < len(sample_times_ns):
                samples[target_index] = image_to_rgb_array(previous[1])
                target_index += 1
                progress.update(1)

    missing = [idx for idx, value in enumerate(samples) if value is None]
    if missing:
        raise RuntimeError(f"Could not sample {len(missing)} images from {topic}")
    first = samples[0]
    assert first is not None
    LOG.info(
        "Image topic=%s sampled_frames=%d shape=%s dtype=%s",
        topic,
        len(samples),
        tuple(first.shape),
        first.dtype,
    )
    return [value for value in samples if value is not None]


def select_episode_interval(
    ctx: BagContext,
    *,
    boundary_arm: ResetSeries,
    base_action: TopicSeries,
    base_action_nonzero_eps: float,
) -> tuple[int, int, dict[str, Any]]:
    arm_non_reset_indices = np.flatnonzero(~boundary_arm.reset)
    first_arm_non_reset_ns = (
        int(boundary_arm.times[int(arm_non_reset_indices[0])]) if len(arm_non_reset_indices) else None
    )
    last_arm_target_ns = int(boundary_arm.times[-1])
    base_nonzero = np.any(np.abs(base_action.values) > base_action_nonzero_eps, axis=1)
    base_nonzero_indices = np.flatnonzero(base_nonzero)
    first_base_nonzero_ns = (
        int(base_action.times[int(base_nonzero_indices[0])]) if len(base_nonzero_indices) else None
    )
    last_base_nonzero_ns = (
        int(base_action.times[int(base_nonzero_indices[-1])]) if len(base_nonzero_indices) else None
    )
    start_candidates = []
    if first_arm_non_reset_ns is not None:
        start_candidates.append(("first_boundary_arm_reset_false", first_arm_non_reset_ns))
    if first_base_nonzero_ns is not None:
        start_candidates.append(("first_nonzero_base_action", first_base_nonzero_ns))
    if not start_candidates:
        raise RuntimeError(
            "Cannot select episode start: no reset=false boundary arm messages and no non-zero base action messages"
        )
    start_source, start_ns = min(start_candidates, key=lambda item: item[1])

    end_candidates = [last_arm_target_ns]
    if last_base_nonzero_ns is not None:
        end_candidates.append(last_base_nonzero_ns)
    end_ns = max(end_candidates)
    if end_ns <= start_ns:
        raise RuntimeError(
            "Invalid selected episode interval: "
            f"start={rel_sec(ctx, start_ns):.3f}s end={rel_sec(ctx, end_ns):.3f}s"
        )
    info = {
        "start_source": start_source,
        "end_source": "max(last_boundary_arm_message,last_nonzero_base_action)",
        "start_s": rel_sec(ctx, start_ns),
        "end_s": rel_sec(ctx, end_ns),
        "first_boundary_arm_reset_false_s": rel_sec(ctx, first_arm_non_reset_ns)
        if first_arm_non_reset_ns is not None
        else None,
        "first_nonzero_base_action_s": rel_sec(ctx, first_base_nonzero_ns)
        if first_base_nonzero_ns is not None
        else None,
        "last_boundary_arm_s": rel_sec(ctx, last_arm_target_ns),
        "last_nonzero_base_action_s": rel_sec(ctx, last_base_nonzero_ns)
        if last_base_nonzero_ns is not None
        else None,
    }
    return start_ns, end_ns, info


def make_sample_times(start_ns: int, end_ns: int, *, fps: int) -> np.ndarray:
    if end_ns <= start_ns:
        raise RuntimeError("Invalid conversion interval")
    step_ns = int(round(1e9 / fps))
    return np.arange(start_ns, end_ns + 1, step_ns, dtype=np.int64)


ACTION_NAMES = (
    "height_invariant_ee_rot6d_col0_x",
    "height_invariant_ee_rot6d_col0_y",
    "height_invariant_ee_rot6d_col0_z",
    "height_invariant_ee_rot6d_col1_x",
    "height_invariant_ee_rot6d_col1_y",
    "height_invariant_ee_rot6d_col1_z",
    "height_invariant_ee_x",
    "height_invariant_ee_y",
    "height_invariant_ee_z",
    "b2_vx",
    "b2_vy",
    "b2_omega_z",
    "gripper_target",
)
TRUNK_STATE_NAMES = ("b2_trunk_roll", "b2_trunk_pitch", "b2_body_height")
ARM_STATE_7D_NAMES = (
    "arm_q_1",
    "arm_q_2",
    "arm_q_3",
    "arm_q_4",
    "arm_q_5",
    "arm_q_6",
    "arm_gripper_feedback",
)


def build_features(image_shapes: dict[str, tuple[int, int]]) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        STATE_KEY: {
            "dtype": "float32",
            "shape": (40,),
            "names": [
                *[f"arm_q_{i + 1}" for i in range(6)],
                *[f"arm_qd_{i + 1}" for i in range(6)],
                "arm_gripper_feedback",
                *[f"b2_joint_pos_{name}" for name in B2_JOINT_NAMES],
                *[f"b2_joint_vel_{name}" for name in B2_JOINT_NAMES],
                "b2_trunk_roll",
                "b2_trunk_pitch",
                "b2_body_height",
            ],
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (len(ACTION_NAMES),),
            "names": list(ACTION_NAMES),
        },
    }
    for key, (height, width) in image_shapes.items():
        features[key] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


B2_JOINT_NAMES = (
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
)


def diagnostic_filename(path: Path) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in path.stem)
    return f"{safe}_timeseries.png"


def manifest_path(output_root: Path) -> Path:
    return output_root / "diagnostics" / "conversion_manifest.json"


def load_manifest(output_root: Path) -> dict[str, Any]:
    path = manifest_path(output_root)
    if not path.exists():
        return {"version": 1, "bags": {}}
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest.setdefault("version", 1)
    manifest.setdefault("bags", {})
    return manifest


def save_manifest(output_root: Path, manifest: dict[str, Any]) -> None:
    path = manifest_path(output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(path)


def bag_manifest_key(path: Path) -> str:
    return str(path.resolve())


def bag_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def manifest_done_for_bag(manifest: dict[str, Any], bag: Path) -> bool:
    entry = manifest.get("bags", {}).get(bag_manifest_key(bag))
    if not entry or entry.get("status") != "done":
        return False
    return entry.get("fingerprint") == bag_fingerprint(bag)


def update_manifest_entry(output_root: Path, manifest: dict[str, Any], bag: Path, **updates: Any) -> None:
    bags = manifest.setdefault("bags", {})
    key = bag_manifest_key(bag)
    entry = dict(bags.get(key, {}))
    entry.update(
        {
            "bag_path": str(bag.resolve()),
            "bag_name": bag.name,
            "fingerprint": bag_fingerprint(bag),
            "updated_unix": time.time(),
        }
    )
    entry.update(updates)
    bags[key] = entry
    save_manifest(output_root, manifest)


def validate_resume_features(dataset: LeRobotDataset, expected_features: dict[str, dict[str, Any]]) -> None:
    for key, expected in expected_features.items():
        actual = dataset.meta.features.get(key)
        if actual is None:
            raise RuntimeError(f"Cannot resume: existing dataset is missing feature {key!r}")
        if tuple(actual["shape"]) != tuple(expected["shape"]):
            raise RuntimeError(
                f"Cannot resume: feature {key!r} shape mismatch: existing={actual['shape']} expected={expected['shape']}"
            )
        if actual["dtype"] != expected["dtype"]:
            raise RuntimeError(
                f"Cannot resume: feature {key!r} dtype mismatch: existing={actual['dtype']} expected={expected['dtype']}"
            )


def save_timeseries_diagnostic(
    *,
    output_root: Path,
    ctx: BagContext,
    sample_times: np.ndarray,
    state_array: np.ndarray,
    action_array: np.ndarray,
) -> Path:
    """Save a sidecar diagnostic plot without touching the LeRobot dataset schema."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if state_array.shape[1] != 40:
        raise RuntimeError(f"Expected state dim 40 for diagnostic plot, got {state_array.shape[1]}")
    if action_array.shape[1] != len(ACTION_NAMES):
        raise RuntimeError(f"Expected action dim {len(ACTION_NAMES)} for diagnostic plot, got {action_array.shape[1]}")

    time_s = (sample_times.astype(np.float64) - float(sample_times[0])) / 1e9
    arm_state_7d = np.concatenate([state_array[:, :6], state_array[:, 12:13]], axis=1)
    trunk_state = state_array[:, 37:40]

    diagnostics_dir = output_root / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    output_path = diagnostics_dir / diagnostic_filename(ctx.path)

    if len(time_s) > 1:
        sampled_hz = 1 / float(np.median(np.diff(time_s)))
    else:
        sampled_hz = 0.0
    plot_specs = [
        *[(name, action_array[:, index], "action") for index, name in enumerate(ACTION_NAMES)],
        *[(name, trunk_state[:, index], "trunk") for index, name in enumerate(TRUNK_STATE_NAMES)],
        *[(name, arm_state_7d[:, index], "arm") for index, name in enumerate(ARM_STATE_7D_NAMES)],
    ]
    expected_diagnostic_signals = len(ACTION_NAMES) + len(TRUNK_STATE_NAMES) + len(ARM_STATE_7D_NAMES)
    if len(plot_specs) != expected_diagnostic_signals:
        raise RuntimeError(f"Expected {expected_diagnostic_signals} diagnostic signals, got {len(plot_specs)}")

    ncols = 5
    nrows = int(np.ceil(len(plot_specs) / ncols))
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(25, 3.2 * nrows), sharex=True, constrained_layout=True)
    axes = axes_grid.reshape(-1)
    fig.suptitle(f"{ctx.path.name} sampled @ {sampled_hz:.1f}Hz", fontsize=13)

    for axis, (name, values, group) in zip(axes[: len(plot_specs)], plot_specs, strict=True):
        axis.plot(time_s, values, linewidth=0.9)
        axis.set_title(f"{group}: {name}", fontsize=9)
        axis.grid(True, alpha=0.3)
        axis.tick_params(axis="both", labelsize=8)

    for axis in axes[len(plot_specs) :]:
        axis.set_visible(False)

    last_row_start = (nrows - 1) * ncols
    for axis in axes[last_row_start : last_row_start + ncols]:
        if axis.get_visible():
            axis.set_xlabel("time (s)", fontsize=8)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    LOG.info("Saved diagnostic timeseries plot: %s", output_path)
    return output_path


def convert_bag_episode(dataset: LeRobotDataset, ctx: BagContext, args: argparse.Namespace) -> dict[str, Any]:
    LOG.info("========== Converting bag: %s ==========", ctx.path)
    LOG.info(
        "Bag time: start=0.000s end=%.3fs duration=%.3fs",
        (ctx.end_time_ns - ctx.start_time_ns) / 1e9,
        (ctx.end_time_ns - ctx.start_time_ns) / 1e9,
    )
    LOG.info("Reading state/action topics")
    arm_state = read_series(ctx, args.arm_state_topic, vector_arm_state)
    b2_joint_state = read_series(ctx, args.b2_joint_state_topic, vector_b2_joint_state)
    trunk_state = read_series(ctx, args.trunk_state_topic, vector_trunk_state)
    arm_action = read_series(ctx, args.arm_action_topic, vector_arm_action)
    base_action = read_series(ctx, args.base_action_topic, vector_base_action)
    gripper_action = read_series(ctx, args.gripper_action_topic, vector_gripper_action)
    boundary_arm = read_reset_series(ctx, args.episode_boundary_arm_topic)

    log_series_summary(ctx, "arm_state", args.arm_state_topic, arm_state)
    log_series_summary(ctx, "b2_joint_state", args.b2_joint_state_topic, b2_joint_state)
    log_series_summary(ctx, "trunk_state", args.trunk_state_topic, trunk_state)
    log_series_summary(ctx, "arm_action", args.arm_action_topic, arm_action)
    log_series_summary(ctx, "base_action", args.base_action_topic, base_action)
    log_series_summary(ctx, "gripper_action", args.gripper_action_topic, gripper_action)
    LOG.info(
        "episode_boundary_arm topic=%s count=%d reset_false=%d time=[%.3fs, %.3fs]",
        args.episode_boundary_arm_topic,
        len(boundary_arm.times),
        int(np.count_nonzero(~boundary_arm.reset)),
        rel_sec(ctx, int(boundary_arm.times[0])),
        rel_sec(ctx, int(boundary_arm.times[-1])),
    )

    start_ns, end_ns, interval_info = select_episode_interval(
        ctx,
        boundary_arm=boundary_arm,
        base_action=base_action,
        base_action_nonzero_eps=args.base_action_nonzero_eps,
    )
    sample_times = make_sample_times(start_ns, end_ns, fps=args.fps)
    LOG.info(
        "Selected sampling interval: start=%.3fs end=%.3fs fps=%d -> candidate_frames=%d",
        interval_info["start_s"],
        interval_info["end_s"],
        args.fps,
        len(sample_times),
    )
    LOG.info(
        "Interval details: start_source=%s first_boundary_arm_reset_false=%s first_nonzero_base_action=%s "
        "end_source=%s last_boundary_arm=%.3fs last_nonzero_base_action=%s",
        interval_info["start_source"],
        "None"
        if interval_info["first_boundary_arm_reset_false_s"] is None
        else f"{interval_info['first_boundary_arm_reset_false_s']:.3f}s",
        "None"
        if interval_info["first_nonzero_base_action_s"] is None
        else f"{interval_info['first_nonzero_base_action_s']:.3f}s",
        interval_info["end_source"],
        interval_info["last_boundary_arm_s"],
        "None"
        if interval_info["last_nonzero_base_action_s"] is None
        else f"{interval_info['last_nonzero_base_action_s']:.3f}s",
    )
    if len(sample_times) == 0:
        raise RuntimeError("No sample times selected")

    LOG.info(
        "Final episode sampling: frames=%d time=[%.3fs, %.3fs] duration≈%.3fs fps=%d",
        len(sample_times),
        rel_sec(ctx, int(sample_times[0])),
        rel_sec(ctx, int(sample_times[-1])),
        len(sample_times) / float(args.fps),
        args.fps,
    )
    camera_topics: dict[str, str] = {}
    for key, topic in (
        (WRIST_IMAGE_KEY, args.wrist_camera_topic),
        (BASE_IMAGE_KEY, args.base_camera_topic),
        (RIGHT_WRIST_IMAGE_KEY, args.right_wrist_camera_topic),
    ):
        if topic is not None:
            camera_topics[key] = topic
    if camera_topics:
        LOG.info(
            "Sampling %d image topic(s): %s",
            len(camera_topics),
            ", ".join(camera_topics.values()),
        )
    else:
        LOG.info("No image topics enabled; converting state/action-only dataset")
    sampled_images = {
        key: sample_images(ctx, topic, sample_times, label=key)
        for key, topic in camera_topics.items()
    }

    zero_gripper = np.zeros(1, dtype=np.float32)
    state_values: list[np.ndarray] = []
    action_values: list[np.ndarray] = []
    episode_index = int(dataset.meta.total_episodes)
    for frame_index, timestamp_ns in enumerate(tqdm(sample_times, desc="write episode frames", unit="frame")):
        state = np.concatenate(
            [
                arm_state.nearest(int(timestamp_ns)),
                b2_joint_state.nearest(int(timestamp_ns)),
                trunk_state.nearest(int(timestamp_ns)),
            ]
        ).astype(np.float32)
        action = np.concatenate(
            [
                arm_action.ffill(int(timestamp_ns), default=arm_action.values[0]),
                base_action.ffill(int(timestamp_ns), default=np.zeros(3, dtype=np.float32)),
                gripper_action.ffill(int(timestamp_ns), default=zero_gripper),
            ]
        ).astype(np.float32)
        state_values.append(state)
        action_values.append(action)
        frame = {
            STATE_KEY: state,
            ACTION_KEY: action,
            "task": args.task,
        }
        for image_key in camera_topics:
            frame[image_key] = sampled_images[image_key][frame_index]
        dataset.add_frame(frame)
    state_array = np.stack(state_values)
    action_array = np.stack(action_values)
    diagnostic_path = save_timeseries_diagnostic(
        output_root=args.output_root,
        ctx=ctx,
        sample_times=sample_times,
        state_array=state_array,
        action_array=action_array,
    )
    dataset.save_episode()
    LOG.info(
        "Saved episode: frames=%d state_dim=%d action_dim=%d",
        len(sample_times),
        state_array.shape[1],
        action_array.shape[1],
    )
    LOG.info(
        "Episode action range min=%s max=%s std=%s",
        format_vector(action_array.min(axis=0)),
        format_vector(action_array.max(axis=0)),
        format_vector(action_array.std(axis=0)),
    )
    LOG.info(
        "Episode state range min=%s max=%s std=%s",
        format_vector(state_array.min(axis=0)),
        format_vector(state_array.max(axis=0)),
        format_vector(state_array.std(axis=0)),
    )
    return {
        "episode_index": episode_index,
        "num_frames": len(sample_times),
        "state_dim": int(state_array.shape[1]),
        "action_dim": int(action_array.shape[1]),
        "diagnostic_path": str(diagnostic_path),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
        force=True,
    )
    try:
        import av

        av.logging.set_level(av.logging.ERROR)
    except Exception:
        LOG.debug("Could not adjust PyAV logging", exc_info=True)
    if args.fps < 1:
        raise ValueError("--fps must be positive")
    if args.output_root.exists():
        if args.overwrite:
            shutil.rmtree(args.output_root)
        elif not args.resume:
            raise FileExistsError(f"Output exists; pass --overwrite to replace it: {args.output_root}")
        elif not manifest_path(args.output_root).exists():
            raise FileExistsError(
                f"Output exists but has no resume manifest: {manifest_path(args.output_root)}. "
                "Use --overwrite to rebuild it from scratch."
            )
    elif args.resume:
        raise FileNotFoundError(f"Cannot resume because output does not exist: {args.output_root}")

    LOG.info("Resolved %d bag(s) for conversion", len(args.bag))
    for idx, bag in enumerate(args.bag[:10]):
        LOG.info("  bag[%d]=%s", idx, bag)
    if len(args.bag) > 10:
        LOG.info("  ... %d more bag(s)", len(args.bag) - 10)

    manifest = load_manifest(args.output_root)
    pending_bags = [bag for bag in args.bag if not (args.resume and manifest_done_for_bag(manifest, bag))]
    skipped_bags = len(args.bag) - len(pending_bags)
    if skipped_bags:
        LOG.info("Resume: skipping %d already completed bag(s)", skipped_bags)
    if not pending_bags:
        LOG.info("No pending bags to convert; manifest already marks all matched bags as done.")
        return

    first_ctx = open_bag_context(args.bag[0])
    configured_camera_topics: dict[str, str] = {}
    for key, topic in (
        (WRIST_IMAGE_KEY, args.wrist_camera_topic),
        (BASE_IMAGE_KEY, args.base_camera_topic),
        (RIGHT_WRIST_IMAGE_KEY, args.right_wrist_camera_topic),
    ):
        if topic is not None:
            configured_camera_topics[key] = topic
    image_shapes = {
        key: image_metadata(first_ctx, topic)[:2]
        for key, topic in configured_camera_topics.items()
    }
    expected_features = build_features(image_shapes)
    rgb_encoder = RGBEncoderConfig(vcodec=args.codec, crf=args.video_crf, preset=args.video_preset, g=args.video_gop)
    if args.resume:
        dataset = LeRobotDataset.resume(
            repo_id=args.repo_id,
            root=args.output_root,
            rgb_encoder=rgb_encoder,
        )
        if dataset.meta.fps != args.fps:
            raise RuntimeError(f"Cannot resume: existing fps={dataset.meta.fps} but requested fps={args.fps}")
        validate_resume_features(dataset, expected_features)
    else:
        manifest = {
            "version": 1,
            "repo_id": args.repo_id,
            "fps": args.fps,
            "action_names": list(ACTION_NAMES),
            "image_keys": list(configured_camera_topics.keys()),
            "episode_boundary_arm_topic": args.episode_boundary_arm_topic,
            "base_action_nonzero_eps": args.base_action_nonzero_eps,
            "bags": {},
        }
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            features=expected_features,
            root=args.output_root,
            robot_type="b2_z1",
            use_videos=True,
            rgb_encoder=rgb_encoder,
        )
        save_manifest(args.output_root, manifest)

    for bag in pending_bags:
        update_manifest_entry(
            args.output_root,
            manifest,
            bag,
            status="running",
            episode_index=int(dataset.meta.total_episodes),
            action_dim=len(ACTION_NAMES),
        )
        ctx = open_bag_context(bag)
        try:
            episode_info = convert_bag_episode(dataset, ctx, args)
        except Exception as exc:
            update_manifest_entry(
                args.output_root,
                manifest,
                bag,
                status="failed",
                error=repr(exc),
                action_dim=len(ACTION_NAMES),
            )
            raise
        update_manifest_entry(
            args.output_root,
            manifest,
            bag,
            status="done",
            **episode_info,
        )

    dataset.finalize()
    LOG.info("Conversion complete: root=%s episodes=%d frames=%d", args.output_root, dataset.num_episodes, dataset.num_frames)


if __name__ == "__main__":
    main()

# Current full command template. Every ROS topic used by the converter can be
# overridden from the command line.
#
# uv run --with rosbags python -m lerobot.scripts.convert_rosbag_vla_to_lerobot \
#   --bag-dir /data/rosbag/rosbags_0721 \
#   --bag-glob '*.bag*' \
#   --output-root /data/b2_z1_vla_lerobot_0721 \
#   --repo-id local/b2_z1_vla \
#   --fps 10 \
#   --base-camera-topic /camera_usb_0_4_1_2/usb_cam_node/image_raw \
#   --wrist-camera-topic /camera_usb_0_4_1_1/usb_cam_node/image_raw \
#   --right-wrist-camera-topic None \
#   --arm-state-topic /arm_current_state \
#   --b2-joint-state-topic /b2_joint_states \
#   --trunk-state-topic /b2_body_rp_height \
#   --arm-action-topic /height_invariant_raw_ee_target_pose \
#   --base-action-topic /b2_target_velocity \
#   --gripper-action-topic /gripper_target_pos \
#   --episode-boundary-arm-topic /arm_target_pos \
#   --overwrite
#
# The converter automatically selects the usable interval for each bag:
#   start = min(first reset=false message on --episode-boundary-arm-topic,
#               first non-zero message on --base-action-topic)
#   end   = max(last message on --episode-boundary-arm-topic,
#               last non-zero message on --base-action-topic)
#
# Camera data are written with Pi0.5-native keys:
#   primary/front view          -> observation.images.base_0_rgb
#   wrist/arm-mounted view      -> observation.images.left_wrist_0_rgb
#   optional secondary/rear view -> observation.images.right_wrist_0_rgb, if enabled
#
# If a large conversion is interrupted, re-run the same command with --resume
# instead of --overwrite. The converter skips bags marked done in:
#   <output-root>/diagnostics/conversion_manifest.json
#
# The arm action topic is read as [roll, pitch, yaw, x, y, z]. The converter
# stores action orientation as the 6D rotation representation from the first
# two columns of:
#   scipy.spatial.transform.Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
#
# For each converted bag, a sidecar diagnostic plot is saved under:
#   <output-root>/diagnostics/<bag-stem>_timeseries.png
# This plot is not part of the LeRobot dataset schema.
