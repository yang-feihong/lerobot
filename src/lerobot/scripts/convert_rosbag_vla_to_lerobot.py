#!/usr/bin/env python3
"""Convert teleoperation ROS1 bags to a LeRobot VLA dataset.

This converter is intentionally small and schema-first.  It does not require a
ROS installation; it reads ROS1 bags with the pure Python ``rosbags`` package.

Default semantics match the current B2+Z1 teleoperation logs:

* wrist RGB image: RealSense wrist camera
* front/rear fisheye images: configurable USB cameras
* state: arm q/qd/gripper + B2 joint pos/vel + trunk roll/pitch/height
* action: height-invariant EE target [roll,pitch,yaw,x,y,z] + B2 [vx,vy,wz]
  + gripper target

Install-free invocation example:

    uv run --no-project --with rosbags --with "lerobot[all] @ ." \
      python src/lerobot/scripts/convert_rosbag_vla_to_lerobot.py ...

In a synced LeRobot environment with rosbags available:

    uv run python -m lerobot.scripts.convert_rosbag_vla_to_lerobot ...
"""

from __future__ import annotations

import argparse
import bisect
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
WRIST_IMAGE_KEY = "observation.images.wrist"
FRONT_FISHEYE_KEY = "observation.images.front_fisheye"
REAR_FISHEYE_KEY = "observation.images.rear_fisheye"


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
class BagContext:
    path: Path
    typestore: Any
    connections_by_topic: dict[str, list[Any]]
    start_time_ns: int
    end_time_ns: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag-dir", action="append", type=Path, required=True, help="Directory containing ROS1 bag files; repeatable.")
    parser.add_argument("--bag-glob", default="*.bag", help="Glob used inside each --bag-dir. Default: *.bag")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/b2_z1_vla")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--task", default="b2 z1 teleoperation")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--wrist-camera-topic", default="/camera/color/image_raw")
    parser.add_argument("--front-fisheye-topic", default="/camera0/usb_cam_node/image_raw")
    parser.add_argument("--rear-fisheye-topic", default="/camera1/usb_cam_node/image_raw")

    parser.add_argument("--arm-state-topic", default="/arm_current_state")
    parser.add_argument("--b2-joint-state-topic", default="/b2_joint_states")
    parser.add_argument("--trunk-state-topic", default="/b2_body_rp_height")
    parser.add_argument("--arm-action-topic", default="/height_invariant_ee_target_pose")
    parser.add_argument("--base-action-topic", default="/b2_target_velocity")
    parser.add_argument("--gripper-action-topic", default="/gripper_target_pos")

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
    return value


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


def sample_images(ctx: BagContext, topic: str, sample_times_ns: np.ndarray) -> list[np.ndarray]:
    samples: list[np.ndarray | None] = [None] * len(sample_times_ns)
    target_index = 0
    previous: tuple[int, Any] | None = None

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
        previous = (timestamp_ns, msg)

    if previous is not None:
        while target_index < len(sample_times_ns):
            samples[target_index] = image_to_rgb_array(previous[1])
            target_index += 1

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


def make_sample_times(arm_action: TopicSeries, *, fps: int) -> np.ndarray:
    start_ns = int(arm_action.times[0])
    end_ns = int(arm_action.times[-1])
    if end_ns <= start_ns:
        raise RuntimeError("Invalid conversion interval")
    step_ns = int(round(1e9 / fps))
    return np.arange(start_ns, end_ns + 1, step_ns, dtype=np.int64)


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
            "shape": (10,),
            "names": [
                "hi_ee_roll",
                "hi_ee_pitch",
                "hi_ee_yaw",
                "hi_ee_x",
                "hi_ee_y",
                "hi_ee_z",
                "b2_vx",
                "b2_vy",
                "b2_omega_z",
                "gripper_target",
            ],
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


def convert_bag_episode(dataset: LeRobotDataset, ctx: BagContext, args: argparse.Namespace) -> None:
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

    log_series_summary(ctx, "arm_state", args.arm_state_topic, arm_state)
    log_series_summary(ctx, "b2_joint_state", args.b2_joint_state_topic, b2_joint_state)
    log_series_summary(ctx, "trunk_state", args.trunk_state_topic, trunk_state)
    log_series_summary(ctx, "arm_action", args.arm_action_topic, arm_action)
    log_series_summary(ctx, "base_action", args.base_action_topic, base_action)
    log_series_summary(ctx, "gripper_action", args.gripper_action_topic, gripper_action)

    sample_times = make_sample_times(arm_action, fps=args.fps)
    full_start_s = rel_sec(ctx, int(sample_times[0]))
    full_end_s = rel_sec(ctx, int(sample_times[-1]))
    LOG.info(
        "Selected sampling interval: start=%.3fs end=%.3fs fps=%d -> candidate_frames=%d "
        "(from first to last %s message)",
        full_start_s,
        full_end_s,
        args.fps,
        len(sample_times),
        args.arm_action_topic,
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
    camera_topics = {
        WRIST_IMAGE_KEY: args.wrist_camera_topic,
        FRONT_FISHEYE_KEY: args.front_fisheye_topic,
        REAR_FISHEYE_KEY: args.rear_fisheye_topic,
    }
    LOG.info("Sampling image topics")
    sampled_images = {key: sample_images(ctx, topic, sample_times) for key, topic in camera_topics.items()}

    zero_gripper = np.zeros(1, dtype=np.float32)
    state_values: list[np.ndarray] = []
    action_values: list[np.ndarray] = []
    for frame_index, timestamp_ns in enumerate(sample_times):
        state = np.concatenate(
            [
                arm_state.nearest(int(timestamp_ns)),
                b2_joint_state.nearest(int(timestamp_ns)),
                trunk_state.nearest(int(timestamp_ns)),
            ]
        ).astype(np.float32)
        action = np.concatenate(
            [
                arm_action.ffill(int(timestamp_ns)),
                base_action.ffill(int(timestamp_ns), default=np.zeros(3, dtype=np.float32)),
                gripper_action.ffill(int(timestamp_ns), default=zero_gripper),
            ]
        ).astype(np.float32)
        state_values.append(state)
        action_values.append(action)
        frame = {
            STATE_KEY: state,
            ACTION_KEY: action,
            WRIST_IMAGE_KEY: sampled_images[WRIST_IMAGE_KEY][frame_index],
            FRONT_FISHEYE_KEY: sampled_images[FRONT_FISHEYE_KEY][frame_index],
            REAR_FISHEYE_KEY: sampled_images[REAR_FISHEYE_KEY][frame_index],
            "task": args.task,
        }
        dataset.add_frame(frame)
    dataset.save_episode()
    state_array = np.stack(state_values)
    action_array = np.stack(action_values)
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
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace it: {args.output_root}")
        shutil.rmtree(args.output_root)

    LOG.info("Resolved %d bag(s) for conversion", len(args.bag))
    for idx, bag in enumerate(args.bag[:10]):
        LOG.info("  bag[%d]=%s", idx, bag)
    if len(args.bag) > 10:
        LOG.info("  ... %d more bag(s)", len(args.bag) - 10)

    first_ctx = open_bag_context(args.bag[0])
    image_shapes = {
        WRIST_IMAGE_KEY: image_metadata(first_ctx, args.wrist_camera_topic)[:2],
        FRONT_FISHEYE_KEY: image_metadata(first_ctx, args.front_fisheye_topic)[:2],
        REAR_FISHEYE_KEY: image_metadata(first_ctx, args.rear_fisheye_topic)[:2],
    }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=build_features(image_shapes),
        root=args.output_root,
        robot_type="b2_z1",
        use_videos=True,
        rgb_encoder=RGBEncoderConfig(vcodec=args.codec, crf=args.video_crf, preset=args.video_preset, g=args.video_gop),
    )

    for bag in args.bag:
        ctx = open_bag_context(bag)
        convert_bag_episode(dataset, ctx, args)

    dataset.finalize()
    LOG.info("Conversion complete: root=%s episodes=%d frames=%d", args.output_root, dataset.num_episodes, dataset.num_frames)


if __name__ == "__main__":
    main()

# Batch conversion: one LeRobot episode per bag. Each bag's interval is selected automatically
# from its first arm action timestamp to its last arm action timestamp.
#
# uv run --with rosbags python -m lerobot.scripts.convert_rosbag_vla_to_lerobot \
#   --bag-dir /data/rosbag \
#   --bag-glob '*.bag' \
#   --output-root /data/b2_z1_vla_lerobot \
#   --repo-id local/b2_z1_vla \
#   --fps 10
#
