#!/usr/bin/env python3
"""Convert VLN-CE smooth RGB trajectories to a LeRobot v3 video dataset.

The source datasets are expected to contain trajectories shaped like::

    <source>/images/<trajectory>/rgb/000000.jpg
    <source>/images/<trajectory>/action/poses_with_start.npy

Only RGB frames are required for MEM-ViT distillation. Source files are opened
read-only. Multiple trajectories are packed into each MP4 to avoid creating an
excessive number of small video files.

The source capture rate is unknown, so this converter deliberately uses a
synthetic frame clock: by default ``fps=1``, meaning one source-frame step is
one dataset time unit. Use integer gap values during distillation to sample by
frame distance rather than pretending those values are physical seconds.

This script needs PyArrow, pandas, and ffmpeg. In a fully synced LeRobot environment::

    uv run python -m lerobot.scripts.convert_vlnce_smooth_to_lerobot ...

For conversion without installing PyTorch and the other training dependencies::

    uv run --no-project --with pyarrow --with pandas python \
      src/lerobot/scripts/convert_vlnce_smooth_to_lerobot.py ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
import random

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - exercised by user environments
    raise SystemExit(
        "PyArrow and pandas are required. Run with the LeRobot environment, or use "
        "`uv run --no-project --with pyarrow --with pandas python <this-script> ...`."
    ) from exc


LOG = logging.getLogger("convert_vlnce_smooth_to_lerobot")
CODEBASE_VERSION = "v3.0"
DEFAULT_SOURCES = (
    Path("/data/ScaleVLN_smooth"),
    Path("/data/R2R_v1-3_smooth"),
    Path("/data/EnvDrop_smooth"),
    Path("/data/RxR_smooth"),
)


@dataclass(frozen=True)
class EpisodeRef:
    source: str
    trajectory: str
    directory: Path
    split: str = "train"


@dataclass(frozen=True)
class Episode:
    source: str
    trajectory: str
    directory: Path
    split: str
    frames: tuple[Path, ...]

    @property
    def length(self) -> int:
        return len(self.frames)


@dataclass
class ConversionState:
    sources: list[str]
    fps: int
    image_key: str
    episodes_per_video: int
    sampling_strategy: str = "sequential"
    sample_stride: int = 1
    sample_offset: int = 0
    sample_seed: int = 42
    max_episodes: int | None = None
    val_fraction: float = 0.1
    test_sources: list[str] | None = None
    total_selected_episodes: int = 0
    selected_episodes_checksum: str = ""
    split_counts: dict[str, int] | None = None
    next_batch_index: int = 0
    total_episodes: int = 0
    total_frames: int = 0
    completed: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        type=Path,
        dest="sources",
        help="Source root; repeat for multiple datasets. Defaults to the four /data/*_smooth roots.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/vlnce_smooth_memvit")
    parser.add_argument("--image-key", default="observation.images.rgb")
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Synthetic frame-clock rate. Keep 1 when physical capture time is unknown.",
    )
    parser.add_argument("--episodes-per-video", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--codec", default="libx264")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--encoder-threads", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum selected trajectories to convert. Omit for every selected trajectory.",
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=("sequential", "stride", "random"),
        default="sequential",
        help=(
            "Trajectory selection mode before conversion. sequential keeps the old sorted order; "
            "stride selects every --sample-stride trajectory; random samples globally with --sample-seed."
        ),
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=10,
        help="For --sampling-strategy=stride, select one trajectory every N sorted trajectories.",
    )
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        help="For --sampling-strategy=stride, start selecting at this zero-based offset.",
    )
    parser.add_argument("--sample-seed", type=int, default=42, help="Seed for random trajectory sampling.")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Validation fraction sampled from non-test trajectories, stratified by source.",
    )
    parser.add_argument(
        "--test-source",
        action="append",
        dest="test_sources",
        default=[],
        help=(
            "Hold out one full source dataset as test distribution; repeat for multiple sources. "
            "Use source directory names such as RxR_smooth."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def write_parquet_atomic(table: pa.Table, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, tmp, compression="snappy", use_dictionary=True)
    os.replace(tmp, path)


def validate_paths(sources: Sequence[Path], output_root: Path) -> None:
    output_resolved = output_root.resolve(strict=False)
    for source in sources:
        if not source.is_dir() or not (source / "images").is_dir():
            raise FileNotFoundError(f"Expected source images directory: {source / 'images'}")
        source_resolved = source.resolve()
        if output_resolved == source_resolved or source_resolved in output_resolved.parents:
            raise ValueError(f"Output must not be inside a source directory: {source}")
        if output_resolved in source_resolved.parents:
            raise ValueError(f"Source must not be inside the output directory: {source}")


def sorted_trajectory_dirs(source: Path) -> list[Path]:
    """Return trajectory dirs in a deterministic order without recursively scanning."""
    with os.scandir(source / "images") as entries:
        return sorted(
            (Path(e.path) for e in entries if e.is_dir(follow_symlinks=False)), key=lambda p: p.name
        )


def get_rgb_frames(trajectory_dir: Path) -> tuple[Path, ...]:
    rgb_dir = trajectory_dir / "rgb"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"Missing RGB directory: {rgb_dir}")
    with os.scandir(rgb_dir) as entries:
        frames = sorted(
            (
                Path(e.path)
                for e in entries
                if e.is_file(follow_symlinks=False) and e.name.lower().endswith(".jpg")
            ),
            key=lambda p: p.name,
        )
    if not frames:
        raise ValueError(f"No JPG frames in {rgb_dir}")
    for index, frame in enumerate(frames):
        expected = f"{index:06d}.jpg"
        if frame.name != expected:
            raise ValueError(f"Non-contiguous frames in {rgb_dir}: expected {expected}, found {frame.name}")
    return tuple(frames)


def iter_episode_refs(sources: Sequence[Path]) -> Iterator[EpisodeRef]:
    for source in sources:
        LOG.info("Indexing trajectory names in %s (one directory level only)", source / "images")
        for trajectory_dir in sorted_trajectory_dirs(source):
            yield EpisodeRef(source=source.name, trajectory=trajectory_dir.name, directory=trajectory_dir)


def select_episode_refs(args: argparse.Namespace, sources: Sequence[Path]) -> list[EpisodeRef]:
    refs = list(iter_episode_refs(sources))
    if args.sampling_strategy == "sequential":
        selected = refs
    elif args.sampling_strategy == "stride":
        selected = refs[args.sample_offset :: args.sample_stride]
    elif args.sampling_strategy == "random":
        rng = random.Random(args.sample_seed)
        selected = refs[:]
        rng.shuffle(selected)
    else:  # pragma: no cover - argparse choices prevent this
        raise ValueError(f"Unknown sampling strategy: {args.sampling_strategy}")

    if args.max_episodes is not None:
        selected = take_balanced_by_source(selected, args.max_episodes)
    LOG.info(
        "Selected %d/%d trajectories: strategy=%s stride=%d offset=%d seed=%d max=%s",
        len(selected),
        len(refs),
        args.sampling_strategy,
        args.sample_stride,
        args.sample_offset,
        args.sample_seed,
        args.max_episodes,
    )
    return selected


def take_balanced_by_source(refs: Sequence[EpisodeRef], max_episodes: int) -> list[EpisodeRef]:
    by_source: dict[str, list[EpisodeRef]] = {}
    for ref in refs:
        by_source.setdefault(ref.source, []).append(ref)

    selected: list[EpisodeRef] = []
    positions = {source: 0 for source in by_source}
    sources = sorted(by_source)
    while len(selected) < max_episodes:
        added = False
        for source in sources:
            pos = positions[source]
            source_refs = by_source[source]
            if pos >= len(source_refs):
                continue
            selected.append(source_refs[pos])
            positions[source] += 1
            added = True
            if len(selected) >= max_episodes:
                break
        if not added:
            break
    return selected


def build_split_refs(args: argparse.Namespace, selected: Sequence[EpisodeRef]) -> list[EpisodeRef]:
    test_sources = set(args.test_sources or [])
    known_sources = {ref.source for ref in selected}
    unknown_test_sources = test_sources - known_sources
    if unknown_test_sources:
        raise ValueError(
            f"--test-source values not found in selected sources: {sorted(unknown_test_sources)}; "
            f"available={sorted(known_sources)}"
        )

    train_val: dict[str, list[EpisodeRef]] = {}
    test_refs: list[EpisodeRef] = []
    for ref in selected:
        if ref.source in test_sources:
            test_refs.append(EpisodeRef(ref.source, ref.trajectory, ref.directory, "test"))
        else:
            train_val.setdefault(ref.source, []).append(ref)

    rng = random.Random(args.sample_seed + 17)
    train_refs: list[EpisodeRef] = []
    val_refs: list[EpisodeRef] = []
    for source in sorted(train_val):
        source_refs = train_val[source][:]
        rng.shuffle(source_refs)
        if len(source_refs) < 2:
            train_refs.extend(EpisodeRef(ref.source, ref.trajectory, ref.directory, "train") for ref in source_refs)
            continue
        num_val = max(1, min(len(source_refs) - 1, round(len(source_refs) * args.val_fraction)))
        val_refs.extend(EpisodeRef(ref.source, ref.trajectory, ref.directory, "val") for ref in source_refs[:num_val])
        train_refs.extend(EpisodeRef(ref.source, ref.trajectory, ref.directory, "train") for ref in source_refs[num_val:])

    if not train_refs:
        raise ValueError("Split produced no training trajectories")
    if not val_refs:
        raise ValueError("Split produced no validation trajectories")
    if test_sources and not test_refs:
        raise ValueError("Split produced no test trajectories")

    ordered = train_refs + val_refs + test_refs
    LOG.info(
        "Split selected trajectories: train=%d val=%d test=%d test_sources=%s",
        len(train_refs),
        len(val_refs),
        len(test_refs),
        sorted(test_sources),
    )
    return ordered


def split_counts(refs: Sequence[EpisodeRef]) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    for ref in refs:
        counts[ref.split] = counts.get(ref.split, 0) + 1
    return counts


def materialize_episode(ref: EpisodeRef) -> Episode:
    return Episode(
        source=ref.source,
        trajectory=ref.trajectory,
        directory=ref.directory,
        split=ref.split,
        frames=get_rgb_frames(ref.directory),
    )


def iter_episodes(refs: Sequence[EpisodeRef]) -> Iterator[Episode]:
    for ref in refs:
        yield materialize_episode(ref)


def selected_refs_checksum(refs: Sequence[EpisodeRef]) -> str:
    digest = hashlib.sha256()
    for ref in refs:
        digest.update(ref.split.encode("utf-8"))
        digest.update(b"\0")
        digest.update(ref.source.encode("utf-8"))
        digest.update(b"\0")
        digest.update(ref.trajectory.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(ref.directory).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def batched(iterator: Iterator[Episode], size: int) -> Iterator[list[Episode]]:
    batch: list[Episode] = []
    for item in iterator:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def output_file(root: Path, category: str, file_index: int, chunk_size: int, suffix: str) -> Path:
    return (
        root
        / category
        / f"chunk-{file_index // chunk_size:03d}"
        / f"file-{file_index % chunk_size:03d}.{suffix}"
    )


def video_file(root: Path, image_key: str, file_index: int, chunk_size: int) -> Path:
    return output_file(root, f"videos/{image_key}", file_index, chunk_size, "mp4")


def encode_video(
    episodes: Sequence[Episode],
    destination: Path,
    *,
    fps: int,
    codec: str,
    crf: int,
    preset: str,
    threads: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".partial.mp4")
    temporary.unlink(missing_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "image2pipe",
        "-framerate",
        str(fps),
        "-vcodec",
        "mjpeg",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        codec,
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-g",
        "2",
        "-threads",
        str(threads),
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for episode in episodes:
            for frame in episode.frames:
                with frame.open("rb") as source:
                    shutil.copyfileobj(source, process.stdin, length=1024 * 1024)
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        temporary.unlink(missing_ok=True)
        raise
    if return_code != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed ({return_code}): {stderr.decode(errors='replace')[-4000:]}")
    expected_frames = sum(ep.length for ep in episodes)
    actual_frames = probe_frame_count(temporary)
    if actual_frames != expected_frames:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Encoded frame count mismatch: expected {expected_frames}, got {actual_frames}")
    os.replace(temporary, destination)


def probe_frame_count(video: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def make_data_table(episodes: Sequence[Episode], first_episode: int, first_frame: int, fps: int) -> pa.Table:
    timestamps: list[float] = []
    frame_indices: list[int] = []
    episode_indices: list[int] = []
    indices: list[int] = []
    task_indices: list[int] = []
    global_index = first_frame
    for local_episode, episode in enumerate(episodes):
        episode_index = first_episode + local_episode
        for frame_index in range(episode.length):
            timestamps.append(frame_index / fps)
            frame_indices.append(frame_index)
            episode_indices.append(episode_index)
            indices.append(global_index)
            task_indices.append(0)
            global_index += 1
    return pa.table(
        {
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "frame_index": pa.array(frame_indices, type=pa.int64()),
            "episode_index": pa.array(episode_indices, type=pa.int64()),
            "index": pa.array(indices, type=pa.int64()),
            "task_index": pa.array(task_indices, type=pa.int64()),
        }
    )


def make_episode_table(
    episodes: Sequence[Episode],
    *,
    first_episode: int,
    first_frame: int,
    batch_index: int,
    chunk_size: int,
    fps: int,
    image_key: str,
) -> pa.Table:
    rows: dict[str, list] = {
        "episode_index": [],
        "tasks": [],
        "length": [],
        "meta/episodes/chunk_index": [],
        "meta/episodes/file_index": [],
        "data/chunk_index": [],
        "data/file_index": [],
        "dataset_from_index": [],
        "dataset_to_index": [],
        f"videos/{image_key}/chunk_index": [],
        f"videos/{image_key}/file_index": [],
        f"videos/{image_key}/from_timestamp": [],
        f"videos/{image_key}/to_timestamp": [],
    }
    video_frame_offset = 0
    global_frame = first_frame
    chunk_index = batch_index // chunk_size
    file_index = batch_index % chunk_size
    for local_episode, episode in enumerate(episodes):
        rows["episode_index"].append(first_episode + local_episode)
        rows["tasks"].append(["visual representation distillation"])
        rows["length"].append(episode.length)
        rows["meta/episodes/chunk_index"].append(chunk_index)
        rows["meta/episodes/file_index"].append(file_index)
        rows["data/chunk_index"].append(chunk_index)
        rows["data/file_index"].append(file_index)
        rows["dataset_from_index"].append(global_frame)
        rows["dataset_to_index"].append(global_frame + episode.length)
        rows[f"videos/{image_key}/chunk_index"].append(chunk_index)
        rows[f"videos/{image_key}/file_index"].append(file_index)
        rows[f"videos/{image_key}/from_timestamp"].append(video_frame_offset / fps)
        rows[f"videos/{image_key}/to_timestamp"].append((video_frame_offset + episode.length) / fps)
        video_frame_offset += episode.length
        global_frame += episode.length
    return pa.Table.from_pydict(rows)


def write_static_metadata(root: Path) -> None:
    tasks_path = root / "meta/tasks.parquet"
    tasks_tmp = tasks_path.with_suffix(tasks_path.suffix + ".tmp")
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = pd.DataFrame(
        {"task_index": [0]},
        index=pd.Index(["visual representation distillation"], name="task"),
    )
    tasks.to_parquet(tasks_tmp)
    os.replace(tasks_tmp, tasks_path)
    atomic_json(root / "meta/stats.json", {})


def write_info(root: Path, args: argparse.Namespace, state: ConversionState) -> None:
    counts = state.split_counts or {"train": state.total_episodes, "val": 0, "test": 0}
    train_end = min(counts.get("train", 0), state.total_episodes)
    val_end = min(train_end + counts.get("val", 0), state.total_episodes)
    test_end = min(val_end + counts.get("test", 0), state.total_episodes)
    splits = {"train": f"0:{train_end}"}
    if val_end > train_end:
        splits["val"] = f"{train_end}:{val_end}"
    if test_end > val_end:
        splits["test"] = f"{val_end}:{test_end}"
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": "vlnce_navigation",
        "total_episodes": state.total_episodes,
        "total_frames": state.total_frames,
        "total_tasks": 1,
        "chunks_size": args.chunk_size,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": state.fps,
        "splits": splits,
        "vlnce_conversion": {
            "sampling_strategy": state.sampling_strategy,
            "sample_stride": state.sample_stride,
            "sample_offset": state.sample_offset,
            "sample_seed": state.sample_seed,
            "max_episodes": state.max_episodes,
            "val_fraction": state.val_fraction,
            "test_sources": state.test_sources or [],
            "total_selected_episodes": state.total_selected_episodes,
            "selected_episodes_checksum": state.selected_episodes_checksum,
            "split_counts": state.split_counts or {},
        },
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            state.image_key: {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": 480,
                    "video.width": 640,
                    "video.channels": 3,
                    "video.fps": state.fps,
                    "video.codec": "h264" if args.codec == "libx264" else args.codec,
                    "video.pix_fmt": "yuv420p",
                    "is_depth_map": False,
                    "has_audio": False,
                },
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    atomic_json(root / "meta/info.json", info)


def load_or_create_state(
    args: argparse.Namespace, sources: Sequence[Path], selected_refs: Sequence[EpisodeRef]
) -> ConversionState:
    state_path = args.output_root / "conversion_state.json"
    expected = ConversionState(
        sources=[str(p.resolve()) for p in sources],
        fps=args.fps,
        image_key=args.image_key,
        episodes_per_video=args.episodes_per_video,
        sampling_strategy=args.sampling_strategy,
        sample_stride=args.sample_stride,
        sample_offset=args.sample_offset,
        sample_seed=args.sample_seed,
        max_episodes=args.max_episodes,
        val_fraction=args.val_fraction,
        test_sources=list(args.test_sources or []),
        total_selected_episodes=len(selected_refs),
        selected_episodes_checksum=selected_refs_checksum(selected_refs),
        split_counts=split_counts(selected_refs),
    )
    if state_path.exists():
        if not args.resume:
            raise FileExistsError(f"Output already contains state; pass --resume: {state_path}")
        with state_path.open(encoding="utf-8") as f:
            state = ConversionState(**json.load(f))
        for field in (
            "sources",
            "fps",
            "image_key",
            "episodes_per_video",
            "sampling_strategy",
            "sample_stride",
            "sample_offset",
            "sample_seed",
            "max_episodes",
            "val_fraction",
            "test_sources",
            "total_selected_episodes",
            "selected_episodes_checksum",
            "split_counts",
        ):
            if getattr(state, field) != getattr(expected, field):
                raise ValueError(f"Resume configuration mismatch for {field}")
        if state.completed:
            LOG.info("Dataset is already marked complete: %s", args.output_root)
        return state
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty output directory: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_static_metadata(args.output_root)
    atomic_json(state_path, asdict(expected))
    return expected


def append_manifest(root: Path, first_episode: int, episodes: Sequence[Episode]) -> None:
    with (root / "source_episodes.jsonl").open("a", encoding="utf-8") as f:
        for offset, episode in enumerate(episodes):
            f.write(
                json.dumps(
                    {
                        "episode_index": first_episode + offset,
                        "source": episode.source,
                        "trajectory": episode.trajectory,
                        "source_directory": str(episode.directory),
                        "split": episode.split,
                        "length": episode.length,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        f.flush()
        os.fsync(f.fileno())


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="[%(asctime)s] %(levelname)s %(message)s")
    sources = tuple(args.sources or DEFAULT_SOURCES)
    if args.fps < 1 or args.episodes_per_video < 1 or args.chunk_size < 1:
        raise ValueError("fps, episodes-per-video, and chunk-size must be positive")
    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError("max-episodes must be positive")
    if args.sample_stride < 1:
        raise ValueError("sample-stride must be positive")
    if args.sample_offset < 0:
        raise ValueError("sample-offset must be non-negative")
    if args.sample_offset >= args.sample_stride:
        raise ValueError("sample-offset must be smaller than sample-stride")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("val-fraction must be between 0 and 1")
    if not args.test_sources:
        raise ValueError("At least one --test-source is required for a held-out test distribution")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe must be installed")
    validate_paths(sources, args.output_root)
    selected_refs = select_episode_refs(args, sources)
    selected_refs = build_split_refs(args, selected_refs)
    if not selected_refs:
        raise ValueError("No trajectories selected")
    state = load_or_create_state(args, sources, selected_refs)
    if state.completed:
        return

    episode_iterator = iter_episodes(selected_refs)
    to_skip = state.total_episodes
    for _ in range(to_skip):
        try:
            next(episode_iterator)
        except StopIteration as exc:
            raise RuntimeError("Resume state contains more episodes than the selected trajectories") from exc

    remaining = state.total_selected_episodes - state.total_episodes
    for episodes in batched(episode_iterator, args.episodes_per_video):
        episodes = episodes[:remaining]
        if not episodes:
            break
        batch_index = state.next_batch_index
        first_episode = state.total_episodes
        first_frame = state.total_frames
        num_frames = sum(ep.length for ep in episodes)
        vid_path = video_file(args.output_root, args.image_key, batch_index, args.chunk_size)
        data_path = output_file(args.output_root, "data", batch_index, args.chunk_size, "parquet")
        episode_path = output_file(args.output_root, "meta/episodes", batch_index, args.chunk_size, "parquet")
        LOG.info(
            "Batch %d: episodes %d..%d, frames=%d -> %s",
            batch_index,
            first_episode,
            first_episode + len(episodes) - 1,
            num_frames,
            vid_path,
        )
        for completed_path in (vid_path, data_path, episode_path):
            if completed_path.exists():
                raise FileExistsError(f"Untracked output exists; refusing to overwrite: {completed_path}")
        encode_video(
            episodes,
            vid_path,
            fps=args.fps,
            codec=args.codec,
            crf=args.crf,
            preset=args.preset,
            threads=args.encoder_threads,
        )
        write_parquet_atomic(make_data_table(episodes, first_episode, first_frame, args.fps), data_path)
        write_parquet_atomic(
            make_episode_table(
                episodes,
                first_episode=first_episode,
                first_frame=first_frame,
                batch_index=batch_index,
                chunk_size=args.chunk_size,
                fps=args.fps,
                image_key=args.image_key,
            ),
            episode_path,
        )
        append_manifest(args.output_root, first_episode, episodes)
        state.next_batch_index += 1
        state.total_episodes += len(episodes)
        state.total_frames += num_frames
        atomic_json(args.output_root / "conversion_state.json", asdict(state))
        write_info(args.output_root, args, state)
        remaining -= len(episodes)

    state.completed = True
    atomic_json(args.output_root / "conversion_state.json", asdict(state))
    write_info(args.output_root, args, state)
    LOG.info(
        "Conversion complete: episodes=%d frames=%d root=%s",
        state.total_episodes,
        state.total_frames,
        args.output_root,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.warning("Interrupted. Re-run with --resume to continue from the last completed batch.")
        sys.exit(130)

# uv run python -m lerobot.scripts.convert_vlnce_smooth_to_lerobot \
#   --output-root /data/VLNCE_smooth_lerobot_final \
#   --repo-id local/vlnce_smooth_memvit \
#   --episodes-per-video 128 \
#   --sampling-strategy stride \
#   --sample-stride 10 \
#   --val-fraction 0.1 \
#   --test-source RxR_smooth
