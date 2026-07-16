#!/usr/bin/env python3
"""Convert drone videos into a standalone MEM-ViT test dataset.

The output is a small standalone LeRobot dataset root containing only a
``test`` split.  It is intended to be used together with
``train_mem_vit_distill.py --test-dataset-root`` so the train/val dataset and
the held-out drone test dataset do not need to live in the same directory.

The script reads schema/fps/video settings from an existing MEM-ViT dataset,
but it does not copy or hard-link that dataset's train/val files.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import statistics
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


LOG = logging.getLogger("convert_drone_videos_to_lerobot")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schema-root",
        type=Path,
        default=Path("/data/VLNCE_smooth_lerobot_final"),
        help="Existing MEM-ViT LeRobot dataset used only as schema/template.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("/data/mem_vit_drone_test"))
    parser.add_argument("--repo-id", default="local/vlnce_smooth_memvit")
    parser.add_argument("--drone-root", type=Path, default=Path("/data/drone_videos"))
    parser.add_argument("--image-key", default="observation.images.rgb")
    parser.add_argument("--sample-hz", type=float, default=5.0)
    parser.add_argument("--skip-start-sec", type=float, default=20.0)
    parser.add_argument("--skip-end-sec", type=float, default=20.0)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--codec", default="libx264")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--encoder-threads", type=int, default=0)
    parser.add_argument("--max-videos", type=int, default=None, help="Debug limit.")
    parser.add_argument("--dry-run", action="store_true", help="Only scan videos and estimate output size.")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Resume an interrupted conversion by keeping complete prefix files in --output-root.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(temporary, path)


def write_parquet_atomic(table: pa.Table, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, temporary, compression="snappy", use_dictionary=True)
    os.replace(temporary, path)


def output_file(root: Path, category: str, file_index: int, chunk_size: int, suffix: str) -> Path:
    return (
        root
        / category
        / f"chunk-{file_index // chunk_size:03d}"
        / f"file-{file_index % chunk_size:03d}.{suffix}"
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def find_videos(root: Path) -> list[Path]:
    videos = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise FileNotFoundError(f"No videos found under {root}")
    return videos


def ffprobe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    info = json.loads(result.stdout)
    stream = (info.get("streams") or [{}])[0]
    duration = float(stream.get("duration") or info.get("format", {}).get("duration") or 0.0)
    return {
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "duration": duration,
        "avg_frame_rate": stream.get("avg_frame_rate") or stream.get("r_frame_rate"),
    }


def estimate_drone_test(videos: list[Path], *, skip_start_sec: float, skip_end_sec: float, sample_hz: float) -> dict[str, Any]:
    usable_durations: list[float] = []
    skipped: list[dict[str, Any]] = []
    frame_count = 0
    resolutions: dict[str, int] = {}
    for video in videos:
        meta = ffprobe_video(video)
        duration = float(meta["duration"])
        usable_duration = duration - skip_start_sec - skip_end_sec
        resolution = f"{meta['width']}x{meta['height']}"
        resolutions[resolution] = resolutions.get(resolution, 0) + 1
        if usable_duration <= 0:
            skipped.append({"path": str(video), "duration": duration, "reason": "too_short"})
            continue
        usable_durations.append(usable_duration)
        frame_count += math.ceil(usable_duration * sample_hz)

    if usable_durations:
        duration_summary = {
            "min_sec": min(usable_durations),
            "median_sec": statistics.median(usable_durations),
            "max_sec": max(usable_durations),
            "total_sec": sum(usable_durations),
        }
    else:
        duration_summary = {"min_sec": 0.0, "median_sec": 0.0, "max_sec": 0.0, "total_sec": 0.0}
    return {
        "input_videos": len(videos),
        "convertible_videos": len(usable_durations),
        "skipped_videos": len(skipped),
        "sample_hz": sample_hz,
        "estimated_frames": frame_count,
        "estimated_dataset_seconds_at_schema_fps_1": frame_count,
        "usable_duration": duration_summary,
        "top_resolutions": dict(sorted(resolutions.items(), key=lambda item: item[1], reverse=True)[:20]),
        "skipped": skipped,
    }


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
    text = result.stdout.strip()
    if text and text != "N/A":
        return int(text)
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def encode_sampled_video(
    src: Path,
    dst: Path,
    *,
    start_sec: float,
    duration_sec: float,
    sample_hz: float,
    dataset_fps: int,
    width: int,
    height: int,
    codec: str,
    crf: int,
    preset: str,
    threads: int,
) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = dst.with_name(dst.stem + ".partial.mp4")
    temporary.unlink(missing_ok=True)
    vf = (
        f"fps={sample_hz},"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        f"setsar=1,setpts=N/({dataset_fps}*TB)"
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_sec:.6f}",
        "-t",
        f"{duration_sec:.6f}",
        "-i",
        str(src),
        "-an",
        "-vf",
        vf,
        "-c:v",
        codec,
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(dataset_fps),
        "-threads",
        str(threads),
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    subprocess.run(command, check=True)
    frame_count = probe_frame_count(temporary)
    if frame_count <= 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Encoded zero frames: {src}")
    os.replace(temporary, dst)
    return frame_count


def make_data_table(episode_index: int, first_frame: int, length: int, fps: int) -> pa.Table:
    return pa.table(
        {
            "timestamp": pa.array([i / fps for i in range(length)], type=pa.float32()),
            "frame_index": pa.array(range(length), type=pa.int64()),
            "episode_index": pa.array([episode_index] * length, type=pa.int64()),
            "index": pa.array(range(first_frame, first_frame + length), type=pa.int64()),
            "task_index": pa.array([0] * length, type=pa.int64()),
        }
    )


def make_episode_table(
    *,
    episode_index: int,
    length: int,
    first_frame: int,
    file_index: int,
    chunk_size: int,
    fps: int,
    image_key: str,
) -> pa.Table:
    chunk_index = file_index // chunk_size
    file_index_in_chunk = file_index % chunk_size
    return pa.Table.from_pydict(
        {
            "episode_index": [episode_index],
            "tasks": [["visual representation distillation"]],
            "length": [length],
            "meta/episodes/chunk_index": [chunk_index],
            "meta/episodes/file_index": [file_index_in_chunk],
            "data/chunk_index": [chunk_index],
            "data/file_index": [file_index_in_chunk],
            "dataset_from_index": [first_frame],
            "dataset_to_index": [first_frame + length],
            f"videos/{image_key}/chunk_index": [chunk_index],
            f"videos/{image_key}/file_index": [file_index_in_chunk],
            f"videos/{image_key}/from_timestamp": [0.0],
            f"videos/{image_key}/to_timestamp": [length / fps],
        }
    )


def copy_schema_sidecars(schema_root: Path, output_root: Path) -> None:
    for name in ("tasks.parquet", "stats.json"):
        src = schema_root / "meta" / name
        if src.exists():
            dst = output_root / "meta" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def read_existing_episode_length(root: Path, file_index: int, chunk_size: int) -> int:
    episode_path = output_file(root, "meta/episodes", file_index, chunk_size, "parquet")
    table = pq.read_table(episode_path, columns=["length"])
    return int(table["length"][0].as_py())


def existing_complete_prefix(root: Path, *, image_key: str, chunk_size: int) -> tuple[int, int]:
    count = 0
    total_frames = 0
    while True:
        video_path = output_file(root, f"videos/{image_key}", count, chunk_size, "mp4")
        data_path = output_file(root, "data", count, chunk_size, "parquet")
        episode_path = output_file(root, "meta/episodes", count, chunk_size, "parquet")
        if not (video_path.exists() and data_path.exists() and episode_path.exists()):
            break
        total_frames += read_existing_episode_length(root, count, chunk_size)
        count += 1
    return count, total_frames


def remove_incomplete_suffix(root: Path, *, image_key: str, chunk_size: int, start_file_index: int) -> None:
    for category, suffix in (
        (f"videos/{image_key}", "mp4"),
        ("data", "parquet"),
        ("meta/episodes", "parquet"),
    ):
        path = output_file(root, category, start_file_index, chunk_size, suffix)
        path.unlink(missing_ok=True)
        if suffix == "mp4":
            path.with_name(path.stem + ".partial.mp4").unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="[%(asctime)s] %(levelname)s %(message)s")
    if args.sample_hz <= 0:
        raise ValueError("--sample-hz must be positive")
    if args.skip_start_sec < 0 or args.skip_end_sec < 0:
        raise ValueError("skip values must be non-negative")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe must be installed")

    schema_info = load_json(args.schema_root / "meta/info.json")
    features = dict(schema_info["features"])
    if args.image_key not in features:
        raise ValueError(f"image-key {args.image_key!r} not found in schema features: {list(features)}")
    dataset_fps = int(schema_info["fps"])
    chunk_size = int(schema_info.get("chunks_size") or 1000)
    video_feature = dict(features[args.image_key])
    video_info = dict(video_feature.get("info") or {})
    width = args.width or int(video_info.get("video.width") or video_feature["shape"][1])
    height = args.height or int(video_info.get("video.height") or video_feature["shape"][0])

    videos = find_videos(args.drone_root)
    if args.max_videos is not None:
        videos = videos[: args.max_videos]
    LOG.info("Found %d drone videos", len(videos))
    if args.dry_run:
        summary = estimate_drone_test(
            videos,
            skip_start_sec=args.skip_start_sec,
            skip_end_sec=args.skip_end_sec,
            sample_hz=args.sample_hz,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.output_root.exists() and not args.resume_existing:
        if not args.overwrite:
            raise FileExistsError(
                f"Output exists; pass --overwrite to replace it or --resume-existing to continue: {args.output_root}"
            )
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    copy_schema_sidecars(args.schema_root, args.output_root)

    episode_index = 0
    frame_index = 0
    file_index = 0
    drone_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    if args.resume_existing:
        episode_index, frame_index = existing_complete_prefix(
            args.output_root, image_key=args.image_key, chunk_size=chunk_size
        )
        file_index = episode_index
        remove_incomplete_suffix(
            args.output_root,
            image_key=args.image_key,
            chunk_size=chunk_size,
            start_file_index=file_index,
        )
        LOG.info(
            "Resuming existing conversion: complete_episodes=%d complete_frames=%d",
            episode_index,
            frame_index,
        )
        for existing_index, video in enumerate(videos[:episode_index]):
            meta = ffprobe_video(video)
            length = read_existing_episode_length(args.output_root, existing_index, chunk_size)
            drone_rows.append(
                {
                    "episode_index": existing_index,
                    "source": "drone_videos",
                    "trajectory": str(video.relative_to(args.drone_root)),
                    "source_video": str(video),
                    "split": "test",
                    "length": length,
                    "source_duration_sec": float(meta["duration"]),
                    "sample_hz": args.sample_hz,
                    "skip_start_sec": args.skip_start_sec,
                    "skip_end_sec": args.skip_end_sec,
                }
            )

    for video in videos[episode_index:]:
        meta = ffprobe_video(video)
        duration = float(meta["duration"])
        usable_duration = duration - args.skip_start_sec - args.skip_end_sec
        if usable_duration <= 0:
            skipped.append({"path": str(video), "duration": duration, "reason": "too_short"})
            continue

        vid_path = output_file(args.output_root, f"videos/{args.image_key}", file_index, chunk_size, "mp4")
        data_path = output_file(args.output_root, "data", file_index, chunk_size, "parquet")
        episode_path = output_file(args.output_root, "meta/episodes", file_index, chunk_size, "parquet")
        LOG.info(
            "Drone test episode %d/%d: %s start=%.1fs duration=%.1fs",
            episode_index + 1,
            len(videos),
            video,
            args.skip_start_sec,
            usable_duration,
        )
        length = encode_sampled_video(
            video,
            vid_path,
            start_sec=args.skip_start_sec,
            duration_sec=usable_duration,
            sample_hz=args.sample_hz,
            dataset_fps=dataset_fps,
            width=width,
            height=height,
            codec=args.codec,
            crf=args.crf,
            preset=args.preset,
            threads=args.encoder_threads,
        )
        write_parquet_atomic(make_data_table(episode_index, frame_index, length, dataset_fps), data_path)
        write_parquet_atomic(
            make_episode_table(
                episode_index=episode_index,
                length=length,
                first_frame=frame_index,
                file_index=file_index,
                chunk_size=chunk_size,
                fps=dataset_fps,
                image_key=args.image_key,
            ),
            episode_path,
        )
        drone_rows.append(
            {
                "episode_index": episode_index,
                "source": "drone_videos",
                "trajectory": str(video.relative_to(args.drone_root)),
                "source_video": str(video),
                "split": "test",
                "length": length,
                "source_duration_sec": duration,
                "sample_hz": args.sample_hz,
                "skip_start_sec": args.skip_start_sec,
                "skip_end_sec": args.skip_end_sec,
            }
        )
        episode_index += 1
        frame_index += length
        file_index += 1

    if not drone_rows:
        raise RuntimeError("No drone videos were converted")

    video_feature["shape"] = [height, width, 3]
    video_info.update(
        {
            "video.height": height,
            "video.width": width,
            "video.channels": 3,
            "video.fps": dataset_fps,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "is_depth_map": False,
            "has_audio": False,
        }
    )
    video_feature["info"] = video_info
    features[args.image_key] = video_feature

    info = {
        "codebase_version": schema_info.get("codebase_version", "v3.0"),
        "robot_type": schema_info.get("robot_type", "vlnce_navigation"),
        "total_episodes": episode_index,
        "total_frames": frame_index,
        "total_tasks": schema_info.get("total_tasks", 1),
        "chunks_size": chunk_size,
        "fps": dataset_fps,
        "splits": {"test": f"0:{episode_index}"},
        "data_path": schema_info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"),
        "video_path": schema_info.get(
            "video_path",
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        ),
        "features": features,
        "drone_test_conversion": {
            "schema_root": str(args.schema_root),
            "drone_root": str(args.drone_root),
            "repo_id": args.repo_id,
            "sample_hz": args.sample_hz,
            "skip_start_sec": args.skip_start_sec,
            "skip_end_sec": args.skip_end_sec,
            "num_test_episodes": episode_index,
            "num_test_frames": frame_index,
            "skipped": skipped,
            "note": "Frames are sampled at sample_hz but encoded at the schema dataset fps.",
        },
    }
    atomic_json(args.output_root / "meta/info.json", info)
    with (args.output_root / "source_episodes.jsonl").open("w", encoding="utf-8") as f:
        for row in drone_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    atomic_json(
        args.output_root / "drone_test_conversion_state.json",
        {
            "schema_root": str(args.schema_root),
            "output_root": str(args.output_root),
            "drone_root": str(args.drone_root),
            "repo_id": args.repo_id,
            "sample_hz": args.sample_hz,
            "skip_start_sec": args.skip_start_sec,
            "skip_end_sec": args.skip_end_sec,
            "num_test_episodes": episode_index,
            "num_test_frames": frame_index,
            "skipped": skipped,
            "completed": True,
        },
    )
    LOG.info("Done: output=%s test=%d total_frames=%d", args.output_root, episode_index, frame_index)


if __name__ == "__main__":
    main()

# Estimate without converting:
#
# uv run python -m lerobot.scripts.convert_drone_videos_to_lerobot \
#   --schema-root /data/VLNCE_smooth_lerobot_final \
#   --output-root /data/mem_vit_drone_test \
#   --drone-root /data/drone_videos \
#   --sample-hz 5 \
#   --skip-start-sec 20 \
#   --skip-end-sec 20 \
#   --dry-run
#
# Full drone test conversion at 5Hz:
#
# uv run python -m lerobot.scripts.convert_drone_videos_to_lerobot \
#   --schema-root /data/VLNCE_smooth_lerobot_final \
#   --output-root /data/mem_vit_drone_test \
#   --drone-root /data/drone_videos \
#   --sample-hz 5 \
#   --skip-start-sec 20 \
#   --skip-end-sec 20
#
# Train with train/val from the original dataset and test from the drone-only dataset:
#
# uv run python -m lerobot.scripts.train_mem_vit_distill \
#   --dataset-repo-id local/vlnce_smooth_memvit \
#   --dataset-root /data/VLNCE_smooth_lerobot_final \
#   --test-dataset-repo-id local/vlnce_smooth_memvit \
#   --test-dataset-root /data/mem_vit_drone_test
#
