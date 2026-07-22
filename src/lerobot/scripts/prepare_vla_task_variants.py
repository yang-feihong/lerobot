#!/usr/bin/env python3
"""Convert raw per-video VLA annotations into LeRobot task variants.

This script is intentionally a small adapter: training consumes the stable
dataset-side format ``<dataset-root>/meta/task_variants.json`` while raw
annotation files are allowed to keep their collection-specific structure.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any


LOG = logging.getLogger("prepare_vla_task_variants")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="Raw annotations.json file.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Converted LeRobot dataset root containing diagnostics/conversion_manifest.json.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional explicit conversion_manifest.json path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Default: <dataset-root>/meta/task_variants.json.",
    )
    parser.add_argument(
        "--detail-levels",
        nargs="*",
        default=None,
        help="Optional whitelist of instruction detail_level values, e.g. task_level concise.",
    )
    parser.add_argument("--min-variants", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_bag_stem(path_like: str) -> str:
    """Return the common task_... stem from a bag/video/diagnostic path."""
    path = Path(path_like)
    for part in reversed(path.parts):
        if part.startswith("task_"):
            stem = part
            break
    else:
        stem = path.stem
    if stem.endswith(".bag"):
        stem = stem[: -len(".bag")]
    if stem.endswith(".bag.active"):
        stem = stem[: -len(".bag.active")]
    return stem


def manifest_episode_by_stem(manifest: dict[str, Any]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for entry in manifest.get("bags", {}).values():
        if entry.get("status") != "done":
            continue
        episode_index = entry.get("episode_index")
        if episode_index is None:
            continue
        for key in ("bag_name", "bag_path"):
            value = entry.get(key)
            if value:
                mapping[normalize_bag_stem(str(value))] = int(episode_index)
    return mapping


def instructions_from_record(record: dict[str, Any], detail_levels: set[str] | None) -> list[str]:
    instructions = []
    for item in record.get("instructions") or []:
        if not isinstance(item, dict):
            continue
        if detail_levels is not None and item.get("detail_level") not in detail_levels:
            continue
        text = item.get("instruction")
        if isinstance(text, str) and text.strip():
            instructions.append(text.strip())

    # Preserve order while deduplicating.
    return list(dict.fromkeys(instructions))


def build_task_variants(
    *,
    annotations: dict[str, Any],
    manifest: dict[str, Any],
    detail_levels: set[str] | None,
    min_variants: int,
) -> dict[str, list[str]]:
    episode_by_stem = manifest_episode_by_stem(manifest)
    variants: dict[str, list[str]] = {}
    unmatched: list[str] = []
    too_few: list[str] = []

    for record in annotations.get("records") or []:
        if not isinstance(record, dict) or record.get("status") != "ok":
            continue
        video_path = str(record.get("video_path") or "")
        stem = normalize_bag_stem(video_path)
        episode_index = episode_by_stem.get(stem)
        if episode_index is None:
            unmatched.append(stem)
            continue
        instructions = instructions_from_record(record, detail_levels)
        if len(instructions) < min_variants:
            too_few.append(stem)
            continue
        variants[str(episode_index)] = instructions

    if unmatched:
        LOG.warning("Unmatched annotation records: %d; examples=%s", len(unmatched), unmatched[:5])
    if too_few:
        LOG.warning("Records with fewer than %d variants: %d; examples=%s", min_variants, len(too_few), too_few[:5])
    if not variants:
        raise RuntimeError("No task variants matched the conversion manifest.")
    return dict(sorted(variants.items(), key=lambda item: int(item[0])))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s")

    annotations = load_json(args.annotations)
    manifest_path = args.manifest or args.dataset_root / "diagnostics" / "conversion_manifest.json"
    manifest = load_json(manifest_path)
    output = args.output or args.dataset_root / "meta" / "task_variants.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output}")

    detail_levels = set(args.detail_levels) if args.detail_levels else None
    variants = build_task_variants(
        annotations=annotations,
        manifest=manifest,
        detail_levels=detail_levels,
        min_variants=args.min_variants,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(variants, f, indent=2, ensure_ascii=False)
        f.write("\n")

    counts = [len(v) for v in variants.values()]
    LOG.info(
        "Wrote %s with %d episode(s), variants per episode min=%d max=%d avg=%.2f",
        output,
        len(variants),
        min(counts),
        max(counts),
        sum(counts) / len(counts),
    )


if __name__ == "__main__":
    main()

# Example:
#
# uv run python -m lerobot.scripts.prepare_vla_task_variants \
#   --annotations /data/rosbag/rosbags_0721/annotations.json \
#   --dataset-root /data/b2_z1_vla_lerobot_0721 \
#   --overwrite
