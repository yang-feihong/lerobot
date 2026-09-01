import json

import pytest

from lerobot.configs.default import DatasetConfig
from lerobot.datasets.paired_image_source import PairedImageSource


def make_pair(tmp_path):
    rollout = tmp_path / "rollout"
    media = rollout / "paired_media"
    media.mkdir(parents=True)
    for name in ("sim_base.mp4", "sim_wrist.mp4"):
        (media / name).write_bytes(b"video")
    (media / "frame_correspondence.csv").write_text(
        "sim_step,source_frame_index,source_time_s\n0,0,0.0\n13,13,0.26\n25,25,0.5\n"
    )
    manifest = tmp_path / "accepted.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "episode_index": 4,
                "success": True,
                "rollout_dir": f"/another/machine/{rollout.name}",
                "sim_base_video": "paired_media/sim_base.mp4",
                "sim_wrist_video": "paired_media/sim_wrist.mp4",
                "sim_video_fps": 4.0,
                "frame_correspondence": "paired_media/frame_correspondence.csv",
            }
        )
        + "\n"
    )
    return manifest


def test_paired_image_source_maps_source_time_to_nearest_sim_frame(tmp_path):
    manifest = make_pair(tmp_path)
    source = PairedImageSource(
        mode="sim",
        manifest=manifest,
        root=tmp_path,
        mixed_sim_probability=0.5,
        seed=3,
        episodes=[4],
    )

    path, timestamps = source.resolve(4, "observation.images.base_0_rgb", [0.02, 0.24, 0.49])

    assert path == tmp_path / "rollout/paired_media/sim_base.mp4"
    assert timestamps == [0.0, 0.25, 0.5]


def test_paired_image_source_requires_every_selected_episode(tmp_path):
    manifest = make_pair(tmp_path)
    with pytest.raises(ValueError, match="does not cover 1 requested episode"):
        PairedImageSource(
            mode="sim",
            manifest=manifest,
            root=tmp_path,
            mixed_sim_probability=0.5,
            seed=3,
            episodes=[4, 5],
        )


def test_paired_image_source_preserves_relative_rollout_subdirectories(tmp_path):
    rollout = tmp_path / "nested/rollout"
    media = rollout / "paired_media"
    media.mkdir(parents=True)
    for name in ("sim_base.mp4", "sim_wrist.mp4"):
        (media / name).write_bytes(b"video")
    (media / "frame_correspondence.csv").write_text(
        "sim_step,source_frame_index,source_time_s\n0,0,0.0\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "relative.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "episode_index": 0,
                "success": True,
                "rollout_dir": "nested/rollout",
                "sim_base_video": "paired_media/sim_base.mp4",
                "sim_wrist_video": "paired_media/sim_wrist.mp4",
                "sim_video_fps": 10.0,
                "frame_correspondence": "paired_media/frame_correspondence.csv",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    source = PairedImageSource(
        mode="sim",
        manifest=manifest,
        root=tmp_path,
        mixed_sim_probability=0.5,
        seed=3,
        episodes=[0],
    )

    path, _ = source.resolve(0, "observation.images.base_0_rgb", [0.0])
    assert path == tmp_path / "nested/rollout/paired_media/sim_base.mp4"


def test_dataset_config_requires_pairing_for_sim_images():
    with pytest.raises(ValueError, match="sim_image_manifest"):
        DatasetConfig(repo_id="local/test", image_source="sim")
