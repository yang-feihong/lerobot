import json

import numpy as np
import pytest
import torch

from lerobot.datasets.ee_frame_adapter import EpisodeEEFrameAdapter
from lerobot.utils.ee_frames import EE_FRAME_CONTRACT, convert_ee_pose


def fixture(tmp_path):
    suffixes = ("rot6d_col0_x", "rot6d_col0_y", "rot6d_col0_z", "rot6d_col1_x", "rot6d_col1_y", "rot6d_col1_z", "x", "y", "z")
    features = {key: {"names": [prefix + s for s in suffixes]} for key, prefix in (
        ("action", "control_height_invariant_ee_"), ("observation.state", "height_invariant_ee_state_"))}
    (tmp_path / "meta").mkdir()
    (tmp_path / "diagnostics").mkdir()
    (tmp_path / "meta/ee_frames.json").write_text(json.dumps({"version": 1, "ee_frame_contract": EE_FRAME_CONTRACT,
                                                            "recordings": {"old": "legacy", "new": "physical"}}))
    (tmp_path / "diagnostics/conversion_manifest.json").write_text(json.dumps({"bags": {
        "old": {"status": "done", "episode_index": 0}, "new": {"status": "done", "episode_index": 1}}}))
    return features


@pytest.mark.parametrize("frame", ["legacy", "physical"])
def test_mixed_samples_match_same_physical_pose(tmp_path, frame):
    features = fixture(tmp_path)
    adapter = EpisodeEEFrameAdapter(tmp_path, features, frame, {0, 1})
    legacy = torch.tensor([1., 0, 0, 0, 1, 0, .2, .3, .4])
    physical = convert_ee_pose(legacy, "legacy", "physical")
    for key in features:
        torch.testing.assert_close(adapter.feature(legacy, key, 0), adapter.feature(physical, key, 1))
        np.testing.assert_allclose(adapter.feature(legacy.numpy(), key, 0), adapter.feature(physical.numpy(), key, 1))


def test_missing_episode_metadata_rejected(tmp_path):
    features = fixture(tmp_path)
    with pytest.raises(ValueError):
        EpisodeEEFrameAdapter(tmp_path, features, "physical", {0, 1, 2})


def test_statistics_follow_converted_samples_without_rewriting_hf(tmp_path):
    from types import SimpleNamespace
    from datasets import Dataset
    from lerobot.datasets.compute_stats import get_feature_stats
    from lerobot.datasets.ee_frame_adapter import configure_dataset_ee_frame

    features = fixture(tmp_path)
    legacy = np.array([1., 0, 0, 0, 1, 0, .2, .3, .4], dtype=np.float32)
    physical = convert_ee_pose(legacy, "legacy", "physical")
    values = np.array([legacy] * 8 + [physical] * 8)
    data = Dataset.from_dict({"episode_index": [0] * 8 + [1] * 8,
                              "action": values.tolist(), "observation.state": values.tolist()})
    stats = {key: get_feature_stats(values, axis=0, keepdims=False) for key in features}
    dataset = SimpleNamespace(root=tmp_path, episodes=None, hf_dataset=data, reader=SimpleNamespace(),
                              meta=SimpleNamespace(features=features, stats=stats, total_episodes=2))
    configure_dataset_ee_frame(dataset, "physical")
    for key in features:
        np.testing.assert_allclose(dataset.meta.stats[key]["mean"][3:6], physical[3:6], atol=1e-6)
    np.testing.assert_array_equal(np.asarray(data["action"]), values)
