import hashlib
import json

import pytest
import torch

from lerobot.datasets.reset_presets import ResetPresetCatalog
from lerobot.utils.reset_presets import (
    DEFAULT_RESET_UNKNOWN_TARGET_POLICY,
    PRESET_ID_KEY,
    PRESET_VALID_KEY,
    default_reset_presets,
)


def test_training_config_defaults_to_masking_only_audited_unknown_targets():
    from lerobot.configs.train import TrainPipelineConfig

    assert DEFAULT_RESET_UNKNOWN_TARGET_POLICY == "mask_audited_unknown"
    assert (TrainPipelineConfig.__dataclass_fields__["reset_preset_unknown_target_policy"].default
            == DEFAULT_RESET_UNKNOWN_TARGET_POLICY)


def prepare(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "diagnostics").mkdir()
    manifest = tmp_path / "diagnostics/conversion_manifest.json"
    manifest.write_text("{}")
    info = tmp_path / "meta/info.json"
    info.write_text("{}")
    data = tmp_path / "data.parquet"
    data.write_bytes(b"test data identity")
    payload = {
        "format": "lerobot.reset_presets",
        "version": 1,
        "catalog": default_reset_presets(),
        "conversion_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "info_sha256": hashlib.sha256(info.read_bytes()).hexdigest(),
        "episodes": {"7": [{"frame_start": 12, "frame_stop": 15, "preset": "front"}]},
        "provenance": {
            "7": {
                "data_path": "data.parquet",
                "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                "num_frames": 30,
            }
        },
    }
    path = tmp_path / "meta/reset_presets.json"
    path.write_text(json.dumps(payload))
    return path


def test_frame_alignment_padding_and_semantic_hold(tmp_path):
    path = prepare(tmp_path)
    catalog = ResetPresetCatalog(path, default_reset_presets())
    # 25 Hz dataset, 50 Hz action chunk: repeated source frames are expected.
    batch = {
        "action": torch.ones(1, 6, 1),
        "episode_index": torch.tensor([7]),
        "frame_index": torch.tensor([12]),
        "action_is_pad": torch.tensor([[False, False, False, False, False, True]]),
    }
    batch["action"][0, 0] = -1  # synthetic hold clears reset even inside its recorded range
    catalog.attach(batch, action_names=["arm_reset"], fps=25, action_dt=0.02)
    assert batch[PRESET_ID_KEY].tolist() == [[0, 0, 0, 0, 0, 0]]
    assert batch[PRESET_VALID_KEY].tolist() == [[False, True, True, True, True, False]]


def test_unknown_effective_reset_never_defaults_to_middle(tmp_path):
    catalog = ResetPresetCatalog(prepare(tmp_path), default_reset_presets())
    batch = {
        "action": torch.ones(1, 2, 1),
        "episode_index": torch.tensor([7]),
        "frame_index": torch.tensor([0]),
    }
    with pytest.raises(ValueError, match="no proven"):
        catalog.attach(batch, action_names=["arm_reset"], fps=50, action_dt=0.02)
    batch["action"] *= -1  # cropped/reset-free effective grasp needs no target
    catalog.attach(batch, action_names=["arm_reset"], fps=50, action_dt=0.02)
    assert not batch[PRESET_VALID_KEY].any()


def test_stale_sidecar_or_catalog_rejected(tmp_path):
    path = prepare(tmp_path)
    with pytest.raises(ValueError, match="differs"):
        ResetPresetCatalog(path, default_reset_presets()[::-1])
    (tmp_path / "data.parquet").write_bytes(b"updated actions")
    with pytest.raises(ValueError, match="Stale"):
        ResetPresetCatalog(path, default_reset_presets())


def test_frame_identity_survives_the_training_preprocessor_boundary():
    from lerobot.processor.converters import batch_to_transition, transition_to_batch

    batch = {
        "action": torch.zeros(1, 3, 1),
        "episode_index": torch.tensor([7]),
        "frame_index": torch.tensor([12]),
    }
    restored = transition_to_batch(batch_to_transition(batch))
    assert restored["frame_index"].tolist() == [12]
    assert restored["episode_index"].tolist() == [7]


def test_only_audited_unknown_targets_can_mask_classification(tmp_path):
    path = prepare(tmp_path)
    data = json.loads(path.read_text())
    data["unresolved"] = {"7": [{"frame_start": 15, "frame_stop": 17}]}
    path.write_text(json.dumps(data))
    batch = {
        "action": torch.ones(1, 5, 2),
        "episode_index": torch.tensor([7]), "frame_index": torch.tensor([12]),
        "action_is_pad": torch.zeros(1, 5, dtype=torch.bool),
    }
    actions, padding = batch["action"].clone(), batch["action_is_pad"].clone()
    with pytest.raises(ValueError, match="no proven"):
        ResetPresetCatalog(path, default_reset_presets(), unknown_target_policy="error").attach(
            batch, action_names=["arm_reset", "task_complete"], fps=50, action_dt=.02)
    catalog = ResetPresetCatalog(path, default_reset_presets())
    catalog.attach(batch, action_names=["arm_reset", "task_complete"], fps=50, action_dt=.02)
    assert batch[PRESET_ID_KEY].tolist() == [[0, 0, 0, -1, -1]]
    assert batch[PRESET_VALID_KEY].tolist() == [[True, True, True, False, False]]
    assert torch.equal(actions, batch["action"])
    assert torch.equal(padding, batch["action_is_pad"])
    batch["frame_index"][:] = 17
    with pytest.raises(ValueError, match="no proven"):
        catalog.attach(batch, action_names=["arm_reset", "task_complete"], fps=50, action_dt=.02)


def test_audited_unknown_cannot_overlap_known_target(tmp_path):
    path = prepare(tmp_path)
    data = json.loads(path.read_text())
    data["unresolved"] = {"7": [{"frame_start": 14, "frame_stop": 17}]}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Overlapping"):
        ResetPresetCatalog(path, default_reset_presets(), unknown_target_policy="mask_audited_unknown")
