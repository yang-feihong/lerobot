from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig


@pytest.mark.parametrize("change", [None, "enable_selector", "action", "observation"])
def test_old_resume_accepts_only_missing_disabled_selector(tmp_path, monkeypatch, change):
    root = tmp_path / "150000/pretrained_model"
    root.mkdir(parents=True)
    config_path = root / "train_config.json"
    config_path.write_text(json.dumps({"dataset": {}}))
    saved = {"action": {"model_names": ["arm_reset"]}, "observation": {"cameras": ["base", "wrist"]}}
    metadata_path = root / "pi05_deployment_metadata.json"
    metadata_path.write_text(json.dumps(saved))
    current = deepcopy(saved)
    current["action"]["reset_preset_selector"] = {
        "enabled": change == "enable_selector", "representation": "candidate_classification_v1",
        "catalog": [], "legacy_default": "middle",
    }
    if change == "action":
        current["action"]["model_names"].append("another_action")
    elif change == "observation":
        current["observation"]["cameras"].reverse()
    policy = SimpleNamespace(
        model_ee_frame=None,
        io_schema_resolved=True, action_predict_reset_preset=change == "enable_selector",
        deployment_metadata=lambda: deepcopy(current), restore_camera_metadata=lambda metadata: None,
    )
    monkeypatch.setattr("lerobot.configs.train.parser.parse_arg",
                        lambda key: str(config_path) if key == "config_path" else None)
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="test"), policy=policy, resume=True)
    if change is None:
        cfg._resolve_resume_checkpoint()
        assert cfg.checkpoint_path == root.parent
    else:
        with pytest.raises(ValueError, match="deployment metadata disagrees"):
            cfg._resolve_resume_checkpoint()
    assert json.loads(metadata_path.read_text()) == saved
