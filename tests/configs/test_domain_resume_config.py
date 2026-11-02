import json

import pytest

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig


@pytest.mark.parametrize("fork", [False, True])
def test_direct_resume_cannot_silently_change_domain_policy(tmp_path, monkeypatch, fork):
    config = tmp_path / "pretrained_model/train_config.json"
    config.parent.mkdir()
    config.write_text(json.dumps({"dataset": {"semantic_domain_weights": {"simulation": 3, "real": 1}}}))
    monkeypatch.setattr(
        "lerobot.configs.train.parser.parse_arg", lambda key: str(config) if key == "config_path" else None
    )
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test"), resume=True, resume_with_updated_dataset=fork
    )
    if fork:
        cfg._resolve_resume_checkpoint()
    else:
        with pytest.raises(ValueError, match="Resume domain sampling"):
            cfg._resolve_resume_checkpoint()
