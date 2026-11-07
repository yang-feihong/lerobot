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


def test_direct_resume_cannot_silently_drop_instruction_domain_override(tmp_path, monkeypatch):
    config = tmp_path / "pretrained_model/train_config.json"
    config.parent.mkdir()
    config.write_text(json.dumps({"dataset": {
        "semantic_instruction_domain_weights": {"full_episode_command": {"simulation": 1}}
    }}))
    monkeypatch.setattr(
        "lerobot.configs.train.parser.parse_arg", lambda key: str(config) if key == "config_path" else None
    )
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="test", semantic_domain_weights={}), resume=True)
    with pytest.raises(ValueError, match="Resume instruction domain sampling"):
        cfg._resolve_resume_checkpoint()


@pytest.mark.parametrize('saved,expected', [({}, {}),
    ({'semantic_domain_weights': None}, None),
    ({'semantic_domain_weights': {'real': 1}}, {'real': 1})])
def test_checkpoint_loading_preserves_historical_and_new_domain_policies(tmp_path, saved, expected):
    path = tmp_path / 'train_config.json'
    path.write_text(json.dumps({'dataset': {'repo_id': 'test', 'semantic_views_path': 'unused.json',
        'semantic_phases': ['approach'],
        'semantic_state_instruction_matrix': {'approach': {'goal_approach': {
            'status': 'active', 'action_supervision': 'demonstrated',
            'completion_boundary': 'goal_end'}}}, **saved}}))
    config = TrainPipelineConfig.from_pretrained(path)
    assert config.dataset.semantic_domain_weights == expected


def test_new_dataset_config_defaults_to_automatic_domain_policy():
    assert DatasetConfig(repo_id='test').semantic_domain_weights is None
