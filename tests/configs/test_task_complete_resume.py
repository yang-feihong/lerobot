from copy import deepcopy

from lerobot.configs.train import (
    is_training_loss_metadata_update,
    is_ee_supervision_metadata_update,
    is_task_complete_deployment_metadata_extension,
    is_task_status_deployment_metadata_extension,
)


def test_source_update_keeps_tensor_layout_and_anchor_fixed():
    saved = {'action': {'ee_supervision_source': 'control_action', 'model_names': ['ee_x'], 'ee_deployment_anchor': 'actual'},
             'memory': {'action_history': {'enabled': False}}}
    current = deepcopy(saved)
    current['action']['ee_supervision_source'] = 'recorded_action'
    assert is_ee_supervision_metadata_update(saved, current)
    assert saved['action']['ee_supervision_source'] == 'control_action'
    for key, value in [('model_names', ['different']), ('ee_deployment_anchor', 'command')]:
        invalid = deepcopy(current)
        invalid['action'][key] = value
        assert not is_ee_supervision_metadata_update(saved, invalid)
    current['action']['ee_supervision_source'] = 'typo'
    assert not is_ee_supervision_metadata_update(saved, current)


def test_source_update_changes_history_provenance_but_not_history_layout():
    saved = {'action': {'ee_supervision_source': 'control_action'}, 'memory': {'action_history': {
        'enabled': True, 'source_representation': 'dataset_executed_command', 'feature_names': ['ee_x']}}}
    current = deepcopy(saved)
    current['action']['ee_supervision_source'] = 'recorded_action'
    assert not is_ee_supervision_metadata_update(saved, current)
    current['memory']['action_history']['source_representation'] = 'dataset_recorded_command'
    assert is_ee_supervision_metadata_update(saved, current)
    current['memory']['action_history']['feature_names'] += ['extra']
    assert not is_ee_supervision_metadata_update(saved, current)


def test_config_requires_new_run_and_both_updates_for_source_switch(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import pytest
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig

    root = tmp_path / '000500/pretrained_model'
    root.mkdir(parents=True)
    config = root / 'train_config.json'
    config.write_text(json.dumps({'dataset': {'semantic_domain_weights': {}}}))
    saved = {'action': {'ee_supervision_source': 'control_action'}}
    (root / 'pi05_deployment_metadata.json').write_text(json.dumps(saved))
    current = {'action': {'ee_supervision_source': 'recorded_action', 'reset_preset_selector': {'enabled': False}}}
    monkeypatch.setattr('lerobot.configs.train.parser.parse_arg', lambda key: str(config) if key == 'config_path' else None)
    for data, objective, same_run in [(False, True, False), (True, False, False), (True, True, True), (True, True, False)]:
        policy = SimpleNamespace(model_ee_frame=None, io_schema_resolved=True, action_predict_reset_preset=False,
                                 deployment_metadata=lambda: deepcopy(current), restore_camera_metadata=lambda m: None)
        cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id='test', semantic_domain_weights={}), policy=policy,
                                  resume=True, resume_with_updated_dataset=data,
                                  resume_with_updated_training_objective=objective,
                                  wandb=WandBConfig(resume_training_run=same_run))
        if data and objective and not same_run:
            cfg._resolve_resume_checkpoint()
            assert cfg.checkpoint_path == root.parent
        else:
            with pytest.raises(ValueError, match='deployment metadata disagrees'):
                cfg._resolve_resume_checkpoint()


def test_training_loss_metadata_update_only_allows_flow_loss_schema() -> None:
    saved = {
        "policy": {"type": "pi05"},
        "action": {"flow_loss_schema": "group_balanced", "model_names": ["b2_vx"]},
    }
    current = deepcopy(saved)
    current["action"]["flow_loss_schema"] = "group_balanced_v2"
    assert is_training_loss_metadata_update(saved, current)

    changed_io = deepcopy(current)
    changed_io["action"]["model_names"].append("task_complete")
    assert not is_training_loss_metadata_update(saved, changed_io)


def test_task_complete_metadata_extension_only_allows_appended_completion() -> None:
    saved = {
        "policy": {"type": "pi05"},
        "action": {
            "model_names": ["b2_vx", "gripper_target"],
            "predict": {"gripper": True, "task_complete": False},
            "task_complete_deployment_behavior": None,
            "task_complete_semantics": None,
        },
    }
    extended = deepcopy(saved)
    extended["action"]["model_names"].append("task_complete")
    extended["action"]["predict"]["task_complete"] = True
    extended["action"]["task_complete_deployment_behavior"] = (
        "stop_before_executing_later_chunk_elements_at_first_true"
    )
    extended["action"]["task_complete_semantics"] = "explicit_true_in_the_terminal_stage_hold"

    assert is_task_complete_deployment_metadata_extension(saved, extended)

    unresolved = deepcopy(extended)
    unresolved["action"]["model_names"] = list(saved["action"]["model_names"])
    assert is_task_complete_deployment_metadata_extension(saved, unresolved)

    reordered = deepcopy(extended)
    reordered["action"]["model_names"] = ["task_complete", "b2_vx", "gripper_target"]
    assert not is_task_complete_deployment_metadata_extension(saved, reordered)

    changed_policy = deepcopy(extended)
    changed_policy["policy"]["type"] = "different"
    assert not is_task_complete_deployment_metadata_extension(saved, changed_policy)


def test_task_status_metadata_extension_appends_both_flow_channels() -> None:
    saved = {
        "policy": {"type": "pi05"},
        "action": {
            "model_names": ["b2_vx"],
            "predict": {"task_complete": False, "task_blocked": False},
            "task_complete_deployment_behavior": None,
            "task_complete_semantics": None,
            "task_blocked_semantics": None,
            "boolean_decoding": {
                "true_side": {"task_complete": "positive"},
                "output_values": {"task_complete": {"false": 0.0, "true": 1.0}},
            },
        },
    }
    current = deepcopy(saved)
    action = current["action"]
    action["model_names"] += ["task_complete", "task_blocked"]
    action["predict"].update(task_complete=True, task_blocked=True)
    action["task_complete_deployment_behavior"] = (
        "stop_before_executing_later_chunk_elements_at_first_true"
    )
    action["task_complete_semantics"] = "explicit_true_in_the_terminal_stage_hold"
    action["task_blocked_semantics"] = (
        "instruction_precondition_is_not_satisfied_in_the_current_state"
    )
    action["boolean_decoding"]["true_side"]["task_blocked"] = "positive"
    action["boolean_decoding"]["output_values"]["task_blocked"] = {
        "false": 0.0,
        "true": 1.0,
    }

    assert is_task_status_deployment_metadata_extension(saved, current)

    reordered = deepcopy(current)
    reordered["action"]["model_names"][-2:] = ["task_blocked", "task_complete"]
    assert not is_task_status_deployment_metadata_extension(saved, reordered)
