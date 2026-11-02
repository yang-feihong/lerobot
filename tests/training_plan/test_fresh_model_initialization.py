import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import save_file

from lerobot.common.model_initialization import initialize_model_weights
from lerobot.policies.pi05.reset_preset_head import ResetPresetHead
from lerobot.utils.reset_presets import default_reset_presets

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("fresh_release", ROOT / "training_plan/dataset_release.py")
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def model(enabled):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = {
                "tie_word_embeddings": False,
                "action_predict_reset_preset": enabled,
                "action_feature_names": ["arm_reset"],
            }
            self.model = torch.nn.Module()
            self.model.proj = torch.nn.Linear(4, 4)
            if enabled:
                self.model.reset_preset_head = ResetPresetHead(4, default_reset_presets())

    return get_peft_model(
        Tiny(),
        LoraConfig(r=2, target_modules=["proj"], modules_to_save=["reset_preset_head"] if enabled else None),
    )


def checkpoint(tmp_path):
    old = model(False)
    root = tmp_path / "source"
    (root / "pretrained_model").mkdir(parents=True)
    (root / "training_state").mkdir()
    old.save_pretrained(root / "pretrained_model")
    (root / "pretrained_model/config.json").write_text(json.dumps(old.config))
    (root / "pretrained_model/train_config.json").write_text('{"must_not_be_reused":true}')
    (root / "training_state/training_step.json").write_text('{"step":100000}')
    return old, root


def test_only_weights_and_step_reused_new_selector_unchanged(tmp_path):
    old, root = checkpoint(tmp_path)
    current = model(True)
    before = {
        k: v.clone() for k, v in get_peft_model_state_dict(current).items() if ".reset_preset_head." in k
    }
    report = initialize_model_weights(current, root)
    assert report["source_step"] == 100000
    assert report["optimizer_reused"] is False
    assert report["training_config_reused"] is False
    for key, value in get_peft_model_state_dict(old).items():
        torch.testing.assert_close(get_peft_model_state_dict(current)[key], value, rtol=0, atol=0)
    for key, value in before.items():
        torch.testing.assert_close(get_peft_model_state_dict(current)[key], value, rtol=0, atol=0)
    assert set(report["new_parameter_keys"]) == set(before)


def test_missing_old_weight_or_io_drift_rejected(tmp_path):
    old, root = checkpoint(tmp_path)
    current = model(True)
    current.config["action_feature_names"] = ["wrong"]
    with pytest.raises(ValueError, match="I/O"):
        initialize_model_weights(current, root)
    current.config["action_feature_names"] = ["arm_reset"]
    state = get_peft_model_state_dict(old)
    state.pop(next(iter(state)))
    save_file(state, root / "pretrained_model/adapter_model.safetensors")
    with pytest.raises(ValueError, match="tensor mismatch"):
        initialize_model_weights(current, root)
    with pytest.raises(ValueError, match="mutually exclusive"):
        initialize_model_weights(current, root, resume=True)


def test_release_pin_is_independent_of_training_supervision_configuration(tmp_path):
    presets = json.loads((ROOT / "training_plan/semantic_training_presets.json").read_text())["presets"]
    saved = {"recipe": "staff1", **presets["three_goal_plus_full"]}
    requested = {"recipe": "staff1", **presets["three_stage_plus_full"]}
    inputs = {"request": saved}
    artifact_id = release.digest(inputs)
    root = tmp_path / "training" / f"staff1--{artifact_id}"
    (root / "meta").mkdir(parents=True)
    receipt = {
        "format": release.FORMAT,
        "status": "READY",
        "artifact_id": artifact_id,
        "inputs": inputs,
        "assets": release.asset_identity(root),
    }
    (root / release.RECEIPT).write_text(json.dumps(receipt))
    assert release.resolve_release(tmp_path, requested, artifact_id=artifact_id) == root
    assert release.resolve_release(tmp_path, requested, pinned_root=root)==root
    requested["state_instruction_matrix"]["approach"]["goal_approach"]["completion_boundary"] = "none"
    # Coverage/rule validation is the training resolver's responsibility, not
    # the physical identity lock's. A different recipe is still forbidden.
    assert release.resolve_release(tmp_path, requested, artifact_id=artifact_id)==root
    with pytest.raises(ValueError, match="differs"):
        release.resolve_release(tmp_path, {**requested,'recipe':'other'}, pinned_root=root)


def test_actual_tokenized_input_audit_rejects_semantic_keys_and_wrong_cells(tmp_path):
    from lerobot.common.semantic_input_audit import SemanticInputAudit
    from lerobot.datasets.semantic_views import SemanticSegment
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    matrix = {"approach": {"goal_approach": {}}}
    catalog = SimpleNamespace(segments={0: [SemanticSegment("approach", 0, 10)]})
    tokenizer = SimpleNamespace(decode=lambda ids, skip_special_tokens: "Task: Approach the door.; Action:")
    path = tmp_path / "audit.json"
    audit = SemanticInputAudit(catalog, matrix, tokenizer, path)
    batch = {
        "episode_index": torch.tensor([0]),
        "frame_index": torch.tensor([2]),
        "semantic_view_name": ["goal_approach/active"],
        "task": ["Task: Approach the door.;\nAction: "],
        OBS_LANGUAGE_TOKENS: torch.tensor([[1, 2]]),
        OBS_LANGUAGE_ATTENTION_MASK: torch.tensor([[1, 1]]),
    }
    audit.record(batch, ["Approach the door."], 100000)
    assert json.loads(path.read_text())["observed_cells"] == ["approach/goal_approach"]
    assert json.loads(path.read_text())["examples"]["approach/goal_approach"][0]["token_ids"] == [1, 2]
    full_mem_audit = SemanticInputAudit(catalog, matrix, tokenizer, path, continuous_state_memory=True)
    batch["task"] = ["Task: Approach the door., State: 1 2;\nAction: "]
    with pytest.raises(ValueError, match="discrete-state"):
        full_mem_audit.record(batch, ["Approach the door."], 100000)
    batch["task"] = ["Task: Approach the door.;\nAction: "]
    with pytest.raises(ValueError, match="semantic key"):
        audit.record(batch, ["goal_approach"], 100001)
    batch["semantic_view_name"] = ["goal_open/active"]
    with pytest.raises(ValueError, match="outside preset"):
        audit.record(batch, ["Open the door."], 100002)
