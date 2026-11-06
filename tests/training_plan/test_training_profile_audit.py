import copy
import json
from pathlib import Path

import pytest

from lerobot.common.training_profile import compare_profile

PROFILE = json.loads((Path(__file__).parents[2] / "training_plan/training_profile.json").read_text())


def values():
    training = copy.deepcopy(PROFILE["training"])
    training["dataset"] = copy.deepcopy(PROFILE["dataset"])
    policy = copy.deepcopy(PROFILE["policy"])
    return training, policy


def test_complete_reference_contract_passes_and_ignores_only_declared_data_fields():
    training, policy = values()
    training["steps"] = 50000
    training["dataset"]["root"] = "/a/different/dataset"
    policy["semantic_view_manifest_sha256"] = "new-sidecar"
    policy["semantic_state_instruction_matrix"] = {"object_grasp": {}}
    policy["action_bool_true_fractions"] = {"task_complete": 0.2}
    assert compare_profile(PROFILE, training, policy, PROFILE["adapter_schema"])["ok"]


@pytest.mark.parametrize("frame", ["legacy", "physical"])
def test_model_tool_frame_is_a_recorded_run_choice_not_architecture_drift(frame):
    training, policy = values()
    policy["model_ee_frame"] = frame
    report = compare_profile(PROFILE, training, policy, PROFILE["adapter_schema"])
    assert report["ok"]
    assert report["model_ee_frame"] == frame


@pytest.mark.parametrize(
    "field,value",
    [
        ("state_action_encoding", "text"),
        ("mem_vit_enabled", False),
        ("state_num_frames", 1),
        ("action_predict_task_complete", False),
        ("dataset_camera_keys", ["observation.images.left_wrist_0_rgb", "observation.images.base_0_rgb"]),
        ("b2_action_representation", "velocity"),
    ],
)
def test_any_model_input_or_output_drift_fails(field, value):
    training, policy = values()
    policy[field] = value
    report = compare_profile(PROFILE, training, policy, PROFILE["adapter_schema"])
    assert not report["ok"]
    assert field in report["differences"]["policy"]


def test_unknown_model_field_optimizer_sampling_and_real_adapter_schema_are_checked():
    training, policy = values()
    policy["surprise_future_default"] = 1
    training["optimizer"]["lr"] *= 2
    training["dataset"]["semantic_exclude_incomplete_static_starts"] = False
    report = compare_profile(PROFILE, training, policy, {"tensor_count": 1, "sha256": "wrong"})
    assert all(
        report["differences"][section] for section in ("policy", "training", "dataset", "adapter_schema")
    )


def test_explicit_action_history_extension_preserves_base_profile_checks():
    training, policy = values()
    policy.update(action_history_enabled=True, action_history_feature_names=["b2_vx"], max_action_history_dim=8)
    assert compare_profile(PROFILE, training, policy, PROFILE["adapter_schema"])["ok"]
    policy["mem_vit_enabled"] = False
    assert not compare_profile(PROFILE, training, policy, PROFILE["adapter_schema"])["ok"]
    policy["action_history_feature_names"] = ["task_complete"]
    with pytest.raises(ValueError):
        compare_profile(PROFILE, training, policy, PROFILE["adapter_schema"])


def test_new_default_history_fields_do_not_change_disabled_profile():
    training, policy = values()
    policy.update(action_history_feature_names=None, max_action_history_dim=32)
    assert compare_profile(PROFILE, training, policy, PROFILE["adapter_schema"])["ok"]


def test_actual_history_projection_shapes_are_audited(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace

    import torch

    from lerobot.common import training_profile

    profile = copy.deepcopy(PROFILE)
    shapes = {"base": [1]}
    profile["adapter_schema"] = {"tensor_count": 1, "sha256": hashlib.sha256(
        json.dumps(shapes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    training, policy = values()
    policy.update(action_history_enabled=True, action_history_feature_names=["b2_vx"],
                  max_action_history_dim=8, action_predict_reset_preset=False)
    actual_policy = SimpleNamespace(config=SimpleNamespace(**policy),
        model=SimpleNamespace(state_memory_proj=torch.nn.Linear(32, 16)))
    tensors = {"base": torch.ones(1), "model.action_memory_proj.weight": torch.zeros(16, 8),
               "model.action_memory_proj.bias": torch.zeros(16)}
    monkeypatch.setattr(training_profile, "get_peft_model_state_dict", lambda *a, **k: tensors)
    monkeypatch.setattr(training_profile.draccus, "encode", lambda _: policy)
    cfg = SimpleNamespace(resume=True, to_dict=lambda: training)
    assert training_profile.audit_training_profile(cfg, actual_policy, path, 4)["ok"]
    tensors["model.action_memory_proj.weight"] = torch.zeros(16, 9)
    with pytest.raises(ValueError, match="history adapter schema"):
        training_profile.audit_training_profile(cfg, actual_policy, path, 4)


def test_gpu_count_may_change_but_not_effective_batch():
    training, policy = values()
    training["gradient_accumulation_steps"] = 12
    assert compare_profile(PROFILE, training, policy, PROFILE["adapter_schema"], 2)["ok"]
    assert not compare_profile(PROFILE, training, policy, PROFILE["adapter_schema"], 4)["ok"]


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"])
def test_ddp_local_cuda_index_is_not_a_model_difference(device):
    training, policy = values()
    policy["device"] = device
    assert compare_profile(PROFILE, training, policy, PROFILE["adapter_schema"])["ok"]


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda:invalid"])
def test_device_backend_drift_is_still_rejected(device):
    training, policy = values()
    policy["device"] = device
    report = compare_profile(PROFILE, training, policy, PROFILE["adapter_schema"])
    assert not report["ok"]
    assert "device" in report["differences"]["policy"]


def test_reviewed_selector_extension_does_not_relax_the_base_contract():
    from lerobot.utils.reset_presets import default_reset_presets

    training, policy = values()
    policy.update(
        action_predict_reset_preset=True,
        reset_preset_catalog=default_reset_presets(),
        reset_preset_loss_weight=1.0,
        reset_preset_manifest_sha256="audited-data",
    )
    assert compare_profile(PROFILE, training, policy, PROFILE["adapter_schema"])["ok"]
    policy["max_action_dim"] += 1
    assert not compare_profile(PROFILE, training, policy, PROFILE["adapter_schema"])["ok"]
    policy["reset_preset_manifest_sha256"] = None
    with pytest.raises(ValueError, match="provenance"):
        compare_profile(PROFILE, training, policy, PROFILE["adapter_schema"])
