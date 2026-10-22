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
