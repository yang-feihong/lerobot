"""Initialize current PEFT training from model tensors, never from a saved train config."""

import hashlib
import json
from pathlib import Path

import draccus
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file


def initialize_model_weights(policy, checkpoint, *, resume=False):
    if resume:
        raise ValueError("Weights-only initialization and full training-state resume are mutually exclusive")
    checkpoint = Path(checkpoint)
    root = checkpoint / "pretrained_model"
    source = json.loads((root / "config.json").read_text())
    current = draccus.encode(policy.config)
    fields = (
        "action_feature_names",
        "dataset_camera_keys",
        "resolved_state_feature_names",
        "state_action_encoding",
        "b2_action_representation",
        "z1_action_representation",
        "max_action_dim",
        "max_state_dim",
        "mem_vit_finetune_mode",
        "policy_base_artifact_id",
        "policy_base_sha256",
        "mem_vit_artifact_id",
        "mem_vit_sha256",
        "policy_tokenizer_artifact_id",
        "policy_tokenizer_sha256",
    )
    mismatches = [key for key in fields if source.get(key) != current.get(key)]
    if source.get("ee_delta_reference", "current_state") != current.get("ee_delta_reference", "current_state"):
        mismatches.append("ee_delta_reference")
    source_history = source.get("action_history_enabled", False)
    target_history = current.get("action_history_enabled", False)
    if source_history and not target_history:
        mismatches.append("action_history_enabled")
    if source_history and target_history:
        from types import SimpleNamespace

        from lerobot.policies.pi05.action_history import action_history_schema

        def history_identity(config):
            capacity = config.get("max_action_history_dim", 32)
            names = action_history_schema(SimpleNamespace(
                action_history_feature_names=config.get("action_history_feature_names"),
                dataset_action_feature_names=config.get("dataset_action_feature_names"),
                max_action_history_dim=capacity,
                ee_supervision_source=config.get("ee_supervision_source", "control_action"),
            ))[0]
            return (names, capacity, config.get("action_history_ee_semantics") or "legacy_control_v0",
                    config.get("ee_supervision_source", "control_action"))

        if history_identity(source) != history_identity(current):
            mismatches.append("action_history_schema")
    if mismatches:
        raise ValueError(f"Model initialization I/O or base artifact identity differs: {mismatches}")
    path = root / "adapter_model.safetensors"
    state = load_file(path, device="cpu")
    expected = get_peft_model_state_dict(policy)
    missing, unexpected = set(expected) - set(state), set(state) - set(expected)
    selector_added = current["action_predict_reset_preset"] and not source.get(
        "action_predict_reset_preset", False
    )
    allowed_missing = {key for key in expected if ".reset_preset_head." in key} if selector_added else set()
    if target_history and not source_history:
        history_added = {key for key in expected if ".action_memory_proj." in key}
        if len(history_added) != 2:
            raise ValueError("New action history requires a saved projection weight and bias")
        allowed_missing |= history_added
    wrong_shapes = [key for key in state.keys() & expected.keys() if state[key].shape != expected[key].shape]
    if missing != allowed_missing or unexpected or wrong_shapes:
        raise ValueError(
            f"Model initialization tensor mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}, shapes={wrong_shapes}"
        )
    combined = {**state, **{key: expected[key] for key in allowed_missing}}
    set_peft_model_state_dict(policy, combined, adapter_name="default")
    # Verify actual tensors, including modules_to_save, not just the loader return status.
    loaded = get_peft_model_state_dict(policy)
    for key, value in state.items():
        actual = loaded[key].detach().to(device="cpu", dtype=value.dtype)
        if not actual.equal(value):
            raise ValueError(f"Model initialization did not restore tensor {key}")
    source_step = int(json.loads((checkpoint / "training_state/training_step.json").read_text())["step"])
    if source_step < 0:
        raise ValueError("Negative source model step")
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "mode": "model_weights_only",
        "source_checkpoint": str(checkpoint),
        "source_step": source_step,
        "source_adapter_sha256": sha,
        "restored_tensor_count": len(state),
        "new_parameter_keys": sorted(missing),
        "training_config_reused": False,
        "optimizer_reused": False,
        "scheduler_reused": False,
        "rng_reused": False,
        "wandb_run_reused": False,
    }
