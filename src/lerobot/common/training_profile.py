"""Compare actual initialized models with the one reviewed mainline contract."""

import copy
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import draccus
from peft import get_peft_model_state_dict

from lerobot.utils.reset_presets import default_reset_presets, validate_reset_presets

RESET_SELECTOR_FIELDS = {
    "action_predict_reset_preset",
    "reset_preset_catalog",
    "reset_preset_loss_weight",
    "reset_preset_manifest_sha256",
}


def selector_contract(config):
    """An explicit, additive extension; all preexisting profile fields stay strict."""
    catalog = config.get("reset_preset_catalog", default_reset_presets())
    validate_reset_presets(catalog)
    enabled = config.get("action_predict_reset_preset", False)
    if type(enabled) is not bool:
        raise ValueError("Mainline reset selector requires a boolean enable")
    if enabled and (not config["action_predict_arm_reset"] or not config.get("reset_preset_manifest_sha256")):
        raise ValueError("Reset selector requires reset output and audited data provenance")
    return {
        "enabled": enabled,
        "representation": "candidate_classification_v1",
        "catalog": catalog,
        "loss_weight": config.get("reset_preset_loss_weight", 1.0),
    }


POLICY_DATA_FIELDS = {
    "action_bool_true_fractions",
    "semantic_state_instruction_matrix",
    "semantic_view_manifest_sha256",
}
DATASET_IDENTITY_FIELDS = {
    "semantic_instruction_domain_weights",
    "semantic_domain_weights",
    "episodes",
    "repo_id",
    "root",
    "semantic_phase_weights",
    "semantic_phases",
    "semantic_state_instruction_matrix",
    "semantic_views_path",
}
RUN_FIELDS = {
    "model_init_checkpoint",
    "reset_preset_sidecar",
    "reset_preset_unknown_target_policy",
    "dataset",
    "policy",
    "job_name",
    "output_dir",
    "checkpoint_path",
    "resume",
    "resume_with_updated_dataset",
    "resume_with_updated_training_objective",
    "steps",
    "wandb",
    "job",
    "gradient_accumulation_steps",
}


def canonical_policy(config: dict) -> dict:
    result = copy.deepcopy(config)
    # Optional input extension; absent/default fields keep old audits identical.
    result.pop("action_history_feature_names", None)
    result.pop("max_action_history_dim", None)
    history_semantics = result.pop("action_history_ee_semantics", None)
    if history_semantics not in {None, "active_only_v1", "legacy_control_v0", "mode_reset_target_v2"}:
        raise ValueError("Unknown action history EE semantics")
    pause_steps = result.pop("history_pause_max_steps", 0)
    if type(pause_steps) is not int or pause_steps < 0:
        raise ValueError("history_pause_max_steps must be a nonnegative integer")
    if pause_steps and not config.get("action_history_enabled", False):
        raise ValueError("history_pause_max_steps requires action history")
    # This is a resolved per-run representation choice, not an architecture
    # override. Checkpoint metadata and the resume gate lock its actual value.
    frame = result.pop("model_ee_frame", None)
    if frame is not None:
        from lerobot.utils.ee_frames import validate_ee_frame
        validate_ee_frame(frame)
    selector_contract(result)
    for key in RESET_SELECTOR_FIELDS:
        result.pop(key, None)
    # DDP assigns a different local CUDA index to each otherwise identical replica.
    if re.fullmatch(r"cuda(?::[0-9]+)?", result["device"]):
        result["device"] = "cuda"
    for key in POLICY_DATA_FIELDS:
        result.pop(key, None)
    for path, identity in (
        ("pretrained_path", "policy_base_artifact_id"),
        ("mem_vit_checkpoint", "mem_vit_artifact_id"),
    ):
        if result[identity]:
            result[path] = "artifact://" + result[identity]
    return result


def compare_profile(
    profile: dict, training: dict, policy: dict, adapter_schema: dict, num_processes=4
) -> dict:
    selector = selector_contract(policy)
    extension = profile["reset_preset_selector_extension"]
    if (
        selector["representation"] != extension["representation"]
        or selector["loss_weight"] != extension["loss_weight"]
    ):
        raise ValueError("Reset selector objective differs from the reviewed profile extension")
    actual = {
        "policy": canonical_policy(policy),
        "training": {k: v for k, v in training.items() if k not in RUN_FIELDS},
        "dataset": {k: v for k, v in training["dataset"].items() if k not in DATASET_IDENTITY_FIELDS},
        "adapter_schema": adapter_schema,
    }
    expected = {name: profile[name] for name in actual}
    expected["training"] = {k: v for k, v in expected["training"].items() if k not in RUN_FIELDS}
    expected["training"]["effective_batch_size"] = profile["global_batch_size"]
    actual["training"]["effective_batch_size"] = (
        training["batch_size"] * training["gradient_accumulation_steps"] * num_processes
    )
    expected["policy"] = canonical_policy(expected["policy"])
    if policy.get("action_history_enabled", False):
        from lerobot.policies.pi05.action_history import action_history_schema

        if type(policy["action_history_enabled"]) is not bool:
            raise ValueError("Action history enable must be boolean")
        action_history_schema(SimpleNamespace(
            action_history_feature_names=policy.get("action_history_feature_names"),
            dataset_action_feature_names=policy.get("dataset_action_feature_names"),
            max_action_history_dim=policy.get("max_action_history_dim", 32),
        ))
        expected["policy"]["action_history_enabled"] = True
    elif policy.get("action_history_feature_names") is not None:
        raise ValueError("Disabled action history cannot declare an input extension")
    differences = {
        section: {
            key: {"expected": expected[section].get(key), "actual": actual[section].get(key)}
            for key in expected[section].keys() | actual[section].keys()
            if expected[section].get(key) != actual[section].get(key)
            or (key in expected[section]) != (key in actual[section])
        }
        for section in actual
    }
    return {
        "reference_run": profile["reference_run"],
        "differences": differences,
        "ok": not any(differences.values()),
        "checked_fields": {section: len(values) for section, values in actual.items()},
        "dataset_specific_policy": {k: policy[k] for k in POLICY_DATA_FIELDS},
        "reset_selector": selector_contract(policy),
        "model_ee_frame": policy.get("model_ee_frame") or "legacy",
    }


def audit_training_profile(cfg, policy, path: Path, num_processes: int) -> dict:
    raw = path.read_bytes()
    profile = json.loads(raw)
    state = get_peft_model_state_dict(policy, adapter_name="default")
    shapes = {key: list(value.shape) for key, value in state.items()}
    selector_shapes = {key: shape for key, shape in shapes.items() if ".reset_preset_head." in key}
    if (
        not cfg.resume
        and policy.config.action_predict_reset_preset
        != profile["reset_preset_selector_extension"]["new_run_enabled"]
    ):
        raise ValueError(
            "New formal runs must enable the reviewed reset selector; legacy resume preserves its structure"
        )
    if policy.config.action_predict_reset_preset:
        expected = profile["reset_preset_selector_extension"]["adapter_shapes"]
        actual = {key.split(".reset_preset_head.", 1)[1]: shape for key, shape in selector_shapes.items()}
        if actual != expected or len(selector_shapes) != len(expected):
            raise ValueError(f"Reset selector adapter schema mismatch: {selector_shapes}")
    elif selector_shapes:
        raise ValueError("Disabled selector unexpectedly has saved weights")
    shapes = {key: shape for key, shape in shapes.items() if key not in selector_shapes}
    history_shapes = {key: shape for key, shape in shapes.items() if ".action_memory_proj." in key}
    if policy.config.action_history_enabled:
        model = policy.get_base_model().model if hasattr(policy, "get_base_model") else policy.model
        width = model.state_memory_proj.out_features
        expected_history = {"weight": [width, policy.config.max_action_history_dim], "bias": [width]}
        actual_history = {key.split(".action_memory_proj.", 1)[1]: shape
                          for key, shape in history_shapes.items()}
        if actual_history != expected_history or len(history_shapes) != 2:
            raise ValueError(f"Action history adapter schema mismatch: {history_shapes}")
    elif history_shapes:
        raise ValueError("Disabled action history unexpectedly has saved weights")
    shapes = {key: shape for key, shape in shapes.items() if key not in history_shapes}
    schema = {
        "tensor_count": len(shapes),
        "sha256": hashlib.sha256(
            json.dumps(shapes, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    report = compare_profile(profile, cfg.to_dict(), draccus.encode(policy.config), schema, num_processes)
    report["profile_sha256"] = hashlib.sha256(raw).hexdigest()
    report["reset_selector_adapter_shapes"] = selector_shapes
    if policy.config.action_history_enabled:
        report["action_history_adapter_shapes"] = history_shapes
    if not report["ok"]:
        raise ValueError("Mainline training profile mismatch: " + json.dumps(report["differences"]))
    return report
