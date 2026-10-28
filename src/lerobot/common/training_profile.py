"""Compare actual initialized models with the one reviewed mainline contract."""

import copy
import hashlib
import json
import re
from pathlib import Path

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
    "episodes",
    "repo_id",
    "root",
    "semantic_phase_weights",
    "semantic_phases",
    "semantic_state_instruction_matrix",
    "semantic_views_path",
}
RUN_FIELDS = {
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
    schema = {
        "tensor_count": len(shapes),
        "sha256": hashlib.sha256(
            json.dumps(shapes, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    report = compare_profile(profile, cfg.to_dict(), draccus.encode(policy.config), schema, num_processes)
    report["profile_sha256"] = hashlib.sha256(raw).hexdigest()
    report["reset_selector_adapter_shapes"] = selector_shapes
    if not report["ok"]:
        raise ValueError("Mainline training profile mismatch: " + json.dumps(report["differences"]))
    return report
