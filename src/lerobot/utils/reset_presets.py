"""Versioned reset targets. IDs are names, never ordinal regression targets."""

import math
from pathlib import Path

PRESET_ID_KEY = "action_reset_preset_id"
PRESET_VALID_KEY = "action_reset_preset_valid"
PRESET_EXECUTION_KEY = "arm_reset_preset_id"
PRESET_SIDECAR = "reset_presets.json"


def default_reset_presets():
    return [
        {"name": "front", "joint_target": [-1.57, 2.6, -1.5, 0.4, 0.0, 0.0]},
        {"name": "middle", "joint_target": [-1.57, 1.75, -0.785, -0.93, 0.0, 0.0]},
        {"name": "back", "joint_target": [1.57, 1.57, -1.3, 1.1, 0.0, 0.0]},
    ]


def validate_reset_presets(catalog):
    if not catalog:
        raise ValueError("Reset preset catalog must not be empty")
    names = []
    targets = []
    for entry in catalog:
        name, target = entry["name"], entry["joint_target"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Reset preset names must be unique nonempty strings")
        if len(target) != 6 or not all(math.isfinite(v) for v in target):
            raise ValueError(f"Reset preset {name!r} requires six finite joint angles")
        if list(target) in targets:
            raise ValueError("Identical reset targets cannot have different class labels")
        names.append(name)
        targets.append(list(target))
    return names


def validate_selector_checkpoint(path, enabled):
    """PEFT otherwise tolerates missing modules_to_save, leaving random selectors."""
    from safetensors import safe_open

    path = Path(path)
    weights = path / "adapter_model.safetensors"
    if not weights.is_file():
        if enabled:
            raise ValueError("Reset selector requires auditable safetensors adapter weights")
        return
    with safe_open(weights, framework="pt", device="cpu") as stream:
        selected = [key for key in stream.keys() if ".reset_preset_head." in key]  # noqa: SIM118
        shapes = {
            key.split(".reset_preset_head.", 1)[1]: list(stream.get_slice(key).get_shape())
            for key in selected
        }
    query = shapes.get("query.weight", [])
    width = query[1] if len(query) == 2 and query[1] > 0 else 0
    expected = (
        {
            "query.weight": [64, width],
            "query.bias": [64],
            "candidate.0.weight": [64, 12],
            "candidate.0.bias": [64],
            "candidate.2.weight": [64, 64],
            "candidate.2.bias": [64],
        }
        if enabled
        else {}
    )
    if shapes != expected or len(selected) != len(expected):
        raise ValueError(f"Checkpoint reset selector weights/config mismatch: {shapes}")
