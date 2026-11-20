"""Executed-command input schema, independent of the predicted action schema."""

import torch

from .b2_action_transform import CONTROL_EXTENDED_DATASET_ACTION_NAMES, DATASET_ACTION_NAMES

RESET_TARGET_NAMES = tuple(f"arm_reset_joint_target_{i}" for i in range(1, 7))
RESET_HISTORY_KEY = "observation.reset_joint_history"
DEFAULT_ACTION_HISTORY_NAMES = (*DATASET_ACTION_NAMES[:-1], *RESET_TARGET_NAMES)


def command_reference_anchor(current_state, command_pose, valid):
    """Use the last visible historical EE command, otherwise the measured pose.

    Masked commands cannot supply an anchor; selection stays inside the visible window.
    All arguments are in the same unnormalized model tool frame.
    """
    if (current_state.shape[-1] != 9 or command_pose.ndim != current_state.ndim + 1
            or command_pose.shape[:-2] != current_state.shape[:-1]
            or command_pose.shape[-1] != 9 or command_pose.shape[-2] < 1):
        raise ValueError("EE command reference requires a nonempty history of matching 9D poses")
    if valid.shape != command_pose.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError("EE command reference validity shape mismatch")
    slots = torch.arange(command_pose.shape[-2], device=valid.device)
    latest = torch.where(valid, slots, -1).amax(dim=-1)
    indices = latest.clamp(min=0)[..., None, None].expand(*latest.shape, 1, 9)
    selected = command_pose.gather(-2, indices).squeeze(-2)
    return torch.where((latest >= 0)[..., None], selected, current_state)


def action_history_schema(config):
    """Return command names, source indices (-1 for sidecar targets), source width.

    Completion/blocked are supervision, not commands. EE history follows the
    same stored source as future supervision.
    """
    requested = getattr(config, "action_history_feature_names", None)
    names = DEFAULT_ACTION_HISTORY_NAMES if requested is None else tuple(requested)
    if not names or len(names) != len(set(names)):
        raise ValueError("Action history needs nonempty, unique command names")
    if set(names) & {"task_complete", "task_blocked"}:
        raise ValueError("Action history cannot contain generated task supervision")
    if len(names) > config.max_action_history_dim:
        raise ValueError("Action history exceeds its projection capacity")
    source = tuple(config.dataset_action_feature_names or CONTROL_EXTENDED_DATASET_ACTION_NAMES)
    if len(source) != len(set(source)):
        raise ValueError("Dataset action names must be unique")
    ee_source = config.ee_supervision_source
    if ee_source not in {"control_action", "recorded_action"}:
        raise ValueError(f"Unsupported supervision source {ee_source!r}")
    stored = tuple("control_" + name if ee_source == "control_action"
                   and name.startswith("height_invariant_ee_") else name for name in names)
    missing = set(stored) - set(source) - set(RESET_TARGET_NAMES)
    if missing:
        raise ValueError(f"Missing executed-command history fields: {sorted(missing)}")
    return names, tuple(-1 if name in RESET_TARGET_NAMES else source.index(name) for name in stored), len(source)
