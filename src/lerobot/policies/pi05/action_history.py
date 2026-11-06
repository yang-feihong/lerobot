"""Executed-command input schema, independent of the predicted action schema."""

from .b2_action_transform import CONTROL_EXTENDED_DATASET_ACTION_NAMES, DATASET_ACTION_NAMES

RESET_TARGET_NAMES = tuple(f"arm_reset_joint_target_{i}" for i in range(1, 7))
RESET_HISTORY_KEY = "observation.reset_joint_history"
DEFAULT_ACTION_HISTORY_NAMES = (*DATASET_ACTION_NAMES[:-1], *RESET_TARGET_NAMES)


def action_history_schema(config):
    """Return command names, source indices (-1 for sidecar targets), source width.

    Completion/blocked are supervision, not executed commands. EE history uses
    control targets, never the legacy feedback-derived action columns.
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
    stored = tuple("control_" + name if name.startswith("height_invariant_ee_") else name for name in names)
    missing = set(stored) - set(source) - set(RESET_TARGET_NAMES)
    if missing:
        raise ValueError(f"Missing executed-command history fields: {sorted(missing)}")
    return names, tuple(-1 if name in RESET_TARGET_NAMES else source.index(name) for name in stored), len(source)
