"""Training-only ideal hold augmentation, before input normalization.

Future supervision and the current state/anchor are never changed. This is a
frozen-observation approximation, not a simulation of settling/contact dynamics.
"""

import torch

from lerobot.utils.constants import ACTION, OBS_STATE

from .action_history import RESET_HISTORY_KEY, action_history_schema
from .b2_action_transform import CONTROL_EXTENDED_DATASET_ACTION_NAMES


def augment_history_pause(batch, config):
    """Move recent history earlier by a per-example uniform integer in [0, max].

    One step is the state/history-command interval, not one dataset frame.
    Commands during the inserted hold have zero base velocity, inactive arm,
    no reset, and the last recorded EE/gripper targets. Images remain untouched.
    """
    maximum = config.history_pause_max_steps
    if maximum == 0:
        return batch
    state = batch[OBS_STATE]
    action = batch[ACTION]
    length = config.state_num_frames - 1
    if state.ndim != 3 or state.shape[1] != length + 1:
        raise ValueError("Pause augmentation requires the full batched state history")
    expected = length + config.chunk_size + int(config.z1_action_representation == "ee_delta")
    if action.ndim != 3 or action.shape[:2] != (state.shape[0], expected):
        raise ValueError("Pause augmentation requires raw history followed by future actions")

    names, columns, _ = action_history_schema(config)
    shifts = torch.randint(maximum + 1, (state.shape[0],), device=state.device)
    rows = torch.arange(state.shape[0], device=state.device)[:, None]
    slots = torch.arange(length + 1, device=state.device)[None]
    state_indices = (slots + shifts[:, None]).clamp(max=length)
    result = dict(batch)
    result[OBS_STATE] = state[rows, state_indices]
    state_pad = f"{OBS_STATE}_is_pad"
    if state_pad in batch:
        result[state_pad] = batch[state_pad][rows, state_indices]

    command_slots = slots[:, :-1]
    command_indices = (command_slots + shifts[:, None]).clamp(max=length - 1)
    history = action[:, :length][rows, command_indices].clone()
    inserted = command_slots + shifts[:, None] >= length
    for name, column in zip(names, columns, strict=True):
        if column < 0:
            continue
        if name in {"b2_vx", "b2_vy", "b2_omega_z", "arm_reset"}:
            history[..., column] = torch.where(inserted, 0.0, history[..., column])
        elif name == "arm_teleop_inactive":
            history[..., column] = torch.where(inserted, 1.0, history[..., column])
    source_names = config.dataset_action_feature_names or CONTROL_EXTENDED_DATASET_ACTION_NAMES
    # Modes are internal validity evidence even when absent from model features.
    for name, value in (("arm_teleop_inactive", 1.0), ("arm_reset", 0.0)):
        if name in source_names:
            column = source_names.index(name)
            history[..., column] = torch.where(inserted, value, history[..., column])
    result[ACTION] = torch.cat((history, action[:, length:]), dim=1)
    if RESET_HISTORY_KEY in batch:
        result[RESET_HISTORY_KEY] = batch[RESET_HISTORY_KEY][rows, command_indices]
        result[RESET_HISTORY_KEY + "_is_pad"] = batch[RESET_HISTORY_KEY + "_is_pad"][rows, command_indices] | inserted
    action_pad = f"{ACTION}_is_pad"
    if action_pad in batch:
        # Unknown terminal commands remain unknown; future padding is unchanged.
        result[action_pad] = torch.cat(
            (batch[action_pad][:, :length][rows, command_indices], batch[action_pad][:, length:]), dim=1
        )
    return result
