"""Tool-axis contract shared by recording, training and serving.

Physical and legacy share a TCP and world reference. Only the tool basis
changes: R_physical = R_legacy @ Rx(-pi/2). Joint angles never change.
"""

EE_FRAME_CONTRACT = "same_tcp_tool_rx_minus_90_v1"
EE_FRAMES = ("legacy", "physical")


def validate_ee_frame(frame):
    if frame not in EE_FRAMES:
        raise ValueError(f"Expected explicit EE frame legacy|physical, got {frame!r}")
    return frame


def checkpoint_ee_frame(metadata):
    """Absent contract means a historical legacy checkpoint, not unknown data."""
    contract = metadata.get("ee_frame_contract")
    if contract is None:
        if metadata.get("version", 10) >= 11:
            raise ValueError("New checkpoint lacks its required EE frame contract")
        if "model_ee_frame" in metadata:
            raise ValueError("Model EE frame is missing its definition contract")
        return "legacy"
    if contract != EE_FRAME_CONTRACT:
        raise ValueError(f"Unsupported EE frame contract: {contract!r}")
    return validate_ee_frame(metadata["model_ee_frame"])


def resume_ee_frame(requested, metadata):
    saved = checkpoint_ee_frame(metadata)
    if requested is not None and validate_ee_frame(requested) != saved:
        raise ValueError("Resume cannot change model_ee_frame; create an explicit new training run")
    return saved


def convert_ee_pose(poses, source, target):
    """Convert absolute [...,rot6d+xyz] NumPy arrays or Torch tensors."""
    validate_ee_frame(source)
    validate_ee_frame(target)
    if poses.shape[-1] != 9:
        raise ValueError("Expected absolute EE rot6d+xyz (9D)")
    if hasattr(poses, "clone"):
        import torch

        values = poses.clone()
        third = torch.linalg.cross(poses[..., :3], poses[..., 3:6], dim=-1)
    else:
        import numpy as np

        values = np.array(poses, copy=True)
        third = np.cross(poses[..., :3], poses[..., 3:6])
    if source != target:
        values[..., 3:6] = third if source == "physical" else -third
    return values


def convert_named_features(values, names, source, target):
    """Convert all absolute EE blocks; leave joints, positions and flags intact."""
    validate_ee_frame(source)
    validate_ee_frame(target)
    result = values.clone() if hasattr(values, "clone") else values.copy()
    for prefix in ("height_invariant_ee_state_", "height_invariant_ee_", "control_height_invariant_ee_"):
        fields = [prefix + suffix for suffix in (
            "rot6d_col0_x", "rot6d_col0_y", "rot6d_col0_z",
            "rot6d_col1_x", "rot6d_col1_y", "rot6d_col1_z", "x", "y", "z",
        )]
        present = [field in names for field in fields]
        if any(present):
            if not all(present):
                raise ValueError(f"Incomplete absolute EE feature block: {prefix}")
            indices = [names.index(field) for field in fields]
            result[..., indices] = convert_ee_pose(values[..., indices], source, target)
    return result
