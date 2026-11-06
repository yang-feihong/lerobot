"""Load prepared reset provenance and attach labels AFTER semantic materialization."""

import hashlib
import json
from pathlib import Path

import torch

from lerobot.utils.reset_presets import (
    DEFAULT_RESET_UNKNOWN_TARGET_POLICY,
    PRESET_ID_KEY,
    PRESET_VALID_KEY,
    validate_reset_presets,
)


class ResetPresetCatalog:
    def __init__(self, path, catalog, *, dataset_root=None,
                 unknown_target_policy=DEFAULT_RESET_UNKNOWN_TARGET_POLICY):
        if unknown_target_policy not in {"error", "mask_audited_unknown"}:
            raise ValueError("Invalid reset preset unknown target policy")
        self.unknown_target_policy = unknown_target_policy
        self.path = Path(path)
        payload = self.path.read_bytes()
        self.payload = payload
        self.sha256 = hashlib.sha256(payload).hexdigest()
        data = json.loads(payload)
        if data["format"] != "lerobot.reset_presets" or data["version"] != 1:
            raise ValueError("Unsupported reset preset sidecar")
        validate_reset_presets(catalog)
        self.joint_targets = torch.tensor([entry["joint_target"] for entry in catalog], dtype=torch.float32)
        if data["catalog"] != catalog:
            raise ValueError("Reset preset catalog differs from policy config")
        names = validate_reset_presets(catalog)
        root = Path(dataset_root).resolve() if dataset_root is not None else self.path.parent.parent.resolve()
        files = {"meta/info.json": data["info_sha256"]}
        for ep in data["episodes"]:
            evidence = data["provenance"][ep]
            relative = evidence["data_path"]
            resolved = (root / relative).resolve()
            if not resolved.is_relative_to(root):
                raise ValueError("Reset provenance escapes dataset root")
            if relative in files and files[relative] != evidence["data_sha256"]:
                raise ValueError("Contradictory reset provenance")
            files[relative] = evidence["data_sha256"]
        for relative, expected in files.items():
            with (root / relative).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"Stale reset preset provenance: {relative}")
        self.episodes = {}
        for ep, spans in data["episodes"].items():
            previous = 0
            for span in spans:
                start, stop = span["frame_start"], span["frame_stop"]
                if (
                    type(start) is not int
                    or type(stop) is not int
                    or not 0 <= previous <= start < stop <= data["provenance"][ep]["num_frames"]
                    or span["preset"] not in names
                ):
                    raise ValueError(f"Invalid/overlapping reset ranges for episode {ep}")
                previous = stop
            self.episodes[int(ep)] = [
                (s["frame_start"], s["frame_stop"], names.index(s["preset"])) for s in spans
            ]
        self.unresolved = {}
        for ep, span in (
            (ep, span) for ep, spans in data.get("unresolved", {}).items() for span in spans
        ):
            start, stop = span["frame_start"], span["frame_stop"]
            if (
                ep not in data["provenance"]
                or type(start) is not int or type(stop) is not int
                or not 0 <= start < stop <= data["provenance"][ep]["num_frames"]
            ):
                raise ValueError("Invalid audited unresolved reset range")
            existing = self.unresolved.setdefault(int(ep), [])
            if any(start < b and a < stop for a, b in existing) or any(
                start < b and a < stop for a, b, _ in self.episodes.get(int(ep), [])
            ):
                raise ValueError("Overlapping audited unresolved reset range")
            existing.append((start, stop))

    def attach_history(self, batch, config):
        """Attach proven targets at STRICTLY PAST dataset frames, before pause augmentation."""
        from lerobot.policies.pi05.action_history import RESET_HISTORY_KEY
        from lerobot.policies.pi05.b2_action_transform import CONTROL_EXTENDED_DATASET_ACTION_NAMES

        k = config.state_num_frames - 1
        raw = batch["action"]
        source = config.dataset_action_feature_names or CONTROL_EXTENDED_DATASET_ACTION_NAMES
        reset = raw[:, :k, source.index("arm_reset")] >= .5
        inactive = raw[:, :k, source.index("arm_teleop_inactive")] >= .5
        reset = reset & ~inactive
        if "action_is_pad" in batch:
            reset = reset & ~batch["action_is_pad"][:, :k].bool()
        targets = torch.zeros((*reset.shape, 6), device=raw.device, dtype=raw.dtype)
        valid = torch.zeros_like(reset)
        q = self.joint_targets.to(device=raw.device, dtype=raw.dtype)
        offsets = torch.arange(-k, 0, device=raw.device) * config.state_history_frame_stride
        for row, (ep, first) in enumerate(zip(batch["episode_index"].reshape(-1).tolist(),
                                            batch["frame_index"].reshape(-1).tolist(), strict=True)):
            frames = int(first) + offsets
            labels = torch.full((k,), -1, device=raw.device, dtype=torch.long)
            unknown = torch.zeros(k, device=raw.device, dtype=torch.bool)
            for start, stop, label in self.episodes.get(int(ep), []):
                labels[(frames >= start) & (frames < stop)] = label
            if self.unknown_target_policy == "mask_audited_unknown":
                for start, stop in self.unresolved.get(int(ep), []):
                    unknown |= (frames >= start) & (frames < stop)
            if (reset[row] & (labels < 0) & ~unknown).any():
                raise ValueError("Historical reset has no proven target; audit reset_presets.json")
            valid[row] = reset[row] & (labels >= 0) & ~unknown
            targets[row, valid[row]] = q[labels[valid[row]]]
        batch[RESET_HISTORY_KEY] = targets
        batch[RESET_HISTORY_KEY + "_is_pad"] = ~valid

    def attach(self, batch, *, action_names, fps, action_dt):
        actions = batch["action"]
        b, t = actions.shape[:2]
        labels = torch.full((b, t), -1, dtype=torch.long, device=actions.device)
        audited_unknown = torch.zeros((b, t), dtype=torch.bool, device=actions.device)
        # Bool normalization uses [-1,+1]; this is the GT reset gate, not a model prediction.
        reset = actions[..., action_names.index("arm_reset")] > 0
        pad = batch.get("action_is_pad")
        if pad is not None:
            reset = reset & ~pad.to(actions.device).bool()
        for row, (ep, first) in enumerate(
            zip(
                batch["episode_index"].reshape(-1).tolist(),
                batch["frame_index"].reshape(-1).tolist(),
                strict=True,
            )
        ):
            frames = torch.round(torch.arange(t, device=actions.device) * fps * action_dt).long() + int(first)
            for start, stop, target in self.episodes.get(int(ep), []):
                labels[row, (frames >= start) & (frames < stop)] = target
            if self.unknown_target_policy == "mask_audited_unknown":
                for start, stop in self.unresolved.get(int(ep), []):
                    audited_unknown[row] |= (frames >= start) & (frames < stop)
        if (reset & (labels < 0) & ~audited_unknown).any():
            raise ValueError(
                "Effective training reset has no proven preset target; prepare/audit reset_presets.json"
            )
        batch[PRESET_ID_KEY] = labels
        batch[PRESET_VALID_KEY] = reset & ~audited_unknown
