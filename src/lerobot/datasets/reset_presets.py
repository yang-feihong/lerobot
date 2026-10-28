"""Load prepared reset provenance and attach labels AFTER semantic materialization."""

import hashlib
import json
from pathlib import Path

import torch

from lerobot.utils.reset_presets import PRESET_ID_KEY, PRESET_VALID_KEY, validate_reset_presets


class ResetPresetCatalog:
    def __init__(self, path, catalog):
        self.path = Path(path)
        payload = self.path.read_bytes()
        self.payload = payload
        self.sha256 = hashlib.sha256(payload).hexdigest()
        data = json.loads(payload)
        if data["format"] != "lerobot.reset_presets" or data["version"] != 1:
            raise ValueError("Unsupported reset preset sidecar")
        validate_reset_presets(catalog)
        if data["catalog"] != catalog:
            raise ValueError("Reset preset catalog differs from policy config")
        names = validate_reset_presets(catalog)
        root = self.path.parent.parent.resolve()
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

    def attach(self, batch, *, action_names, fps, action_dt):
        actions = batch["action"]
        b, t = actions.shape[:2]
        labels = torch.full((b, t), -1, dtype=torch.long, device=actions.device)
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
        if (reset & (labels < 0)).any():
            raise ValueError(
                "Effective training reset has no proven preset target; prepare/audit reset_presets.json"
            )
        batch[PRESET_ID_KEY] = labels
        batch[PRESET_VALID_KEY] = reset
