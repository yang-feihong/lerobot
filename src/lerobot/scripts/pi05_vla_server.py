#!/usr/bin/env python
"""Serve asynchronous PI0.5 RTC chunks to the visual-wholebody simulator.

The simulator is the clock authority.  Observations carry a monotonically
increasing 50 Hz simulation step and the currently executing chunk/index.
Inference always consumes the newest observation and never blocks simulation.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from threading import Condition, Event, Lock, Thread
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import torch

from lerobot.configs import PreTrainedConfig, RTCAttentionSchedule
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.rtc import RTCConfig
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.utils import init_logging

LOG = logging.getLogger("pi05_vla_server")
PROTOCOL_VERSION = 1


def _resolve_policy_path(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    if (path / "config.json").exists():
        return path
    if (path / "pretrained_model" / "config.json").exists():
        return path / "pretrained_model"
    checkpoint_root = path / "checkpoints"
    if checkpoint_root.exists():
        steps = sorted(
            (item for item in checkpoint_root.iterdir() if item.is_dir() and item.name.isdigit()),
            key=lambda item: int(item.name),
        )
        if steps and (steps[-1] / "pretrained_model" / "config.json").exists():
            return steps[-1] / "pretrained_model"
    raise FileNotFoundError(f"Cannot resolve a policy checkpoint from {path}")


def _decode_jpeg(encoded: np.ndarray, name: str) -> np.ndarray:
    image = cv2.imdecode(np.asarray(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode {name} JPEG")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def decode_observation_packet(payload: bytes, max_state_dim: int = 4096) -> ObservationPacket:
    with np.load(BytesIO(payload), allow_pickle=False) as packet:
        version = int(packet["protocol_version"].item())
        if version != PROTOCOL_VERSION:
            raise ValueError(f"Protocol version {version} != {PROTOCOL_VERSION}")
        state = np.asarray(packet["state"], dtype=np.float32).reshape(-1)
        if state.size == 0 or state.size > max_state_dim or not np.isfinite(state).all():
            raise ValueError(f"Invalid state vector shape/content: {state.shape}")
        return ObservationPacket(
            sim_step=int(packet["sim_step"].item()),
            active_sequence=int(packet["active_sequence"].item()),
            active_index=int(packet["active_index"].item()),
            task=str(packet["task"].item()),
            state=state,
            base_rgb=_decode_jpeg(packet["base_jpeg"], "base"),
            wrist_rgb=_decode_jpeg(packet["wrist_jpeg"], "wrist"),
        )


def encode_action_packet(record: ActionRecord) -> bytes:
    output = BytesIO()
    np.savez(
        output,
        protocol_version=np.asarray(PROTOCOL_VERSION, dtype=np.int64),
        sequence=np.asarray(record.sequence, dtype=np.int64),
        source_step=np.asarray(record.source_step, dtype=np.int64),
        inference_seconds=np.asarray(record.inference_seconds, dtype=np.float64),
        action_names=np.asarray(record.action_names, dtype=np.str_),
        processed_actions=record.processed.numpy().astype(np.float32, copy=False),
    )
    return output.getvalue()


@dataclass(frozen=True)
class ObservationPacket:
    sim_step: int
    active_sequence: int
    active_index: int
    task: str
    state: np.ndarray
    base_rgb: np.ndarray
    wrist_rgb: np.ndarray


@dataclass(frozen=True)
class ActionRecord:
    sequence: int
    source_step: int
    inference_seconds: float
    action_names: tuple[str, ...]
    original: torch.Tensor
    processed: torch.Tensor


class AsyncRTCPolicy:
    """Latest-only observation mailbox plus a single asynchronous GPU worker."""

    def __init__(self, args: argparse.Namespace):
        policy_path = _resolve_policy_path(args.policy_path)
        metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
        config = PreTrainedConfig.from_pretrained(policy_path)
        config.pretrained_path = policy_path
        config.device = args.device
        if args.num_inference_steps is not None:
            config.num_inference_steps = args.num_inference_steps

        LOG.info("Loading policy=%s dataset=%s", policy_path, args.dataset_root)
        self.policy = make_policy(config, ds_meta=metadata)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(policy_path),
            pretrained_revision=getattr(config, "pretrained_revision", None),
            dataset_stats=metadata.stats,
            preprocessor_overrides={
                "device_processor": {"device": torch.device(args.device).type},
                "normalizer_processor": {
                    "stats": metadata.stats,
                    "features": {**self.policy.config.input_features, **self.policy.config.output_features},
                    "norm_map": self.policy.config.normalization_mapping,
                },
            },
            postprocessor_overrides={
                "unnormalizer_processor": {
                    "stats": metadata.stats,
                    "features": self.policy.config.output_features,
                    "norm_map": self.policy.config.normalization_mapping,
                },
            },
        )

        rtc_config = RTCConfig(
            enabled=True,
            prefix_attention_schedule=RTCAttentionSchedule(args.rtc_schedule.upper()),
            max_guidance_weight=args.rtc_max_guidance_weight,
            execution_horizon=args.rtc_execution_horizon,
        )
        self.policy.config.rtc_config = rtc_config
        self.policy.init_rtc_processor()
        self.rtc_config = rtc_config
        self.device = torch.device(args.device)
        self.low_level_hz = float(args.low_level_hz)
        self.period_s = 1.0 / float(args.high_level_hz)
        self.camera_keys = tuple(metadata.camera_keys)
        if len(self.camera_keys) != 2:
            raise ValueError(f"Expected exactly two dataset cameras, got {self.camera_keys}")

        names = getattr(self.policy.config, "action_feature_names", None)
        if getattr(self.policy.config, "io_schema_resolved", False):
            from lerobot.policies.pi05.b2_action_transform import (
                action_schema_kwargs,
                b2_execution_action_names,
            )

            names = b2_execution_action_names(
                metadata.features["action"].get("names"),
                **action_schema_kwargs(self.policy.config),
            )
        if names is None:
            names = metadata.features["action"].get("names")
        if names is None:
            raise ValueError("Checkpoint/dataset does not define named actions")
        self.action_names = tuple(str(name) for name in names)

        self._mailbox_condition = Condition()
        self._mailbox: ObservationPacket | None = None
        self._mailbox_version = 0
        self._last_inferred_version = 0
        self._records_lock = Lock()
        self._records: dict[int, ActionRecord] = {}
        self._latest_record: ActionRecord | None = None
        self._next_sequence = 1
        self._latencies: deque[float] = deque(maxlen=args.rtc_latency_window)
        self._stop = Event()
        self._worker = Thread(target=self._run, name="pi05-rtc-worker", daemon=True)
        self._last_error: str | None = None
        self._inference_count = 0

    def start(self) -> None:
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        with self._mailbox_condition:
            self._mailbox_condition.notify_all()
        self._worker.join(timeout=5.0)

    def submit(self, packet: ObservationPacket) -> None:
        with self._mailbox_condition:
            if self._mailbox is not None and packet.sim_step < self._mailbox.sim_step:
                return
            self._mailbox = packet
            self._mailbox_version += 1
            self._mailbox_condition.notify()

    def latest_after(self, sequence: int) -> ActionRecord | None:
        with self._records_lock:
            record = self._latest_record
            return record if record is not None and record.sequence > sequence else None

    def health(self) -> dict[str, object]:
        with self._records_lock:
            latest = self._latest_record
        return {
            "ok": self._worker.is_alive() and self._last_error is None,
            "protocol_version": PROTOCOL_VERSION,
            "worker_alive": self._worker.is_alive(),
            "inference_count": self._inference_count,
            "latest_sequence": latest.sequence if latest else 0,
            "latest_source_step": latest.source_step if latest else -1,
            "last_error": self._last_error,
        }

    def _previous_prefix(self, packet: ObservationPacket) -> torch.Tensor | None:
        if packet.active_sequence < 0:
            return None
        with self._records_lock:
            record = self._records.get(packet.active_sequence)
        if record is None:
            LOG.warning("RTC prefix sequence %d is no longer available", packet.active_sequence)
            return None
        index = max(0, min(packet.active_index, len(record.original)))
        prefix = record.original[index:].clone().to(self.device)
        if prefix.numel() == 0:
            return None
        horizon = self.rtc_config.execution_horizon
        if len(prefix) >= horizon:
            return prefix[:horizon]
        padded = torch.zeros((horizon, prefix.shape[-1]), dtype=prefix.dtype, device=prefix.device)
        padded[: len(prefix)] = prefix
        return padded

    def _estimated_delay_steps(self) -> int:
        if not self._latencies:
            return 0
        return int(math.ceil(max(self._latencies) * self.low_level_hz))

    def _make_batch(self, packet: ObservationPacket) -> dict[str, object]:
        images = (packet.base_rgb, packet.wrist_rgb)
        batch: dict[str, object] = {OBS_STATE: torch.from_numpy(packet.state).unsqueeze(0)}
        for key, image in zip(self.camera_keys, images, strict=True):
            batch[key] = (
                torch.from_numpy(np.ascontiguousarray(image))
                .permute(2, 0, 1)
                .to(dtype=torch.float32)
                .div_(255.0)
                .unsqueeze(0)
            )
        batch["task"] = [packet.task]
        return batch

    def _infer(self, packet: ObservationPacket) -> ActionRecord:
        started = time.perf_counter()
        batch = self.preprocessor(self._make_batch(packet))
        prefix = self._previous_prefix(packet)
        # RTC temporarily re-enables autograd to compute its prefix-guidance
        # correction. ``inference_mode`` cannot be overridden that way, while
        # ``no_grad`` can and still avoids retaining the normal inference graph.
        with torch.no_grad():
            actions = self.policy.predict_action_chunk(
                batch,
                inference_delay=self._estimated_delay_steps(),
                prev_chunk_left_over=prefix,
            )
            original = actions.squeeze(0).detach().cpu().clone()
            processed = self.postprocessor(actions).squeeze(0).detach().cpu().to(torch.float32)
        elapsed = time.perf_counter() - started
        if processed.ndim != 2 or processed.shape[-1] != len(self.action_names):
            raise ValueError(
                f"Processed action shape {tuple(processed.shape)} does not match names {self.action_names}"
            )
        self._latencies.append(elapsed)
        record = ActionRecord(
            sequence=self._next_sequence,
            source_step=packet.sim_step,
            inference_seconds=elapsed,
            action_names=self.action_names,
            original=original,
            processed=processed,
        )
        self._next_sequence += 1
        return record

    def _run(self) -> None:
        next_start = 0.0
        while not self._stop.is_set():
            with self._mailbox_condition:
                self._mailbox_condition.wait_for(
                    lambda: self._stop.is_set()
                    or (self._mailbox is not None and self._mailbox_version != self._last_inferred_version),
                    timeout=0.25,
                )
                if self._stop.is_set():
                    return
                packet = self._mailbox
                version = self._mailbox_version
            if packet is None or version == self._last_inferred_version:
                continue
            remaining = next_start - time.perf_counter()
            if remaining > 0 and self._stop.wait(remaining):
                return
            # A newer observation may have arrived while enforcing the 4 Hz
            # start-rate cap. Refresh once so an older mailbox item is never
            # inferred merely because it was current before the wait.
            with self._mailbox_condition:
                if self._mailbox_version != version:
                    packet = self._mailbox
                    version = self._mailbox_version
            if packet is None or version == self._last_inferred_version:
                continue
            inference_started = time.perf_counter()
            try:
                record = self._infer(packet)
                with self._records_lock:
                    self._records[record.sequence] = record
                    self._latest_record = record
                    while len(self._records) > 16:
                        self._records.pop(min(self._records))
                self._last_inferred_version = version
                self._inference_count += 1
                self._last_error = None
                LOG.info(
                    "chunk=%d obs_step=%d latency=%.3fs delay=%d actions=%s",
                    record.sequence,
                    record.source_step,
                    record.inference_seconds,
                    self._estimated_delay_steps(),
                    tuple(record.processed.shape),
                )
            except Exception as exc:  # keep the server observable, but stop retry storms
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._last_inferred_version = version
                LOG.exception("Inference failed for sim_step=%d", packet.sim_step)
                if self._stop.wait(0.5):
                    return
            # Cap inference starts at high_level_hz. If inference itself is
            # slower than the period, immediately consume the newest mailbox.
            next_start = inference_started + self.period_s


class VLARequestHandler(BaseHTTPRequestHandler):
    server_version = "Pi05VLA/1"

    @property
    def engine(self) -> AsyncRTCPolicy:
        return self.server.engine  # type: ignore[attr-defined,no-any-return]

    @property
    def max_payload_bytes(self) -> int:
        return self.server.max_payload_bytes  # type: ignore[attr-defined,no-any-return]

    def log_message(self, fmt: str, *args: object) -> None:
        LOG.debug("http %s - %s", self.address_string(), fmt % args)

    def _write_json(self, status: int, value: dict[str, object]) -> None:
        payload = json.dumps(value, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/v1/observations":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > self.max_payload_bytes:
                raise ValueError(f"Invalid Content-Length: {length}")
            packet = decode_observation_packet(self.rfile.read(length))
            self.engine.submit(packet)
            self._write_json(HTTPStatus.ACCEPTED, {"accepted_step": packet.sim_step})
        except Exception as exc:
            LOG.warning("Rejected observation: %s", exc)
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/healthz":
            health = self.engine.health()
            self._write_json(HTTPStatus.OK if health["ok"] else HTTPStatus.SERVICE_UNAVAILABLE, health)
            return
        if parsed.path != "/v1/actions":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            after = int(parse_qs(parsed.query).get("after", ["0"])[0])
        except ValueError:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "after must be an integer"})
            return
        record = self.engine.latest_after(after)
        if record is None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        payload = encode_action_packet(record)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-npz")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-repo-id", default="local/b2_z1_vla")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--high-level-hz", type=float, default=4.0)
    parser.add_argument("--low-level-hz", type=float, default=50.0)
    parser.add_argument("--rtc-execution-horizon", type=int, default=13)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=10.0)
    parser.add_argument("--rtc-schedule", choices=[item.value for item in RTCAttentionSchedule], default="EXP")
    parser.add_argument("--rtc-latency-window", type=int, default=10)
    parser.add_argument("--max-payload-mb", type=float, default=12.0)
    args = parser.parse_args()
    if args.high_level_hz <= 0 or args.low_level_hz <= 0:
        parser.error("both frequencies must be positive")
    if args.rtc_execution_horizon <= 0 or args.rtc_latency_window <= 0:
        parser.error("RTC horizon/window must be positive")
    if args.num_inference_steps is not None and args.num_inference_steps <= 0:
        parser.error("num-inference-steps must be positive")
    return args


def main() -> None:
    args = parse_args()
    init_logging()
    engine = AsyncRTCPolicy(args)
    server = ThreadingHTTPServer((args.host, args.port), VLARequestHandler)
    server.engine = engine  # type: ignore[attr-defined]
    server.max_payload_bytes = int(args.max_payload_mb * 1024 * 1024)  # type: ignore[attr-defined]
    engine.start()
    LOG.info("Listening on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        engine.stop()


if __name__ == "__main__":
    main()
