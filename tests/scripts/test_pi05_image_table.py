from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from lerobot.scripts.pi05_image_table import FIELDS, encode_image_table, decode_image_table
from lerobot.scripts.pi05_npz import dump_arrays, load_arrays
from lerobot.scripts.pi05_vla_server import (
    ActionRecord,
    AsyncRTCPolicy,
    FlatImageHistory,
    RolloutRecorder,
    decode_observation_packet,
    encode_action_packet,
    _validated_model_image_history,
    _decode_model_image_history,
)
from test_pi05_vla_server import _complete_packet


def inputs():
    first = cv2.imencode(".jpg", np.full((9, 13, 3), 80, dtype=np.uint8))[1]
    second = cv2.imencode(".jpg", np.full((9, 13, 3), 180, dtype=np.uint8))[1]
    history = dict(
        format_version=np.int64(1),
        source_step=np.int64(12),
        base_camera_key=np.asarray("base"),
        wrist_camera_key=np.asarray("wrist"),
        is_pad=np.asarray([True, False, False]),
    )
    history.update({name: np.asarray([10, 20, 30], dtype=np.int64) for name in FIELDS})
    for i in range(3):
        history[f"base_jpeg_{i}"] = first if i < 2 else second
        history[f"wrist_jpeg_{i}"] = first
    legacy = load_arrays(
        _complete_packet(
            base_jpeg=first,
            wrist_jpeg=second,
            model_image_history=np.frombuffer(dump_arrays(history), dtype=np.uint8),
        )
    )
    flat = {
        key: value
        for key, value in legacy.items()
        if key not in {"base_jpeg", "wrist_jpeg", "model_image_history"}
    }
    flat.update(encode_image_table(history, (first, second)))
    return legacy, flat, history


def action(history):
    return ActionRecord(
        sequence=1,
        source_step=12,
        inference_seconds=0.1,
        action_names=("x",),
        model_action_names=("x",),
        control_epoch=7,
        b2_execution_mode="velocity_chunk",
        z1_action_representation="ee_state_delta",
        b2_anchor_kind="none",
        b2_anchor=np.zeros(3),
        b2_reference=None,
        ee_anchor_kind="actual_ee_state",
        ee_anchor=np.zeros(9),
        original=torch.zeros(50, 1),
        unsmoothed=torch.zeros(50, 1),
        processed=torch.zeros(50, 1),
        inference_started_ns=1,
        inference_finished_ns=2,
        preprocess_seconds=0.01,
        predict_seconds=0.06,
        postprocess_seconds=0.01,
        observation_received_ns=0,
        inference_delay_steps=0,
        velocity_smoothing_transition_step=0,
        sim_steps_per_wall_second=50,
        model_image_history=history,
    )


def test_flat_decoder_preserves_images_metadata_and_legacy_path():
    legacy, flat, expected = inputs()
    old = decode_observation_packet(dump_arrays(legacy))
    payload = dump_arrays(flat)
    new = decode_observation_packet(payload)
    assert isinstance(new.model_image_history, FlatImageHistory)
    assert new.model_image_history.input_payload is payload
    assert new.model_image_history.input_sha256 == hashlib.sha256(payload).hexdigest()
    for key in (
        "state",
        "prepared_state",
        "prepared_state_is_pad",
        "base_rgb",
        "wrist_rgb",
        "base_jpeg",
        "wrist_jpeg",
    ):
        np.testing.assert_array_equal(getattr(new, key), getattr(old, key))
    args = dict(source_step=12, count=3, base_key="base", wrist_key="wrist")
    old_records, old_pad = _validated_model_image_history(old.model_image_history, **args)
    records, pad = _validated_model_image_history(new.model_image_history, **args)
    np.testing.assert_array_equal(old_pad, pad)
    for a, b in zip(old_records, records, strict=True):
        assert a.capture_ns == b.capture_ns and a.base_capture_ns == b.base_capture_ns
        np.testing.assert_array_equal(a.base_jpeg, b.base_jpeg)
        np.testing.assert_array_equal(a.wrist_jpeg, b.wrist_jpeg)
    for value in new.model_image_history.data.values():
        if isinstance(value, np.ndarray):
            with pytest.raises(ValueError):
                value.setflags(write=True)
    public, public_pad = _decode_model_image_history(new.model_image_history, **args)
    public[0].base_jpeg[:] = 0
    public_pad[:] = False
    np.testing.assert_array_equal(records[0].base_jpeg, expected["base_jpeg_0"])
    assert pad.tolist() == [True, False, False]


def test_output_contains_reference_not_repeated_jpegs_and_recorder_replays(tmp_path):
    legacy, flat, expected = inputs()
    payload = dump_arrays(flat)
    packet = decode_observation_packet(payload)
    record = action(packet.model_image_history)
    output = load_arrays(encode_action_packet(record))
    assert output["model_input_sha256"].item() == hashlib.sha256(payload).hexdigest()
    assert "model_image_history" not in output
    old = load_arrays(
        encode_action_packet(replace(record, model_image_history=legacy["model_image_history"].tobytes()))
    )
    for key in output.keys() - {"model_input_sha256"}:
        np.testing.assert_array_equal(output[key], old[key])
    assert len(encode_action_packet(record)) < len(
        encode_action_packet(replace(record, model_image_history=legacy["model_image_history"].tobytes()))
    )
    recorder = RolloutRecorder(str(tmp_path))
    recorder.observation(packet)
    recorder.action(record)
    saved = tmp_path / "vla/inputs" / (packet.model_image_history.input_sha256 + ".npz")
    assert saved.read_bytes() == payload
    assert saved.stat().st_ino == (tmp_path / "vla/observations/000012.npz").stat().st_ino
    another = dict(flat, task=np.asarray("another instruction"))
    recorder.observation(decode_observation_packet(dump_arrays(another)))
    assert saved.read_bytes() == payload
    image_history, current = decode_image_table(load_arrays(saved.read_bytes()))
    for key, value in expected.items():
        np.testing.assert_array_equal(image_history[key], value)
    assert len(current) == 2
    assert "model_input_sha256" in load_arrays((tmp_path / "vla/actions/000001.npz").read_bytes())


def test_flat_current_jpeg_still_validated_when_history_is_good():
    _, flat, _ = inputs()
    history, current = decode_image_table(flat)
    broken = encode_image_table(history, (np.zeros(4, dtype=np.uint8), current[1]))
    with pytest.raises(ValueError, match="base JPEG"):
        decode_observation_packet(dump_arrays(flat | broken))
    with pytest.raises(ValueError, match="Mixed"):
        decode_observation_packet(dump_arrays(flat | {"base_jpeg": current[0]}))


@pytest.mark.parametrize(
    "changes", [dict(source_step=13), dict(count=2), dict(base_key="wrist"), dict(wrist_key="base")]
)
def test_flat_history_contract_mismatch_rejected(changes):
    _, flat, _ = inputs()
    packet = decode_observation_packet(dump_arrays(flat))
    with pytest.raises(ValueError):
        _validated_model_image_history(
            packet.model_image_history,
            **(dict(source_step=12, count=3, base_key="base", wrist_key="wrist") | changes),
        )


def test_transport_capability_does_not_change_model_schema_identity():
    engine = AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    engine.policy = SimpleNamespace(config=SimpleNamespace(ee_frame_metadata=lambda: {}))
    engine.source_state_names = engine.selected_state_names = ("state",)
    engine.base_camera_key, engine.wrist_camera_key = "base", "wrist"
    engine.mem_image_num_frames, engine.mem_image_interval_ns = 3, 500_000_000
    engine.state_num_frames, engine.state_history_stride, engine.low_level_hz = 3, 2, 50
    engine.action_history_enabled, engine.action_history_names = False, ()
    spec = engine.input_spec()
    assert spec["image_transport"] == "jpeg-table-v1"
    legacy = {key: value for key, value in spec.items() if key not in {"image_transport", "image_resources", "result_wait_ms", "array_transport", "persistent_http", "schema_id"}}
    assert spec["schema_id"] == hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest()


def test_array_wire_input_output_and_raw_recording(tmp_path):
    from lerobot.scripts.pi05_array_wire import dump_arrays as encode_wire, load_arrays as decode_wire
    _, flat, expected = inputs()
    payload = encode_wire(flat)
    packet = decode_observation_packet(payload)
    record = action(packet.model_image_history)
    result = decode_wire(encode_action_packet(record, binary_arrays=True))
    old = load_arrays(encode_action_packet(record))
    assert result.keys() == old.keys()
    for key in old:
        np.testing.assert_array_equal(result[key], old[key])
    recorder = RolloutRecorder(str(tmp_path))
    recorder.observation(packet)
    recorder.action(record)
    saved = tmp_path / "vla/inputs" / (hashlib.sha256(payload).hexdigest() + ".awb")
    assert saved.read_bytes() == payload
    assert saved.stat().st_ino == (tmp_path / "vla/observations/000012.awb").stat().st_ino
    history, _ = decode_image_table(decode_wire(saved.read_bytes()))
    for key in expected:
        np.testing.assert_array_equal(history[key], expected[key])
