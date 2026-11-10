from io import BytesIO

import cv2
import numpy as np
import pytest
import torch

import lerobot.scripts.pi05_vla_server as server


def jpeg(value):
    return cv2.imencode(".jpg", np.full((9, 13, 3), value, dtype=np.uint8))[1]


def history():
    data = dict(format_version=np.int64(1), source_step=np.int64(7),
                base_camera_key=np.asarray("base"), wrist_camera_key=np.asarray("wrist"),
                is_pad=np.array([True, False]))
    for field in ("capture_ns", "base_capture_ns", "wrist_capture_ns", "encoded_ns", "server_received_ns", "sim_step"):
        data[field] = np.array([3, 7], dtype=np.int64)
    for index in range(2):
        data[f"base_jpeg_{index}"] = jpeg(index * 80)
        data[f"wrist_jpeg_{index}"] = jpeg(255 - index * 80)
    buffer = BytesIO()
    np.savez(buffer, **data)
    return buffer.getvalue()


def test_history_npz_materialized_once_and_contract_rechecked():
    server._load_model_image_history.cache_clear()
    server._validated_model_image_history.cache_clear()
    payload = history()
    args = dict(source_step=7, count=2, base_key="base", wrist_key="wrist")
    for _ in range(3):
        server._decode_model_image_history(payload, **args)
    assert server._load_model_image_history.cache_info().misses == 1
    assert server._validated_model_image_history.cache_info().hits == 2
    for changes in [dict(source_step=8), dict(count=3), dict(base_key="wrong"), dict(wrist_key="wrong")]:
        with pytest.raises(ValueError):
            server._decode_model_image_history(payload, **(args | changes))


def test_history_cached_arrays_are_not_exposed_to_callers():
    payload = history()
    args = dict(source_step=7, count=2, base_key="base", wrist_key="wrist")
    records, pad = server._decode_model_image_history(payload, **args)
    original = records[0].base_jpeg.copy()
    records[0].base_jpeg[:] = 0
    pad[:] = False
    records, pad = server._decode_model_image_history(payload, **args)
    assert np.array_equal(records[0].base_jpeg, original)
    assert pad.tolist() == [[True, False]]


def test_jpeg_cache_uses_content_and_returns_independent_arrays():
    server._decode_jpeg_bytes.cache_clear()
    encoded = jpeg(80)
    result = server._decode_jpeg(encoded, "base")
    expected = result.copy()
    result[:] = 0
    assert np.array_equal(server._decode_jpeg(encoded.copy(), "wrist"), expected)
    assert server._decode_jpeg_bytes.cache_info().misses == 1
    assert server._decode_jpeg_bytes.cache_info().hits == 1
    assert not np.array_equal(server._decode_jpeg(jpeg(180), "base"), expected)


def test_decode_caches_are_bounded():
    server._decode_jpeg_bytes.cache_clear()
    for value in range(50):
        server._decode_jpeg(jpeg(value * 5), "base")
    assert server._decode_jpeg_bytes.cache_info().currsize == 32
    assert server._load_model_image_history.cache_info().maxsize == 4


def test_internal_jpeg_views_are_immutable_and_reuse_pixels():
    encoded = jpeg(80)
    first = server._decode_jpeg(encoded, "base", writable=False)
    second = server._decode_jpeg(encoded, "wrist", writable=False)
    assert first is second
    with pytest.raises(ValueError):
        first.setflags(write=True)
    with pytest.raises(ValueError):
        first[:] = 0
    public = server._decode_jpeg(encoded, "base")
    public[:] = 0
    assert np.array_equal(second, server._decode_jpeg(encoded, "base", writable=False))
    with pytest.raises(ValueError, match="broken"):
        server._decode_jpeg_window([(np.zeros(4, dtype=np.uint8), "broken")], writable=False)


def test_parallel_jpeg_window_preserves_order_and_errors():
    values = [80, 10, 250, 120, 40, 200]
    images = [(jpeg(value), str(index)) for index, value in enumerate(values)]
    expected = [server._decode_jpeg(*image) for image in images]
    actual = server._decode_jpeg_window(images)
    assert all(np.array_equal(left, right) for left, right in zip(actual, expected))
    with pytest.raises(ValueError, match="broken"):
        server._decode_jpeg_window([(np.zeros(4, dtype=np.uint8), "broken")])
    assert np.array_equal(server._decode_jpeg_window(images)[0], expected[0])


def test_batched_image_conversion_matches_per_frame_values_and_layout():
    images = [server._decode_jpeg(jpeg(value), "base") for value in [0, 10, 80, 128, 200, 255]]
    original = torch.stack([torch.from_numpy(image).permute(2, 0, 1).to(torch.float32).div_(255) for image in images])[None]
    batched = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous().to(torch.float32).div_(255)[None]
    assert torch.equal(original, batched)
    assert original.stride() == batched.stride()
