from concurrent.futures import Future
from threading import get_ident
from types import SimpleNamespace
from unittest.mock import Mock
import math

import cv2
import numpy as np
import pytest
import torch

from lerobot.scripts import pi05_apxinf as apx
from lerobot.scripts import pi05_cpu_images as rgb
from lerobot.scripts import pi05_vla_server as api
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.configs import FeatureType, NormalizationMode


@pytest.fixture
def built_rgb(tmp_path, monkeypatch):
    monkeypatch.setenv("LEROBOT_CPU_IMAGE_OPS_CACHE", str(tmp_path))
    rgb._kernel.cache_clear()
    path = rgb.build()
    yield path
    rgb._kernel.cache_clear()


def test_fused_rgb_all_levels_frames_and_readonly_ownership(built_rgb):
    pixels = np.arange(256, dtype=np.uint8).reshape(16, 16)
    image = np.stack((pixels, np.flip(pixels, 0), np.flip(pixels, 1)), axis=-1)
    image.setflags(write=False)
    images = [image, np.ascontiguousarray(np.flip(image, 0))]
    expected = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous().float().div_(255)[None]
    actual = rgb.rgb_window(images)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    actual.zero_()
    assert image.max() == 255
    assert rgb.build() == built_rgb


def test_fused_rgb_retains_fallback_without_build_or_for_unsupported_views(tmp_path, monkeypatch):
    monkeypatch.setenv("LEROBOT_CPU_IMAGE_OPS_CACHE", str(tmp_path))
    rgb._kernel.cache_clear()
    image = np.zeros((5, 7, 3), dtype=np.uint8)
    assert rgb.rgb_window([image]) is None
    assert rgb.rgb_window([image[:, ::2]]) is None
    assert rgb.rgb_window([image.astype(np.float32)]) is None
    assert rgb.rgb_window([]) is None
    rgb._kernel.cache_clear()


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("gray", [False, True])
def test_jpeg_rgb_matches_original_and_cannot_be_made_writable(direct, gray, monkeypatch):
    if direct and not hasattr(cv2, "IMREAD_COLOR_RGB"):
        pytest.skip("OpenCV has no direct RGB mode")
    source = np.random.default_rng(17).integers(0, 256, (23, 37) if gray else (23, 37, 3), np.uint8)
    encoded = cv2.imencode(".jpg", source)[1].tobytes()
    expected = cv2.cvtColor(cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if not direct:
        monkeypatch.delattr(cv2, "IMREAD_COLOR_RGB", raising=False)
    api._decode_jpeg_bytes.cache_clear()
    actual = api._decode_jpeg_bytes(encoded)
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError):
        actual.setflags(write=True)
    writable = api._decode_jpeg(np.frombuffer(encoded, np.uint8), "test", writable=True)
    writable[:] = 0
    np.testing.assert_array_equal(actual, expected)
    api._decode_jpeg_bytes.cache_clear()


def reference_filter(actions, names, initial, *, dt, tau, transition, prefix):
    result = actions.clone()
    if tau == 0 or len(actions) == 0:
        return result
    indices = [names.index(name) for name in api.B2_EXECUTION_VELOCITY_NAMES]
    alpha = -math.expm1(-dt/tau)
    filtered = initial
    for step in range(len(result)):
        if step < transition:
            if prefix is not None and step < len(prefix):
                filtered = prefix[step]
        else:
            filtered = filtered + alpha*(actions[step, indices]-filtered)
        result[step, indices] = filtered
    return result


@pytest.mark.parametrize("length", [0, 1, 50])
@pytest.mark.parametrize("tau", [0., .04, .15])
@pytest.mark.parametrize("transition", [0, 1, 7, 60])
@pytest.mark.parametrize("prefix_length", [None, 0, 3, 50])
def test_cpu_filter_bitwise_preserves_original_arithmetic(length, tau, transition, prefix_length):
    names = ("other", "b2_omega_z", "b2_vx", "b2_vy", "untouched")
    generator = torch.Generator().manual_seed(27)
    actions = torch.randn(length, len(names), generator=generator)
    if length:
        actions[0] = -0.
    initial = torch.randn(3, generator=generator)
    prefix = None if prefix_length is None else torch.randn(prefix_length, 3, generator=generator)
    original = actions.clone()
    expected = reference_filter(actions, names, initial, dt=.02, tau=tau, transition=transition, prefix=prefix)
    actual = api._smooth_b2_execution_velocity(actions, names, initial, dt=.02, time_constant_s=tau,
                                              transition_start_step=transition, prefix_velocity=prefix)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(actions.view(torch.int32), original.view(torch.int32))


@pytest.mark.parametrize("kind", ["float64", "grad", "negative_view"])
def test_cpu_filter_unsupported_numpy_inputs_keep_torch_path(kind):
    actions = torch.arange(18, dtype=torch.float64 if kind == "float64" else torch.float32).reshape(6, 3)
    if kind == "grad":
        actions.requires_grad_()
    elif kind == "negative_view":
        actions = torch._neg_view(actions)
    initial = torch.zeros(3, dtype=actions.dtype)
    names = api.B2_EXECUTION_VELOCITY_NAMES
    expected = reference_filter(actions, names, initial, dt=.02, tau=.04, transition=0, prefix=None)
    actual = api._smooth_b2_execution_velocity(actions, names, initial, dt=.02, time_constant_s=.04)
    assert torch.equal(actual, expected)
    if kind == "grad":
        actual.sum().backward()
        assert actions.grad is not None


def test_async_images_are_request_local_and_native_call_stays_on_caller(monkeypatch):
    caller = get_ident()
    threads = []
    images = [torch.zeros(1, 3, 224, 224)]*2
    def prepare(*args):
        threads.append(get_ident())
        return images, [torch.ones(1, dtype=torch.bool)]*2, None
    monkeypatch.setattr(apx, "preprocess_pi05_images", prepare)
    policy = apx.ApxInfPreparedPolicy()
    policy.config = SimpleNamespace()
    policy._rtc_inputs = Mock(return_value={})
    policy.runner = object()
    def native(*args):
        assert get_ident() == caller
        return np.zeros((50, 32), np.float32)
    policy._infer_prepared = native
    batch = {OBS_LANGUAGE_TOKENS: torch.zeros(1, 200, dtype=torch.int64),
             OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 200, dtype=torch.bool)}
    try:
        future = policy.prepare_images_async(batch)
        result = policy.predict_action_chunk(batch, prepared_images=future, noise=torch.zeros(1, 50, 32))
        assert result.shape == (1, 50, 10)
        assert len(threads) == 1 and threads[0] != caller
        failed = Future()
        failed.set_exception(ValueError("image failed"))
        with pytest.raises(ValueError, match="image failed"):
            policy.predict_action_chunk(batch, prepared_images=failed)
    finally:
        policy.close()
    assert policy._image_executor is None


def test_rtc_input_error_cancels_unused_image_work():
    policy = apx.ApxInfPreparedPolicy()
    policy._rtc_inputs = Mock(side_effect=ValueError("bad prefix"))
    future = Future()
    with pytest.raises(ValueError, match="bad prefix"):
        policy.predict_action_chunk({}, prepared_images=future)
    assert future.cancelled()


@pytest.mark.parametrize("seed", range(4))
def test_promoted_fused_and_async_path_preserves_every_native_input(built_rgb, seed):
    rng = np.random.default_rng(seed)
    pixels = {"base": rng.integers(0, 256, (37, 59, 3), np.uint8),
              "wrist": rng.integers(0, 256, (61, 29, 3), np.uint8)}
    baseline = {key: torch.from_numpy(image.copy()).permute(2, 0, 1).contiguous().float().div_(255)[None]
                for key, image in pixels.items()}
    optimized = {key: rgb.rgb_window([image])[:, -1] for key, image in pixels.items()}
    tokens = torch.arange(200, dtype=torch.int64)[None]
    mask = tokens < 123
    for batch in (baseline, optimized):
        batch.update({OBS_LANGUAGE_TOKENS: tokens, OBS_LANGUAGE_ATTENTION_MASK: mask})
    captured = []
    policy = apx.ApxInfPreparedPolicy()
    policy.config = SimpleNamespace(image_features={"base": None, "wrist": None},
                                    image_resolution=(224, 224), mem_vit_enabled=False)
    policy.runner = object()
    policy._rtc_inputs = lambda prefix, delay: {}
    def native(*args):
        captured.append([value.copy() if isinstance(value, np.ndarray) else value for value in args])
        return np.zeros((50, 32), dtype=np.float32)
    policy._infer_prepared = native
    noise = torch.randn(1, 50, 32, generator=torch.Generator().manual_seed(seed))
    try:
        first = policy.predict_action_chunk(baseline, noise=noise)
        future = policy.prepare_images_async(optimized)
        second = policy.predict_action_chunk(optimized, noise=noise, prepared_images=future)
        assert torch.equal(first, second)
        for original, actual in zip(captured[0][:4], captured[1][:4], strict=True):
            assert original.dtype == actual.dtype and original.shape == actual.shape
            assert original.tobytes() == actual.tobytes()
        assert captured[0][1].shape == captured[0][2].shape == (1, 200)
        assert captured[0][4] == captured[1][4] == {}
    finally:
        policy.close()


@pytest.mark.parametrize("device,mem,identity,unchanged,expected", [
    ("cpu", False, True, True, True),
    ("cuda", False, True, True, False),
    ("cpu", True, True, True, False),
    ("cpu", False, False, True, False),
    ("cpu", False, True, False, False),
])
@pytest.mark.parametrize("failure", ["preprocess", "rtc"])
def test_engine_async_guard_and_failure_cancellation(device, mem, identity, unchanged, expected, failure, monkeypatch):
    future = Future()
    class NativePolicy:
        config = SimpleNamespace(normalization_mapping={
            FeatureType.VISUAL: NormalizationMode.IDENTITY if identity else NormalizationMode.MEAN_STD})
        prepare_images_async = Mock(return_value=future)
    engine = api.AsyncRTCPolicy.__new__(api.AsyncRTCPolicy)
    engine.chunk_scheduling_mode = "RTC"
    engine.device, engine.mem_enabled = torch.device(device), mem
    engine.policy = NativePolicy()
    engine._make_batch = lambda packet: {}
    def fail(*args):
        raise ValueError("preparation failed")
    engine.preprocessor = fail if failure == "preprocess" else lambda batch: batch
    engine._rtc_conditioning = fail
    monkeypatch.setattr(api, "images_are_unchanged", lambda _: unchanged)
    packet = SimpleNamespace(scheduling_mode="RTC", inference_delay_steps=4, source_steps_per_wall_second=50.)
    with pytest.raises(ValueError, match="preparation failed"):
        engine._infer_unlocked(packet)
    assert engine.policy.prepare_images_async.call_count == int(expected)
    assert future.cancelled() == expected
