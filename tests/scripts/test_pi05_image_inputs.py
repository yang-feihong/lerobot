from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from lerobot.configs import FeatureType, NormalizationMode
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, resize_with_pad_torch
from lerobot.processor import DeviceProcessorStep, NormalizerProcessorStep, RenameObservationsProcessorStep
from lerobot.scripts.pi05_image_inputs import ImageTensorCache, images_are_unchanged
from lerobot.scripts.pi05_vla_server import _decode_jpeg_window


def jpeg(seed, shape=(23, 39, 3)):
    pixels = np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)
    return cv2.imencode(".jpg", pixels)[1]


def check_gpu(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")


def policy(device, mem, wrist_first=False):
    result = PI05Policy.__new__(PI05Policy)
    torch.nn.Module.__init__(result)
    result.register_parameter("p", torch.nn.Parameter(torch.zeros(1, device=device)))
    result.config = SimpleNamespace(
        image_resolution=(16, 16), image_features={"base": None, "wrist": None}, mem_vit_enabled=mem
    )
    if wrist_first:
        result.config.image_features = {"wrist": None, "base": None}
    return result


def original_windows(windows):
    return {
        key: torch.from_numpy(np.stack(_decode_jpeg_window([(image, key) for image in values])))
        .permute(0, 3, 1, 2)
        .contiguous()
        .float()
        .div_(255)[None]
        for key, values in windows.items()
    }


def normalized(model, windows):
    if not model.config.mem_vit_enabled:
        windows = {key: value[:, -1] if value.ndim == 5 else value for key, value in windows.items()}
    return model._preprocess_images(
        windows, mem_vit_window_num_frames=3, mem_vit_memory_mask=torch.ones(1, 3, dtype=torch.bool)
    )[0]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mem", [False, True])
@pytest.mark.parametrize("wrist_first", [False, True])
@pytest.mark.parametrize("shape", [(23, 39, 3), (16, 16, 3)])
def test_cached_images_match_original_model_inputs(device, mem, wrist_first, shape):
    check_gpu(device)
    keys = ("wrist", "base") if wrist_first else ("base", "wrist")
    cache = ImageTensorCache(
        keys, 3, (16, 16), torch.device(device), _decode_jpeg_window, resize_with_pad_torch
    )
    model = policy(device, mem, wrist_first)
    windows = {
        "base": [jpeg(1, shape), jpeg(1, shape), jpeg(2, shape)],
        "wrist": [jpeg(3, shape), jpeg(4, shape), jpeg(5, shape)],
    }
    expected = normalized(model, original_windows(windows))
    for _ in range(3):
        prepared = cache.prepare(windows)
        actual = normalized(model, prepared)
        assert all(torch.equal(a, b) and a.stride() == b.stride() for a, b in zip(actual, expected, strict=True))
        prepared["base"].zero_()
    torch.testing.assert_close(cache.levels.cpu(), torch.arange(256).float().div_(255), rtol=0, atol=0)
    assert len(cache.cache) == 5


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_eviction_fallback_and_shape_changes(device):
    check_gpu(device)
    cache = ImageTensorCache(
        ("base", "wrist"),
        1,
        (16, 16),
        torch.device(device),
        _decode_jpeg_window,
        resize_with_pad_torch,
        max_cache_bytes=2 * 3 * 16 * 16 * 4,
        max_pinned_bytes=0,
    )
    for seed in range(10):
        windows = {key: [jpeg(seed, (13 + seed, 19 + seed, 3))] for key in ("base", "wrist")}
        result = cache.prepare(windows)
        assert all(value.shape == (1, 1, 3, 16, 16) for value in result.values())
        assert len(cache.cache) <= 2
        assert (
            sum(entry[1].untyped_storage().nbytes() for entry in cache.cache.values()) <= 2 * 3 * 16 * 16 * 4
        )
        assert not cache.buffers
    cache.clear()
    assert not cache.cache


def test_invalid_windows_and_decode_failure():
    cache = ImageTensorCache(
        ("base",),
        2,
        (16, 16),
        torch.device("cpu"),
        _decode_jpeg_window,
        resize_with_pad_torch,
        max_cache_bytes=0,
    )
    with pytest.raises(ValueError, match="frame count"):
        cache.prepare({"base": []})
    with pytest.raises(ValueError, match="shapes differ"):
        cache.prepare({"base": [jpeg(1), jpeg(2, (19, 17, 3))]})
    with pytest.raises(ValueError, match="JPEG"):
        cache.prepare({"base": [jpeg(1), np.zeros(4, dtype=np.uint8)]})
    cache.prepare({"base": [jpeg(1), jpeg(2)]})
    assert not cache.cache


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_current_and_both_cameras_share_one_decode_batch(device):
    check_gpu(device)
    calls = []
    def decode(images):
        calls.append([image.tobytes() for image, _key in images])
        return _decode_jpeg_window(images, writable=False)
    cache = ImageTensorCache(("base", "wrist"), 2, (16,16), torch.device(device),
                             decode, resize_with_pad_torch)
    a,b,c,d = [jpeg(i) for i in range(4)]
    windows = {"base":[a,b], "wrist":[c,d]}
    first = cache.prepare(windows, current=(b,d))
    assert len(calls)==1 and len(calls[0])==4
    second = cache.prepare(windows, current=(b,d))
    assert len(calls)==1
    for key in windows:
        torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)
    with pytest.raises(ValueError,match="JPEG"):
        cache.prepare(windows,current=(np.asarray([1,2,3],dtype=np.uint8),))


def test_predecoded_current_pixels_are_reused_without_decode():
    a,b=jpeg(1),jpeg(2)
    calls=[]
    def decode(images):
        calls.extend(image.tobytes() for image,_key in images)
        return _decode_jpeg_window(images,writable=False)
    cache=ImageTensorCache(("base",),2,(16,16),torch.device("cpu"),decode,resize_with_pad_torch)
    rgb=_decode_jpeg_window([(b,"base")],writable=False)[0]
    cache.prepare({"base":[a,b]},current=(b,),decoded={b.tobytes():rgb})
    assert calls==[a.tobytes()]


@pytest.mark.parametrize(
    "dtype,expected",
    [(None, True), ("float32", True), ("float64", True), ("float16", False), ("bfloat16", False)],
)
def test_pipeline_pixel_contract(dtype, expected):
    pipeline = SimpleNamespace(
        steps=[RenameObservationsProcessorStep(), DeviceProcessorStep(float_dtype=dtype)]
    )
    assert images_are_unchanged(pipeline) == expected
    normalizer = NormalizerProcessorStep.__new__(NormalizerProcessorStep)
    normalizer.features = {"base": SimpleNamespace(type=FeatureType.VISUAL)}
    normalizer.norm_map = {FeatureType.VISUAL: NormalizationMode.MEAN_STD}
    pipeline.steps.append(normalizer)
    assert not images_are_unchanged(pipeline)


def test_unknown_image_processors_disable_cache():
    assert not images_are_unchanged(SimpleNamespace(steps=[SimpleNamespace()]))


def test_async_uploads_and_cached_images_across_streams():
    check_gpu("cuda")
    cache = ImageTensorCache(
        ("base", "wrist"), 3, (16, 16), torch.device("cuda"), _decode_jpeg_window, resize_with_pad_torch
    )
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    results = []
    for seed in range(12):
        windows = {key: [jpeg(seed), jpeg(seed + 1), jpeg(seed + 2)] for key in ("base", "wrist")}
        with torch.cuda.stream(streams[seed % 2]):
            results.append((windows, cache.prepare(windows)))
    torch.cuda.current_stream().wait_event(cache.ready)
    model = policy("cuda", True)
    for windows, prepared in results:
        expected, actual = normalized(model, original_windows(windows)), normalized(model, prepared)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert sum(buffer.numel() for buffer, _event in cache.buffers.values()) <= 2 * cache.max_pinned_bytes


def test_decode_failure_preserves_cross_stream_dependencies():
    check_gpu("cuda")
    cache = ImageTensorCache(
        ("base", "wrist"), 1, (16, 16), torch.device("cuda"), _decode_jpeg_window, resize_with_pad_torch
    )
    first, second = torch.cuda.Stream(), torch.cuda.Stream()
    base, wrist = jpeg(51), jpeg(52)
    with torch.cuda.stream(first):
        torch.cuda._sleep(10_000_000)
        with pytest.raises(ValueError, match="JPEG"):
            cache.prepare({"base": [base], "wrist": [np.zeros(4, dtype=np.uint8)]})
    windows = {"base": [base], "wrist": [wrist]}
    with torch.cuda.stream(second):
        result = cache.prepare(windows)
    torch.cuda.current_stream().wait_event(cache.ready)
    model = policy("cuda", False)
    expected = normalized(model, original_windows(windows))
    actual = normalized(model, result)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
