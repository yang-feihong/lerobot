"""Pixel-exact CPU image conversion; build once during environment preparation.

This small C ABI library does not link Torch, CUDA or apxinf. Unsupported images
and installations without a built helper retain the original Torch path.
"""
from functools import lru_cache
import ctypes
import hashlib
import logging
import os
from pathlib import Path
import platform
import subprocess
import tempfile


def library_path():
    source = Path(__file__).with_name("pi05_rgb_fused.cpp")
    identity = platform.machine() + "-" + hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    cache = Path(os.environ.get("LEROBOT_CPU_IMAGE_OPS_CACHE",
                                str(Path(tempfile.gettempdir()) / "lerobot-cpu-image-ops")))
    return cache / identity / "rgb_fused.so"


def build():
    """Compile in a private temporary file; publish atomically after ABI loading."""
    target = library_path()
    if target.is_file():
        ctypes.CDLL(str(target)).rgb_u8_to_chw_f32
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="building-", suffix=".so", dir=target.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        subprocess.run(["c++", "-O3", "-std=c++17", "-shared", "-fPIC",
                        str(Path(__file__).with_name("pi05_rgb_fused.cpp")), "-o", str(temporary)],
                       check=True)
        ctypes.CDLL(str(temporary)).rgb_u8_to_chw_f32
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    _kernel.cache_clear()
    return target


@lru_cache(maxsize=1)
def _kernel():
    try:
        library = ctypes.CDLL(str(library_path()))
    except OSError:
        logging.getLogger(__name__).debug("CPU RGB helper not built; using the original Torch conversion")
        return None
    kernel = library.rgb_u8_to_chw_f32
    kernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    kernel.restype = None
    return kernel


@lru_cache(maxsize=1)
def _levels():
    import torch
    return torch.arange(256, dtype=torch.float32).div_(255.0).numpy()


def rgb_window(images):
    """Return owned float32 [1,T,3,H,W], or None for the original fallback."""
    import numpy as np
    import torch

    if not images or any(image.dtype != np.uint8 or image.ndim != 3
                         or image.shape[-1] != 3 or not image.flags.c_contiguous for image in images):
        return None
    height, width, _ = images[0].shape
    if min(height, width) < 1 or any(image.shape != images[0].shape for image in images):
        return None
    kernel = _kernel()
    if kernel is None:
        return None
    levels = _levels()
    result = np.empty((len(images), 3, height, width), dtype=np.float32)
    for index, image in enumerate(images):
        kernel(image.ctypes.data, result[index].ctypes.data, height * width, levels.ctypes.data)
    return torch.from_numpy(result).unsqueeze(0)


if __name__ == "__main__":
    print(build())
