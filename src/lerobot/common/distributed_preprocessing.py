"""Coordinate long CPU-only startup work without outstanding GPU collectives."""

from datetime import timedelta
from typing import Callable, TypeVar

import torch.distributed as dist

T = TypeVar("T")


def run_main_process_preprocessing(callback: Callable[[], T]) -> T | None:
    if not dist.is_initialized():
        return callback()

    # Exact episode statistics can exceed the training NCCL timeout. Keep that
    # timeout intact for real training collectives and wait on a CPU group here.
    group = dist.new_group(backend="gloo", timeout=timedelta(hours=6))
    result = None
    failure = None
    status = [None]
    try:
        if dist.get_rank() == 0:
            try:
                result = callback()
            except Exception as exc:
                failure = exc
                status[0] = f"{type(exc).__name__}: {exc}"
        dist.broadcast_object_list(status, src=0, group=group)
        if failure is not None:
            raise failure
        if status[0] is not None:
            raise RuntimeError(f"Main-process preprocessing failed: {status[0]}")
        return result
    finally:
        dist.destroy_process_group(group)
