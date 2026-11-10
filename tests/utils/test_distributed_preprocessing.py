import json
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from lerobot.common import distributed_preprocessing as module


def test_serial_callback_result_and_original_failure():
    assert module.run_main_process_preprocessing(lambda: 17) == 17
    with pytest.raises(ValueError, match="original failure"):
        module.run_main_process_preprocessing(lambda: (_ for _ in ()).throw(ValueError("original failure")))


def test_long_work_uses_separate_cpu_group_without_default_barrier(monkeypatch):
    calls = []
    group = object()

    def new_group(**kwargs):
        assert kwargs == {"backend": "gloo", "timeout": timedelta(hours=6)}
        calls.append("new_group")
        return group

    def broadcast(status, *, src, group):
        assert src == 0 and status == [None]
        calls.append("broadcast")

    monkeypatch.setattr(module, "dist", SimpleNamespace(
        is_initialized=lambda: True,
        new_group=new_group,
        get_rank=lambda: 0,
        broadcast_object_list=broadcast,
        destroy_process_group=lambda value: calls.append("destroy") if value is group else pytest.fail(),
    ))
    assert module.run_main_process_preprocessing(lambda: calls.append("work") or 9) == 9
    assert calls == ["new_group", "work", "broadcast", "destroy"]


def _worker(rank, rendezvous, directory, fail):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    root = Path(directory)
    try:
        def work():
            time.sleep(0.5)
            if fail:
                raise ValueError("deliberate preprocessing failure")
            (root / "result.json").write_text(json.dumps({"ready": True}))
            return 23

        try:
            result = module.run_main_process_preprocessing(work)
        except Exception as exc:
            assert fail
            assert isinstance(exc, ValueError if rank == 0 else RuntimeError)
            assert "deliberate preprocessing failure" in str(exc)
            (root / f"rank{rank}.json").write_text(json.dumps({"failed": True}))
        else:
            assert not fail
            assert json.loads((root / "result.json").read_text()) == {"ready": True}
            assert result == (23 if rank == 0 else None)
            (root / f"rank{rank}.json").write_text(json.dumps({"failed": False}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("fail", [False, True])
def test_two_process_gate_and_failure_propagation(tmp_path, fail):
    mp.spawn(_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path), fail), nprocs=2, join=True)
    assert json.loads((tmp_path / "rank0.json").read_text())["failed"] is fail
    assert json.loads((tmp_path / "rank1.json").read_text())["failed"] is fail
