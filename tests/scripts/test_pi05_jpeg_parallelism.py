from threading import Barrier

import numpy as np

import lerobot.scripts.pi05_vla_server as server


def test_full_image_window_decodes_eight_jobs_concurrently_in_source_order(monkeypatch):
    barrier = Barrier(8, timeout=5)

    def decode(encoded, name, *, writable=True):
        barrier.wait()
        return np.asarray([int(name)])

    monkeypatch.setattr(server, "_decode_jpeg", decode)
    actual = server._decode_jpeg_window([(b"", str(index)) for index in range(8)])
    assert [int(value[0]) for value in actual] == list(range(8))
