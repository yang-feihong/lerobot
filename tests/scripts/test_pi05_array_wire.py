import json
import struct
from zlib import crc32

import numpy as np
import pytest

from lerobot.scripts.pi05_array_wire import MAGIC, dump_arrays, load_arrays


@pytest.mark.parametrize("value", [
    np.asarray(2), np.asarray(True), np.asarray("腕部"), np.asarray(b"jpeg"),
    np.arange(24).reshape(4, 6)[:, ::2], np.asfortranarray(np.arange(12).reshape(3, 4)),
    np.asarray([0., -0., np.nan], dtype=">f8"), np.zeros((0, 3), dtype=np.float32),
    np.arange(3, dtype=np.uint8), np.asarray([True, False]),
])
def test_roundtrip_preserves_bits_and_independent_ownership(value):
    payload = dump_arrays({"数据": value})
    first, second = load_arrays(payload), load_arrays(payload)
    for actual in (first["数据"], second["数据"]):
        assert actual.dtype == value.dtype and actual.shape == value.shape
        assert actual.tobytes() == value.tobytes()
        assert actual.flags.writeable
    view = load_arrays(payload, writable=False)["数据"]
    with pytest.raises(ValueError):
        view.setflags(write=True)


def frame(fields, data=b""):
    header = json.dumps(fields).encode()
    body = struct.pack("!I", len(header)) + header + data
    return MAGIC + struct.pack("!I", crc32(body)) + body


@pytest.mark.parametrize("fields,data", [
    ({}, b""), (None, b""), ([["a", "O", [], 8]], b"0" * 8),
    ([["a", "f8", [-1], 8]], b"0" * 8), ([["a", "u1", [True], 1]], b"0"),
    ([["a", "u1", [1], True]], b"0"), ([["a", "u1", [2], 2]], b"0"),
    ([["a", "u1", [1], 1]], b"00"), ([["a", "V8", [1], 8]], b"0" * 8),
    ([["a", "u1", [1], 1], ["a", "u1", [1], 1]], b"00"),
    ([["a", "u1", [1] * 9, 1]], b"0"), ([["a", "u1", [1 << 60], 1]], b"0"),
    ([["a", {}, [], 1]], b"0"), ([["a", "u1", [1], 1, 2]], b"0"),
    ([["", "u1", [1], 1]], b"0"),
])
def test_bad_descriptors_fail_closed(fields, data):
    with pytest.raises(ValueError):
        load_arrays(frame(fields, data))


def test_crc_truncation_version_and_limits():
    payload = dump_arrays({"x": np.arange(5)})
    bad = bytearray(payload); bad[-1] ^= 1
    for value in (bytes(bad), payload[:-1], b"AWB2" + payload[4:], b"AWB1", b"AWB1" + b"0" * 20):
        with pytest.raises(ValueError):
            load_arrays(value)
    with pytest.raises(ValueError):
        dump_arrays({str(i): i for i in range(257)})
    with pytest.raises(ValueError):
        dump_arrays({"object": np.asarray(object())})
    assert load_arrays(dump_arrays({})) == {}
