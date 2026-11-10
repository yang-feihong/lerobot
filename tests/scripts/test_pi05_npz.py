from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import struct
from zipfile import ZipFile, ZIP_STORED

import numpy as np
import pytest

from lerobot.scripts.pi05_npz import dump_arrays, load_arrays, open_arrays, _header


def packet(values, compressed=False):
    output = BytesIO()
    (np.savez_compressed if compressed else np.savez)(output, **values)
    return output.getvalue()


def baseline(payload):
    with np.load(BytesIO(payload), allow_pickle=False) as values:
        return {key: np.array(values[key]) for key in values.files}


def assert_equal(payload):
    expected, actual = baseline(payload), load_arrays(payload)
    assert list(actual) == list(expected)
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])
        assert actual[key].dtype == expected[key].dtype
        assert actual[key].flags.writeable
        assert actual[key].flags.f_contiguous == expected[key].flags.f_contiguous
        assert actual[key].flags.c_contiguous == expected[key].flags.c_contiguous
    return actual


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("value", [
    np.asarray(4), np.asarray("抓取水瓶"), np.arange(30).reshape(5, 6),
    np.asfortranarray(np.arange(30).reshape(5, 6)), np.ones((2, 3), dtype=">f8"),
    np.arange(12, dtype=np.uint8)[::-2], np.zeros((0, 3)), np.asarray(True),
    np.array([1 + 2j, np.nan + 1j]), np.zeros(4, dtype="S0"),
    np.zeros(4, dtype="U0"), np.zeros(3, dtype=[("q", ">f4"), ("flag", "i8")]),
    np.zeros(3, dtype=[("机械臂", "f4")]), np.zeros(3, dtype="datetime64[ns]"),
])
def test_matches_numpy(value, compressed):
    assert_equal(packet({"value": value, "names": np.asarray(["image", "腕部"])}, compressed))


def test_cached_header_does_not_share_writable_data():
    payload = packet({"value": np.arange(20)})
    first, second = load_arrays(payload), load_arrays(payload)
    first["value"][:] = 17
    np.testing.assert_array_equal(second["value"], np.arange(20))
    np.testing.assert_array_equal(load_arrays(payload)["value"], np.arange(20))


def test_concurrent_readers_are_independent():
    payload = packet({"value": np.arange(20)})
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(load_arrays, [payload] * 100))
    for item in values:
        np.testing.assert_array_equal(item["value"], np.arange(20))
    assert not np.shares_memory(values[0]["value"], values[1]["value"])


def test_empty_and_nested_and_ambiguous_member_names():
    for values in ({}, {"nested/value": np.arange(3)}, {"x": np.arange(3), "x.npy": np.arange(7)}):
        assert_equal(packet(values))


def test_duplicate_and_non_npy_members_follow_numpy_lookup():
    content = BytesIO()
    np.save(content, np.arange(3))
    output = BytesIO()
    with ZipFile(output, "w") as archive:
        archive.writestr("x.npy", content.getvalue())
        archive.writestr("notes", b"notes")
        with pytest.warns(UserWarning):
            archive.writestr("x.npy", content.getvalue())
    assert_equal(output.getvalue())


def raw_member(content):
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_STORED) as archive:
        archive.writestr("value.npy", content)
    return output.getvalue()


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
def test_npy_versions(version):
    output = BytesIO()
    np.lib.format.write_array(output, np.arange(30).reshape(5, 6), version=version)
    assert_equal(raw_member(output.getvalue()))


def test_non_npy_content_in_npy_member():
    assert_equal(raw_member(b"not a numpy array"))


def test_pickle_is_rejected():
    payload = packet({"value": np.asarray([{"q": 3}], dtype=object)})
    for reader in (baseline, load_arrays):
        with pytest.raises(ValueError, match="Object arrays"):
            reader(payload)


def test_crc_is_checked():
    original = packet({"value": np.arange(10000, dtype=np.int64)})
    load_arrays(original)
    with ZipFile(BytesIO(original)) as archive:
        info = archive.infolist()[0]
    data = bytearray(original)
    names, extra = struct.unpack_from("<HH", data, info.header_offset + 26)
    data[info.header_offset + 30 + names + extra + info.file_size - 1] ^= 1
    from zipfile import BadZipFile
    for reader in (baseline, load_arrays):
        with pytest.raises(BadZipFile):
            reader(bytes(data))


@pytest.mark.parametrize("cut", [7, 10, 90, 139])
def test_truncated_npy_is_rejected(cut):
    output = BytesIO()
    np.save(output, np.arange(20))
    payload = raw_member(output.getvalue()[:cut])
    for reader in (baseline, load_arrays):
        with pytest.raises((ValueError, EOFError)):
            reader(payload)


def test_header_limits_and_impossible_shape():
    for count in (2**40, -3):
        output = BytesIO()
        np.lib.format.write_array_header_1_0(output, dict(descr="<f4", fortran_order=False, shape=(count,)))
        with pytest.raises(ValueError):
            load_arrays(raw_member(output.getvalue()))
    payload = raw_member(b"\x93NUMPY\x01\x00" + struct.pack("<H", 10001) + b" " * 10001)
    with pytest.raises(ValueError, match="size limit"):
        load_arrays(payload)


def test_header_cache_is_bounded():
    for size in range(1, 180):
        load_arrays(packet({"value": np.zeros(1, dtype=f"U{size}")}))
    assert _header.cache_info().currsize <= 128


def test_lazy_archive_matches_numpy_lookup_and_closes():
    payload = packet({"x": np.arange(3), "x.npy": np.arange(7), "object": np.array([{}], dtype=object)})
    with open_arrays(payload) as actual, np.load(BytesIO(payload), allow_pickle=False) as expected:
        assert actual.files == expected.files
        assert list(actual) == list(expected)
        assert "object" in actual and "missing" not in actual
        for key in ["x", "x.npy", "x.npy.npy"]:
            np.testing.assert_array_equal(actual[key], expected[key])
        with pytest.raises(ValueError, match="Object arrays"):
            actual["object"]
        with pytest.raises(KeyError):
            actual["missing"]
    assert actual.zip is None


def test_lazy_archive_returns_opaque_members_as_bytes():
    payload = raw_member(b"not an array")
    with open_arrays(payload) as actual:
        assert actual["value"] == b"not an array"


@pytest.mark.parametrize("descr", ["|S0", "<U0", "|V0"])
def test_zero_itemsize_matches_numpy_array_conversion(descr):
    output = BytesIO()
    np.lib.format.write_array_header_1_0(output, dict(descr=descr, fortran_order=False, shape=(3,)))
    assert_equal(raw_member(output.getvalue()))

@pytest.mark.parametrize("value", [
    np.asarray(4), np.asarray("抓取水瓶"), np.arange(30).reshape(5, 6),
    np.asfortranarray(np.arange(30).reshape(5, 6)), np.ones((2, 3), dtype=">f8"),
    np.arange(12, dtype=np.uint8)[::-2], np.zeros((0, 3)), np.asarray(True),
    np.array([1 + 2j, np.nan + 1j]), np.zeros(4, dtype="S0"),
    np.zeros(4, dtype="U0"), np.zeros(3, dtype=[("q", ">f4"), ("flag", "i8")]),
    np.zeros(3, dtype=[("机械臂", "f4")]), np.zeros(3, dtype="datetime64[ns]"),
])
def test_encoder_is_readable_by_numpy(value):
    payload = dump_arrays(dict(value=value, names=np.asarray(["image", "腕部"])))
    expected = {"value": value, "names": np.asarray(["image", "腕部"])}
    actual = assert_equal(payload)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
        assert actual[key].dtype == expected[key].dtype


def test_encoder_snapshots_input_and_concurrent_headers():
    values = {"state": np.arange(58, dtype=np.float32)}
    payload = dump_arrays(values)
    values["state"][:] = -1
    np.testing.assert_array_equal(load_arrays(payload)["state"], np.arange(58))
    with ThreadPoolExecutor(max_workers=8) as pool:
        payloads = list(pool.map(lambda n: dump_arrays({"x": np.arange(n)}), range(1, 201)))
    for n, content in enumerate(payloads, 1):
        np.testing.assert_array_equal(load_arrays(content)["x"], np.arange(n))
