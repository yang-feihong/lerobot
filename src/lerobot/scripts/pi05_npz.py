"""NPZ encoding and writable array decoding with bounded header caches."""

from functools import lru_cache
from io import BytesIO
from collections.abc import Mapping
import math
import struct
from zipfile import ZipFile
from zlib import crc32

import numpy as np
from lerobot.scripts.pi05_array_wire import MAGIC as ARRAY_MAGIC, ArrayArchive as WireArchive

_MAX_HEADER_SIZE = 10_000


@lru_cache(maxsize=128)
def _output_header(dtype, shape, fortran):
    output = BytesIO()
    fields = dict(descr=np.lib.format.dtype_to_descr(dtype), fortran_order=fortran, shape=shape)
    try:
        np.lib.format.write_array_header_1_0(output, fields)
    except ValueError:
        output = BytesIO()
        np.lib.format.write_array_header_2_0(output, fields)
    return output.getvalue()


def dump_arrays(values):
    arrays = {key: np.asanyarray(value) for key, value in values.items()}
    output = BytesIO()
    if any(value.dtype.hasobject or value.dtype.fields is not None for value in arrays.values()):
        np.savez(output, **values)
        return output.getvalue()
    with ZipFile(output, mode="w", allowZip64=True) as archive:
        for key, value in arrays.items():
            fortran = bool(value.flags.f_contiguous and not value.flags.c_contiguous)
            header = _output_header(value.dtype, value.shape, fortran)
            with archive.open(key + ".npy", "w", force_zip64=True) as target:
                target.write(header)
                if value.size:
                    data = memoryview(value).cast("B") if value.flags.c_contiguous and value.dtype.kind not in "mM" else value.tobytes(order="F" if fortran else "C")
                    target.write(data)
    return output.getvalue()


def dump_action_arrays(values):
    """Write small stored NPZ members without per-member stream objects."""
    arrays = {key: np.asarray(value) for key, value in values.items()}
    if (len(arrays) > 256 or sum(value.nbytes for value in arrays.values()) > 16 * 1024 * 1024
            or any(value.dtype.hasobject or value.dtype.fields is not None for value in arrays.values())):
        return dump_arrays(values)
    names = {key: (key + ".npy").encode("utf-8") for key in arrays}
    if any(len(name) > 65535 for name in names.values()):
        return dump_arrays(values)
    output, directory = BytesIO(), []
    for key, value in arrays.items():
        name = names[key]
        fortran = bool(value.flags.f_contiguous and not value.flags.c_contiguous)
        header = _output_header(value.dtype, value.shape, fortran)
        data = (memoryview(value).cast("B") if value.flags.c_contiguous and value.dtype.kind not in "mM"
                else value.tobytes(order="F" if fortran else "C")) if value.nbytes else b""
        size = len(header) + value.nbytes
        checksum = crc32(data, crc32(header))
        offset = output.tell()
        output.write(struct.pack("<I5H3I2H", 0x04034B50, 20, 0x800, 0, 0, 33,
                                 checksum, size, size, len(name), 0))
        output.write(name)
        output.write(header)
        output.write(data)
        directory.append(struct.pack("<I6H3I5H2I", 0x02014B50, 20, 20, 0x800, 0, 0, 33,
                                     checksum, size, size, len(name), 0, 0, 0, 0, 0, offset) + name)
    start = output.tell()
    output.write(b"".join(directory))
    output.write(struct.pack("<I4H2IH", 0x06054B50, 0, 0, len(arrays), len(arrays),
                             output.tell() - start, start, 0))
    return output.getvalue()


@lru_cache(maxsize=128)
def _header(prefix):
    source = BytesIO(prefix)
    version = np.lib.format.read_magic(source)
    reader = {
        (1, 0): np.lib.format.read_array_header_1_0,
        (2, 0): np.lib.format.read_array_header_2_0,
    }[version]
    shape, fortran, dtype = reader(source)
    if dtype.hasobject:
        raise ValueError("Object arrays cannot be loaded when allow_pickle=False")
    return shape, fortran, dtype


def _member(archive, info):
    with archive.open(info) as source:
        magic = source.read(8)
        if not magic.startswith(b"\x93NUMPY"):
            return magic + source.read()
        version = tuple(magic[6:])
        if version not in {(1, 0), (2, 0)}:
            source.seek(0)
            return np.lib.format.read_array(source, allow_pickle=False)
        length_size = 2 if version == (1, 0) else 4
        length = source.read(length_size)
        if len(length) != length_size:
            raise ValueError("Truncated NPY header length")
        header_size = struct.unpack("<H" if length_size == 2 else "<I", length)[0]
        if header_size > _MAX_HEADER_SIZE:
            raise ValueError("NPY header exceeds size limit")
        header = source.read(header_size)
        if len(header) != header_size:
            raise ValueError("Truncated NPY header")
        prefix = magic + length + header
        shape, fortran, dtype = _header(prefix)
        if dtype.fields is not None or dtype.subdtype is not None or dtype.itemsize == 0:
            source.seek(0)
            return np.lib.format.read_array(source, allow_pickle=False)
        if any(type(size) is not int or size < 0 for size in shape):
            raise ValueError("Invalid NPY shape")
        count = math.prod(shape)
        byte_count = count * dtype.itemsize
        remaining = info.file_size - len(prefix)
        if byte_count > remaining:
            raise ValueError("Truncated NPY array data")
        if byte_count != remaining:
            source.seek(0)
            return np.lib.format.read_array(source, allow_pickle=False)
        data = source.read(byte_count)
        if len(data) != byte_count:
            raise ValueError("Truncated NPY array data")
        return np.frombuffer(data, dtype=dtype, count=count).reshape(
            shape, order="F" if fortran else "C"
        ).copy(order="K")


class ArrayArchive(Mapping):
    def __init__(self, payload):
        self._source = BytesIO(payload)
        self.zip = ZipFile(self._source)
        self._members = self.zip.namelist()
        self.files = [name[:-4] if name.endswith(".npy") else name for name in self._members]

    def __getitem__(self, key):
        if key in self._members:
            member = key
        elif key in self.files:
            member = key + ".npy"
        else:
            raise KeyError(key)
        return _member(self.zip, self.zip.getinfo(member))

    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)

    def __contains__(self, key):
        return key in self._members or key in self.files

    def close(self):
        if self.zip is not None:
            self.zip.close()
            self.zip = None
            self._source.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def open_arrays(payload):
    if isinstance(payload, bytes) and payload.startswith(ARRAY_MAGIC):
        return WireArchive(payload)
    return ArrayArchive(payload)


def load_arrays(payload):
    with open_arrays(payload) as arrays:
        result = {}
        for key in arrays.files:
            value = arrays[key]
            result[key] = value if isinstance(value, np.ndarray) and value.dtype.itemsize else np.array(value)
        return result
