"""Bounded typed arrays in one CRC-protected binary frame."""

import json
import math
import struct
from collections.abc import Mapping
from zlib import crc32

import numpy as np

MAGIC = b"AWB1"
FORMAT = "array-bundle-v1"
CONTENT_TYPE = "application/x-array-bundle"
MAX_BYTES = 32 * 1024 * 1024
MAX_HEADER = 65536
MAX_FIELDS = 256


def _dtype(value):
    try:
        dtype = np.dtype(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Invalid array dtype") from exc
    if dtype.hasobject or dtype.fields is not None or dtype.kind not in "biufUS":
        raise ValueError("Unsupported array dtype")
    return dtype


def dump_arrays(values):
    if len(values) > MAX_FIELDS:
        raise ValueError("Too many array fields")
    descriptions, chunks, size = [], [], 0
    for key, value in values.items():
        if not isinstance(key, str) or not key or len(key) > 256:
            raise ValueError("Invalid array field name")
        array = np.asarray(value)
        _dtype(array.dtype)
        if array.ndim > 8 or array.dtype.itemsize == 0:
            raise ValueError("Unsupported array layout")
        size += array.nbytes
        if size > MAX_BYTES:
            raise ValueError("Array frame exceeds limit")
        descriptions.append([key, array.dtype.str, list(array.shape), array.nbytes])
        chunks.append(array.tobytes(order="C"))
    header = json.dumps(descriptions, ensure_ascii=False, separators=(",", ":")).encode()
    if len(header) > MAX_HEADER or size + len(header) + 12 > MAX_BYTES:
        raise ValueError("Array frame exceeds limit")
    body = struct.pack("!I", len(header)) + header + b"".join(chunks)
    return MAGIC + struct.pack("!I", crc32(body)) + body


def load_arrays(payload, *, writable=True):
    if not isinstance(payload, bytes) or not 12 <= len(payload) <= MAX_BYTES or not payload.startswith(MAGIC):
        raise ValueError("Invalid array frame")
    if crc32(memoryview(payload)[8:]) != struct.unpack("!I", payload[4:8])[0]:
        raise ValueError("Array frame CRC mismatch")
    length = struct.unpack("!I", payload[8:12])[0]
    if not 0 < length <= MAX_HEADER or length + 12 > len(payload):
        raise ValueError("Invalid array header length")
    try:
        fields = json.loads(payload[12:12 + length])
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("Invalid array header") from exc
    if not isinstance(fields, list) or len(fields) > MAX_FIELDS:
        raise ValueError("Invalid array field count")
    result, offset = {}, 12 + length
    for field in fields:
        if not isinstance(field, list) or len(field) != 4:
            raise ValueError("Invalid array descriptor")
        key, descriptor, shape, size = field
        if not isinstance(key, str) or not key or len(key) > 256 or key in result or not isinstance(descriptor, str):
            raise ValueError("Invalid or duplicate array name/dtype")
        dtype = _dtype(descriptor)
        if (not isinstance(shape, list) or len(shape) > 8 or dtype.itemsize == 0
                or any(type(n) is not int or n < 0 for n in shape)
                or type(size) is not int or size < 0
                or math.prod(shape) * dtype.itemsize != size or offset + size > len(payload)):
            raise ValueError("Invalid array shape or byte count")
        array = np.frombuffer(payload, dtype=dtype, count=math.prod(shape), offset=offset).reshape(shape)
        result[key] = array.copy() if writable else array
        offset += size
    if offset != len(payload):
        raise ValueError("Trailing array data")
    return result


class ArrayArchive(Mapping):
    def __init__(self, payload):
        self.values = load_arrays(payload, writable=False)
        self.files = list(self.values)

    def __getitem__(self, key):
        return self.values[key].copy()

    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)

    def __enter__(self):
        return self

    def close(self):
        pass

    def __exit__(self, *_args):
        pass
