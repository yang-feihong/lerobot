"""Flat, content-deduplicated JPEG windows with independent source metadata."""

import numpy as np

FORMAT = "jpeg-table-v1"
FIELDS = ("capture_ns", "base_capture_ns", "wrist_capture_ns", "encoded_ns", "server_received_ns", "sim_step")
PREFIX = "model_images_"


def encode_image_table(history, current):
    if int(history["format_version"]) != 1:
        raise ValueError("Unsupported source image history version")
    blobs, lookup = [], {}

    def intern(value):
        content = np.asarray(value, dtype=np.uint8).tobytes()
        if content not in lookup:
            lookup[content] = len(blobs)
            blobs.append(np.frombuffer(content, dtype=np.uint8))
        return lookup[content]

    count = len(history["is_pad"])
    values = {PREFIX + "format": np.asarray(FORMAT)}
    for name in ("source_step", "base_camera_key", "wrist_camera_key", "is_pad", *FIELDS):
        values[PREFIX + name] = history[name]
    for role in ("base", "wrist"):
        values[PREFIX + role + "_indices"] = np.asarray(
            [intern(history[f"{role}_jpeg_{i}"]) for i in range(count)], dtype=np.int64
        )
    values[PREFIX + "current_indices"] = np.asarray([intern(image) for image in current], dtype=np.int64)
    values[PREFIX + "blob_count"] = np.int64(len(blobs))
    values.update({PREFIX + f"blob_{i}": blob for i, blob in enumerate(blobs)})
    return values


def _scalar(value, kind):
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in kind:
        raise ValueError("Invalid image-table scalar")
    return array.item()


def _frozen(value):
    array = np.asarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def decode_image_table(values):
    if _scalar(values[PREFIX + "format"], "U") != FORMAT:
        raise ValueError("Unsupported image-table format")
    pad = np.asarray(values[PREFIX + "is_pad"])
    if pad.dtype != np.bool_ or pad.ndim != 1 or not 1 <= len(pad) <= 64:
        raise ValueError("Invalid image-table padding/frame count")
    count = len(pad)
    blob_count = _scalar(values[PREFIX + "blob_count"], "iu")
    if not 1 <= blob_count <= 2 * (count + 1):
        raise ValueError("Invalid image-table blob count")
    expected = {
        PREFIX + name
        for name in (
            "format",
            "source_step",
            "base_camera_key",
            "wrist_camera_key",
            "is_pad",
            *FIELDS,
            "base_indices",
            "wrist_indices",
            "current_indices",
            "blob_count",
        )
    }
    expected.update(PREFIX + f"blob_{i}" for i in range(blob_count))
    if {key for key in values if key.startswith(PREFIX)} != expected:
        raise ValueError("Unexpected or missing image-table fields")
    source_step = _scalar(values[PREFIX + "source_step"], "iu")
    if not 0 <= source_step <= np.iinfo(np.int64).max:
        raise ValueError("Invalid image-table source step")
    history = dict(
        format_version=np.int64(1),
        source_step=np.int64(source_step),
        base_camera_key=_frozen(np.asarray(_scalar(values[PREFIX + "base_camera_key"], "U"))),
        wrist_camera_key=_frozen(np.asarray(_scalar(values[PREFIX + "wrist_camera_key"], "U"))),
        is_pad=_frozen(pad),
    )
    for name in FIELDS:
        value = np.asarray(values[PREFIX + name])
        if value.shape != (count,) or value.dtype.kind not in "iu":
            raise ValueError(f"Invalid image-table {name}")
        history[name] = _frozen(value)
    blobs = []
    for i in range(blob_count):
        value = np.asarray(values[PREFIX + f"blob_{i}"])
        if value.dtype != np.uint8 or value.ndim != 1 or not 0 < value.size <= 4 * 1024 * 1024:
            raise ValueError("Invalid image-table JPEG")
        blobs.append(_frozen(value))
    used = set()

    def references(name, size):
        indices = np.asarray(values[PREFIX + name])
        if indices.dtype.kind not in "iu" or indices.shape != (size,):
            raise ValueError("Invalid image-table reference layout")
        if np.any(indices < 0) or np.any(indices >= blob_count):
            raise ValueError("Image-table reference out of bounds")
        used.update(int(i) for i in indices)
        return [blobs[int(i)] for i in indices]

    for role in ("base", "wrist"):
        for i, image in enumerate(references(role + "_indices", count)):
            history[f"{role}_jpeg_{i}"] = image
    current = references("current_indices", 2)
    if used != set(range(blob_count)):
        raise ValueError("Unreferenced image-table JPEG")
    return history, current
