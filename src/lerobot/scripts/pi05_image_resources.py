"""Bounded JPEG references to previously accepted, immutable input packets."""

from collections import OrderedDict
import hashlib
import re
import threading

import numpy as np

FEATURE = "jpeg-input-references-v1"
SOURCES = "model_images_blob_sources"


def image_bytes(value):
    image = np.asarray(value)
    owner = image.base
    if (image.dtype == np.uint8 and image.ndim == 1 and image.flags.c_contiguous
            and isinstance(owner, bytes) and len(owner) == image.nbytes):
        return owner
    return image.tobytes()


def identity(values):
    epoch, schema = np.asarray(values["control_epoch"]), np.asarray(values["model_input_schema"])
    if epoch.shape != () or epoch.dtype.kind not in "iu" or schema.shape != () or schema.dtype.kind != "U":
        raise ValueError("Invalid image resource identity")
    return int(epoch), str(schema.item())


def source_identity(value):
    match = re.fullmatch(r"([0-9a-f]{64}):(0|[1-9][0-9]{0,2})", value)
    if match is None or int(match[2]) >= 130:
        raise ValueError("Invalid JPEG input reference")
    return match[1], int(match[2])


def expand(values, fetch):
    if SOURCES not in values:
        return values
    count = np.asarray(values["model_images_blob_count"])
    if count.shape != () or count.dtype.kind not in "iu" or not 1 <= int(count) <= 130:
        raise ValueError("Invalid referenced JPEG count")
    sources = np.asarray(values[SOURCES])
    if sources.dtype.kind != "U" or sources.shape != (int(count),):
        raise ValueError("Invalid JPEG input references")
    result = dict(values)
    del result[SOURCES]
    for index, source in enumerate(sources.tolist()):
        key = f"model_images_blob_{index}"
        if source:
            source_identity(source)
            if key in values:
                raise ValueError("Ambiguous inline/referenced JPEG")
            result[key] = fetch(source)
        elif key not in values:
            raise ValueError("Missing inline JPEG")
    return result


def resolve_recorded(values, load_payload, decode_arrays):
    def fetch(source):
        digest, index = source_identity(source)
        payload = load_payload(digest)
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("JPEG source input checksum mismatch")
        parent = decode_arrays(payload)
        if identity(parent) != identity(values) or int(parent["sim_step"]) > int(values["sim_step"]):
            raise ValueError("JPEG source input identity mismatch")
        key = f"model_images_blob_{index}"
        count = np.asarray(parent["model_images_blob_count"])
        if count.shape != () or count.dtype.kind not in "iu" or not index < int(count):
            raise ValueError("JPEG source index outside its original table")
        if key not in parent or (SOURCES in parent and parent[SOURCES][index]):
            raise ValueError("JPEG references must point directly to inline data")
        return parent[key]
    return expand(values, fetch)


class ImageResources:
    def __init__(self, max_entries=64, max_bytes=32 * 1024 * 1024):
        self.limit, self.byte_limit = max_entries, max_bytes
        self.entries = OrderedDict()
        self.lookup = {}
        self.bytes = 0
        self.identity = None
        self.lock = threading.RLock()

    def clear(self):
        with self.lock:
            self.entries.clear()
            self.lookup.clear()
            self.bytes = 0
            self.identity = None

    def pack(self, values):
        with self.lock:
            if identity(values) != self.identity:
                return values
            result, sources = dict(values), []
            for index in range(int(values["model_images_blob_count"])):
                key = f"model_images_blob_{index}"
                source = self.lookup.get(image_bytes(values[key]), "")
                sources.append(source)
                if source:
                    del result[key]
            if any(sources):
                result[SOURCES] = np.asarray(sources)
            return result

    def expand(self, values):
        with self.lock:
            def fetch(source):
                if identity(values) != self.identity or source not in self.entries:
                    raise ValueError("Missing JPEG input reference; start a new session")
                return self.entries[source]
            return expand(values, fetch)

    def commit(self, values, digest):
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid acknowledged input checksum")
        with self.lock:
            resolved = self.expand(values)
            current_identity = identity(values)
            if current_identity != self.identity:
                self.clear()
                self.identity = current_identity
            sources = values.get(SOURCES)
            for index in range(int(values["model_images_blob_count"])):
                source = str(sources[index]) if sources is not None and sources[index] else f"{digest}:{index}"
                image = np.asarray(resolved[f"model_images_blob_{index}"])
                if image.dtype != np.uint8 or image.ndim != 1 or not 0 < image.size <= 4 * 1024 * 1024:
                    raise ValueError("Invalid JPEG resource")
                content = image_bytes(image)
                if source in self.entries:
                    if image is not self.entries[source] and image_bytes(self.entries[source]) != content:
                        raise ValueError("JPEG input reference content mismatch")
                    self.entries.move_to_end(source)
                else:
                    self.entries[source] = np.frombuffer(content, dtype=np.uint8)
                    self.bytes += len(content)
                self.lookup[content] = source
                while len(self.entries) > self.limit or self.bytes > self.byte_limit:
                    evicted, image = self.entries.popitem(last=False)
                    content = image_bytes(image)
                    self.bytes -= len(content)
                    if self.lookup.get(content) == evicted:
                        del self.lookup[content]
