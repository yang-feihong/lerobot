import hashlib

import numpy as np
import pytest

from lerobot.scripts.pi05_array_wire import dump_arrays, load_arrays
from lerobot.scripts.pi05_image_table import FIELDS, encode_image_table, decode_image_table
from lerobot.scripts.pi05_image_resources import SOURCES, ImageResources, resolve_recorded


def test_immutable_bytes_reuse_does_not_reuse_partial_or_reversed_owner():
    from lerobot.scripts.pi05_image_resources import image_bytes

    content = b"abcdef"
    image = np.frombuffer(content, dtype=np.uint8)
    assert image_bytes(image) is content
    assert image_bytes(image[1:]) == b"bcdef"
    assert image_bytes(image[::-1]) == b"fedcba"
    writable = image.copy()
    saved = image_bytes(writable)
    writable[:] = 0
    assert saved == content


def test_recommitting_an_inline_identity_with_changed_content_is_rejected():
    cache = ImageResources()
    original = values()
    digest = "a" * 64
    cache.commit(original, digest)
    changed = dict(original, model_images_blob_0=np.frombuffer(b"forged", dtype=np.uint8))
    with pytest.raises(ValueError, match="content mismatch"):
        cache.commit(changed, digest)


def values(images=(b"image-a", b"image-b"), step=1, epoch=7):
    history = dict(format_version=np.int64(1), source_step=np.int64(step),
        base_camera_key=np.asarray("base"), wrist_camera_key=np.asarray("wrist"),
        is_pad=np.asarray([False, False]))
    history.update({key:np.asarray([10,20], dtype=np.int64) for key in FIELDS})
    for role in ("base", "wrist"):
        for index, image in enumerate(images):
            history[f"{role}_jpeg_{index}"] = np.frombuffer(image, dtype=np.uint8)
    return dict(encode_image_table(history, (history['base_jpeg_1'], history['wrist_jpeg_1'])),
                control_epoch=np.int64(epoch), model_input_schema=np.asarray("schema"), sim_step=np.int64(step))


def commit(sender, receiver, original, packets):
    wire = sender.pack(original)
    payload = dump_arrays(wire)
    digest = hashlib.sha256(payload).hexdigest()
    expanded = receiver.expand(load_arrays(payload))
    for key in original:
        np.testing.assert_array_equal(expanded[key], original[key])
    decode_image_table(expanded)
    receiver.commit(wire, digest)
    sender.commit(wire, digest)
    packets[digest] = payload
    assert list(sender.entries) == list(receiver.entries)
    return wire, digest


def test_references_are_acknowledged_bounded_and_direct():
    sender, receiver, packets = ImageResources(), ImageResources(), {}
    first, first_sha = commit(sender, receiver, values(), packets)
    assert SOURCES not in first
    second, _ = commit(sender, receiver, values((b"image-b",b"image-c"),2), packets)
    assert second[SOURCES].tolist() == [first_sha+':1', '']
    third, _ = commit(sender, receiver, values((b"image-b",b"image-d"),3), packets)
    assert third[SOURCES][0] == first_sha+':1'
    restored = resolve_recorded(third, packets.__getitem__, load_arrays)
    expected = values((b"image-b",b"image-d"),3)
    for key in expected:
        np.testing.assert_array_equal(restored[key], expected[key])
    assert all(not value.flags.writeable for value in receiver.entries.values())


def test_unacknowledged_data_is_not_referenced():
    cache = ImageResources()
    assert SOURCES not in cache.pack(values())
    assert SOURCES not in cache.pack(values(step=2))


@pytest.mark.parametrize("limit,byte_limit", [(2,1000),(64,14),(0,1000)])
def test_eviction_is_identical_and_falls_back_to_inline(limit,byte_limit):
    sender, receiver, packets = ImageResources(limit,byte_limit), ImageResources(limit,byte_limit), {}
    for i in range(10):
        commit(sender,receiver,values((f'image-{i}'.encode(),f'image-{i+1}'.encode()),i),packets)
        assert sender.bytes == receiver.bytes <= byte_limit
        assert len(sender.entries) <= limit
    commit(sender,receiver,values(step=20),packets)


def test_missing_and_cross_epoch_references_fail_closed():
    sender, receiver, packets = ImageResources(), ImageResources(), {}
    commit(sender,receiver,values(),packets)
    wire=sender.pack(values(step=2))
    with pytest.raises(ValueError,match="Missing JPEG"):
        ImageResources().expand(wire)
    bad=dict(wire,control_epoch=np.int64(8))
    with pytest.raises(ValueError,match="Missing JPEG"):
        receiver.expand(bad)
    assert SOURCES not in sender.pack(values(epoch=8))
    sender.clear()
    assert not sender.entries and SOURCES not in sender.pack(values())


@pytest.mark.parametrize("kind", ["inline", "source", "dtype", "shape"])
def test_invalid_reference_layouts(kind):
    sender, receiver, packets = ImageResources(), ImageResources(), {}
    commit(sender,receiver,values(),packets)
    wire=sender.pack(values(step=2))
    if kind=='inline':
        wire['model_images_blob_0']=np.frombuffer(b'forged',dtype=np.uint8)
    elif kind=='source':
        wire[SOURCES]=np.asarray(['../bad',''])
    elif kind=='dtype':
        wire[SOURCES]=np.asarray([1,2])
    else:
        wire[SOURCES]=np.asarray([''])
    with pytest.raises(ValueError):
        receiver.expand(wire)


def test_recorded_references_reject_corruption_and_cross_epoch():
    sender, receiver, packets = ImageResources(), ImageResources(), {}
    commit(sender,receiver,values(),packets)
    wire=sender.pack(values(step=2))
    with pytest.raises(ValueError,match="checksum"):
        resolve_recorded(wire,lambda _digest:b'corrupt',load_arrays)
    wire['control_epoch']=np.int64(8)
    with pytest.raises(ValueError,match="identity"):
        resolve_recorded(wire,packets.__getitem__,load_arrays)
