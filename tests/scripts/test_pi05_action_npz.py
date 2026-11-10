from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pytest

from lerobot.scripts.pi05_npz import dump_action_arrays, open_arrays


@pytest.mark.parametrize("value", [
    np.asarray(3), np.asarray("中文"), np.asarray(True), np.empty((0, 3)),
    np.zeros((1, 0, 2)), np.arange(30).reshape(5, 6)[:, ::2],
    np.asfortranarray(np.arange(12).reshape(3, 4)),
    np.asarray([1., -0., np.nan, np.inf], dtype=">f8"),
    np.asarray(["2026-11-09"], dtype="datetime64[D]"),
    np.asarray([2], dtype="timedelta64[s]"), np.zeros(3, dtype="V0"),
    np.ones(1, dtype=[("field", "f4")]),
])
def test_action_npz_matches_numpy_and_private_decoder(value):
    values = {"结果": value, "empty": np.empty(0, dtype=np.uint8)}
    payload = dump_action_arrays(values)
    with ZipFile(BytesIO(payload)) as archive:
        assert archive.testzip() is None
    with np.load(BytesIO(payload), allow_pickle=False) as standard, open_arrays(payload) as custom:
        for key, expected in values.items():
            for actual in (standard[key], custom[key]):
                assert actual.shape == expected.shape and actual.dtype == expected.dtype
                assert actual.tobytes() == expected.tobytes()


def test_empty_archive_is_valid():
    with np.load(BytesIO(dump_action_arrays({})), allow_pickle=False) as archive:
        assert archive.files == []
