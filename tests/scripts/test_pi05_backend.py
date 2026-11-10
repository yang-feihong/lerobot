from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from lerobot.scripts import pi05_vla_server as api
from lerobot.scripts.pi05_backend import Pi05Backend
from lerobot.scripts.pi05_array_wire import dump_arrays, load_arrays
from test_pi05_image_table import inputs, action


class Prepared(dict):
    def __bytes__(self):
        return dump_arrays(self)


def test_local_submit_never_parses_array_archive(monkeypatch):
    _, flat, _ = inputs()
    prepared = Prepared(flat)
    backend = Pi05Backend(SimpleNamespace(validate_input=Mock(), submit=Mock(return_value=True)), api)
    monkeypatch.setattr(api, "open_arrays", Mock(side_effect=AssertionError("Unexpected wire decode")))
    backend.submit(prepared)
    packet = backend.engine.submit.call_args.args[0]
    np.testing.assert_array_equal(packet.prepared_state, flat["model_state"])
    backend.engine.validate_input.assert_called_once_with(packet)


def test_local_poll_keeps_complete_output_and_skips_wire_encoding(monkeypatch):
    record = action(b"")
    backend = Pi05Backend(SimpleNamespace(latest_after=lambda after, wait: record), api)
    expected = load_arrays(api.encode_action_packet(record, binary_arrays=True))
    monkeypatch.setattr(api, "encode_action_packet", Mock(side_effect=AssertionError("Unexpected wire encode")))
    result = backend.poll(0, 20)
    assert result.keys() == expected.keys()
    for key in expected:
        assert result[key].dtype == expected[key].dtype
        np.testing.assert_array_equal(result[key], expected[key])
    backend.close()
    with pytest.raises(OSError):
        backend.poll(0)


def test_b2_only_output_profile_is_native_and_holds_arm():
    source = torch.tensor([[0.1, -0.2, 0.3]])
    names = ("b2_vx", "b2_vy", "b2_omega_z")
    actual = api._to_execution_actions(source, names, "ee_state_delta")
    output_names = api.execution_action_names("ee_state_delta")
    torch.testing.assert_close(actual[:, :3], source)
    assert actual[0, output_names.index("b2_active")] == 1
    for name in ("arm_active", "arm_reset", "gripper_target"):
        assert actual[0, output_names.index(name)] == 0


def test_b2_only_rtc_reanchors_without_an_ee_profile():
    engine = object.__new__(api.AsyncRTCPolicy)
    engine.model_action_names = ("b2_delta_x", "b2_delta_y", "b2_delta_yaw")
    engine._action_unnormalizer = engine._action_normalizer = None
    engine._apply_action_processor = lambda step, values: values
    engine._b2_global_pose = lambda packet: np.asarray([0.3, 0.4, 0.2])
    record = SimpleNamespace(b2_anchor=np.asarray([0.1, 0.2, -0.1]))
    values = torch.tensor([[0.2, -0.1, 0.3], [0.1, 0.0, 0.2]])
    expected = api._reanchor_b2_pose_delta(values,
        torch.as_tensor(record.b2_anchor, dtype=values.dtype),
        torch.tensor([0.3, 0.4, 0.2], dtype=values.dtype))
    torch.testing.assert_close(engine._reanchor_rtc_prefix(values, record, None), expected)


def test_known_rejection_is_distinct_from_ambiguous_backend_failure():
    _, flat, _ = inputs()
    engine = SimpleNamespace(validate_input=Mock(side_effect=ValueError("bad shape")), submit=Mock())
    backend = Pi05Backend(engine, api, input_error=lambda message: RuntimeError("rejected: " + message))
    with pytest.raises(RuntimeError, match="rejected: bad shape"):
        backend.submit(Prepared(flat))
    engine.submit.assert_not_called()
