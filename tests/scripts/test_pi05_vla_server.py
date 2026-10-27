import json
import time
from io import BytesIO
from threading import Event, Lock, Thread, get_ident
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.pi05.b2_action_transform import (
    CONTROL_EXTENDED_DATASET_ACTION_NAMES,
    EE_DELTA_ROTVEC_NAMES,
    HEIGHT_INVARIANT_EE_STATE_NAMES,
    action_schema_kwargs,
    b2_execution_action_names,
    b2_pose_delta_action_names,
    integrate_body_twist,
    se2_increment_to_body_twist,
)
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.rtc import RTCConfig, TrainingRTCConfig
from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from lerobot.scripts.pi05_vla_server import (
    B2_EXECUTION_VELOCITY_NAMES,
    B2_GLOBAL_POSE_NAMES,
    EE_DELTA_ACTION_NAMES,
    EE_POSE_ACTION_NAMES,
    PROTOCOL_VERSION,
    ActionRecord,
    AsyncRTCPolicy,
    ImagePacket,
    _apply_completion_stop,
    _b2_velocity_smoothing_transition_step,
    _camera_bindings,
    _checkpoint_hot_swap_signature,
    _decode_discrete_actions,
    _deployment_rtc_config,
    _load_checkpoint_contract,
    _reanchor_b2_pose_delta,
    _resolve_checkpoint_action_representations,
    _rtc_enabled_for_chunk_scheduling,
    _smooth_b2_execution_velocity,
    _to_execution_actions,
    decode_observation_packet,
    encode_action_packet,
    execution_action_names,
    _decode_model_image_history,
    _RecordedImageBatch,
    RolloutRecorder,
)

EXECUTION_ACTION_NAMES = execution_action_names("ee_delta")


def _complete_packet(**changes):
    jpeg = __import__("cv2").imencode(".jpg", np.zeros((4, 4, 3), dtype=np.uint8))[1]
    data = dict(protocol_version=np.int64(PROTOCOL_VERSION), control_epoch=np.int64(7),
        sim_step=np.int64(12), active_sequence=np.int64(-1), active_index=np.int64(0),
        task=np.asarray("test"), state=np.zeros(2, dtype=np.float32),
        state_names=np.asarray(["state_a", "state_b"]),
        executed_ee_target=np.zeros(len(EE_POSE_ACTION_NAMES), dtype=np.float32),
        executed_ee_target_names=np.asarray(EE_POSE_ACTION_NAMES),
        actual_ee_state=np.zeros(len(EE_POSE_ACTION_NAMES), dtype=np.float32),
        base_jpeg=jpeg, wrist_jpeg=jpeg, model_input_schema=np.asarray("test"),
        model_state=np.zeros((3, 2), dtype=np.float32), model_state_is_pad=np.array([True, False, False]),
        rtc_inference_delay_steps=np.int64(2), source_steps_per_wall_second=np.float64(5),
        model_image_history=np.zeros(0, dtype=np.uint8))
    data.update(changes)
    output = BytesIO()
    np.savez(output, **data)
    return output.getvalue()


def test_complete_packet_decoder_has_no_raw_stream_history():
    packet = decode_observation_packet(_complete_packet())
    assert packet.prepared_state.shape == (3, 2)
    assert packet.prepared_state_is_pad.tolist() == [True, False, False]
    assert packet.inference_delay_steps == 2
    assert packet.source_steps_per_wall_second == 5
    assert not hasattr(packet, "telemetry_history")
    with pytest.raises(ValueError, match="Raw sensor histories"):
        decode_observation_packet(_complete_packet(history_states=np.zeros((3, 2))))


@pytest.mark.parametrize("mode", ["SYNC", "RTC"])
def test_worker_decodes_request_scoped_scheduling_mode(mode):
    packet = decode_observation_packet(_complete_packet(chunk_scheduling_mode=np.asarray(mode)))
    assert packet.scheduling_mode == mode
    with pytest.raises(ValueError, match="request scheduling mode"):
        decode_observation_packet(_complete_packet(chunk_scheduling_mode=np.asarray("invalid")))


@pytest.mark.parametrize("changes", [
    {"model_state_is_pad": np.zeros(3, dtype=np.float32)},
    {"model_state_is_pad": np.zeros(2, dtype=np.bool_)},
    {"model_state": np.full((3, 2), np.nan)},
    {"rtc_inference_delay_steps": np.int64(-1)},
    {"source_steps_per_wall_second": np.float64(0)},
    {"source_steps_per_wall_second": np.float64(np.inf)},
])
def test_complete_packet_rejects_invalid_window_and_rtc_conditions(changes):
    with pytest.raises(ValueError):
        decode_observation_packet(_complete_packet(**changes))


def _prepared_images(base_key, wrist_key, frames, steps, pad, source_step):
    data = {"format_version": np.int64(1), "source_step": np.int64(source_step),
            "base_camera_key": np.asarray(base_key), "wrist_camera_key": np.asarray(wrist_key),
            "is_pad": np.asarray(pad, dtype=np.bool_)}
    for field in ("capture_ns", "base_capture_ns", "wrist_capture_ns", "encoded_ns", "server_received_ns", "sim_step"):
        data[field] = np.asarray(steps, dtype=np.int64)
    for index, (base, wrist) in enumerate(frames):
        data[f"base_jpeg_{index}"] = base
        data[f"wrist_jpeg_{index}"] = wrist
    output = BytesIO()
    np.savez(output, **data)
    return output.getvalue()


def test_prepared_image_window_is_consumed_without_sampling_or_cache():
    import cv2
    base = [cv2.imencode(".jpg", np.full((8, 8, 3), color, dtype=np.uint8))[1] for color in (10, 80)]
    engine = AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    engine.state_num_frames = 1
    engine.source_state_names = ("state",)
    engine.mem_image_num_frames = 3
    engine.mem_enabled = True
    engine.action_history_enabled = False
    engine.base_camera_key = "observation.images.base_0_rgb"
    engine.wrist_camera_key = "observation.images.left_wrist_0_rgb"
    engine.input_spec = lambda: {"schema_id": "test"}
    snapshot = _prepared_images(engine.base_camera_key, engine.wrist_camera_key,
        [(base[0], base[0]), (base[0], base[0]), (base[1], base[1])], [0, 0, 5], [True, False, False], 5)
    packet = SimpleNamespace(sim_step=5, task="Open the door.", input_schema="test",
        prepared_state=np.zeros((1, 1), dtype=np.float32), prepared_state_is_pad=np.zeros(1, dtype=np.bool_),
        model_image_history=snapshot)
    batch = engine._make_batch(packet)
    assert batch.model_image_history == snapshot
    assert batch[engine.base_camera_key + "_is_pad"].tolist() == [[True, False, False]]
    assert batch[engine.base_camera_key].shape == (1, 3, 3, 8, 8)
    assert not hasattr(engine, "_image_history")
    assert not hasattr(engine, "_observation_history")
    for changes in ({"source_step": 6}, {"count": 2}, {"wrist_key": "wrong"}):
        args = dict(source_step=5, count=3, base_key=engine.base_camera_key, wrist_key=engine.wrist_camera_key)
        args.update(changes)
        with pytest.raises(ValueError):
            _decode_model_image_history(snapshot, **args)


@pytest.mark.parametrize("wrist_first", [False, True])
def test_real_robot_jpeg_roles_reach_metadata_model_slots(wrist_first):
    import cv2
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    base_key = "observation.images.base_0_rgb"
    wrist_key = "observation.images.left_wrist_0_rgb"
    keys = [wrist_key, base_key] if wrist_first else [base_key, wrist_key]
    jpeg = lambda value: cv2.imencode(".jpg", np.full((224, 224, 3), value, dtype=np.uint8))[1]
    engine = AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    engine.state_num_frames = 1
    engine.source_state_names = ("state",)
    engine.mem_image_num_frames = 1
    engine.mem_enabled = engine.action_history_enabled = False
    engine.base_camera_key, engine.wrist_camera_key = _camera_bindings(tuple(keys))
    engine.input_spec = lambda: {"schema_id": "test"}
    packet = SimpleNamespace(sim_step=0, task="test", input_schema="test",
        prepared_state=np.zeros((1, 1), dtype=np.float32), prepared_state_is_pad=np.zeros(1, dtype=np.bool_),
        model_image_history=_prepared_images(base_key, wrist_key, [(jpeg(0), jpeg(255))], [0], [False], 0))
    batch = engine._make_batch(packet)
    policy = PI05Policy.__new__(PI05Policy)
    torch.nn.Module.__init__(policy)
    policy.register_parameter("dummy", torch.nn.Parameter(torch.zeros(1)))
    policy.config = PI05Config(device="cpu", dataset_camera_keys=keys,
        input_features={key: PolicyFeature(FeatureType.VISUAL, (3, 224, 224)) for key in reversed(keys)})
    images, _, _ = policy._preprocess_images(batch)
    assert [image.mean().item() for image in images] == [1.0 if key == wrist_key else -1.0 for key in keys]












def test_start_runs_compile_warmup_on_the_inference_worker() -> None:
    policy = AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    policy._stop = Event()
    policy._warmup_requested = Event()
    policy._warmup_finished = Event()
    policy._warmup_error = None
    policy._mailbox_condition = __import__("threading").Condition()
    policy._mailbox = None
    policy._mailbox_version = 0
    policy._last_inferred_version = 0
    policy._model_lock = Lock()
    warmup_thread_ids: list[int] = []

    def warmup() -> None:
        warmup_thread_ids.append(get_ident())
        policy._stop.set()

    policy._warmup = warmup
    policy._worker = Thread(target=policy._run)
    caller_thread_id = get_ident()

    policy.start()
    policy._worker.join(timeout=1.0)

    assert warmup_thread_ids
    assert warmup_thread_ids == [policy._worker.ident]
    assert warmup_thread_ids[0] != caller_thread_id


def test_inference_worker_immediately_consumes_an_observation_waiting_in_the_mailbox() -> None:
    policy = AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    policy._stop = Event()
    policy._warmup_requested = Event()
    policy._warmup_finished = Event()
    policy._warmup_error = None
    policy._mailbox_condition = __import__("threading").Condition()
    policy._mailbox = None
    policy._mailbox_version = 0
    policy._last_inferred_version = 0
    policy._records_lock = Lock()
    policy._records = {}
    policy._latest_record = None
    policy._inference_count = 0
    policy._last_error = None
    policy._last_error_traceback = None
    policy.recorder = SimpleNamespace(action=lambda _: None)
    first_started = Event()
    release_first = Event()
    second_started = Event()
    starts: list[float] = []

    def infer(packet: SimpleNamespace) -> SimpleNamespace:
        starts.append(time.monotonic())
        if packet.sim_step == 1:
            first_started.set()
            assert release_first.wait(timeout=1.0)
        else:
            second_started.set()
            policy._stop.set()
        return SimpleNamespace(
            sequence=packet.sim_step,
            source_step=packet.sim_step,
            inference_seconds=0.0,
            inference_delay_steps=0,
            processed=torch.zeros(1, 1),
        )

    policy._infer = infer
    worker = Thread(target=policy._run)
    worker.start()

    with policy._mailbox_condition:
        policy._mailbox = SimpleNamespace(sim_step=1)
        policy._mailbox_version += 1
        policy._mailbox_condition.notify()
    assert first_started.wait(timeout=1.0)
    with policy._mailbox_condition:
        policy._mailbox = SimpleNamespace(sim_step=2)
        policy._mailbox_version += 1
        policy._mailbox_condition.notify()
    released_at = time.monotonic()
    release_first.set()

    assert second_started.wait(timeout=1.0)
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert starts[1] - released_at < 0.1


def test_checkpoint_hot_swap_signature_requires_identical_runtime_schema(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    names = (
        "config.json",
        "adapter_config.json",
        "pi05_deployment_metadata.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    )
    for directory in (first, second):
        directory.mkdir()
        for name in names:
            (directory / name).write_text(json.dumps({"name": name, "schema": 1}))

    (first / "config.json").write_text(json.dumps({"max_action_dim": 32, "pretrained_path": "/base/model"}))
    (second / "config.json").write_text(
        json.dumps({"max_action_dim": 32, "pretrained_path": "/resume/checkpoint"})
    )

    assert _checkpoint_hot_swap_signature(first) == _checkpoint_hot_swap_signature(second)

    (second / "config.json").write_text(json.dumps({"max_action_dim": 64}))
    assert _checkpoint_hot_swap_signature(first) != _checkpoint_hot_swap_signature(second)


def test_checkpoint_hot_swap_signature_ignores_training_only_fields(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    processor = {
        "steps": [
            {
                "registry_name": "pi05_action_representation_processor",
                "config": {"representation": "pose_delta"},
            }
        ]
    }
    for directory in (first, second):
        directory.mkdir()
        (directory / "adapter_config.json").write_text(json.dumps({"r": 16}))
        (directory / "pi05_deployment_metadata.json").write_text(
            json.dumps(
                {
                    "action": {
                        "model_names": ["b2_delta_x"],
                        "flow_loss_schema": ("group_balanced" if directory == first else "group_balanced_v2"),
                    }
                }
            )
        )
        for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
            (directory / name).write_text(json.dumps(processor))

    (first / "config.json").write_text(
        json.dumps(
            {
                "architecture": "pi05",
                "action_bool_true_fractions": {"arm_reset": 0.01},
                "action_supervise_terminal_static_padding": True,
                "mem_vit_random_interval_sampling": False,
                "mem_vit_global_interval_std_seconds": 0.15,
                "mem_vit_local_interval_std_seconds": 0.05,
                "mem_vit_min_interval_seconds": 0.25,
                "mem_vit_max_interval_seconds": 1.0,
                "semantic_state_instruction_matrix": {
                    "approach": {
                        "goal_approach": {
                            "status": "active",
                            "action_supervision": "demonstrated",
                            "completion_boundary": "goal_end",
                        }
                    }
                },
                "semantic_view_kind_weights": {"active": 1.0},
                "semantic_view_manifest_sha256": "first-manifest",
                "semantic_view_manifest_version": 1,
                "task_complete_sample_tail_seconds": 2.0,
                "action_loss_schema": "group_balanced_v2",
                "action_continuous_group_weight": 3.0,
                "action_discrete_group_weight": 1.0,
                "optimizer_lr": 2.5e-5,
            }
        )
    )
    (second / "config.json").write_text(
        json.dumps(
            {
                "architecture": "pi05",
                "action_bool_true_fractions": {"arm_reset": 0.25},
                "semantic_state_instruction_matrix": {
                    "approach": {
                        "goal_approach": {
                            "status": "active",
                            "action_supervision": "demonstrated",
                            "completion_boundary": "goal_end",
                        }
                    },
                    "handle_press": {
                        "goal_open": {
                            "status": "active",
                            "action_supervision": "demonstrated",
                            "completion_boundary": "goal_end",
                        }
                    },
                    "traversal": {
                        "goal_enter": {
                            "status": "active",
                            "action_supervision": "demonstrated",
                            "completion_boundary": "goal_end",
                        }
                    },
                },
                "semantic_view_kind_weights": {"active": 0.5, "complete": 0.5},
                "semantic_view_manifest_sha256": "second-manifest",
                "semantic_view_manifest_version": 2,
                "task_complete_sample_tail_seconds": 4.0,
                "action_loss_schema": "uniform_valid",
                "action_continuous_group_weight": 1.0,
                "action_discrete_group_weight": 1.0,
                "optimizer_lr": 1e-5,
            }
        )
    )
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        document = json.loads((first / name).read_text())
        document["steps"][0]["config"]["supervise_terminal_static_padding"] = True
        (first / name).write_text(json.dumps(document))

    assert _checkpoint_hot_swap_signature(first) == _checkpoint_hot_swap_signature(second)

    # A real inference-structure change must still force a full reload.
    second_config = json.loads((second / "config.json").read_text())
    second_config["action_feature_names"] = ["b2_delta_x", "task_complete"]
    (second / "config.json").write_text(json.dumps(second_config))
    assert _checkpoint_hot_swap_signature(first) != _checkpoint_hot_swap_signature(second)

    second_config.pop("action_feature_names")
    second_config["mem_vit_num_frames"] = 5
    first_config = json.loads((first / "config.json").read_text())
    first_config["mem_vit_num_frames"] = 6
    (first / "config.json").write_text(json.dumps(first_config))
    (second / "config.json").write_text(json.dumps(second_config))
    assert _checkpoint_hot_swap_signature(first) != _checkpoint_hot_swap_signature(second)


def test_action_packet_contains_replayable_model_and_execution_outputs(tmp_path) -> None:
    record = ActionRecord(
        sequence=3,
        source_step=11,
        inference_seconds=0.2,
        action_names=("b2_vx",),
        model_action_names=("model_b2",),
        control_epoch=7,
        b2_execution_mode="velocity_chunk",
        z1_action_representation="ee_state_delta",
        b2_anchor_kind="actual_state",
        b2_anchor=np.asarray([1.0, 2.0, 3.0]),
        b2_reference=torch.ones((2, 3)),
        ee_anchor_kind="actual_ee_state",
        ee_anchor=np.arange(9, dtype=np.float32),
        original=torch.full((2, 1), 0.25),
        unsmoothed=torch.full((2, 1), 0.5),
        processed=torch.full((2, 1), 0.75),
        inference_started_ns=100,
        inference_finished_ns=200,
        preprocess_seconds=0.01,
        predict_seconds=0.15,
        postprocess_seconds=0.04,
        observation_received_ns=80,
        inference_delay_steps=2,
        velocity_smoothing_transition_step=1,
        sim_steps_per_wall_second=50.0,
        rtc_mode="training",
        rtc_conditioning_delay_steps=2,
        model_image_history=b"recorded-image-history",
    )

    with np.load(BytesIO(encode_action_packet(record)), allow_pickle=False) as packet:
        np.testing.assert_array_equal(packet["original_actions"], record.original.numpy())
        np.testing.assert_array_equal(packet["unsmoothed_execution_actions"], record.unsmoothed.numpy())
        np.testing.assert_array_equal(packet["processed_actions"], record.processed.numpy())
        np.testing.assert_array_equal(packet["b2_reference"], record.b2_reference.numpy())
        assert int(packet["control_epoch"]) == 7
        assert float(packet["predict_seconds"]) == pytest.approx(0.15)
        assert str(packet["rtc_mode"]) == "training"
        assert int(packet["rtc_conditioning_delay_steps"]) == 2
        assert packet["model_image_history"].tobytes() == record.model_image_history
    RolloutRecorder(str(tmp_path)).action(record)
    with np.load(tmp_path / "vla/actions/000003.npz", allow_pickle=False) as saved:
        assert saved["model_image_history"].tobytes() == record.model_image_history


@pytest.mark.parametrize(
    ("training_enabled", "expected_mode"),
    [(False, "inference"), (True, "training")],
)
def test_server_derives_rtc_mode_from_checkpoint(training_enabled: bool, expected_mode: str) -> None:
    training_config = TrainingRTCConfig(enabled=training_enabled)
    config = SimpleNamespace(training_rtc_config=training_config)
    args = SimpleNamespace(
        chunk_scheduling_mode="RTC",
        rtc_schedule="EXP",
        rtc_max_guidance_weight=10.0,
        rtc_execution_horizon=25,
    )

    rtc_config = _deployment_rtc_config(config, args)

    assert rtc_config.enabled
    assert rtc_config.mode == expected_mode
    assert config.training_rtc_config is training_config


def test_sync_scheduling_disables_training_rtc_conditioning() -> None:
    config = SimpleNamespace(training_rtc_config=TrainingRTCConfig(enabled=True))
    args = SimpleNamespace(
        chunk_scheduling_mode="SYNC",
        rtc_schedule="EXP",
        rtc_max_guidance_weight=10.0,
        rtc_execution_horizon=25,
    )

    rtc_config = _deployment_rtc_config(config, args)

    assert not rtc_config.enabled
    assert rtc_config.mode == "training"


@pytest.mark.parametrize("rtc_mode", ["inference", "training"])
def test_server_preserves_prefix_padding_only_for_inference_rtc(rtc_mode: str) -> None:
    policy = AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    policy.chunk_scheduling_mode = "RTC"
    policy.rtc_config = RTCConfig(mode=rtc_mode, execution_horizon=4)
    policy._records_lock = Lock()
    policy.device = torch.device("cpu")
    policy.b2_action_representation = "velocity"
    policy.z1_action_representation = "ee_delta"
    original = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    policy._records = {7: SimpleNamespace(original=original)}

    prefix = policy._previous_prefix(SimpleNamespace(active_sequence=7, active_index=2))

    if rtc_mode == "training":
        torch.testing.assert_close(prefix, original[2:])
    else:
        torch.testing.assert_close(prefix, torch.cat([original[2:], torch.zeros(2, 3)]))


@pytest.mark.parametrize("rtc_mode", ["inference", "training"])
def test_server_reanchors_prefix_before_rtc_conditioning(rtc_mode: str) -> None:
    policy = AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    policy.chunk_scheduling_mode = "RTC"
    policy.rtc_config = RTCConfig(mode=rtc_mode, execution_horizon=2)
    policy._records_lock = Lock()
    policy.device = torch.device("cpu")
    policy.b2_action_representation = "pose_delta"
    policy.z1_action_representation = "ee_delta"
    original = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    record = SimpleNamespace(
        sequence=7,
        original=original,
        b2_anchor=np.zeros(3),
        ee_anchor=np.zeros(9),
    )
    policy._records = {7: record}
    calls = []

    def reanchor(prefix, previous, packet):
        calls.append((prefix.clone(), previous, packet))
        return prefix + 1

    policy._reanchor_rtc_prefix = reanchor
    policy._rtc_reanchor_status = lambda: (True, False, (), "")
    policy._b2_global_pose = lambda _packet: np.ones(3)
    policy.recorder = SimpleNamespace(event=lambda *_args, **_kwargs: None)
    packet = SimpleNamespace(active_sequence=7, active_index=1, sim_step=10, actual_ee_state=np.ones(9))

    prefix = policy._previous_prefix(packet)

    assert len(calls) == 1
    assert calls[0][1] is record
    assert calls[0][2] is packet
    torch.testing.assert_close(calls[0][0], original[1:])
    torch.testing.assert_close(prefix, original[1:] + 1)


@pytest.mark.parametrize("delay", [0, 4])
def test_server_training_rtc_passes_supported_delay_and_real_prefix(delay: int) -> None:
    policy = AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    policy.chunk_scheduling_mode = "RTC"
    policy.rtc_config = RTCConfig(mode="training")
    policy.training_rtc_max_delay = 4
    prefix = torch.randn(4, 3)
    policy._previous_prefix = lambda _packet: prefix

    conditioning = policy._rtc_conditioning(SimpleNamespace(), delay)

    assert conditioning["inference_delay"] == delay
    assert conditioning["prev_chunk_left_over"] is prefix


@pytest.mark.parametrize(
    ("delay", "prefix_length", "error"),
    [(5, 8, "trained maximum 4"), (4, 3, "available queued prefix")],
)
def test_server_training_rtc_rejects_unsupported_delay(delay, prefix_length, error) -> None:
    policy = AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    policy.chunk_scheduling_mode = "RTC"
    policy.rtc_config = RTCConfig(mode="training")
    policy.training_rtc_max_delay = 4
    policy._previous_prefix = lambda _packet: torch.zeros(prefix_length, 3)

    with pytest.raises(ValueError, match=error):
        policy._rtc_conditioning(SimpleNamespace(), delay)


def test_server_training_rtc_cold_start_does_not_freeze_an_unavailable_prefix() -> None:
    policy = AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    policy.chunk_scheduling_mode = "RTC"
    policy.rtc_config = RTCConfig(mode="training")
    policy.training_rtc_max_delay = 4
    policy._previous_prefix = lambda _packet: None

    assert policy._rtc_conditioning(SimpleNamespace(), 70) == {
        "inference_delay": 0,
        "prev_chunk_left_over": None,
    }


def test_b2_rtc_pose_delta_prefix_is_reexpressed_in_new_inference_frame() -> None:
    old_anchor = torch.tensor([10.0, 20.0, np.pi / 2])
    new_anchor = torch.tensor([10.0, 21.0, np.pi / 2])
    old_relative_targets = torch.tensor([[1.0, 0.0, 0.1], [2.0, 0.0, 0.2]])

    rebased = _reanchor_b2_pose_delta(old_relative_targets, old_anchor, new_anchor)

    torch.testing.assert_close(rebased[:, :2], torch.tensor([[0.0, 0.0], [1.0, 0.0]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(rebased[:, 2], torch.tensor([0.1, 0.2]), atol=1e-6, rtol=0)








def test_z1_state_delta_rtc_prefix_is_reexpressed_from_current_actual_state() -> None:
    model_names = (*B2_EXECUTION_VELOCITY_NAMES, *EE_DELTA_ROTVEC_NAMES, *EE_DELTA_ACTION_NAMES[6:9])
    policy = AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    policy._action_unnormalizer = lambda transition: transition
    policy._action_normalizer = lambda transition: transition
    policy.b2_action_representation = "velocity"
    policy.b2_execution_mode = "velocity_chunk"
    policy.z1_action_representation = "ee_state_delta"
    policy.model_action_names = model_names
    prefix = torch.zeros(2, len(model_names))
    prefix[:, model_names.index("height_invariant_ee_delta_x")] = torch.tensor([1.0, 2.0])
    identity = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    old_anchor = np.concatenate((identity, np.zeros(3, dtype=np.float32)))
    new_anchor = np.concatenate((identity, np.asarray([0.5, 0.0, 0.0], dtype=np.float32)))

    rebased = policy._reanchor_rtc_prefix(
        prefix,
        SimpleNamespace(ee_anchor=old_anchor),
        SimpleNamespace(actual_ee_state=new_anchor),
    )

    x_index = model_names.index("height_invariant_ee_delta_x")
    torch.testing.assert_close(rebased[:, x_index], torch.tensor([0.5, 1.5]))


def test_rtc_reanchor_skips_coupled_b2_group_with_degenerate_quantile_scale() -> None:
    model_names = (
        "b2_delta_x",
        "b2_delta_y",
        "b2_delta_yaw",
        *EE_DELTA_ROTVEC_NAMES,
        *EE_DELTA_ACTION_NAMES[6:9],
    )
    stats = {
        "action": {
            "q01": np.asarray([0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
            "q99": np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        }
    }
    processor_kwargs = {
        "features": {"action": PolicyFeature(FeatureType.ACTION, (len(model_names),))},
        "norm_map": {FeatureType.ACTION: NormalizationMode.QUANTILES},
        "stats": stats,
    }
    policy = AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    policy._action_unnormalizer = UnnormalizerProcessorStep(**processor_kwargs)
    policy._action_normalizer = NormalizerProcessorStep(**processor_kwargs)
    policy.b2_action_representation = "pose_delta"
    policy.z1_action_representation = "ee_state_delta"
    policy.model_action_names = model_names
    prefix = torch.zeros((2, len(model_names)))
    prefix[:, :3] = -1.0
    prefix[:, model_names.index("height_invariant_ee_delta_x")] = torch.tensor([0.5, 1.0])
    identity = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    old_ee_anchor = np.concatenate((identity, np.zeros(3, dtype=np.float32)))
    new_ee_anchor = np.concatenate((identity, np.asarray([0.25, 0.0, 0.0], dtype=np.float32)))

    rebased = policy._reanchor_rtc_prefix(
        prefix,
        SimpleNamespace(
            b2_anchor=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            ee_anchor=old_ee_anchor,
        ),
        SimpleNamespace(
            state_names=B2_GLOBAL_POSE_NAMES,
            state=np.asarray([0.02, -0.01, 0.01], dtype=np.float32),
            actual_ee_state=new_ee_anchor,
        ),
    )

    torch.testing.assert_close(rebased[:, :3], prefix[:, :3])
    x_index = model_names.index("height_invariant_ee_delta_x")
    torch.testing.assert_close(rebased[:, x_index], torch.tensor([0.25, 0.75]))
    assert torch.isfinite(rebased).all()
    assert torch.max(torch.abs(rebased)) <= 1.0


def test_metadata_v2_is_rejected() -> None:
    with pytest.raises(ValueError, match="retired contract"):
        _resolve_checkpoint_action_representations(
            2,
            {"representation": "velocity"},
            {"b2_action_representation": "velocity"},
        )


def test_metadata_v2_rejects_retired_cumulative_local_trajectory() -> None:
    with pytest.raises(ValueError, match="retired contract"):
        _resolve_checkpoint_action_representations(
            2,
            {"representation": "local_trajectory"},
            {"b2_action_representation": "local_trajectory"},
        )


def test_metadata_v4_is_rejected_even_when_representations_are_named() -> None:
    with pytest.raises(ValueError, match="retired contract"):
        _resolve_checkpoint_action_representations(
            4,
            {"representation": "pose_delta", "z1_representation": "ee_delta"},
            {"b2_action_representation": "pose_delta"},
        )


@pytest.mark.parametrize("b2_representation", ["velocity", "pose_delta"])
@pytest.mark.parametrize("z1_representation", ["ee_delta", "ee_state_delta"])
def test_server_loads_current_contract_for_all_representation_pairs(
    tmp_path, b2_representation: str, z1_representation: str
) -> None:
    state_names = ["arm_q_1", *HEIGHT_INVARIANT_EE_STATE_NAMES]
    config = PI05Config(
        b2_action_representation=b2_representation,
        z1_action_representation=z1_representation,
        ee_delta_rotation_representation="rotvec",
        control_frequency_hz=50.0,
        action_dt_seconds=0.02,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (len(state_names),)),
            "observation.images.base": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
            "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
        },
    )
    name_fn = b2_pose_delta_action_names if b2_representation == "pose_delta" else b2_execution_action_names
    config.action_feature_names = name_fn(
        list(CONTROL_EXTENDED_DATASET_ACTION_NAMES), **action_schema_kwargs(config)
    )
    config.output_features = {
        "action": PolicyFeature(FeatureType.ACTION, (len(config.action_feature_names),))
    }
    config.io_schema_resolved = True
    config.dataset_state_feature_names = state_names
    config.resolved_state_feature_names = state_names
    config.state_feature_indices = list(range(len(state_names)))
    config.ee_state_anchor_indices = list(range(1, 10))
    config.dataset_action_feature_names = list(CONTROL_EXTENDED_DATASET_ACTION_NAMES)
    config.dataset_camera_keys = ["observation.images.base", "observation.images.wrist"]
    config._save_pretrained(tmp_path)

    contract = _load_checkpoint_contract(tmp_path, config, 50.0)

    assert contract.metadata_version == 10
    assert contract.model_action_representation == b2_representation
    assert contract.z1_action_representation == z1_representation
    assert contract.arm_mode_encoding == "paired_flow_channels"

    metadata_path = tmp_path / "pi05_deployment_metadata.json"
    legacy_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    legacy_metadata["action"].pop("arm_mode_encoding")
    metadata_path.write_text(json.dumps(legacy_metadata), encoding="utf-8")
    legacy_contract = _load_checkpoint_contract(tmp_path, config, 50.0)
    assert legacy_contract.arm_mode_encoding is None

    pre_task_blocked_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pre_task_blocked_metadata["action"]["predict"].pop("task_blocked")
    pre_task_blocked_metadata["action"]["boolean_decoding"]["true_side"].pop("task_blocked")
    pre_task_blocked_metadata["action"]["boolean_decoding"]["output_values"].pop("task_blocked")
    pre_task_blocked_metadata["action"].pop("task_blocked_semantics")
    metadata_path.write_text(json.dumps(pre_task_blocked_metadata), encoding="utf-8")
    pre_task_blocked_contract = _load_checkpoint_contract(tmp_path, config, 50.0)
    assert pre_task_blocked_contract.arm_mode_encoding is None


def _schema_v2_names() -> tuple[str, ...]:
    names = list(EXECUTION_ACTION_NAMES)
    names[names.index("arm_active")] = "arm_teleop_inactive"
    return tuple(names)


def test_schema_v2_arm_gate_is_converted_before_low_level() -> None:
    names = _schema_v2_names()
    actions = torch.zeros((2, len(names)))
    gate_index = names.index("arm_teleop_inactive")
    actions[:, gate_index] = torch.tensor([0.0, 1.0])

    execution = _to_execution_actions(actions, names, "ee_delta")

    assert execution[:, EXECUTION_ACTION_NAMES.index("arm_active")].tolist() == [1.0, 0.0]


@pytest.mark.parametrize("mode", ["continuous_flow", "structured_temporal"])
def test_server_decodes_normalized_discrete_outputs_to_physical_commands(mode: str) -> None:
    names = _schema_v2_names()
    normalized = torch.zeros((4, len(names)))
    postprocessed = torch.full_like(normalized, 0.37)
    normalized[:, names.index("arm_teleop_inactive")] = torch.tensor([-0.2, 0.3, -0.1, -0.4])
    normalized[:, names.index("arm_reset")] = torch.tensor([-0.4, -0.2, 0.6, -0.3])
    normalized[:, names.index("gripper_target")] = torch.tensor([-0.01, 0.2, -0.7, 0.9])
    normalized[:, names.index("task_complete")] = torch.tensor([-0.5, 0.2, -0.3, -0.4])

    decoded = _decode_discrete_actions(
        normalized,
        postprocessed,
        names,
        mode=mode,
        gripper_target_representation="binary_position",
        gripper_negative_value=-1.0471976,
        gripper_nonnegative_value=0.0,
    )

    assert decoded[:, names.index("arm_teleop_inactive")].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert decoded[:, names.index("arm_reset")].tolist() == [0.0, 0.0, 1.0, 0.0]
    torch.testing.assert_close(
        decoded[:, names.index("gripper_target")],
        torch.tensor([-1.0471976, 0.0, -1.0471976, 0.0]),
    )
    assert decoded[:, names.index("task_complete")].tolist() == [0.0, 1.0, 1.0, 1.0]


def test_server_projects_paired_arm_flow_channels_to_ik_bridge_states() -> None:
    names = _schema_v2_names()
    normalized = torch.zeros((4, len(names)))
    postprocessed = torch.zeros_like(normalized)
    inactive_index = names.index("arm_teleop_inactive")
    reset_index = names.index("arm_reset")
    normalized[:, [inactive_index, reset_index]] = torch.tensor(
        [
            [-0.8, -0.7],
            [0.9, -0.6],
            [-0.4, 0.8],
            [0.9, 0.8],
        ]
    )

    decoded = _decode_discrete_actions(
        normalized,
        postprocessed,
        names,
        mode="continuous_flow",
        arm_mode_encoding="paired_flow_channels",
        gripper_target_representation="binary_position",
        gripper_negative_value=-1.0471976,
        gripper_nonnegative_value=0.0,
    )
    execution = _to_execution_actions(decoded, names, "ee_delta")

    assert decoded[:, inactive_index].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert decoded[:, reset_index].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert execution[:, EXECUTION_ACTION_NAMES.index("arm_active")].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert execution[:, EXECUTION_ACTION_NAMES.index("arm_reset")].tolist() == [0.0, 0.0, 1.0, 0.0]


def test_continuous_gripper_is_not_thresholded() -> None:
    names = ("gripper_target",)
    normalized = torch.tensor([[-0.2], [0.3]])
    physical = torch.tensor([[-0.27], [-0.81]])

    decoded = _decode_discrete_actions(
        normalized,
        physical,
        names,
        mode="continuous_flow",
        gripper_target_representation="continuous_position",
        gripper_negative_value=-1.0471976,
        gripper_nonnegative_value=0.0,
    )

    torch.testing.assert_close(decoded, physical)


def test_continuous_ee_schema_defaults_missing_arm_modes_to_active() -> None:
    names = tuple(["b2_vx", "b2_vy", "b2_omega_z"] + list(EE_DELTA_ACTION_NAMES) + ["gripper_target"])
    actions = torch.zeros((2, len(names)))

    execution = _to_execution_actions(actions, names, "ee_delta")

    assert execution[:, EXECUTION_ACTION_NAMES.index("arm_active")].tolist() == [1.0, 1.0]
    assert execution[:, EXECUTION_ACTION_NAMES.index("arm_reset")].tolist() == [0.0, 0.0]
    assert execution[:, EXECUTION_ACTION_NAMES.index("task_complete")].tolist() == [0.0, 0.0]


@pytest.mark.parametrize(
    ("z1_representation", "expected_anchor_kind", "expected_anchor"),
    [
        ("ee_delta", "executed_ee_target", np.zeros(9, dtype=np.float32)),
        ("ee_state_delta", "actual_ee_state", np.ones(9, dtype=np.float32)),
    ],
)
@pytest.mark.parametrize("rtc_mode", ["inference", "training"])
@pytest.mark.parametrize("scheduling_mode", ["RTC", "SYNC"])
def test_infer_records_model_action_names_and_source_step_anchor(
    z1_representation: str,
    expected_anchor_kind: str,
    expected_anchor: np.ndarray,
    rtc_mode: str,
    scheduling_mode: str,
) -> None:
    model_names = (*B2_EXECUTION_VELOCITY_NAMES, *EE_DELTA_ACTION_NAMES, "gripper_target")
    policy = AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    policy._model_lock = Lock()
    policy.chunk_scheduling_mode = "SYNC" if scheduling_mode == "RTC" else "RTC"
    policy.rtc_config = RTCConfig(mode=rtc_mode)
    apply_mode = policy.set_chunk_scheduling_mode

    def locked_mode_switch(mode):
        assert policy._model_lock.locked()
        return apply_mode(mode)

    policy.set_chunk_scheduling_mode = locked_mode_switch
    policy.training_rtc_max_delay = 4
    policy.preprocessor = lambda batch: batch
    policy.postprocessor = lambda actions: actions
    predict_calls = []

    def predict(_batch, **kwargs):
        predict_calls.append(kwargs)
        return torch.zeros((1, 2, len(model_names)))

    policy.policy = SimpleNamespace(predict_action_chunk=predict)
    raw_batch = _RecordedImageBatch()
    raw_batch.model_image_history = b"selected-image-window"
    policy._make_batch = lambda _packet: raw_batch
    policy._previous_prefix = lambda _packet: None
    policy._b2_velocity_filter_context = lambda _packet: (torch.zeros(3), None)
    policy.postprocessed_action_names = model_names
    policy.model_action_names = model_names
    policy.discrete_action_training_mode = "continuous_flow"
    policy.arm_mode_encoding = None
    policy.gripper_target_representation = "continuous_position"
    policy.gripper_negative_value = -1.0471976
    policy.gripper_nonnegative_value = 0.0
    policy.z1_action_representation = z1_representation
    policy.b2_action_representation = "velocity"
    policy.b2_execution_mode = "velocity_chunk"
    policy.action_names = EXECUTION_ACTION_NAMES
    policy.stop_on_model_task_complete = False
    policy._task_complete_latched = False
    policy.b2_velocity_smoothing_time_constant_s = 0.0
    policy.low_level_hz = 50.0
    policy._next_sequence = 1

    record = policy._infer(
        SimpleNamespace(
            scheduling_mode=scheduling_mode,
            control_epoch=3,
            sim_step=25,
            server_received_ns=0,
            inference_delay_steps=7,
            source_steps_per_wall_second=50.0,
            executed_ee_target=np.zeros(9, dtype=np.float32),
            actual_ee_state=np.ones(9, dtype=np.float32),
            state_names=B2_GLOBAL_POSE_NAMES,
            state=np.asarray([1.0, 2.0, 0.3], dtype=np.float32),
        )
    )

    assert record.model_action_names == model_names
    assert record.model_image_history == raw_batch.model_image_history
    assert record.original.shape == (2, len(model_names))
    assert record.processed.shape == (2, len(EXECUTION_ACTION_NAMES))
    assert record.source_step == 25
    assert record.control_epoch == 3
    assert record.b2_execution_mode == "velocity_chunk"
    assert record.b2_reference is None
    assert record.ee_anchor_kind == expected_anchor_kind
    np.testing.assert_array_equal(record.ee_anchor, expected_anchor)
    assert record.b2_anchor_kind == "actual_world_pose"
    np.testing.assert_allclose(record.b2_anchor, [1.0, 2.0, 0.3])
    assert record.rtc_mode == rtc_mode
    assert policy.chunk_scheduling_mode == scheduling_mode
    assert policy.rtc_config.enabled == (scheduling_mode == "RTC")
    assert record.inference_delay_steps == (7 if scheduling_mode == "RTC" else 0)
    assert policy._next_sequence == 2
    conditioning_delay = 7 if rtc_mode == "inference" and scheduling_mode == "RTC" else 0
    expected_conditioning = {"inference_delay": conditioning_delay, "prev_chunk_left_over": None} if scheduling_mode == "RTC" else {}
    assert predict_calls == [expected_conditioning]
    assert record.rtc_conditioning_delay_steps == conditioning_delay


def test_formal_server_rejects_legacy_arm_active_semantics() -> None:
    actions = torch.zeros((2, len(EXECUTION_ACTION_NAMES)))

    with pytest.raises(ValueError, match="must not output legacy arm_active"):
        _to_execution_actions(actions, EXECUTION_ACTION_NAMES, "ee_delta")


def test_completion_stops_motion_at_the_first_true_row() -> None:
    actions = torch.ones((4, len(EXECUTION_ACTION_NAMES)))
    actions[:, EXECUTION_ACTION_NAMES.index("task_complete")] = torch.tensor([0.0, 1.0, 0.0, 0.0])

    stopped = _apply_completion_stop(actions, EXECUTION_ACTION_NAMES)

    for name in (*B2_EXECUTION_VELOCITY_NAMES, "b2_active", "arm_active", "arm_reset"):
        assert stopped[:, EXECUTION_ACTION_NAMES.index(name)].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert stopped[:, EXECUTION_ACTION_NAMES.index("task_complete")].tolist() == [0.0, 1.0, 1.0, 1.0]


def _completion_engine(enabled):
    policy = AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    policy._records_lock = Lock()
    policy._mailbox_condition = __import__("threading").Condition()
    actions = torch.zeros((3, len(EXECUTION_ACTION_NAMES)))
    actions[1, EXECUTION_ACTION_NAMES.index("task_complete")] = 1.0
    policy._records = {4: SimpleNamespace(processed=actions, action_names=EXECUTION_ACTION_NAMES)}
    policy._mailbox = None
    policy._mailbox_version = 0
    policy._control_epoch = None
    policy._task_complete_latched = False
    policy.stop_on_model_task_complete = enabled
    policy.recorder = SimpleNamespace(observation=lambda _: None)
    return policy


def test_server_latches_an_executed_completion_until_sim_step_resets():
    policy = _completion_engine(True)
    def packet(step, index, epoch=0):
        return SimpleNamespace(sim_step=step, active_sequence=4, active_index=index, control_epoch=epoch)
    policy.submit(packet(10, 2))
    assert policy._task_complete_latched
    policy.submit(packet(9, 0))
    assert policy._task_complete_latched
    assert policy._mailbox.sim_step == 10
    policy.submit(packet(11, 0, 1))
    assert not policy._task_complete_latched
    assert policy._mailbox.sim_step == 11
    assert not hasattr(policy, "_observation_history")
    assert not hasattr(policy, "_image_history")


def test_server_does_not_latch_completion_when_model_stop_is_disabled():
    policy = _completion_engine(False)
    policy.submit(SimpleNamespace(control_epoch=0, sim_step=10, active_sequence=4, active_index=2))
    assert not policy._task_complete_latched


def test_direct_inference_is_not_enqueued_and_preserves_completion_context():
    from lerobot.scripts.pi05_vla_server import InferenceBusyError
    policy = _completion_engine(True)
    policy._request_gate = Lock()
    policy._model_lock = Lock()
    policy._latest_record = None
    policy._inference_count = 0
    policy.recorder.action = lambda _: None
    policy._infer_unlocked = lambda p: SimpleNamespace(sequence=5, control_epoch=p.control_epoch)
    packet = SimpleNamespace(control_epoch=0, sim_step=10, active_sequence=4, active_index=2)
    record = policy.infer_request(packet)
    assert policy._task_complete_latched
    assert policy._mailbox is None
    assert policy._mailbox_version == 0
    assert policy.latest_after(4) is record
    assert policy._inference_count == 1
    policy._request_gate.acquire()
    try:
        with pytest.raises(InferenceBusyError):
            policy.infer_request(packet)
    finally:
        policy._request_gate.release()


@pytest.fixture
def model_http(monkeypatch):
    from http.server import ThreadingHTTPServer
    from unittest.mock import Mock
    import lerobot.scripts.pi05_vla_server as server_module
    server = ThreadingHTTPServer(("127.0.0.1", 0), server_module.VLARequestHandler)
    server.max_payload_bytes = 1024 * 1024
    server.engine = Mock()
    server.engine.input_spec.return_value = {"schema_id": "test"}
    server.engine.latest_after.return_value = None
    server.engine.policy = SimpleNamespace(config=SimpleNamespace(chunk_size=50))
    server.engine.action_names = EXECUTION_ACTION_NAMES
    server.engine.model_action_names = ("b2_delta_x",)
    server.engine.b2_execution_mode = "se2_feedback"
    server.engine.z1_action_representation = "ee_state_delta"
    server.engine.infer_request.side_effect = lambda packet: packet
    monkeypatch.setattr(server_module, "encode_action_packet", lambda p: _complete_packet(control_epoch=np.int64(p.control_epoch)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", server.engine
    server.shutdown()
    server.server_close()
    thread.join()


def test_http_direct_inference_and_async_submission_share_complete_packet(model_http):
    from urllib import request
    url, engine = model_http
    for path, status in (("/v1/infer", 200), ("/v1/observations", 202)):
        req = request.Request(url + path, data=_complete_packet(), headers={"Content-Type": "application/x-npz"})
        with request.urlopen(req) as response:
            assert response.status == status
            if status == 200:
                assert response.headers.get_content_type() == "application/x-npz"
            else:
                assert json.load(response) == {"accepted_step": 12, "control_epoch": 7}
    engine.infer_request.assert_called_once()
    engine.submit.assert_called_once()
    with request.urlopen(url + "/v1/actions?after=0") as response:
        assert response.status == 204
    with request.urlopen(url + "/v1/schema") as response:
        schema = json.load(response)
        assert schema["protocol_version"] == PROTOCOL_VERSION
        assert schema["output"]["chunk_size"] == 50
        assert schema["output"]["action_names"] == list(EXECUTION_ACTION_NAMES)
        assert schema["endpoints"]["infer"] == "/v1/infer"


@pytest.mark.parametrize("path,body,content_type,status,code", [
    ("/v1/infer", b"bad", "application/json", 415, "unsupported_media_type"),
    ("/v1/infer", b"bad", "application/x-npz", 400, "invalid_input"),
    ("/v1/infer", b"x" * (1024 * 1024 + 1), "application/x-npz", 413, "payload_too_large"),
    ("/unknown", b"x", "application/x-npz", 404, "not_found"),
])
def test_http_errors_have_machine_readable_codes(model_http, path, body, content_type, status, code):
    from urllib import request, error
    url, _ = model_http
    req = request.Request(url + path, data=body, headers={"Content-Type": content_type})
    with pytest.raises(error.HTTPError) as caught:
        request.urlopen(req)
    assert caught.value.code == status
    assert json.load(caught.value)["error"]["code"] == code


def test_http_schema_conflict_and_busy(model_http):
    from urllib import request, error
    from lerobot.scripts.pi05_vla_server import InferenceBusyError
    url, engine = model_http
    for schema, exception, status, code in (("other", None, 409, "schema_mismatch"),
            ("test", InferenceBusyError("busy"), 429, "inference_busy")):
        engine.infer_request.side_effect = exception
        req = request.Request(url + "/v1/infer", data=_complete_packet(model_input_schema=np.asarray(schema)),
            headers={"Content-Type": "application/x-npz"})
        with pytest.raises(error.HTTPError) as caught:
            request.urlopen(req)
        assert caught.value.code == status
        assert json.load(caught.value)["error"]["code"] == code


def test_ee_delta_is_forwarded_with_an_explicit_low_level_schema() -> None:
    delta_names = execution_action_names("ee_delta")
    postprocessed_names = list(delta_names)
    postprocessed_names[postprocessed_names.index("arm_active")] = "arm_teleop_inactive"
    actions = torch.arange(len(delta_names), dtype=torch.float32).repeat(2, 1)

    execution = _to_execution_actions(actions, tuple(postprocessed_names), "ee_delta")

    assert execution.shape == actions.shape
    assert execution_action_names("ee_delta")[6].startswith("height_invariant_ee_delta_")
    torch.testing.assert_close(execution[:, 6:15], actions[:, 6:15])


def test_rotvec_model_delta_is_expanded_to_low_level_rot6d() -> None:
    names = (
        "b2_vx",
        "b2_vy",
        "b2_omega_z",
        "arm_teleop_inactive",
        "arm_reset",
        *EE_DELTA_ROTVEC_NAMES,
        "height_invariant_ee_delta_x",
        "height_invariant_ee_delta_y",
        "height_invariant_ee_delta_z",
        "gripper_target",
        "task_complete",
    )
    actions = torch.zeros((1, len(names)))
    actions[0, names.index("height_invariant_ee_delta_rotvec_z")] = torch.pi / 2

    execution = _to_execution_actions(actions, names, "ee_delta")

    torch.testing.assert_close(
        execution[0, 6:12],
        torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0, 0.0]),
        atol=1e-6,
        rtol=1e-6,
    )


def test_b2_velocity_smoothing_is_causal_and_leaves_other_actions_unchanged() -> None:
    actions = torch.zeros((3, len(EXECUTION_ACTION_NAMES)))
    velocity_indices = [EXECUTION_ACTION_NAMES.index(name) for name in B2_EXECUTION_VELOCITY_NAMES]
    actions[:, velocity_indices] = torch.tensor([1.0, -1.0, 2.0])
    gripper_index = EXECUTION_ACTION_NAMES.index("gripper_target")
    actions[:, gripper_index] = torch.tensor([0.1, 0.2, 0.3])

    smoothed = _smooth_b2_execution_velocity(
        actions,
        EXECUTION_ACTION_NAMES,
        torch.tensor([0.0, 0.0, 0.0]),
        dt=0.02,
        time_constant_s=0.02 / torch.log(torch.tensor(2.0)).item(),
    )

    torch.testing.assert_close(
        smoothed[:, velocity_indices],
        torch.tensor([[0.5, -0.5, 1.0], [0.75, -0.75, 1.5], [0.875, -0.875, 1.75]]),
    )
    torch.testing.assert_close(smoothed[:, gripper_index], actions[:, gripper_index])
    torch.testing.assert_close(actions[:, velocity_indices], torch.tensor([[1.0, -1.0, 2.0]] * 3))


def test_zero_time_constant_disables_b2_velocity_smoothing() -> None:
    actions = torch.randn((4, len(EXECUTION_ACTION_NAMES)))

    smoothed = _smooth_b2_execution_velocity(
        actions,
        EXECUTION_ACTION_NAMES,
        torch.tensor([10.0, 10.0, 10.0]),
        dt=0.02,
        time_constant_s=0.0,
    )

    torch.testing.assert_close(smoothed, actions)
    assert smoothed.data_ptr() != actions.data_ptr()


def test_rtc_chunk_smoothing_continues_from_last_executed_command() -> None:
    policy = AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    policy.chunk_scheduling_mode = "RTC"
    policy.action_names = EXECUTION_ACTION_NAMES
    policy._records_lock = Lock()
    processed = torch.zeros((4, len(EXECUTION_ACTION_NAMES)))
    velocity_indices = [EXECUTION_ACTION_NAMES.index(name) for name in B2_EXECUTION_VELOCITY_NAMES]
    processed[:, velocity_indices] = torch.tensor(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]
    )
    policy._records = {7: SimpleNamespace(processed=processed)}
    packet = SimpleNamespace(active_sequence=7, active_index=3)

    anchor, prefix = policy._b2_velocity_filter_context(packet)

    torch.testing.assert_close(anchor, torch.tensor([0.7, 0.8, 0.9]))
    torch.testing.assert_close(prefix, torch.tensor([[1.0, 1.1, 1.2]]))


def test_first_chunk_smoothing_uses_observed_body_velocity() -> None:
    policy = AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    policy.chunk_scheduling_mode = "RTC"
    policy.action_names = EXECUTION_ACTION_NAMES
    policy._records_lock = Lock()
    policy._records = {}
    names = ("unused", "b2_body_wz", "b2_body_vy", "b2_body_vx")
    packet = SimpleNamespace(
        active_sequence=-1,
        active_index=0,
        state_names=names,
        state=np.asarray([99.0, 0.3, 0.2, 0.1], dtype=np.float32),
    )

    anchor, prefix = policy._b2_velocity_filter_context(packet)

    torch.testing.assert_close(anchor, torch.tensor([0.1, 0.2, 0.3]))
    assert prefix is None


def test_sync_server_disables_rtc_prefix_delay_and_previous_velocity_context() -> None:
    policy = AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    policy.chunk_scheduling_mode = "SYNC"
    policy.rtc_config = RTCConfig(mode="inference")
    policy.low_level_hz = 50.0
    packet = SimpleNamespace(active_sequence=7, active_index=25)

    assert not _rtc_enabled_for_chunk_scheduling("SYNC")
    assert policy._previous_prefix(packet) is None
    assert policy._rtc_conditioning(packet, 99) == {}
    anchor, prefix = policy._b2_velocity_filter_context(packet)
    torch.testing.assert_close(anchor, torch.zeros(3))
    assert prefix is None


def test_chunk_scheduling_mode_switch_updates_rtc_without_reloading_policy() -> None:
    policy = AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    policy.chunk_scheduling_mode = "RTC"
    policy.rtc_config = RTCConfig(mode="inference")

    result = policy.set_chunk_scheduling_mode("SYNC")

    assert result == {
        "accepted": True,
        "previous_mode": "RTC",
        "chunk_scheduling_mode": "SYNC",
        "rtc_enabled": False,
    }
    assert policy.chunk_scheduling_mode == "SYNC"
    assert not policy.rtc_config.enabled
    assert policy.set_chunk_scheduling_mode("SYNC")["accepted"] is False
    rtc_result = policy.set_chunk_scheduling_mode("RTC")
    assert rtc_result["accepted"] is True
    assert policy.chunk_scheduling_mode == "RTC"
    assert policy.rtc_config.enabled
    with pytest.raises(ValueError, match="Unknown chunk scheduling mode"):
        policy.set_chunk_scheduling_mode("invalid")


def test_task_complete_termination_switch_updates_worker_without_reloading_policy() -> None:
    policy = AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    policy.stop_on_model_task_complete = False
    policy._task_complete_latched = True

    enabled = policy.set_stop_on_model_task_complete(True)
    assert enabled == {
        "accepted": True,
        "previous_enabled": False,
        "stop_on_model_task_complete": True,
    }
    assert policy._task_complete_latched is True

    disabled = policy.set_stop_on_model_task_complete(False)
    assert disabled["accepted"] is True
    assert policy.stop_on_model_task_complete is False
    assert policy._task_complete_latched is False
    with pytest.raises(ValueError, match="enabled must be a boolean"):
        policy.set_stop_on_model_task_complete(1)


def test_rtc_server_keeps_prefix_and_measured_delay_behavior() -> None:
    policy = AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    policy.chunk_scheduling_mode = "RTC"
    policy.device = torch.device("cpu")
    policy.low_level_hz = 50.0
    policy.rtc_config = RTCConfig(mode="inference", execution_horizon=2)
    policy.b2_action_representation = "velocity"
    policy.z1_action_representation = "ee_delta"
    policy._records_lock = Lock()
    policy._records = {7: SimpleNamespace(original=torch.arange(12).reshape(4, 3).float())}
    policy._mailbox_condition = Lock()
    packet = SimpleNamespace(active_sequence=7, active_index=1)

    assert _rtc_enabled_for_chunk_scheduling("RTC")
    torch.testing.assert_close(
        policy._previous_prefix(packet),
        torch.tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]),
    )
    assert policy._rtc_conditioning(packet, 11)["inference_delay"] == 11






def test_first_chunk_smoothing_does_not_fill_cold_start_delay_with_zeros() -> None:
    assert _b2_velocity_smoothing_transition_step(71, None) == 0
    assert _b2_velocity_smoothing_transition_step(12, torch.zeros(5, 3)) == 12






def test_rtc_skipped_prefix_keeps_previous_plan_and_delays_new_transition() -> None:
    actions = torch.zeros((5, len(EXECUTION_ACTION_NAMES)))
    velocity_indices = [EXECUTION_ACTION_NAMES.index(name) for name in B2_EXECUTION_VELOCITY_NAMES]
    actions[:, velocity_indices] = 1.0
    previous_prefix = torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])

    smoothed = _smooth_b2_execution_velocity(
        actions,
        EXECUTION_ACTION_NAMES,
        torch.tensor([0.1, 0.1, 0.1]),
        dt=0.02,
        time_constant_s=0.02 / torch.log(torch.tensor(2.0)).item(),
        transition_start_step=3,
        prefix_velocity=previous_prefix,
    )

    torch.testing.assert_close(
        smoothed[:, velocity_indices],
        torch.tensor(
            [
                [0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7],
                [0.5, 0.6, 0.7],
                [0.75, 0.8, 0.85],
                [0.875, 0.9, 0.925],
            ]
        ),
    )


def test_se2_increment_decode_and_velocity_modes_share_identical_execution_smoothing() -> None:
    velocity = torch.tensor([[0.2, -0.1, 0.3], [0.4, 0.0, -0.2], [-0.1, 0.2, 0.1]], dtype=torch.float32)
    increments = integrate_body_twist(velocity, dt=0.02)
    differentiated_velocity = se2_increment_to_body_twist(increments, dt=0.02)
    velocity_actions = torch.zeros((len(velocity), len(EXECUTION_ACTION_NAMES)))
    trajectory_actions = velocity_actions.clone()
    velocity_indices = [EXECUTION_ACTION_NAMES.index(name) for name in B2_EXECUTION_VELOCITY_NAMES]
    velocity_actions[:, velocity_indices] = velocity
    trajectory_actions[:, velocity_indices] = differentiated_velocity

    direct_result = _smooth_b2_execution_velocity(
        velocity_actions,
        EXECUTION_ACTION_NAMES,
        torch.zeros(3),
        dt=0.02,
        time_constant_s=0.1,
    )
    trajectory_result = _smooth_b2_execution_velocity(
        trajectory_actions,
        EXECUTION_ACTION_NAMES,
        torch.zeros(3),
        dt=0.02,
        time_constant_s=0.1,
    )

    torch.testing.assert_close(trajectory_result, direct_result, atol=1e-6, rtol=1e-6)
