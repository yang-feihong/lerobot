from io import BytesIO
from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
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
from lerobot.scripts.pi05_vla_server import (
    B2_EXECUTION_VELOCITY_NAMES,
    B2_GLOBAL_POSE_NAMES,
    EE_DELTA_ACTION_NAMES,
    EE_POSE_ACTION_NAMES,
    PROTOCOL_VERSION,
    ActionRecord,
    AsyncRTCPolicy,
    MemoryObservation,
    SE2TrajectoryController,
    _apply_completion_stop,
    _b2_velocity_smoothing_transition_step,
    _decode_discrete_actions,
    _load_checkpoint_contract,
    _reanchor_b2_pose_delta,
    _resolve_checkpoint_action_representations,
    _smooth_b2_execution_velocity,
    _to_execution_actions,
    decode_observation_packet,
    encode_action_packet,
    execution_action_names,
)

EXECUTION_ACTION_NAMES = execution_action_names("ee_delta")


def test_action_packet_contains_replayable_model_and_execution_outputs() -> None:
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
    )

    with np.load(BytesIO(encode_action_packet(record)), allow_pickle=False) as packet:
        np.testing.assert_array_equal(packet["original_actions"], record.original.numpy())
        np.testing.assert_array_equal(packet["unsmoothed_execution_actions"], record.unsmoothed.numpy())
        np.testing.assert_array_equal(packet["processed_actions"], record.processed.numpy())
        np.testing.assert_array_equal(packet["b2_reference"], record.b2_reference.numpy())
        assert int(packet["control_epoch"]) == 7
        assert float(packet["predict_seconds"]) == pytest.approx(0.15)


def test_b2_rtc_pose_delta_prefix_is_reexpressed_in_new_inference_frame() -> None:
    old_anchor = torch.tensor([10.0, 20.0, np.pi / 2])
    new_anchor = torch.tensor([10.0, 21.0, np.pi / 2])
    old_relative_targets = torch.tensor([[1.0, 0.0, 0.1], [2.0, 0.0, 0.2]])

    rebased = _reanchor_b2_pose_delta(old_relative_targets, old_anchor, new_anchor)

    torch.testing.assert_close(rebased[:, :2], torch.tensor([[0.0, 0.0], [1.0, 0.0]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(rebased[:, 2], torch.tensor([0.1, 0.2]), atol=1e-6, rtol=0)


def test_se2_feedback_tracks_world_target_in_actual_body_frame_with_rate_limits() -> None:
    controller = SE2TrajectoryController(50.0)
    actual = np.asarray([3.5, 19.5, np.pi], dtype=np.float64)
    target = np.asarray([3.0, 19.5, np.pi], dtype=np.float64)
    commands = []

    for sim_step in range(50):
        command = controller.command(
            control_epoch=0,
            sim_step=sim_step,
            actual_pose=actual,
            target_pose=target,
        )
        commands.append(command)
        actual[0] += np.cos(actual[2]) * command[0] / 50.0
        actual[1] += np.sin(actual[2]) * command[0] / 50.0
        actual[2] += command[2] / 50.0

    commands = np.stack(commands)
    assert commands[0, 0] == pytest.approx(0.03)
    assert np.abs(np.diff(commands[:, :2], axis=0)).max() <= 1.5 / 50.0 + 1e-9
    assert np.abs(np.diff(commands[:, 2], axis=0)).max() <= 2.0 / 50.0 + 1e-9
    assert actual[0] < 3.5


def test_se2_feedback_resets_rate_limiter_on_control_epoch_change() -> None:
    controller = SE2TrajectoryController(50.0)
    actual = np.zeros(3, dtype=np.float64)
    target = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    controller.command(control_epoch=0, sim_step=100, actual_pose=actual, target_pose=target)
    command = controller.command(
        control_epoch=1,
        sim_step=100,
        actual_pose=actual,
        target_pose=None,
    )

    np.testing.assert_array_equal(command, np.zeros(3))


def test_server_tracks_rtc_indexed_anchor_trajectory() -> None:
    policy = AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    policy.b2_execution_mode = "se2_feedback"
    policy._records_lock = Lock()
    policy._b2_controller_lock = Lock()
    policy._b2_controller = SE2TrajectoryController(50.0)
    reference = torch.zeros((50, 3), dtype=torch.float32)
    reference[:, 0] = torch.linspace(0.01, 0.50, 50)
    policy._records = {
        4: SimpleNamespace(
            sequence=4,
            control_epoch=2,
            b2_reference=reference,
            b2_anchor=np.asarray([3.5, 19.5, np.pi], dtype=np.float32),
            processed=torch.zeros((50, len(EXECUTION_ACTION_NAMES))),
        )
    }

    result = policy.b2_feedback_control(
        control_epoch=2,
        sim_step=10,
        sequence=4,
        action_index=3,
        actual_pose=np.asarray([3.5, 19.5, np.pi], dtype=np.float64),
    )

    assert result["target_index"] == 8
    assert result["target_pose"][0] < 3.5
    assert result["command"][0] == pytest.approx(0.03)


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
def test_infer_records_model_action_names_and_source_step_anchor(
    z1_representation: str,
    expected_anchor_kind: str,
    expected_anchor: np.ndarray,
) -> None:
    model_names = (*B2_EXECUTION_VELOCITY_NAMES, *EE_DELTA_ACTION_NAMES, "gripper_target")
    policy = AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    policy.preprocessor = lambda batch: batch
    policy.postprocessor = lambda actions: actions
    policy.policy = SimpleNamespace(
        predict_action_chunk=lambda *_args, **_kwargs: torch.zeros((1, 2, len(model_names)))
    )
    policy._make_batch = lambda _packet: {}
    policy._previous_prefix = lambda _packet: None
    policy._estimated_delay = lambda: (0, 50.0)
    policy._b2_velocity_filter_context = lambda _packet: (torch.zeros(3), None)
    policy.postprocessed_action_names = model_names
    policy.model_action_names = model_names
    policy.discrete_action_training_mode = "continuous_flow"
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
    policy._latencies = []
    policy._next_sequence = 1

    record = policy._infer(
        SimpleNamespace(
            control_epoch=3,
            sim_step=25,
            server_received_ns=0,
            executed_ee_target=np.zeros(9, dtype=np.float32),
            actual_ee_state=np.ones(9, dtype=np.float32),
            state_names=B2_GLOBAL_POSE_NAMES,
            state=np.asarray([1.0, 2.0, 0.3], dtype=np.float32),
        )
    )

    assert record.model_action_names == model_names
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


def test_server_latches_an_executed_completion_until_sim_step_resets() -> None:
    policy = AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    policy._records_lock = Lock()
    policy._mailbox_condition = __import__("threading").Condition()
    policy._observation_history = __import__("collections").deque(maxlen=4)
    policy._image_history = __import__("collections").deque(maxlen=4)
    policy._records = {}
    policy._mailbox = None
    policy._mailbox_version = 0
    policy._previous_observation_clock = None
    policy._control_epoch = None
    policy._task_complete_latched = False
    policy.stop_on_model_task_complete = True
    policy.recorder = SimpleNamespace(observation=lambda _: None)
    actions = torch.zeros((3, len(EXECUTION_ACTION_NAMES)))
    actions[1, EXECUTION_ACTION_NAMES.index("task_complete")] = 1.0
    policy._records[4] = SimpleNamespace(processed=actions, action_names=EXECUTION_ACTION_NAMES)

    def packet(step: int, active_index: int, control_epoch: int = 0) -> SimpleNamespace:
        telemetry = MemoryObservation(
            sim_step=step,
            active_sequence=4,
            active_index=active_index,
            executed_b2_command=np.zeros(3, dtype=np.float32),
            state=np.zeros(1, dtype=np.float32),
            state_names=("state",),
            executed_ee_target=np.zeros(9, dtype=np.float32),
        )
        return SimpleNamespace(
            control_epoch=control_epoch,
            sim_step=step,
            active_sequence=4,
            active_index=active_index,
            telemetry_history=(telemetry,),
            base_jpeg=np.zeros(1, dtype=np.uint8),
            wrist_jpeg=np.zeros(1, dtype=np.uint8),
            server_received_ns=step + 1,
        )

    policy.submit(packet(10, 2))
    assert policy._task_complete_latched is True
    policy.submit(packet(9, 0))
    assert policy._task_complete_latched is True
    assert policy._mailbox.sim_step == 10
    policy.submit(packet(11, 0, control_epoch=1))
    assert policy._task_complete_latched is False
    assert policy._mailbox.sim_step == 11
    assert [record.sim_step for record in policy._observation_history] == [11]


def test_server_does_not_latch_completion_when_model_stop_is_disabled() -> None:
    policy = AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    policy._records_lock = Lock()
    policy._mailbox_condition = __import__("threading").Condition()
    policy._observation_history = __import__("collections").deque(maxlen=4)
    policy._image_history = __import__("collections").deque(maxlen=4)
    actions = torch.zeros((3, len(EXECUTION_ACTION_NAMES)))
    actions[1, EXECUTION_ACTION_NAMES.index("task_complete")] = 1.0
    policy._records = {4: SimpleNamespace(processed=actions, action_names=EXECUTION_ACTION_NAMES)}
    policy._mailbox = None
    policy._mailbox_version = 0
    policy._previous_observation_clock = None
    policy._control_epoch = None
    policy._task_complete_latched = False
    policy.stop_on_model_task_complete = False
    policy.recorder = SimpleNamespace(observation=lambda _: None)
    telemetry = MemoryObservation(
        sim_step=10,
        active_sequence=4,
        active_index=2,
        executed_b2_command=np.zeros(3, dtype=np.float32),
        state=np.zeros(1, dtype=np.float32),
        state_names=("state",),
        executed_ee_target=np.zeros(9, dtype=np.float32),
    )
    packet = SimpleNamespace(
        control_epoch=0,
        sim_step=10,
        active_sequence=4,
        active_index=2,
        telemetry_history=(telemetry,),
        base_jpeg=np.zeros(1, dtype=np.uint8),
        wrist_jpeg=np.zeros(1, dtype=np.uint8),
        server_received_ns=11,
    )

    policy.submit(packet)

    assert policy._task_complete_latched is False


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


def test_source_state_requires_model_fields_but_not_unused_dataset_storage_fields() -> None:
    policy = AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    policy.source_state_names = ("arm_q_1", "unused_ee_label", "b2_body_height")
    policy.selected_state_names = ("arm_q_1", "b2_body_height")
    record = SimpleNamespace(
        state_names=("b2_body_height", "arm_q_1"),
        state=np.asarray([0.48, 0.25], dtype=np.float32),
    )

    source = policy._source_state(record)

    torch.testing.assert_close(source, torch.tensor([0.25, 0.0, 0.48]))


def test_source_state_rejects_missing_model_field() -> None:
    policy = AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    policy.source_state_names = ("arm_q_1", "unused_ee_label", "b2_body_height")
    policy.selected_state_names = ("arm_q_1", "b2_body_height")
    record = SimpleNamespace(
        state_names=("arm_q_1",),
        state=np.asarray([0.25], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="missing model state fields.*b2_body_height"):
        policy._source_state(record)


def test_first_chunk_smoothing_does_not_fill_cold_start_delay_with_zeros() -> None:
    assert _b2_velocity_smoothing_transition_step(71, None) == 0
    assert _b2_velocity_smoothing_transition_step(12, torch.zeros(5, 3)) == 12


def test_action_history_uses_last_executed_row_and_low_level_ee_target_feedback() -> None:
    policy = AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    policy._records_lock = Lock()
    names = execution_action_names("ee_delta")
    processed = torch.zeros((3, len(names)))
    processed[:, names.index("b2_vx")] = torch.tensor([10.0, 20.0, 30.0])
    processed[:, names.index("arm_active")] = 1.0
    policy._records = {
        4: SimpleNamespace(
            processed=processed,
            action_names=names,
            b2_execution_mode="velocity_chunk",
        )
    }
    policy.source_action_names = ("b2_vx", "height_invariant_ee_x")
    policy.action_history_indices = (0, 1)
    ee_target = np.zeros(9, dtype=np.float32)
    ee_target[-3] = 0.42
    observation = SimpleNamespace(
        active_sequence=4,
        active_index=2,
        executed_ee_target=ee_target,
    )

    history = policy._executed_source_action(observation)

    torch.testing.assert_close(history, torch.tensor([20.0, 0.42]))


def test_feedback_action_history_uses_low_level_executed_b2_command() -> None:
    policy = AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    policy._records_lock = Lock()
    names = execution_action_names("ee_delta")
    processed = torch.zeros((3, len(names)))
    processed[:, names.index("b2_vx")] = 20.0
    processed[:, names.index("arm_active")] = 1.0
    policy._records = {
        4: SimpleNamespace(
            sequence=4,
            processed=processed,
            action_names=names,
            b2_execution_mode="se2_feedback",
        )
    }
    policy.source_action_names = ("b2_vx", "b2_vy", "b2_omega_z")
    policy.action_history_indices = (0, 1, 2)
    observation = SimpleNamespace(
        active_sequence=4,
        active_index=2,
        executed_b2_command=np.asarray([0.2, -0.1, 0.3], dtype=np.float32),
        executed_ee_target=np.zeros(9, dtype=np.float32),
    )

    history = policy._executed_source_action(observation)

    torch.testing.assert_close(history, torch.tensor([0.2, -0.1, 0.3]))


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


def test_memory_sampling_is_right_aligned_and_marks_episode_start_padding() -> None:
    history = tuple(
        MemoryObservation(
            sim_step=step,
            state=np.asarray([step], dtype=np.float32),
            state_names=("state",),
            active_sequence=-1,
            active_index=0,
            executed_b2_command=np.zeros(3, dtype=np.float32),
            executed_ee_target=np.zeros(9, dtype=np.float32),
        )
        for step in range(10, 21)
    )

    records, is_pad = AsyncRTCPolicy._sample_memory_records(history, current_step=20, count=6, stride=5)

    assert [record.sim_step for record in records] == [10, 10, 10, 10, 15, 20]
    assert is_pad.tolist() == [[True, True, True, False, False, False]]


def test_memory_sampling_rejects_a_missing_50hz_sample() -> None:
    history = tuple(
        MemoryObservation(
            sim_step=step,
            state=np.asarray([step], dtype=np.float32),
            state_names=("state",),
            active_sequence=-1,
            active_index=0,
            executed_b2_command=np.zeros(3, dtype=np.float32),
            executed_ee_target=np.zeros(9, dtype=np.float32),
        )
        for step in (10, 11, 12, 14)
    )

    with pytest.raises(ValueError, match="Missing exact 50 Hz telemetry sample"):
        AsyncRTCPolicy._sample_memory_records(history, current_step=14, count=2, stride=1)


def test_protocol_v6_decodes_consecutive_50hz_telemetry_history() -> None:
    state_names = ("state_a", "state_b")
    states = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    ee_targets = np.zeros((3, len(EE_POSE_ACTION_NAMES)), dtype=np.float32)
    b2_commands = np.asarray([[0.1, 0.0, 0.0], [0.2, -0.1, 0.0], [0.3, -0.1, 0.2]], dtype=np.float32)
    ok, jpeg = __import__("cv2").imencode(".jpg", np.zeros((4, 4, 3), dtype=np.uint8))
    assert ok
    body = BytesIO()
    np.savez(
        body,
        protocol_version=np.asarray(PROTOCOL_VERSION),
        control_epoch=np.asarray(7),
        sim_step=np.asarray(12),
        active_sequence=np.asarray(3),
        active_index=np.asarray(2),
        task=np.asarray("test"),
        state=states[-1],
        state_names=np.asarray(state_names),
        executed_ee_target=ee_targets[-1],
        executed_ee_target_names=np.asarray(EE_POSE_ACTION_NAMES),
        actual_ee_state=ee_targets[-1],
        history_sim_steps=np.asarray([10, 11, 12]),
        history_states=states,
        history_active_sequences=np.asarray([2, 3, 3]),
        history_active_indices=np.asarray([12, 1, 2]),
        history_executed_b2_commands=b2_commands,
        history_executed_ee_targets=ee_targets,
        base_jpeg=jpeg,
        wrist_jpeg=jpeg,
    )

    packet = decode_observation_packet(body.getvalue())

    assert packet.control_epoch == 7
    assert [record.sim_step for record in packet.telemetry_history] == [10, 11, 12]
    np.testing.assert_array_equal(packet.telemetry_history[-1].state, states[-1])
    np.testing.assert_array_equal(packet.telemetry_history[-1].executed_b2_command, b2_commands[-1])
