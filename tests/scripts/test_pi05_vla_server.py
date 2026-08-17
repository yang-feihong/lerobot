import json
from io import BytesIO
from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.pi05.b2_action_transform import (
    EE_DELTA_ROTVEC_NAMES,
    differentiate_local_trajectory,
    integrate_body_twist,
)
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.scripts.pi05_vla_server import (
    B2_EXECUTION_VELOCITY_NAMES,
    EE_POSE_ACTION_NAMES,
    PROTOCOL_VERSION,
    AsyncRTCPolicy,
    MemoryObservation,
    _apply_completion_stop,
    _b2_velocity_smoothing_transition_step,
    _decode_discrete_actions,
    _load_checkpoint_contract,
    _resolve_checkpoint_action_representations,
    _smooth_b2_execution_velocity,
    _to_execution_actions,
    decode_observation_packet,
    execution_action_names,
)

EXECUTION_ACTION_NAMES = execution_action_names("ee_pose")


def test_metadata_v2_velocity_explicitly_migrates_to_ee_pose() -> None:
    assert _resolve_checkpoint_action_representations(
        2,
        {"representation": "velocity"},
        {"b2_action_representation": "velocity"},
    ) == ("velocity", "ee_pose")


def test_metadata_v2_rejects_retired_cumulative_local_trajectory() -> None:
    with pytest.raises(ValueError, match="retired cumulative-pose semantics"):
        _resolve_checkpoint_action_representations(
            2,
            {"representation": "local_trajectory"},
            {"b2_action_representation": "local_trajectory"},
        )


def test_metadata_v4_uses_both_explicit_representations() -> None:
    assert _resolve_checkpoint_action_representations(
        4,
        {"representation": "local_trajectory", "z1_representation": "ee_delta"},
        {
            "b2_action_representation": "local_trajectory",
            "z1_action_representation": "ee_delta",
        },
    ) == ("local_trajectory", "ee_delta")


def test_server_loads_missing_discrete_mode_in_historical_metadata_as_continuous_flow(tmp_path) -> None:
    state_names = ["arm_q_1"]
    action_names = [
        "b2_vx",
        "b2_vy",
        "b2_omega_z",
        "arm_teleop_inactive",
        "arm_reset",
        *[f"height_invariant_ee_{index}" for index in range(9)],
        "gripper_target",
        "task_complete",
    ]
    config = PI05Config(
        b2_action_representation="velocity",
        control_frequency_hz=50.0,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (1,)),
            "observation.images.base": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
            "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (16,))},
    )
    config.io_schema_resolved = True
    config.dataset_state_feature_names = state_names
    config.resolved_state_feature_names = state_names
    config.state_feature_indices = [0]
    config.dataset_action_feature_names = action_names
    config.action_feature_names = action_names
    config.dataset_camera_keys = ["observation.images.base", "observation.images.wrist"]
    config._save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    raw_config = json.loads(config_path.read_text())
    raw_config.pop("discrete_action_training_mode")
    config_path.write_text(json.dumps(raw_config))
    metadata_path = tmp_path / "pi05_deployment_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["version"] = 4
    metadata["action"].pop("discrete_training_mode")
    metadata["action"].pop("discrete_temporal_structure")
    metadata["action"]["boolean_decoding"].pop("output_values")
    metadata_path.write_text(json.dumps(metadata))

    contract = _load_checkpoint_contract(tmp_path, config, 50.0)

    assert contract.discrete_action_training_mode == "continuous_flow"
    assert contract.gripper_negative_value == pytest.approx(-1.0471976)
    assert contract.gripper_nonnegative_value == 0.0


def _schema_v2_names() -> tuple[str, ...]:
    names = list(EXECUTION_ACTION_NAMES)
    names[names.index("arm_active")] = "arm_teleop_inactive"
    return tuple(names)


def test_schema_v2_arm_gate_is_converted_before_low_level() -> None:
    names = _schema_v2_names()
    actions = torch.zeros((2, len(names)))
    gate_index = names.index("arm_teleop_inactive")
    actions[:, gate_index] = torch.tensor([0.0, 1.0])

    execution = _to_execution_actions(actions, names, "ee_pose")

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


def test_formal_server_rejects_legacy_arm_active_semantics() -> None:
    actions = torch.zeros((2, len(EXECUTION_ACTION_NAMES)))

    with pytest.raises(ValueError, match="must not output legacy arm_active"):
        _to_execution_actions(actions, EXECUTION_ACTION_NAMES, "ee_pose")


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
    policy._records = {4: SimpleNamespace(processed=processed, action_names=names)}
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


def test_local_trajectory_and_velocity_modes_share_identical_execution_smoothing() -> None:
    velocity = torch.tensor([[0.2, -0.1, 0.3], [0.4, 0.0, -0.2], [-0.1, 0.2, 0.1]], dtype=torch.float32)
    local_trajectory = integrate_body_twist(velocity, dt=0.02)
    differentiated_velocity = differentiate_local_trajectory(local_trajectory, dt=0.02)
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
            executed_ee_target=np.zeros(9, dtype=np.float32),
        )
        for step in (10, 11, 12, 14)
    )

    with pytest.raises(ValueError, match="Missing exact 50 Hz telemetry sample"):
        AsyncRTCPolicy._sample_memory_records(history, current_step=14, count=2, stride=1)


def test_protocol_v4_decodes_consecutive_50hz_telemetry_history() -> None:
    state_names = ("state_a", "state_b")
    states = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    ee_targets = np.zeros((3, len(EE_POSE_ACTION_NAMES)), dtype=np.float32)
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
        history_sim_steps=np.asarray([10, 11, 12]),
        history_states=states,
        history_active_sequences=np.asarray([2, 3, 3]),
        history_active_indices=np.asarray([12, 1, 2]),
        history_executed_ee_targets=ee_targets,
        base_jpeg=jpeg,
        wrist_jpeg=jpeg,
    )

    packet = decode_observation_packet(body.getvalue())

    assert packet.control_epoch == 7
    assert [record.sim_step for record in packet.telemetry_history] == [10, 11, 12]
    np.testing.assert_array_equal(packet.telemetry_history[-1].state, states[-1])
