from random import Random
from types import SimpleNamespace

import pytest
import torch

from lerobot.datasets.factory import resolve_delta_timestamps, resolve_random_history_sampling
from lerobot.datasets.temporal_history import RandomHistorySamplingConfig
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05.processor_pi05 import (
    OBS_ACTION_HISTORY,
    Pi05SplitActionHistoryProcessorStep,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE


def test_hierarchical_history_sampling_is_ordered_bounded_and_nonuniform():
    config = RandomHistorySamplingConfig(
        keys=("observation.images.base", "observation.images.wrist"),
        num_frames=6,
        fps=50,
        nominal_interval_seconds=0.5,
        global_interval_std_seconds=0.15,
        local_interval_std_seconds=0.05,
        min_interval_seconds=0.25,
        max_interval_seconds=1.0,
    )
    rng = Random(7)
    samples = [config.sample_delta_indices(rng) for _ in range(128)]

    assert all(offsets[-1] == 0 for offsets in samples)
    assert all(
        all(left < right for left, right in zip(offsets, offsets[1:], strict=False))
        for offsets in samples
    )
    gaps = [
        right - left
        for offsets in samples
        for left, right in zip(offsets, offsets[1:], strict=False)
    ]
    assert min(gaps) >= 13
    assert max(gaps) <= 50
    assert len({tuple(offsets) for offsets in samples}) > 100


def test_random_history_sampling_resolves_only_configured_cameras():
    selected_key = "observation.images.base"
    ignored_key = "observation.images.debug"
    config = SimpleNamespace(
        mem_vit_random_interval_sampling=True,
        mem_vit_num_frames=6,
        mem_vit_frame_interval_seconds=0.5,
        mem_vit_global_interval_std_seconds=0.15,
        mem_vit_local_interval_std_seconds=0.05,
        mem_vit_min_interval_seconds=0.25,
        mem_vit_max_interval_seconds=1.0,
        image_features={selected_key: object()},
    )
    metadata = SimpleNamespace(fps=50, camera_keys=[selected_key, ignored_key])

    sampling = resolve_random_history_sampling(config, metadata, seed=123)

    assert sampling is not None
    assert sampling.keys == (selected_key,)
    assert sampling.seed == 123


def test_mem_modalities_use_independent_50hz_sampling_clocks():
    image_key = "observation.images.base"
    state_names = ["b2_position_x", "b2_position_y", "b2_yaw"]
    config = SimpleNamespace(
        control_frequency_hz=50,
        mem_vit_enabled=True,
        mem_vit_num_frames=6,
        mem_vit_frame_interval_seconds=0.5,
        mem_vit_frame_stride=1,
        state_action_encoding="continuous",
        state_num_frames=13,
        state_history_frame_interval_seconds=0.04,
        state_history_frame_stride=1,
        action_history_enabled=True,
        io_schema_resolved=False,
        b2_action_representation="pose_delta",
        z1_action_representation="ee_delta",
        b2_global_pose_state_indices=None,
        chunk_size=50,
        action_delta_indices=list(range(50)),
        reward_delta_indices=None,
        observation_delta_indices=None,
        image_features={image_key: object()},
    )
    metadata = SimpleNamespace(
        fps=50,
        features={ACTION: {}, OBS_STATE: {"names": state_names}, image_key: {}},
        camera_keys=[image_key],
    )

    timestamps = resolve_delta_timestamps(config, metadata)

    image_history = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0]
    state_history = [-0.48 + 0.04 * index for index in range(13)]
    assert timestamps[image_key] == pytest.approx(image_history)
    assert timestamps[OBS_STATE][:13] == pytest.approx(state_history)
    assert timestamps[OBS_STATE] == pytest.approx(state_history)
    assert timestamps[ACTION][:12] == pytest.approx(state_history[:-1])
    assert timestamps[ACTION][12:] == pytest.approx([index / 50 for index in range(51)])
    assert config.mem_vit_frame_stride == 25
    assert config.state_history_frame_stride == 2


def test_action_history_split_keeps_target_and_padding_independent():
    step = Pi05SplitActionHistoryProcessorStep(
        history_length=12,
        target_length=50,
        action_indices=(0, 2, 4),
    )
    combined = torch.arange(2 * 62 * 16, dtype=torch.float32).reshape(2, 62, 16)
    combined_pad = torch.zeros(2, 62, dtype=torch.bool)
    combined_pad[0, :5] = True
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: torch.zeros(2, 13, 4)},
        TransitionKey.ACTION: combined,
        TransitionKey.COMPLEMENTARY_DATA: {f"{ACTION}_is_pad": combined_pad},
    }

    result = step(transition)

    assert torch.equal(result[TransitionKey.OBSERVATION][OBS_ACTION_HISTORY], combined[:, :12, (0, 2, 4)])
    assert torch.equal(result[TransitionKey.ACTION], combined[:, 12:])
    complementary = result[TransitionKey.COMPLEMENTARY_DATA]
    assert torch.equal(complementary[f"{OBS_ACTION_HISTORY}_is_pad"], combined_pad[:, :12])
    assert torch.equal(complementary[f"{ACTION}_is_pad"], combined_pad[:, 12:])


def test_deployment_memory_plans_preserve_fixed_shapes_while_masking_startup_padding():
    policy = PI05Policy.__new__(PI05Policy)
    policy.config = SimpleNamespace(
        mem_vit_enabled=True,
        mem_vit_num_frames=6,
        mem_vit_min_num_frames=None,
        image_features={"observation.images.base": object()},
    )
    image_key = "observation.images.base"
    image_pad = torch.tensor([[True, True, True, True, True, False]])
    batch = {
        image_key: torch.zeros(1, 6, 3, 8, 8),
        f"{image_key}_is_pad": image_pad,
        OBS_STATE: torch.zeros(1, 13, 4),
        f"{OBS_STATE}_is_pad": torch.tensor([[True] * 12 + [False]]),
        OBS_ACTION_HISTORY: torch.zeros(1, 12, 3),
        f"{OBS_ACTION_HISTORY}_is_pad": torch.ones(1, 12, dtype=torch.bool),
    }

    image_window, image_lengths, image_mask = policy._mem_vit_window_plan(
        batch, randomize=False, preserve_input_window=True
    )
    state_window, state_lengths, state_mask = policy._fixed_memory_window_plan(
        batch,
        key=OBS_STATE,
        max_frames=13,
        preserve_input_window=True,
    )
    action_window, action_lengths, action_mask = policy._fixed_memory_window_plan(
        batch,
        key=OBS_ACTION_HISTORY,
        max_frames=12,
        allow_empty=True,
        preserve_input_window=True,
    )

    assert image_window == 6
    assert image_lengths.tolist() == [1]
    assert image_mask.tolist() == [[False, False, False, False, False, True]]
    assert state_window == 13
    assert state_lengths.tolist() == [1]
    assert state_mask.tolist() == [[False] * 12 + [True]]
    assert action_window == 12
    assert action_lengths.tolist() == [0]
    assert action_mask.tolist() == [[False] * 12]


def test_random_state_history_samples_visible_suffix_per_example_reproducibly():
    policy = PI05Policy.__new__(PI05Policy)
    batch = {OBS_STATE: torch.arange(1024 * 13 * 4).reshape(1024, 13, 4)}

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        window, lengths, mask = policy._fixed_memory_window_plan(
            batch, key=OBS_STATE, max_frames=13, randomize=True
        )
        torch.manual_seed(17)
        repeated = policy._fixed_memory_window_plan(batch, key=OBS_STATE, max_frames=13, randomize=True)

    assert set(lengths.tolist()) == set(range(1, 14))
    assert window == repeated[0] == 13
    assert torch.equal(lengths, repeated[1])
    assert torch.equal(mask, repeated[2])
    assert torch.equal(mask.sum(dim=1), lengths)
    assert mask[:, -1].all()
    positions = torch.arange(window)[None, :]
    assert torch.equal(mask, positions >= window - lengths[:, None])


def test_random_state_history_respects_real_padding_and_keeps_current_state():
    policy = PI05Policy.__new__(PI05Policy)
    policy.config = SimpleNamespace(state_action_encoding="continuous")
    available = torch.tensor([1, 2, 4, 7, 13] * 64)
    original_valid = torch.arange(13)[None, :] >= 13 - available[:, None]
    batch = {
        OBS_STATE: torch.arange(available.numel() * 13 * 4).reshape(-1, 13, 4),
        f"{OBS_STATE}_is_pad": ~original_valid,
    }

    window, lengths, mask = policy._fixed_memory_window_plan(
        batch, key=OBS_STATE, max_frames=13, randomize=True
    )

    assert (lengths >= 1).all()
    assert (lengths <= available).all()
    assert (lengths[available == 1] == 1).all()
    assert not (mask & ~original_valid[:, -window:]).any()
    assert torch.equal(mask.sum(dim=1), lengths)
    assert mask[:, -1].all()
    # Cropping only removes old frames; the actual current state is unchanged.
    state = policy._prepare_state_history(batch, window)
    assert torch.equal(state[:, -1], batch[OBS_STATE][:, -1])


@pytest.mark.parametrize(
    "entrypoint,training", [("forward", False), ("forward", True), ("predict_action_chunk", True)]
)
def test_state_history_randomization_is_training_only(monkeypatch, entrypoint, training):
    policy = PI05Policy.__new__(PI05Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(state_action_encoding="continuous", state_num_frames=13)
    policy.train(training)
    batch = {OBS_STATE: torch.zeros(2, 13, 4)}
    calls = []
    original_plan = policy._fixed_memory_window_plan

    def record_plan(batch, **kwargs):
        calls.append(kwargs)
        return original_plan(batch, **kwargs)

    class InputsPreparedError(Exception):
        pass

    def stop_after_planning(*args, **kwargs):
        raise InputsPreparedError

    monkeypatch.setattr(policy, "_fixed_memory_window_plan", record_plan)
    monkeypatch.setattr(policy, "_mem_vit_window_plan", lambda *args, **kwargs: (1, None, None))
    monkeypatch.setattr(policy, "_preprocess_images", stop_after_planning)
    with pytest.raises(InputsPreparedError):
        getattr(policy, entrypoint)(batch)
    expected = {"key": OBS_STATE, "max_frames": 13}
    if entrypoint == "forward":
        expected["randomize"] = training
    else:
        expected["preserve_input_window"] = True
    assert calls == [expected]


def test_memory_history_default_remains_deterministic_without_consuming_rng():
    policy = PI05Policy.__new__(PI05Policy)
    batch = {
        OBS_STATE: torch.zeros(2, 13, 4),
        OBS_ACTION_HISTORY: torch.zeros(2, 12, 3),
    }
    rng_before = torch.random.get_rng_state()
    for key, max_frames, allow_empty in [(OBS_STATE, 13, False), (OBS_ACTION_HISTORY, 12, True)]:
        window, lengths, mask = policy._fixed_memory_window_plan(
            batch, key=key, max_frames=max_frames, allow_empty=allow_empty
        )
        assert window == max_frames
        assert lengths.tolist() == [max_frames, max_frames]
        assert mask.all()
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_continuous_state_and_action_history_do_not_require_mem_vit():
    image_key = "observation.images.base"
    config = SimpleNamespace(
        control_frequency_hz=50,
        mem_vit_enabled=False,
        mem_vit_num_frames=6,
        mem_vit_frame_interval_seconds=0.5,
        mem_vit_frame_stride=1,
        state_action_encoding="continuous",
        state_num_frames=13,
        state_history_frame_interval_seconds=0.04,
        state_history_frame_stride=1,
        action_history_enabled=True,
        io_schema_resolved=False,
        b2_action_representation="velocity",
        z1_action_representation="ee_delta",
        b2_global_pose_state_indices=None,
        chunk_size=50,
        action_delta_indices=list(range(50)),
        reward_delta_indices=None,
        observation_delta_indices=None,
        image_features={image_key: object()},
    )
    metadata = SimpleNamespace(
        fps=50,
        features={ACTION: {}, OBS_STATE: {"names": ["state"]}, image_key: {}},
        camera_keys=[image_key],
    )

    timestamps = resolve_delta_timestamps(config, metadata)

    assert image_key not in timestamps
    assert timestamps[OBS_STATE] == pytest.approx([-0.48 + 0.04 * index for index in range(13)])
    assert timestamps[ACTION][:12] == pytest.approx(timestamps[OBS_STATE][:-1])
    assert timestamps[ACTION][12:] == pytest.approx([index / 50 for index in range(51)])


def test_ee_delta_loads_current_plus_one_extra_future_target():
    config = SimpleNamespace(
        control_frequency_hz=50,
        mem_vit_enabled=False,
        mem_vit_num_frames=1,
        mem_vit_frame_interval_seconds=0.5,
        mem_vit_frame_stride=1,
        state_action_encoding="text",
        state_num_frames=1,
        state_history_frame_interval_seconds=0.04,
        state_history_frame_stride=1,
        action_history_enabled=False,
        io_schema_resolved=False,
        b2_action_representation="velocity",
        z1_action_representation="ee_delta",
        b2_global_pose_state_indices=None,
        chunk_size=50,
        action_delta_indices=list(range(50)),
        reward_delta_indices=None,
        observation_delta_indices=None,
        image_features={},
    )
    metadata = SimpleNamespace(
        fps=50,
        features={ACTION: {}, OBS_STATE: {"names": ["state"]}},
        camera_keys=[],
    )

    timestamps = resolve_delta_timestamps(config, metadata)

    assert timestamps[ACTION] == pytest.approx([index / 50 for index in range(51)])


def test_ee_state_delta_uses_only_the_inference_time_state_anchor():
    config = SimpleNamespace(
        control_frequency_hz=50,
        mem_vit_enabled=False,
        mem_vit_num_frames=1,
        mem_vit_frame_interval_seconds=0.5,
        mem_vit_frame_stride=1,
        state_action_encoding="text",
        state_num_frames=1,
        state_history_frame_interval_seconds=0.04,
        state_history_frame_stride=1,
        action_history_enabled=False,
        io_schema_resolved=False,
        b2_action_representation="velocity",
        z1_action_representation="ee_state_delta",
        ee_supervision_source="control_action",
        b2_global_pose_state_indices=None,
        chunk_size=50,
        action_delta_indices=list(range(50)),
        reward_delta_indices=None,
        observation_delta_indices=None,
        image_features={},
    )
    metadata = SimpleNamespace(
        fps=50,
        features={ACTION: {}, OBS_STATE: {"names": ["state"]}},
        camera_keys=[],
    )

    timestamps = resolve_delta_timestamps(config, metadata)

    assert OBS_STATE not in timestamps
    assert timestamps[ACTION] == pytest.approx([index / 50 for index in range(50)])
