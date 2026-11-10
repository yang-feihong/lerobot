"""History must remain factual and independent of the model output schema."""

from pathlib import Path

import pytest
import torch

from lerobot.datasets.semantic_views import SemanticView, SemanticViewCatalog, apply_semantic_views_to_batch
from lerobot.policies.pi05.action_history import action_history_schema
from lerobot.policies.pi05.b2_action_transform import CONTROL_EXTENDED_DATASET_ACTION_NAMES
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch
from lerobot.policies.pi05.processor_pi05 import OBS_ACTION_HISTORY, _action_history_steps
from lerobot.types import TransitionKey


def config(**kwargs):
    # Keep explicit v1 coverage: saved 13D models must not silently become 21D.
    if "action_history_feature_names" not in kwargs:
        from lerobot.policies.pi05.b2_action_transform import DATASET_ACTION_NAMES
        kwargs["action_history_feature_names"] = [name for name in DATASET_ACTION_NAMES[:-1]
            if name not in {"arm_teleop_inactive", "arm_reset"}]
        kwargs.setdefault("action_history_ee_semantics", "active_only_v1")
    return PI05Config(
        device="cpu",
        state_action_encoding="continuous",
        z1_action_representation="ee_state_delta",
        ee_delta_rotation_representation="rotvec",
        action_history_enabled=True,
        **kwargs,
    )


@pytest.mark.parametrize("extra,reorder", [(0, False), (4, False), (7, True)])
@pytest.mark.parametrize("ee_source", ["control_action", "recorded_action"])
@pytest.mark.parametrize("predict_complete,predict_ee", [(True, True), (False, True), (False, False)])
def test_named_selection_stats_and_projection_ignore_output_dimensions(
    extra, reorder, predict_complete, predict_ee, ee_source
):
    source = list(CONTROL_EXTENDED_DATASET_ACTION_NAMES) + [f"future_command_{i}" for i in range(extra)]
    if reorder:
        source.reverse()
    cfg = config(
        dataset_action_feature_names=source,
        action_predict_task_complete=predict_complete,
        action_predict_ee_pose=predict_ee,
        ee_supervision_source=ee_source,
    )
    cfg.max_action_dim = 3  # The output padding capacity is NOT the history capacity.
    names, indices, size = action_history_schema(cfg)
    assert len(names) == 13
    assert "arm_teleop_inactive" not in names
    assert "arm_reset" not in names
    assert "task_complete" not in names
    prefix = "control_" if ee_source == "control_action" else ""
    assert source[indices[names.index("height_invariant_ee_x")]] == prefix + "height_invariant_ee_x"
    low = torch.arange(size, dtype=torch.float32)
    stats = {"action": {"q01": low, "q99": low + 2, "count": torch.tensor([50])}}
    split, normalizer = _action_history_steps(cfg, stats)
    raw = low.expand(2, 62, size).clone() + 1
    before = raw.clone()
    pad = torch.zeros(2, 62, dtype=torch.bool)
    pad[0, :4] = True
    result = split(
        {
            TransitionKey.ACTION: raw,
            TransitionKey.OBSERVATION: {},
            TransitionKey.COMPLEMENTARY_DATA: {"action_is_pad": pad},
        }
    )
    torch.testing.assert_close(
        result[TransitionKey.OBSERVATION][OBS_ACTION_HISTORY], raw[:, :12, list(indices)]
    )
    torch.testing.assert_close(normalizer.stats[OBS_ACTION_HISTORY]["q01"], low[list(indices)])
    result = normalizer(result)
    history = result[TransitionKey.OBSERVATION][OBS_ACTION_HISTORY]
    torch.testing.assert_close(history, torch.zeros(2, 12, 13))
    torch.testing.assert_close(result[TransitionKey.ACTION], before[:, 12:])
    assert torch.equal(result[TransitionKey.COMPLEMENTARY_DATA][OBS_ACTION_HISTORY + "_is_pad"], pad[:, :12])
    model = PI05Pytorch.__new__(PI05Pytorch)
    torch.nn.Module.__init__(model)
    model.config = cfg
    model.gradient_checkpointing_enabled = False
    model.action_memory_proj = torch.nn.Linear(cfg.max_action_history_dim, 24)
    tokens = model.embed_action_memory(history + 1)
    assert tokens.shape == (2, 12, 24)
    tokens.sum().backward()
    assert model.action_memory_proj.weight.grad.abs().sum() > 0


def test_input_commands_can_be_added_removed_and_reordered_by_name():
    cfg = config(
        dataset_action_feature_names=["new_command", "b2_vx"],
        action_history_feature_names=["b2_vx", "new_command"],
    )
    assert action_history_schema(cfg) == (("b2_vx", "new_command"), (1, 0), 2)
    cfg.action_history_feature_names = ["new_command"]
    assert action_history_schema(cfg)[1] == (0,)


def test_default_history_names_are_frozen_into_saved_config():
    import draccus

    cfg = PI05Config(device="cpu", state_action_encoding="continuous", action_history_enabled=True)
    assert draccus.encode(cfg)["action_history_feature_names"] == list(action_history_schema(cfg)[0])
    assert cfg.action_history_ee_semantics == "mode_reset_target_v2"
    assert len(cfg.action_history_feature_names) == 21


def test_explicit_legacy_history_keeps_mode_fields_and_order(tmp_path):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.b2_action_transform import DATASET_ACTION_NAMES

    saved_names = list(DATASET_ACTION_NAMES[:-1])
    cfg = config(action_history_feature_names=saved_names)
    assert list(action_history_schema(cfg)[0]) == saved_names
    assert len(action_history_schema(cfg)[0]) == 15
    cfg.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert list(action_history_schema(restored)[0]) == saved_names
    assert restored.action_history_ee_semantics == "legacy_control_v0"


def test_active_only_ee_mask_is_applied_after_normalization_without_touching_other_commands():
    cfg = config(state_num_frames=5)
    split, _ = _action_history_steps(cfg, None)
    raw = torch.ones(1, 54, len(CONTROL_EXTENDED_DATASET_ACTION_NAMES)) * 7
    raw[:, :4, 3:5] = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    original = raw.clone()
    result = split({TransitionKey.ACTION: raw, TransitionKey.OBSERVATION: {},
                    TransitionKey.COMPLEMENTARY_DATA: {}})
    feature_pad = result[TransitionKey.COMPLEMENTARY_DATA][OBS_ACTION_HISTORY + "_feature_is_pad"]
    names = action_history_schema(cfg)[0]
    ee_columns = [i for i, name in enumerate(names) if name.startswith("height_invariant_ee_")]
    assert not feature_pad[:, 0].any()
    assert feature_pad[:, 1:, ee_columns].all()
    assert not feature_pad[..., [names.index("b2_vx"), names.index("gripper_target")]].any()
    # Represent arbitrary nonzero normalized values; the mask must zero them here,
    # not before normalization where the coordinate origin would become nonzero.
    history = torch.full((1, 4, len(names)), 2.5)
    policy = PI05Policy.__new__(PI05Policy)
    policy.config = cfg
    prepared = policy._prepare_action_history({OBS_ACTION_HISTORY: history,
        OBS_ACTION_HISTORY + "_feature_is_pad": feature_pad}, 4)
    assert not prepared[:, 1:, ee_columns].any()
    torch.testing.assert_close(prepared[:, 0], history[:, 0])
    torch.testing.assert_close(prepared[..., [0, 1, 2, 12]], history[..., [0, 1, 2, 12]])
    assert torch.equal(result[TransitionKey.ACTION], original[:, 4:])
    assert torch.equal(raw, original)
    with pytest.raises(KeyError):
        policy._prepare_action_history({OBS_ACTION_HISTORY: history}, 4)


def test_pause_inserts_internal_hold_modes_even_without_mode_input_features(monkeypatch):
    from lerobot.policies.pi05.history_pause import augment_history_pause

    cfg = config(history_pause_max_steps=5)
    batch = {"action": torch.zeros(1, 62, len(CONTROL_EXTENDED_DATASET_ACTION_NAMES)),
             "observation.state": torch.zeros(1, 13, 58)}
    batch["action"][:, :, 4] = 1  # Reset before the synthetic pause.
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor([5]))
    augmented = augment_history_pause(batch, cfg)
    assert augmented["action"][0, 7:12, 3].eq(1).all()
    assert augmented["action"][0, 7:12, 4].eq(0).all()
    assert torch.equal(augmented["action"][:, 12:], batch["action"][:, 12:])


def test_active_only_subset_without_ee_needs_no_mode_columns():
    cfg = config(action_history_feature_names=["b2_vx"], dataset_action_feature_names=["b2_vx"],
                 action_history_ee_semantics="active_only_v1")
    split, _ = _action_history_steps(cfg, None)
    result = split({TransitionKey.ACTION: torch.ones(1, 62, 1), TransitionKey.OBSERVATION: {},
                    TransitionKey.COMPLEMENTARY_DATA: {}})
    feature_pad = result[TransitionKey.COMPLEMENTARY_DATA][OBS_ACTION_HISTORY + "_feature_is_pad"]
    assert feature_pad.shape == (1, 12, 1)
    assert not feature_pad.any()


@pytest.mark.parametrize("fixed_shape", [True, False])
def test_unknown_execution_slots_remain_at_their_original_times(fixed_shape):
    policy = PI05Policy.__new__(PI05Policy)
    pad = torch.tensor([[False, True, False, True], [True, True, True, True]])
    batch = {OBS_ACTION_HISTORY: torch.arange(2 * 4 * 3).reshape(2, 4, 3),
             OBS_ACTION_HISTORY + "_is_pad": pad}
    window, lengths, mask = policy._fixed_memory_window_plan(batch, key=OBS_ACTION_HISTORY,
        max_frames=4, allow_empty=True, preserve_input_window=fixed_shape)
    assert window == 4
    assert lengths.tolist() == [2, 0]
    assert torch.equal(mask, ~pad)


@pytest.mark.parametrize("names", [[], ["b2_vx", "b2_vx"], ["missing"], ["task_complete"], ["task_blocked"]])
def test_bad_or_synthetic_history_fields_fail_closed(names):
    with pytest.raises(ValueError):
        config(action_history_feature_names=names)


@pytest.mark.parametrize("status,supervision", [("active", "demonstrated"), ("complete", "hold")])
def test_goal_supervision_never_rewrites_history_or_shifts_future_clock(status, supervision):
    view = SemanticView(
        "open",
        "goal_open",
        ("Open the door.",),
        status,
        supervision,
        completion_source="physical_stage_tail",
        completion_frame=110,
    )
    catalog = SemanticViewCatalog(Path("unused"), {0: (view,)}, "", 1, ("goal_open/" + status,), None, {})
    raw = torch.arange(62 * 25, dtype=torch.float32).reshape(1, 62, 25)
    raw[..., 0] = 1
    raw[..., 15] = 0
    history = raw[:, :12].clone()
    batch = {"episode_index": [0], "frame_index": [100], "index": [100], "action": raw}
    apply_semantic_views_to_batch(batch, catalog, step=0, seed=0, randomize=False, action_history_length=12)
    assert torch.equal(batch["action"][:, :12], history)
    assert batch["action_task_blocked"].shape == (1, 50)
    if status == "active":
        assert batch["action"][0, 12, 0] == 1
        assert batch["action"][0, 12, 15] == 0
        assert batch["action"][0, 22, 15] == 1  # t+10, NOT combined index10.
    else:
        assert batch["action"][0, 12, 15] == 1
        assert batch["action"][0, 12, 0] == 0


def test_zero_history_is_bitwise_identical_and_disabled_metadata_has_no_extension():
    view = SemanticView(
        "open",
        "goal_open",
        ("Open.",),
        "active",
        "demonstrated",
        completion_source="physical_stage_tail",
        completion_frame=110,
    )
    catalog = SemanticViewCatalog(Path("unused"), {0: (view,)}, "", 1, ("goal_open/active",), None, {})
    action = torch.randn(1, 50, 25)
    batches = [{"episode_index": [0], "frame_index": [100], "action": action.clone()} for _ in range(2)]
    apply_semantic_views_to_batch(batches[0], catalog, step=0, seed=0, randomize=False)
    apply_semantic_views_to_batch(
        batches[1], catalog, step=0, seed=0, randomize=False, action_history_length=0
    )
    assert torch.equal(batches[0]["action"], batches[1]["action"])
    cfg = PI05Config(device="cpu")
    assert "feature_names" not in cfg.deployment_metadata()["memory"]["action_history"]


def test_saved_history_normalization_survives_deployment_reload(tmp_path):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.processor import DataProcessorPipeline, NormalizerProcessorStep, RelativeActionsProcessorStep

    cfg = config(action_history_feature_names=["b2_vx"], io_schema_resolved=True, action_dt_seconds=0.02)
    stats = {"action": {"q01": torch.zeros(25), "q99": torch.ones(25) * 2}}
    split, norm = _action_history_steps(cfg, stats)
    from lerobot.processor import UnnormalizerProcessorStep

    pre = DataProcessorPipeline(
        steps=[split, RelativeActionsProcessorStep(), NormalizerProcessorStep(features={}, norm_map={}), norm]
    )
    post = DataProcessorPipeline(steps=[UnnormalizerProcessorStep(features={}, norm_map={})])
    pre.save_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    post.save_pretrained(tmp_path, config_filename="policy_postprocessor.json")
    loaded, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(tmp_path))
    loaded_norm = next(
        s
        for s in loaded.steps
        if isinstance(s, NormalizerProcessorStep)
        and set(s.normalize_observation_keys or ()) == {OBS_ACTION_HISTORY}
    )
    input_value = {TransitionKey.OBSERVATION: {OBS_ACTION_HISTORY: torch.ones(1, 12, 1)}}
    torch.testing.assert_close(
        loaded_norm(input_value)[TransitionKey.OBSERVATION][OBS_ACTION_HISTORY], torch.zeros(1, 12, 1)
    )


def test_deployment_rejects_missing_history_statistics(tmp_path):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.processor import (
        DataProcessorPipeline,
        NormalizerProcessorStep,
        RelativeActionsProcessorStep,
        UnnormalizerProcessorStep,
    )

    cfg = config(action_history_feature_names=["b2_vx"], io_schema_resolved=True, action_dt_seconds=0.02)
    split, norm = _action_history_steps(cfg, None)
    pre = DataProcessorPipeline(
        steps=[split, RelativeActionsProcessorStep(), NormalizerProcessorStep(features={}, norm_map={}), norm]
    )
    post = DataProcessorPipeline(steps=[UnnormalizerProcessorStep(features={}, norm_map={})])
    pre.save_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    post.save_pretrained(tmp_path, config_filename="policy_postprocessor.json")
    with pytest.raises(ValueError, match="missing action history normalization"):
        make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(tmp_path))


def test_refresh_history_statistics_uses_new_named_dataset_not_saved_values():
    from lerobot.policies.pi05.processor_pi05 import reconcile_pi05_action_representation_processors
    from lerobot.processor import (
        DataProcessorPipeline,
        NormalizerProcessorStep,
        RelativeActionsProcessorStep,
        UnnormalizerProcessorStep,
    )

    cfg = config(action_history_feature_names=["b2_vx"], io_schema_resolved=True, action_dt_seconds=0.02)
    raw_stats = {"action": {"q01": torch.zeros(25), "q99": torch.ones(25)}}
    split, saved = _action_history_steps(cfg, raw_stats)
    pre = DataProcessorPipeline(
        steps=[
            split,
            RelativeActionsProcessorStep(),
            NormalizerProcessorStep(features={}, norm_map={}),
            saved,
        ]
    )
    post = DataProcessorPipeline(steps=[UnnormalizerProcessorStep(features={}, norm_map={})])
    new_stats = {"action": {"q01": torch.zeros(25) + 2, "q99": torch.ones(25) * 4}}
    pre, _ = reconcile_pi05_action_representation_processors(
        cfg, pre, post, new_stats, transformed_action_stats={"q01": torch.zeros(13), "q99": torch.ones(13)}
    )
    refreshed = next(
        step
        for step in pre.steps
        if isinstance(step, NormalizerProcessorStep)
        and set(step.normalize_observation_keys or ()) == {OBS_ACTION_HISTORY}
    )
    assert refreshed is not saved
    value = {TransitionKey.OBSERVATION: {OBS_ACTION_HISTORY: torch.ones(1, 12, 1) * 3}}
    torch.testing.assert_close(
        refreshed(value)[TransitionKey.OBSERVATION][OBS_ACTION_HISTORY], torch.zeros(1, 12, 1)
    )


@pytest.mark.parametrize("complete,ee", [(True, True), (False, True), (False, False)])
def test_real_training_preprocessor_keeps_history_independent_of_output_schema(monkeypatch, complete, ee):
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.pi05 import processor_pi05
    from lerobot.policies.pi05.b2_action_transform import action_schema_kwargs, b2_pose_delta_action_names
    from lerobot.processor.pipeline import IdentityProcessorStep

    monkeypatch.setattr(processor_pi05, "TokenizerProcessorStep", lambda **_: IdentityProcessorStep())
    cfg = config(action_predict_task_complete=complete, action_predict_ee_pose=ee)
    cfg.io_schema_resolved = True
    cfg.action_dt_seconds = 0.02
    cfg.state_feature_indices = list(range(10)) + list(range(49, 58))
    cfg.ee_state_anchor_indices = list(range(49, 58))
    cfg.action_feature_names = b2_pose_delta_action_names(
        list(CONTROL_EXTENDED_DATASET_ACTION_NAMES), **action_schema_kwargs(cfg)
    )
    dim = len(cfg.action_feature_names)
    cfg.input_features = {"observation.state": PolicyFeature(FeatureType.STATE, (19,))}
    cfg.output_features = {"action": PolicyFeature(FeatureType.ACTION, (dim,))}
    raw_stats = {
        key: {"q01": torch.zeros(d), "q99": torch.ones(d)}
        for key, d in [("action", 25), ("observation.state", 58)]
    }
    transformed = {"q01": -torch.ones(dim), "q99": torch.ones(dim)}
    pre, _ = processor_pi05.make_pi05_pre_post_processors(cfg, raw_stats, transformed)
    actions = torch.zeros(2, 62, 25)
    actions[..., [5, 9, 16, 20]] = 1
    actions[..., 11] = -0.9
    actions[..., 22] = 0.5
    actions[:, 1, 3] = 1  # Hold in the historical prefix only.
    actions[:, 2, 4] = 1  # Reset in the historical prefix only.
    states = torch.zeros(2, 13, 58)
    states[..., [49, 53]] = 1
    batch = pre(
        {
            "action": actions,
            "observation.state": states,
            "task": ["Open.", "Open."],
            "action_is_pad": torch.zeros(2, 62, dtype=torch.bool),
        }
    )
    assert batch["action"].shape == (2, 50, dim)
    assert batch[OBS_ACTION_HISTORY].shape == (2, 12, 13)
    feature_pad = batch[OBS_ACTION_HISTORY + "_feature_is_pad"]
    assert feature_pad.shape == batch[OBS_ACTION_HISTORY].shape
    assert not feature_pad[:, 0].any()
    assert feature_pad[:, 1:3, 3:12].all()
    policy = PI05Policy.__new__(PI05Policy)
    policy.config = cfg
    assert not policy._prepare_action_history(batch, 12)[:, 1:3, 3:12].any()
    names = action_history_schema(cfg)[0]
    torch.testing.assert_close(
        batch[OBS_ACTION_HISTORY][..., names.index("height_invariant_ee_x")], torch.zeros(2, 12)
    )
