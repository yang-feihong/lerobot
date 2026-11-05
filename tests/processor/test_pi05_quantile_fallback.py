import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import (
    OBS_ACTION_HISTORY,
    Pi05PrepareStateTokenizerProcessorStep,
    reconcile_pi05_action_representation_processors,
)
from lerobot.processor import (
    DataProcessorPipeline,
    NormalizerProcessorStep,
    RelativeActionsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION


@pytest.mark.parametrize("saved_config", [{}, {"continuous_state_memory": False}, {"continuous_state_memory": True}])
def test_factory_load_preserves_checkpoint_prompt(tmp_path, saved_config):
    from lerobot.policies.factory import make_pre_post_processors

    config = PI05Config(device="cpu", io_schema_resolved=True, action_dt_seconds=.02,
                        state_action_encoding="continuous")
    pre = DataProcessorPipeline(steps=[RelativeActionsProcessorStep(),
        NormalizerProcessorStep(features={}, norm_map={}),
        Pi05PrepareStateTokenizerProcessorStep(**saved_config)])
    post = DataProcessorPipeline(steps=[UnnormalizerProcessorStep(features={}, norm_map={})])
    pre.save_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    post.save_pretrained(tmp_path, config_filename="policy_postprocessor.json")
    # Reproduce old files which predate get_config serialization.
    import json
    path = tmp_path / "policy_preprocessor.json"
    document = json.loads(path.read_text())
    document["steps"][-1]["config"] = saved_config
    path.write_text(json.dumps(document))
    loaded, _ = make_pre_post_processors(policy_cfg=config, pretrained_path=str(tmp_path))
    step = next(s for s in loaded.steps if isinstance(s, Pi05PrepareStateTokenizerProcessorStep))
    assert step.continuous_state_memory is saved_config.get("continuous_state_memory", False)


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("encoding", ["continuous", "text"])
def test_training_selects_prompt_only_for_new_runs(resume, encoding):
    # Execute the real training branch without importing/starting training.
    path = Path(__file__).parents[2] / "src/lerobot/scripts/lerobot_train.py"
    tree = ast.parse(path.read_text())
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "isinstance(active_cfg, PI05Config) and (not cfg.resume)")
    config = PI05Config(device="cpu", state_action_encoding=encoding)
    overrides = {}
    exec(compile(ast.Module(body=[branch], type_ignores=[]), str(path), "exec"), {
        "PI05Config": PI05Config, "active_cfg": config,
        "cfg": SimpleNamespace(resume=resume), "preprocessor_overrides": overrides,
    })
    if resume:
        assert overrides == {}
    else:
        saved = overrides["pi05_prepare_state_tokenizer_processor_step"]
        step = Pi05PrepareStateTokenizerProcessorStep(**saved)
        assert step.continuous_state_memory is (encoding == "continuous")
        assert step.get_config()["continuous_state_memory"] is (encoding == "continuous")


@pytest.mark.parametrize("encoding", ["continuous", "text"])
@pytest.mark.parametrize("saved_prompt_memory", [False, True])
def test_checkpoint_state_prompt_is_preserved_and_serialized(encoding, saved_prompt_memory):
    config = PI05Config(device="cpu", io_schema_resolved=True, action_dt_seconds=.02,
                        state_action_encoding=encoding)
    inherited = Pi05PrepareStateTokenizerProcessorStep(
        max_state_dim=64, continuous_state_memory=saved_prompt_memory,
    )
    pre, post = DataProcessorPipeline(steps=[
        RelativeActionsProcessorStep(), NormalizerProcessorStep(features={}, norm_map={}), inherited,
    ]), DataProcessorPipeline(steps=[])
    # Reconciliation requires the standard inverse normalizer.
    post.steps = [UnnormalizerProcessorStep(features={}, norm_map={})]
    reconcile_pi05_action_representation_processors(config, pre, post, dataset_stats=None)
    assert inherited.continuous_state_memory is saved_prompt_memory
    assert inherited.max_state_dim == 64
    restored = Pi05PrepareStateTokenizerProcessorStep(**inherited.get_config())
    transition = {TransitionKey.OBSERVATION: {"observation.state": torch.zeros(1, 19)},
                  TransitionKey.COMPLEMENTARY_DATA: {"task": ["Open the door."]}}
    prompt = restored(transition)[TransitionKey.COMPLEMENTARY_DATA]["task"][0]
    assert (", State:" in prompt) is (not saved_prompt_memory)


@pytest.mark.parametrize("saved_fallback", [None, False, True])
def test_refreshing_action_history_preserves_checkpoint_quantile_convention(saved_fallback):
    config = PI05Config(
        device="cpu",
        io_schema_resolved=True,
        action_dt_seconds=0.02,
        state_action_encoding="continuous",
        action_history_enabled=True,
    )
    main_normalizer = NormalizerProcessorStep(
        features={ACTION: PolicyFeature(FeatureType.ACTION, (16,))},
        norm_map={FeatureType.ACTION: NormalizationMode.QUANTILES},
        quantile_fallback_to_min_max=True,
    )
    original_steps = [RelativeActionsProcessorStep(), main_normalizer]
    if saved_fallback is not None:
        original_steps.append(
            NormalizerProcessorStep(
                features={OBS_ACTION_HISTORY: PolicyFeature(FeatureType.STATE, (16,))},
                norm_map={FeatureType.STATE: NormalizationMode.QUANTILES},
                normalize_observation_keys={OBS_ACTION_HISTORY},
                quantile_fallback_to_min_max=saved_fallback,
            )
        )
    preprocessor = DataProcessorPipeline(steps=original_steps)
    postprocessor = DataProcessorPipeline(
        steps=[
            UnnormalizerProcessorStep(
                features={ACTION: PolicyFeature(FeatureType.ACTION, (16,))},
                norm_map={FeatureType.ACTION: NormalizationMode.QUANTILES},
                quantile_fallback_to_min_max=True,
            )
        ]
    )

    refreshed_preprocessor, refreshed_postprocessor = reconcile_pi05_action_representation_processors(
        config, preprocessor, postprocessor, dataset_stats=None
    )

    history_normalizers = [
        step
        for step in refreshed_preprocessor.steps
        if isinstance(step, NormalizerProcessorStep)
        and step.normalize_observation_keys == {OBS_ACTION_HISTORY}
    ]
    assert len(history_normalizers) == 1
    assert history_normalizers[0].quantile_fallback_to_min_max is (saved_fallback is True)
    if saved_fallback is not None:
        assert history_normalizers[0] is not original_steps[-1]
    assert main_normalizer.quantile_fallback_to_min_max is True
    assert refreshed_postprocessor.steps[0].quantile_fallback_to_min_max is True
