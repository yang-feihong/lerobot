import pytest
import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import (
    DataProcessorPipeline,
    IdentityProcessorStep,
    NormalizerProcessorStep,
    UnnormalizerProcessorStep,
    create_transition,
    identity_transition,
)
from lerobot.scripts.lerobot_train import _configure_quantile_fallback_for_training
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION


def _make_training_processors(*, fallback_enabled: bool):
    history_key = "observation.action_history"
    action_feature = PolicyFeature(FeatureType.ACTION, (2,))
    action_stats = {"q01": [0.0, 0.0], "q99": [0.0, 0.0], "min": [0.0, -0.4], "max": [0.2, 0.0]}
    normalizer = NormalizerProcessorStep(
        features={ACTION: action_feature},
        norm_map={FeatureType.ACTION: NormalizationMode.QUANTILES},
        stats={ACTION: action_stats},
        quantile_fallback_to_min_max=fallback_enabled,
    )
    history_normalizer = NormalizerProcessorStep(
        features={history_key: PolicyFeature(FeatureType.STATE, (2,))},
        norm_map={FeatureType.STATE: NormalizationMode.QUANTILES},
        stats={history_key: action_stats},
        normalize_observation_keys={history_key},
        quantile_fallback_to_min_max=fallback_enabled,
    )
    unnormalizer = UnnormalizerProcessorStep(
        features={ACTION: action_feature},
        norm_map={FeatureType.ACTION: NormalizationMode.QUANTILES},
        stats={ACTION: action_stats},
        quantile_fallback_to_min_max=fallback_enabled,
    )
    preprocessor = DataProcessorPipeline(
        steps=[IdentityProcessorStep(), normalizer, history_normalizer],
        to_transition=identity_transition,
        to_output=identity_transition,
    )
    postprocessor = DataProcessorPipeline(
        steps=[unnormalizer],
        to_transition=identity_transition,
        to_output=identity_transition,
    )
    return preprocessor, postprocessor


def test_new_training_uses_same_robust_scales_for_actions_history_and_outputs():
    preprocessor, postprocessor = _make_training_processors(fallback_enabled=False)

    _configure_quantile_fallback_for_training(
        preprocessor, postprocessor, resume=False, is_reward_model_training=False
    )

    actions = torch.tensor([[[0.0, 0.0], [0.2, -0.4]]])
    history_key = "observation.action_history"
    batch = create_transition(action=actions, observation={history_key: actions.clone()})
    normalized = preprocessor(batch)
    expected = torch.tensor([[[-1.0, 1.0], [1.0, -1.0]]])
    torch.testing.assert_close(normalized[TransitionKey.ACTION], expected)
    torch.testing.assert_close(normalized[TransitionKey.OBSERVATION][history_key], expected)
    torch.testing.assert_close(postprocessor(normalized)[TransitionKey.ACTION], actions)
    for step in (*preprocessor.steps[1:], *postprocessor.steps):
        assert step.get_config()["quantile_fallback_to_min_max"] is True
    assert not hasattr(preprocessor.steps[0], "quantile_fallback_to_min_max")


@pytest.mark.parametrize("fallback_enabled", [False, True])
@pytest.mark.parametrize("resume,is_reward_model_training", [(True, False), (False, True), (True, True)])
def test_resume_and_reward_training_preserve_normalization(
    fallback_enabled, resume, is_reward_model_training
):
    preprocessor, postprocessor = _make_training_processors(fallback_enabled=fallback_enabled)
    batch = create_transition(action=torch.tensor([[[0.0, 0.0], [0.2, -0.4]]]))
    normalized_before = preprocessor(batch)[TransitionKey.ACTION]

    _configure_quantile_fallback_for_training(
        preprocessor,
        postprocessor,
        resume=resume,
        is_reward_model_training=is_reward_model_training,
    )

    for step in (*preprocessor.steps[1:], *postprocessor.steps):
        assert step.quantile_fallback_to_min_max is fallback_enabled
    torch.testing.assert_close(preprocessor(batch)[TransitionKey.ACTION], normalized_before)
