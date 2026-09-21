import pytest

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import (
    OBS_ACTION_HISTORY,
    reconcile_pi05_action_representation_processors,
)
from lerobot.processor import (
    DataProcessorPipeline,
    NormalizerProcessorStep,
    RelativeActionsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.utils.constants import ACTION


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
