import json

import pytest

from lerobot.datasets.mixture_sampling import load_dataset_mixture_manifest


def test_load_dataset_mixture_manifest_normalizes_weights(tmp_path):
    path = tmp_path / "mixture.json"
    path.write_text(
        json.dumps(
            {
                "sources": [
                    {"name": "stage12", "weight": 80, "episode_range": [0, 3]},
                    {"name": "stage2", "weight": 20, "episode_range": [3, 5]},
                ]
            }
        )
    )
    sources = load_dataset_mixture_manifest(path, num_episodes=5)
    assert [source.weight for source in sources] == pytest.approx([0.8, 0.2])
    assert sources[0].episode_indices == [0, 1, 2]
    assert sources[1].episode_indices == [3, 4]


@pytest.mark.parametrize(
    "ranges,match",
    [
        ([[0, 3], [2, 5]], "overlaps"),
        ([[0, 2], [3, 5]], "cover every episode"),
    ],
)
def test_load_dataset_mixture_manifest_requires_exact_partition(tmp_path, ranges, match):
    path = tmp_path / "mixture.json"
    path.write_text(
        json.dumps(
            {
                "sources": [
                    {"name": "a", "weight": 0.8, "episode_range": ranges[0]},
                    {"name": "b", "weight": 0.2, "episode_range": ranges[1]},
                ]
            }
        )
    )
    with pytest.raises(ValueError, match=match):
        load_dataset_mixture_manifest(path, num_episodes=5)
