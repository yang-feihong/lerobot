import json

import pytest

from lerobot.datasets.mixture_sampling import load_dataset_mixture_manifest


def test_load_dataset_mixture_manifest_normalizes_weights(tmp_path):
    path = tmp_path / "mixture.json"
    path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "name": "source_a",
                        "semantic_type": "approach",
                        "weight": 80,
                        "episode_range": [0, 3],
                    },
                    {
                        "name": "source_b",
                        "semantic_type": "handle_press",
                        "weight": 20,
                        "episode_range": [3, 5],
                    },
                ]
            }
        )
    )
    sources = load_dataset_mixture_manifest(path, num_episodes=5)
    assert [source.weight for source in sources] == pytest.approx([0.8, 0.2])
    assert [source.semantic_type for source in sources] == ["approach", "handle_press"]
    assert sources[0].episode_indices == [0, 1, 2]
    assert sources[1].episode_indices == [3, 4]


def test_load_dataset_mixture_manifest_requires_explicit_semantics(tmp_path):
    path = tmp_path / "mixture.json"
    path.write_text(
        json.dumps(
            {
                "sources": [
                    {"name": "source_a", "weight": 1.0, "episode_range": [0, 1]},
                    {
                        "name": "source_b",
                        "semantic_type": "handle_press",
                        "weight": 1.0,
                        "episode_range": [1, 2],
                    },
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="explicit non-empty semantic_type"):
        load_dataset_mixture_manifest(path, num_episodes=2)


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
                    {
                        "name": "a",
                        "semantic_type": "approach",
                        "weight": 0.8,
                        "episode_range": ranges[0],
                    },
                    {
                        "name": "b",
                        "semantic_type": "handle_press",
                        "weight": 0.2,
                        "episode_range": ranges[1],
                    },
                ]
            }
        )
    )
    with pytest.raises(ValueError, match=match):
        load_dataset_mixture_manifest(path, num_episodes=5)
