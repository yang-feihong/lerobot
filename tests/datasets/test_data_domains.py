import json

import pytest

from lerobot.data_domains import load_episode_domains, recording_domain, select_domain_episodes


@pytest.mark.parametrize("override", [
    {"typo": {"simulation": 1}},
    {"full_episode_command": {}},
    {"full_episode_command": {"simulation": -1}},
    {"full_episode_command": {"simulation": True}},
    {"full_episode_command": {"real": 1}},
])
def test_instruction_policy_rejects_invalid_or_globally_excluded_domains(override):
    from lerobot.data_domains import validate_instruction_domain_weights

    with pytest.raises(ValueError):
        validate_instruction_domain_weights(
            override, {"simulation": 1}, {"approach": {"full_episode_command": {}}},
        )


def test_historical_origins_do_not_need_raw_mounts_or_directory_heuristics(tmp_path):
    (tmp_path / "diagnostics").mkdir()
    records = {
        "a": {"status": "done", "episode_index": 0, "bag_path": "/offline/new_name/a.bag.active"},
        "b": {
            "status": "done",
            "episode_index": 1,
            "bag_path": "/offline/arbitrary/location",
            "source_type": "accepted_simulation_shard",
        },
    }
    path = tmp_path / "diagnostics/conversion_manifest.json"
    path.write_text(json.dumps({"bags": records}))
    episodes = {str(r["episode_index"]): {"source_path": r["bag_path"]} for r in records.values()}
    assert load_episode_domains(tmp_path / "meta/semantic_views.json", episodes) == {
        0: "real",
        1: "simulation",
    }
    episodes["1"]["data_domain"] = "real"
    with pytest.raises(ValueError, match="disagrees"):
        load_episode_domains(tmp_path / "meta/semantic_views.json", episodes)
    episodes["1"] = {"source_path": "different"}
    with pytest.raises(ValueError, match="source mismatch"):
        load_episode_domains(tmp_path / "meta/semantic_views.json", episodes)


def test_no_origin_guesses_from_dataset_names():
    with pytest.raises(ValueError, match="no authoritative"):
        recording_domain({"bag_path": "/simulation/push_seed100"})
    with pytest.raises(ValueError, match="Conflicting"):
        recording_domain({"bag_path": "/raw/a.bag", "data_domain": "simulation"})


def test_only_domain_filter_precedes_split_and_statistics():
    from types import SimpleNamespace

    catalog = SimpleNamespace(episode_domains={0: "real", 1: "simulation", 2: "real"})
    assert select_domain_episodes(catalog, [0, 1, 2], {"simulation": 1}) == [1]
    assert select_domain_episodes(catalog, [2, 0], {"real": 1}) == [2, 0]
    with pytest.raises(ValueError, match="Missing recording"):
        select_domain_episodes(catalog, [3], {"real": 1})
