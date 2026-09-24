import hashlib
import json

import pytest

from lerobot.scripts.lerobot_train import load_task_variants


def write_json(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def test_staged_dataset_requires_hard_language_binding(tmp_path) -> None:
    write_json(tmp_path / "meta/info.json", {"semantic_type": "approach"})
    write_json(tmp_path / "meta/task_variants.json", {"0": ["Open the door."]})

    with pytest.raises(RuntimeError, match="missing .*task_variant_binding"):
        load_task_variants(tmp_path, None)


def test_staged_dataset_accepts_matching_stage_binding(tmp_path) -> None:
    variants_path = tmp_path / "meta/task_variants.json"
    write_json(tmp_path / "meta/info.json", {"semantic_type": "approach"})
    write_json(variants_path, {"0": ["Approach the door."]})
    write_json(
        tmp_path / "meta/task_variant_binding.json",
        {
            "binding": "stage_annotations",
            "expected_stage": "b2_approach",
            "task_variants_sha256": hashlib.sha256(variants_path.read_bytes()).hexdigest(),
        },
    )

    assert load_task_variants(tmp_path, None) == {0: ["Approach the door."]}


def test_staged_dataset_rejects_modified_variants_after_binding(tmp_path) -> None:
    variants_path = tmp_path / "meta/task_variants.json"
    write_json(tmp_path / "meta/info.json", {"semantic_type": "handle_press"})
    write_json(variants_path, {"0": ["Press the handle."]})
    write_json(
        tmp_path / "meta/task_variant_binding.json",
        {
            "binding": "stage_annotations",
            "expected_stage": "handle_press",
            "task_variants_sha256": "not-the-current-hash",
        },
    )

    with pytest.raises(RuntimeError, match="do not match their hard binding"):
        load_task_variants(tmp_path, None)
