#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from lerobot.common.train_utils import (
    _append_peft_base_weights,
    _peft_modules_to_save_from_state_dict,
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_batch_size,
    load_training_num_processes,
    load_training_state,
    load_training_step,
    prune_checkpoints,
    push_checkpoint_to_hub,
    save_checkpoint,
    save_training_state,
    save_training_step,
    update_last_checkpoint,
)
from lerobot.utils.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    OPTIMIZER_PARAM_GROUPS,
    OPTIMIZER_STATE,
    RNG_STATE,
    SCHEDULER_STATE,
    TRAINING_STATE_DIR,
    TRAINING_STEP,
)


def test_get_step_identifier():
    assert get_step_identifier(5, 1000) == "000005"
    assert get_step_identifier(123, 100_000) == "000123"
    assert get_step_identifier(456789, 1_000_000) == "0456789"


def test_get_step_checkpoint_dir():
    output_dir = Path("/checkpoints")
    step_dir = get_step_checkpoint_dir(output_dir, 1000, 5)
    assert step_dir == output_dir / CHECKPOINTS_DIR / "000005"


def test_save_load_training_step(tmp_path):
    save_training_step(5000, tmp_path)
    assert (tmp_path / TRAINING_STEP).is_file()


def test_load_training_step(tmp_path):
    step = 5000
    save_training_step(step, tmp_path)
    loaded_step = load_training_step(tmp_path)
    assert loaded_step == step


def test_save_training_state_records_num_processes(tmp_path, optimizer, scheduler):
    save_training_state(tmp_path, 10, optimizer, scheduler, num_processes=4)
    assert load_training_num_processes(tmp_path) == 4


def test_load_training_num_processes_absent_returns_none(tmp_path, optimizer, scheduler):
    # Checkpoints written before the world size was recorded must still load (back-compat).
    save_training_state(tmp_path, 10, optimizer, scheduler)
    assert load_training_num_processes(tmp_path) is None


def test_save_training_state_records_batch_size(tmp_path, optimizer, scheduler):
    save_training_state(tmp_path, 10, optimizer, scheduler, batch_size=32)
    assert load_training_batch_size(tmp_path) == 32


def test_load_training_batch_size_absent_returns_none(tmp_path, optimizer, scheduler):
    # Checkpoints written before the batch size was recorded must still load (back-compat).
    save_training_state(tmp_path, 10, optimizer, scheduler)
    assert load_training_batch_size(tmp_path) is None


def test_update_last_checkpoint(tmp_path):
    checkpoint = tmp_path / "0005"
    checkpoint.mkdir()
    update_last_checkpoint(checkpoint)
    last_checkpoint = tmp_path / LAST_CHECKPOINT_LINK
    assert last_checkpoint.is_symlink()
    assert last_checkpoint.resolve() == checkpoint


def test_prune_checkpoints_keeps_latest_and_milestones(tmp_path):
    checkpoints_dir = tmp_path / CHECKPOINTS_DIR
    checkpoints_dir.mkdir()
    for step in range(500, 25_001, 500):
        (checkpoints_dir / f"{step:06d}").mkdir()
    (checkpoints_dir / "notes").mkdir()
    update_last_checkpoint(checkpoints_dir / "025000")

    removed = prune_checkpoints(checkpoints_dir, keep_last=2, keep_every_n_steps=10_000)

    assert {path.name for path in removed} == {
        f"{step:06d}" for step in range(500, 24_001, 500) if step not in {10_000, 20_000}
    }
    assert {path.name for path in checkpoints_dir.iterdir()} == {
        "010000",
        "020000",
        "024500",
        "025000",
        "last",
        "notes",
    }
    assert (checkpoints_dir / "last").resolve() == checkpoints_dir / "025000"


def test_prune_checkpoints_keep_last_zero_disables_pruning(tmp_path):
    for name in ("000500", "001000", "001500"):
        (tmp_path / name).mkdir()

    assert prune_checkpoints(tmp_path, keep_last=0, keep_every_n_steps=1_000) == []
    assert {path.name for path in tmp_path.iterdir()} == {"000500", "001000", "001500"}


@patch("lerobot.common.train_utils.save_training_state")
def test_save_checkpoint(mock_save_training_state, tmp_path, optimizer):
    policy = Mock()
    cfg = Mock()
    save_checkpoint(tmp_path, 10, cfg, policy, optimizer)
    policy.save_pretrained.assert_called_once()
    cfg.save_pretrained.assert_called_once()
    mock_save_training_state.assert_called_once()


@patch("lerobot.common.train_utils.save_training_state")
def test_save_checkpoint_peft(mock_save_training_state, tmp_path, optimizer):
    policy = Mock()
    policy.config = Mock()
    policy.config.save_pretrained = Mock()
    cfg = Mock()
    cfg.use_peft = True
    save_checkpoint(tmp_path, 10, cfg, policy, optimizer)
    policy.save_pretrained.assert_called_once()
    cfg.save_pretrained.assert_called_once()
    policy.config.save_pretrained.assert_called_once()
    mock_save_training_state.assert_called_once()


def test_peft_modules_to_save_uses_supplied_state_dict(monkeypatch):
    peft = pytest.importorskip("peft")
    import torch

    wrapper = peft.utils.other.ModulesToSaveWrapper(torch.nn.Linear(2, 1), "default")
    policy = torch.nn.Sequential(wrapper)
    supplied_weight = torch.randn_like(wrapper.modules_to_save["default"].weight)
    supplied_bias = torch.randn_like(wrapper.modules_to_save["default"].bias)
    supplied_state = {
        "modules_to_save.default.weight": supplied_weight,
        "modules_to_save.default.bias": supplied_bias,
    }

    def unexpected_state_dict_call(*args, **kwargs):
        raise AssertionError("PEFT re-read a wrapped module instead of using the supplied state dict")

    monkeypatch.setattr(wrapper.modules_to_save["default"], "state_dict", unexpected_state_dict_call)
    with _peft_modules_to_save_from_state_dict(policy):
        extracted = wrapper.adapter_state_dict("default", supplied_state)

    assert set(extracted) == {"weight", "bias"}
    assert extracted["weight"] is supplied_weight
    assert extracted["bias"] is supplied_bias
    with pytest.raises(AssertionError, match="PEFT re-read"):
        wrapper.adapter_state_dict("default", supplied_state)


def test_peft_save_pretrained_does_not_reread_modules_to_save(tmp_path, monkeypatch):
    peft = pytest.importorskip("peft")
    import torch

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = torch.nn.Linear(2, 2)
            self.head = torch.nn.Linear(2, 1)

        def forward(self, value):
            return self.head(self.body(value))

    policy = peft.get_peft_model(
        TinyModel(),
        peft.LoraConfig(target_modules=["body"], modules_to_save=["head"]),
    )
    supplied_state = policy.state_dict()
    wrapper = policy.base_model.model.head

    def unexpected_state_dict_call(*args, **kwargs):
        raise AssertionError("PEFT re-read a wrapped module instead of using the supplied state dict")

    monkeypatch.setattr(wrapper.modules_to_save["default"], "state_dict", unexpected_state_dict_call)
    with _peft_modules_to_save_from_state_dict(policy):
        policy.save_pretrained(tmp_path, state_dict=supplied_state)

    assert (tmp_path / "adapter_model.safetensors").is_file()
    assert (tmp_path / "adapter_config.json").is_file()


def test_peft_checkpoint_loads_appended_base_weights(tmp_path):
    peft = pytest.importorskip("peft")
    import torch
    from safetensors import safe_open

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = torch.nn.Linear(2, 2)
            self.memory = torch.nn.Linear(2, 2)

        def forward(self, value):
            return self.memory(self.body(value))

    policy = peft.get_peft_model(TinyModel(), peft.LoraConfig(target_modules=["body"]))
    with torch.no_grad():
        policy.base_model.model.memory.weight.fill_(3.25)
        policy.base_model.model.memory.bias.fill_(-1.5)
    supplied_state = policy.state_dict()
    policy.save_pretrained(tmp_path, state_dict=supplied_state)

    appended = _append_peft_base_weights(tmp_path, supplied_state, key_fragment=".memory.")

    assert appended == 2
    with safe_open(tmp_path / "adapter_model.safetensors", framework="pt") as adapter_file:
        saved_keys = adapter_file.keys()
        assert "base_model.model.memory.weight" in saved_keys
        assert "base_model.model.memory.bias" in saved_keys

    reloaded = peft.PeftModel.from_pretrained(TinyModel(), tmp_path)
    torch.testing.assert_close(
        reloaded.base_model.model.memory.weight,
        torch.full_like(reloaded.base_model.model.memory.weight, 3.25),
    )
    torch.testing.assert_close(
        reloaded.base_model.model.memory.bias,
        torch.full_like(reloaded.base_model.model.memory.bias, -1.5),
    )


@pytest.mark.parametrize("gathered", [False, True])
@pytest.mark.parametrize("mode", ["full", "lora", "frozen"])
def test_save_checkpoint_embeds_mem_base_weights(tmp_path, gathered, mode):
    import torch

    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    policy = Mock()
    policy.config = PI05Config(
        device="cpu",
        mem_vit_enabled=True,
        mem_vit_finetune_mode=mode,
    )
    policy.config.save_pretrained = Mock()
    supplied_state = {"base_model.model.vision_tower.weight": torch.ones(2)}
    policy.state_dict.return_value = supplied_state
    cfg = SimpleNamespace(peft=object(), policy=policy.config, save_pretrained=Mock())
    with (
        patch("lerobot.common.train_utils._peft_modules_to_save_from_state_dict"),
        patch("lerobot.common.train_utils._append_peft_base_weights", return_value=1) as append,
        patch("lerobot.common.train_utils.save_training_state"),
    ):
        save_checkpoint(
            tmp_path, 1, cfg, policy, optimizer=None, model_state_dict=supplied_state if gathered else None
        )
    append.assert_called_once_with(
        tmp_path / "pretrained_model", supplied_state, key_fragment=".vision_tower."
    )
    assert policy.config.mem_vit_base_weights_embedded is True
    assert policy.config.mem_vit_embedded_tensor_count == 1
    if not gathered:
        policy.state_dict.assert_called_once_with()
    else:
        policy.state_dict.assert_not_called()


def test_save_training_state(tmp_path, optimizer, scheduler):
    save_training_state(tmp_path, 10, optimizer, scheduler)
    assert (tmp_path / TRAINING_STATE_DIR).is_dir()
    assert (tmp_path / TRAINING_STATE_DIR / TRAINING_STEP).is_file()
    assert (tmp_path / TRAINING_STATE_DIR / RNG_STATE).is_file()
    assert (tmp_path / TRAINING_STATE_DIR / OPTIMIZER_STATE).is_file()
    assert (tmp_path / TRAINING_STATE_DIR / OPTIMIZER_PARAM_GROUPS).is_file()
    assert (tmp_path / TRAINING_STATE_DIR / SCHEDULER_STATE).is_file()


def test_save_load_training_state(tmp_path, optimizer, scheduler):
    save_training_state(tmp_path, 10, optimizer, scheduler)
    loaded_step, loaded_optimizer, loaded_scheduler = load_training_state(tmp_path, optimizer, scheduler)
    assert loaded_step == 10
    assert loaded_optimizer is optimizer
    assert loaded_scheduler is scheduler


def test_load_training_state_skip_optimizer(tmp_path, optimizer, scheduler):
    # FSDP loads optimizer separately (after accelerator.prepare)
    # load_training_state(load_optimizer=False) must restore step + scheduler but leave the
    # optimizer untouched and never touch the on-disk optimizer state.
    save_training_state(tmp_path, 10, optimizer, scheduler)
    with patch("lerobot.common.train_utils.load_optimizer_state") as mock_load_optimizer_state:
        loaded_step, loaded_optimizer, loaded_scheduler = load_training_state(
            tmp_path, optimizer, scheduler, load_optimizer=False
        )
    mock_load_optimizer_state.assert_not_called()
    assert loaded_step == 10
    assert loaded_optimizer is optimizer
    assert loaded_scheduler is scheduler


def test_push_checkpoint_to_hub_creates_repo_and_uploads(tmp_path, monkeypatch):
    ckpt = tmp_path / "010000"
    (ckpt / "pretrained_model").mkdir(parents=True)
    api = MagicMock()
    monkeypatch.setattr("lerobot.common.train_utils.HfApi", lambda *a, **k: api)
    push_checkpoint_to_hub(ckpt, "user/run", private=True)
    api.create_repo.assert_called_once()
    assert api.create_repo.call_args.kwargs["private"] is True
    assert api.create_repo.call_args.kwargs["repo_type"] == "model"
    api.upload_folder.assert_called_once()
    kwargs = api.upload_folder.call_args.kwargs
    assert kwargs["repo_id"] == "user/run"
    assert kwargs["repo_type"] == "model"
    assert kwargs["path_in_repo"] == "checkpoints/010000"
    assert kwargs["folder_path"] == str(ckpt)
    assert kwargs["commit_message"] == "checkpoint 010000"
    # A tag named after the checkpoint step is created so the checkpoint can be
    # recovered with --policy.pretrained_revision instead of a commit sha.
    api.create_tag.assert_called_once()
    tag_kwargs = api.create_tag.call_args.kwargs
    assert tag_kwargs["tag"] == "010000"
    assert tag_kwargs["revision"] == api.upload_folder.return_value.oid
    assert tag_kwargs["repo_type"] == "model"
    assert tag_kwargs["exist_ok"] is True


def test_push_checkpoint_to_hub_defaults_to_hub_default_visibility(tmp_path, monkeypatch):
    ckpt = tmp_path / "010000"
    (ckpt / "pretrained_model").mkdir(parents=True)
    api = MagicMock()
    monkeypatch.setattr("lerobot.common.train_utils.HfApi", lambda *a, **k: api)
    push_checkpoint_to_hub(ckpt, "user/run")
    api.create_repo.assert_called_once()
    assert api.create_repo.call_args.kwargs["private"] is None


def test_resolve_resume_checkpoint_downloads_latest_and_links(tmp_path, monkeypatch):
    from lerobot.common import train_utils

    out = tmp_path / "run"

    def fake_snapshot_download(repo_id, repo_type, allow_patterns, local_dir):
        # Mimic the Hub layout the real download materializes locally.
        assert allow_patterns == "checkpoints/020000/*"
        (Path(local_dir) / "checkpoints" / "020000" / "pretrained_model").mkdir(parents=True)
        return local_dir

    monkeypatch.setattr("lerobot.common.train_utils.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(
        "lerobot.common.train_utils.find_latest_hub_checkpoint", lambda repo_id: "checkpoints/020000"
    )

    checkpoint_dir = train_utils.resolve_resume_checkpoint("u/run", out)

    assert checkpoint_dir == out / CHECKPOINTS_DIR / "020000"
    last = out / CHECKPOINTS_DIR / LAST_CHECKPOINT_LINK
    assert last.is_symlink()
    # `last` points at the downloaded step dir.
    assert (last.parent / last.readlink()).resolve() == checkpoint_dir.resolve()


def test_resolve_resume_checkpoint_raises_without_checkpoints(tmp_path, monkeypatch):
    from lerobot.common import train_utils

    monkeypatch.setattr("lerobot.common.train_utils.find_latest_hub_checkpoint", lambda repo_id: None)
    with pytest.raises(FileNotFoundError, match="No checkpoint"):
        train_utils.resolve_resume_checkpoint("u/run", tmp_path / "run")
