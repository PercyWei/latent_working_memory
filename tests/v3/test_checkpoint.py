from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

from latent_working_memory.v3.checkpoint import TokenMemoryCheckpointHandler, read_checkpoint
from latent_working_memory.v3.config import TrainingConfig
from latent_working_memory.v3.engine import TokenMemoryEngine


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.3, -0.4]))

    def trainable_state_dict(self):
        return {"weight": self.weight.detach().clone()}

    def load_trainable_state_dict(self, state):
        with torch.no_grad():
            self.weight.copy_(state["weight"])


def engine():
    result = TokenMemoryEngine(
        TinyModel(),
        TrainingConfig(dataset_dir="unused", save_total_limit=2),
        "cpu",
    )
    result.initialize()
    result.model.weight.square().sum().backward()
    result.optimizer.step()
    return result


def loader():
    return StatefulDataLoader(
        list(range(10)), batch_size=2, generator=torch.Generator().manual_seed(123)
    )


def run_record():
    return {"config": {"objective": {"method": "icae_single", "stage": "pretrain"}}}


def test_native_handler_restores_loader_weights_optimizer_cursor_and_rng(tmp_path):
    original_engine = engine()
    train_loader = loader()
    iterator = iter(train_loader)
    next(iterator)
    next(iterator)
    cursor = {"epoch": 0, "sample_offset": 4, "step": 2, "sample_visits": 4}
    handler = TokenMemoryCheckpointHandler(
        original_engine, train_loader, tmp_path, run_record(), cursor
    )
    handler.save_checkpoint(2)
    path = tmp_path / "global_step_2"
    expected_weight = original_engine.model.weight.detach().clone()
    expected_optimizer = deepcopy(original_engine.optimizer.state_dict())
    expected_rng = (random.random(), np.random.random(), torch.rand(2))
    expected_remaining = [batch.tolist() for batch in iterator]

    restored_engine = engine()
    with torch.no_grad():
        restored_engine.model.weight.zero_()
    restored_cursor = {"epoch": 0, "sample_offset": 0, "step": 0, "sample_visits": 0}
    restored_loader = loader()
    restored = TokenMemoryCheckpointHandler(
        restored_engine, restored_loader, tmp_path, run_record(), restored_cursor, path
    )
    assert restored.load_checkpoint() == 2
    assert restored_cursor == cursor
    torch.testing.assert_close(restored_engine.model.weight, expected_weight)
    actual_optimizer = restored_engine.optimizer.state_dict()
    assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
    for parameter, state in expected_optimizer["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(actual_optimizer["state"][parameter][name], value)
    assert random.random() == expected_rng[0]
    assert np.random.random() == expected_rng[1]
    torch.testing.assert_close(torch.rand(2), expected_rng[2])
    assert [batch.tolist() for batch in restored_loader] == expected_remaining
    assert find_latest_ckpt_path(str(tmp_path)) == str(path)
    assert read_checkpoint(path)["cursor"] == cursor


def test_native_retention_removes_old_directories_before_and_after_resume(tmp_path):
    original_engine, cursor = engine(), {"step": 0}
    handler = TokenMemoryCheckpointHandler(
        original_engine, loader(), tmp_path, run_record(), cursor
    )
    for step in range(1, 4):
        cursor["step"] = step
        handler.save_checkpoint(step)
    assert sorted(path.name for path in tmp_path.glob("global_step_*")) == [
        "global_step_2",
        "global_step_3",
    ]

    restored = TokenMemoryCheckpointHandler(
        original_engine, loader(), tmp_path, run_record(), cursor, tmp_path / "global_step_3"
    )
    restored.load_checkpoint()
    cursor["step"] = 4
    restored.save_checkpoint(4)
    assert sorted(path.name for path in tmp_path.glob("global_step_*")) == [
        "global_step_3",
        "global_step_4",
    ]


def test_resuming_older_checkpoint_can_overwrite_an_existing_step_without_duplicate_retention(
    tmp_path,
):
    original_engine, cursor = engine(), {"step": 0}
    handler = TokenMemoryCheckpointHandler(
        original_engine, loader(), tmp_path, run_record(), cursor
    )
    for step in (1, 2):
        cursor["step"] = step
        handler.save_checkpoint(step)

    resumed = TokenMemoryCheckpointHandler(
        original_engine, loader(), tmp_path, run_record(), cursor, tmp_path / "global_step_1"
    )
    assert resumed.load_checkpoint() == 1
    for step in (2, 3):
        cursor["step"] = step
        resumed.save_checkpoint(step)
    assert sorted(path.name for path in tmp_path.glob("global_step_*")) == [
        "global_step_2",
        "global_step_3",
    ]
    for step in (2, 3):
        restored_cursor = {}
        restored = TokenMemoryCheckpointHandler(
            engine(),
            loader(),
            tmp_path,
            run_record(),
            restored_cursor,
            tmp_path / f"global_step_{step}",
        )
        assert restored.load_checkpoint() == step
        assert restored_cursor == {"step": step}


def test_loader_save_failure_keeps_two_complete_checkpoints_and_tracker(tmp_path, monkeypatch):
    original_engine, cursor, train_loader = engine(), {"step": 0}, loader()
    handler = TokenMemoryCheckpointHandler(
        original_engine, train_loader, tmp_path, run_record(), cursor
    )
    for step in (1, 2):
        cursor["step"] = step
        handler.save_checkpoint(step)

    def fail():
        raise OSError("loader save failed")

    monkeypatch.setattr(train_loader, "state_dict", fail)
    cursor["step"] = 3
    with pytest.raises(OSError, match="loader save failed"):
        handler.save_checkpoint(3)
    assert (tmp_path / "global_step_1" / "data_0.pt").is_file()
    assert (tmp_path / "global_step_2" / "data_0.pt").is_file()
    assert find_latest_ckpt_path(str(tmp_path)) == str(tmp_path / "global_step_2")
    restored = TokenMemoryCheckpointHandler(
        original_engine, loader(), tmp_path, run_record(), cursor, tmp_path / "global_step_2"
    )
    restored.load_checkpoint()
    cursor["step"] = 4
    restored.save_checkpoint(4)
    assert not (tmp_path / "global_step_1").exists()
    assert (tmp_path / "global_step_2" / "data_0.pt").is_file()
    assert (tmp_path / "global_step_4" / "data_0.pt").is_file()


@pytest.mark.parametrize("missing", ["state.pt", "data_0.pt"])
def test_incomplete_checkpoint_cannot_silently_restart_loader(tmp_path, missing):
    original_engine, cursor = engine(), {"step": 1}
    handler = TokenMemoryCheckpointHandler(
        original_engine, loader(), tmp_path, run_record(), cursor
    )
    handler.save_checkpoint(1)
    path = tmp_path / "global_step_1"
    (path / missing).unlink()
    restored = TokenMemoryCheckpointHandler(
        original_engine, loader(), tmp_path, run_record(), cursor, path
    )
    with pytest.raises(FileNotFoundError, match=missing):
        restored.load_checkpoint()


def test_checkpoint_metadata_mismatch_precedes_model_or_loader_restore(tmp_path):
    original_engine, cursor = engine(), {"step": 1}
    handler = TokenMemoryCheckpointHandler(
        original_engine, loader(), tmp_path, run_record(), cursor
    )
    handler.save_checkpoint(1)
    with torch.no_grad():
        original_engine.model.weight.zero_()
    original_weight = original_engine.model.weight.detach().clone()
    changed = {**run_record(), "new_setting": True}
    restored = TokenMemoryCheckpointHandler(
        original_engine, loader(), tmp_path, changed, cursor, tmp_path / "global_step_1"
    )
    with pytest.raises(ValueError, match="resume configuration"):
        restored.load_checkpoint()
    torch.testing.assert_close(original_engine.model.weight, original_weight)


def test_checkpoint_rng_world_size_mismatch_precedes_model_restore(tmp_path):
    original_engine, cursor = engine(), {"step": 1}
    handler = TokenMemoryCheckpointHandler(
        original_engine, loader(), tmp_path, run_record(), cursor
    )
    handler.save_checkpoint(1)
    path = tmp_path / "global_step_1"
    checkpoint = read_checkpoint(path)
    checkpoint["rng"] = []
    torch.save(checkpoint, path / "state.pt")
    with torch.no_grad():
        original_engine.model.weight.zero_()
    restored = TokenMemoryCheckpointHandler(
        original_engine, loader(), tmp_path, run_record(), cursor, path
    )
    with pytest.raises(ValueError, match="RNG states do not match world size"):
        restored.load_checkpoint()
    assert not torch.count_nonzero(original_engine.model.weight)


def test_partial_loader_file_is_not_registered_as_a_complete_checkpoint_after_restart(
    tmp_path, monkeypatch
):
    original_engine, cursor = engine(), {"step": 0}
    handler = TokenMemoryCheckpointHandler(
        original_engine, loader(), tmp_path, run_record(), cursor
    )
    for step in (1, 2):
        cursor["step"] = step
        handler.save_checkpoint(step)
    original_save = torch.save

    def fail_partial_data_save(value, path, **kwargs):
        if Path(path).name == "data_0.pt":
            Path(path).write_bytes(b"partial loader state")
            raise OSError("partial data save failed")
        return original_save(value, path, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", fail_partial_data_save)
        cursor["step"] = 3
        with pytest.raises(OSError, match="partial data save failed"):
            handler.save_checkpoint(3)
    assert (tmp_path / "global_step_3" / "data_0.pt").is_file()
    assert not (tmp_path / "global_step_3" / "state.pt").exists()

    restored = TokenMemoryCheckpointHandler(
        original_engine, loader(), tmp_path, run_record(), cursor, tmp_path / "global_step_2"
    )
    assert restored.load_checkpoint() == 2
    cursor["step"] = 25
    restored.save_checkpoint(25)
    assert not (tmp_path / "global_step_1").exists()
    for step in (2, 25):
        restored_cursor = {}
        checkpoint = TokenMemoryCheckpointHandler(
            engine(),
            loader(),
            tmp_path,
            run_record(),
            restored_cursor,
            tmp_path / f"global_step_{step}",
        )
        assert checkpoint.load_checkpoint() == step
        assert restored_cursor == {"step": step}


def test_rewriting_an_existing_step_revokes_its_completion_on_save_failure(tmp_path, monkeypatch):
    original_engine, cursor, train_loader = engine(), {"step": 1}, loader()
    handler = TokenMemoryCheckpointHandler(
        original_engine, train_loader, tmp_path, run_record(), cursor
    )
    handler.save_checkpoint(1)

    def fail():
        raise OSError("loader save failed")

    monkeypatch.setattr(train_loader, "state_dict", fail)
    with pytest.raises(OSError, match="loader save failed"):
        handler.save_checkpoint(1)
    assert not (tmp_path / "global_step_1" / "state.pt").exists()
    restored = TokenMemoryCheckpointHandler(
        original_engine, loader(), tmp_path, run_record(), cursor, tmp_path / "global_step_1"
    )
    with pytest.raises(FileNotFoundError, match="state.pt"):
        restored.load_checkpoint()


@pytest.mark.parametrize("dirname", ["global_step_bad", "prefix_global_step_1"])
def test_valid_files_under_a_noncanonical_directory_name_cannot_resume(tmp_path, dirname):
    original_engine, cursor = engine(), {"step": 1}
    handler = TokenMemoryCheckpointHandler(
        original_engine, loader(), tmp_path, run_record(), cursor
    )
    handler.save_checkpoint(1)
    path = tmp_path / dirname
    (tmp_path / "global_step_1").rename(path)
    restored = TokenMemoryCheckpointHandler(
        original_engine, loader(), tmp_path / "new-checkpoints", run_record(), {}, path
    )
    with pytest.raises(ValueError, match="must be named global_step_<step>"):
        restored.load_checkpoint()
