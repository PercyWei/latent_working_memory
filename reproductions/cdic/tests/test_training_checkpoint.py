from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cdic_repro.training_checkpoint import (
    TrainingProgress,
    load_model_from_training_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)


class FakeModel:
    def __init__(self) -> None:
        self.state = {"weight": 1}

    def trainable_state_dict(self) -> dict[str, object]:
        return dict(self.state)

    def load_trainable_state_dict(
        self,
        state_dict: dict[str, object],
        strict: bool = True,
    ) -> None:
        assert strict
        self.state = dict(state_dict)


class FakeOptimizer:
    def __init__(self) -> None:
        self.state = {"step": 1}

    def state_dict(self) -> dict[str, object]:
        return dict(self.state)

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.state = dict(state_dict)


def test_training_checkpoint_restores_model_optimizer_progress_and_rng(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    model = FakeModel()
    optimizer = FakeOptimizer()
    torch.manual_seed(123)
    expected_rng_state = torch.get_rng_state().clone()
    progress = TrainingProgress(epoch=1, next_episode_position=7, global_step=18)
    save_training_checkpoint(
        path,
        model=model,  # type: ignore[arg-type]
        optimizer=optimizer,
        progress=progress,
        config_fingerprint="fingerprint",
    )
    model.state = {"weight": 9}
    optimizer.state = {"step": 99}
    torch.manual_seed(999)

    restored = load_training_checkpoint(
        path,
        model=model,  # type: ignore[arg-type]
        optimizer=optimizer,
        expected_config_fingerprint="fingerprint",
    )

    assert restored == progress
    assert model.state == {"weight": 1}
    assert optimizer.state == {"step": 1}
    assert torch.equal(torch.get_rng_state(), expected_rng_state)


def test_training_checkpoint_rejects_different_config(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    model = FakeModel()
    optimizer = FakeOptimizer()
    save_training_checkpoint(
        path,
        model=model,  # type: ignore[arg-type]
        optimizer=optimizer,
        progress=TrainingProgress(),
        config_fingerprint="expected",
    )

    with pytest.raises(ValueError, match="fingerprint"):
        load_training_checkpoint(
            path,
            model=model,  # type: ignore[arg-type]
            optimizer=optimizer,
            expected_config_fingerprint="different",
        )


def test_evaluation_restore_loads_only_model_state(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    model = FakeModel()
    optimizer = FakeOptimizer()
    torch.manual_seed(321)
    progress = TrainingProgress(epoch=2, next_episode_position=0, global_step=1002)
    save_training_checkpoint(
        path,
        model=model,  # type: ignore[arg-type]
        optimizer=optimizer,
        progress=progress,
        config_fingerprint="training-fingerprint",
    )
    model.state = {"weight": 9}
    optimizer.state = {"step": 99}
    torch.manual_seed(654)
    changed_rng_state = torch.get_rng_state().clone()

    restored = load_model_from_training_checkpoint(
        path,
        model=model,
    )

    assert restored == progress
    assert model.state == {"weight": 1}
    assert optimizer.state == {"step": 99}
    assert torch.equal(torch.get_rng_state(), changed_rng_state)
