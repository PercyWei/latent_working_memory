from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from cdic_repro.training_checkpoint import (
    TrainingProgress,
    load_model_from_training_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)


class FakeCuda:
    def __init__(self) -> None:
        self.state = ["cuda-state"]

    def is_available(self) -> bool:
        return True

    def get_rng_state(self) -> list[str]:
        return list(self.state)

    def set_rng_state(self, value: list[str]) -> None:
        self.state = list(value)


class FakeTorch:
    def __init__(self) -> None:
        self.state = "cpu-state"
        self.cuda = FakeCuda()

    def get_rng_state(self) -> str:
        return self.state

    def set_rng_state(self, value: str) -> None:
        self.state = value

    def save(self, payload: object, path: Path) -> None:
        path.write_bytes(pickle.dumps(payload))

    def load(self, path: Path, **_: object) -> object:
        return pickle.loads(path.read_bytes())


class FakeModel:
    def __init__(self) -> None:
        self.state = {"weight": 1}

    def trainable_state_dict(self) -> dict[str, object]:
        return dict(self.state)

    def load_trainable_state_dict(
        self,
        state_dict: dict[str, object],
        *,
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
    torch = FakeTorch()
    progress = TrainingProgress(epoch=1, next_episode_position=7, global_step=18)
    save_training_checkpoint(
        path,
        model=model,  # type: ignore[arg-type]
        optimizer=optimizer,
        progress=progress,
        config_fingerprint="fingerprint",
        torch_module=torch,
    )
    model.state = {"weight": 9}
    optimizer.state = {"step": 99}
    torch.state = "changed"
    torch.cuda.state = ["changed"]

    restored = load_training_checkpoint(
        path,
        model=model,  # type: ignore[arg-type]
        optimizer=optimizer,
        expected_config_fingerprint="fingerprint",
        torch_module=torch,
    )

    assert restored == progress
    assert model.state == {"weight": 1}
    assert optimizer.state == {"step": 1}
    assert torch.state == "cpu-state"
    assert torch.cuda.state == ["cuda-state"]


def test_training_checkpoint_rejects_different_config(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    model = FakeModel()
    optimizer = FakeOptimizer()
    torch = FakeTorch()
    save_training_checkpoint(
        path,
        model=model,  # type: ignore[arg-type]
        optimizer=optimizer,
        progress=TrainingProgress(),
        config_fingerprint="expected",
        torch_module=torch,
    )

    with pytest.raises(ValueError, match="fingerprint"):
        load_training_checkpoint(
            path,
            model=model,  # type: ignore[arg-type]
            optimizer=optimizer,
            expected_config_fingerprint="different",
            torch_module=torch,
        )


def test_evaluation_restore_loads_only_model_state(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    model = FakeModel()
    optimizer = FakeOptimizer()
    torch = FakeTorch()
    progress = TrainingProgress(epoch=2, next_episode_position=0, global_step=1002)
    save_training_checkpoint(
        path,
        model=model,  # type: ignore[arg-type]
        optimizer=optimizer,
        progress=progress,
        config_fingerprint="training-fingerprint",
        torch_module=torch,
    )
    model.state = {"weight": 9}
    optimizer.state = {"step": 99}
    torch.state = "changed"
    torch.cuda.state = ["changed"]

    restored = load_model_from_training_checkpoint(
        path,
        model=model,
        torch_module=torch,
    )

    assert restored == progress
    assert model.state == {"weight": 1}
    assert optimizer.state == {"step": 99}
    assert torch.state == "changed"
    assert torch.cuda.state == ["changed"]
