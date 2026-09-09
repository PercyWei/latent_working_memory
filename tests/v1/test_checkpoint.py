from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from latent_working_memory.v1.checkpoint import (
    capture_rng_state,
    load_model_checkpoint,
    load_runtime_memory,
    restore_rng_state,
    save_model_checkpoint,
    save_runtime_memory,
)
from latent_working_memory.v1.config import ExperimentConfig
from latent_working_memory.v1.model import JointMemoryWriter
from latent_working_memory.v1.state import MemoryState


def test_model_checkpoint_round_trip_and_rng_restore(tmp_path: Path) -> None:
    config = ExperimentConfig()
    writer = JointMemoryWriter(d_mem=8, num_layers=1, num_heads=2, ffn_dim=16)
    random.seed(3)
    torch.manual_seed(4)
    rng_state = capture_rng_state()
    expected_python = random.random()
    expected_torch = torch.rand(2)

    path = tmp_path / "model.pt"
    save_model_checkpoint(
        path,
        "pretrain",
        config,
        writer.state_dict(),
        {},
        {"episode": 2},
        rng_state,
    )
    loaded = load_model_checkpoint(path)
    assert loaded.phase == "pretrain"
    assert loaded.config == config
    assert loaded.progress == {"episode": 2}

    restore_rng_state(loaded.rng_state)
    assert random.random() == expected_python
    assert torch.equal(torch.rand(2), expected_torch)


def test_runtime_memory_is_bf16_and_bound_to_model_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "runtime.pt"
    state = MemoryState(torch.randn(3, 8, dtype=torch.bfloat16), seen_tokens=64)
    save_runtime_memory(path, "checkpoints/v1/p1.pt", state)
    loaded = load_runtime_memory(path, "checkpoints/v1/p1.pt", 8)
    assert loaded.seen_tokens == 64
    assert torch.equal(loaded.values, state.values)

    with pytest.raises(ValueError, match="different model checkpoint"):
        load_runtime_memory(path, "checkpoints/v1/other.pt", 8)
    with pytest.raises(TypeError, match="bfloat16"):
        save_runtime_memory(path, "checkpoints/v1/p1.pt", MemoryState(torch.zeros(3, 8), 0))


def test_checkpoint_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "runtime.pt"
    torch.save(
        {
            "unexpected": 1,
            "model_checkpoint": "model.pt",
            "values": torch.zeros(1, 2, dtype=torch.bfloat16),
            "seen_tokens": 0,
        },
        path,
    )
    with pytest.raises(ValueError, match="invalid runtime memory fields"):
        load_runtime_memory(path, "model.pt", 2)
