from __future__ import annotations

import pytest
import torch

from latent_working_memory.devices import validate_device
from latent_working_memory.v1.train import parse_args


def test_train_entry_exposes_only_the_implemented_pretrain_phase() -> None:
    args = parse_args(
        [
            "--phase",
            "pretrain",
            "--config",
            "config.json",
            "--data-dir",
            "data",
            "--output-dir",
            "output",
        ]
    )
    assert args.phase == "pretrain"
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--phase",
                "p1",
                "--config",
                "config.json",
                "--data-dir",
                "data",
                "--output-dir",
                "output",
            ]
        )


def test_cuda_device_must_resolve_to_physical_gpu_zero_or_one(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="explicitly select"):
        validate_device(torch.device("cuda"))

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(RuntimeError, match="only permits"):
        validate_device(torch.device("cuda"))

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    validate_device(torch.device("cuda:1"))
    with pytest.raises(RuntimeError, match="not visible"):
        validate_device(torch.device("cuda:2"))
