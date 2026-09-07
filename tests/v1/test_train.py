from __future__ import annotations

import pytest
import torch

from latent_working_memory.v1.train import _validate_device, parse_args


def test_train_entry_exposes_only_the_implemented_p0_phase() -> None:
    args = parse_args(
        [
            "--phase",
            "p0",
            "--config",
            "config.json",
            "--data-dir",
            "data",
            "--output-dir",
            "output",
        ]
    )
    assert args.phase == "p0"
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
        _validate_device(torch.device("cuda"))

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(RuntimeError, match="only permits"):
        _validate_device(torch.device("cuda"))

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    _validate_device(torch.device("cuda:1"))
    with pytest.raises(RuntimeError, match="not visible"):
        _validate_device(torch.device("cuda:2"))
