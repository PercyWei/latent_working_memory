from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor

from cdic_repro.icae.modeling import LlamaICAE


def save_icae_checkpoint(model: LlamaICAE, checkpoint_path: Path) -> None:

    model_state = model.state_dict()
    checkpoint_state = {
        name: model_state[name].detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(checkpoint_state, temporary_path)
    temporary_path.replace(checkpoint_path)


def load_icae_checkpoint(model: LlamaICAE, checkpoint_path: Path) -> None:

    checkpoint_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint_state, Mapping):
        raise TypeError("ICAE checkpoint must contain a direct state-dict mapping")
    if any(not isinstance(key, str) for key in checkpoint_state):
        raise TypeError("ICAE checkpoint keys must be strings")
    if any(not isinstance(value, Tensor) for value in checkpoint_state.values()):
        raise TypeError("ICAE checkpoint values must be tensors")

    expected_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    checkpoint_names = set(checkpoint_state)
    missing = sorted(expected_names.difference(checkpoint_names))
    unexpected = sorted(checkpoint_names.difference(expected_names))
    if missing or unexpected:
        raise ValueError(
            f"checkpoint key mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    model_state = model.state_dict()
    for name, value in checkpoint_state.items():
        if value.shape != model_state[name].shape:
            raise ValueError(
                f"checkpoint tensor shape mismatch for {name}: "
                f"expected {tuple(model_state[name].shape)}, got {tuple(value.shape)}"
            )
        model_state[name] = value
    model.load_state_dict(model_state, strict=True)
