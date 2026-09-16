"""GMSA checkpoint: model.json, stage.json, model.safetensors."""

import json
from dataclasses import asdict
from pathlib import Path

from safetensors.torch import load_file, save_file

from latent_working_memory.v2.gmsa_config import GMSAConfig


def save_model(model, directory, state_dict=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = model.state_dict() if state_dict is None else state_dict
    tensors, pointers = {}, set()
    for name, value in state.items():
        value = value.detach().cpu().contiguous()
        pointer = value.data_ptr()
        tensors[name] = value.clone() if pointer in pointers else value
        pointers.add(pointer)
    temporary = directory / "model.safetensors.tmp"
    save_file(tensors, str(temporary))
    temporary.replace(directory / "model.safetensors")
    (directory / "model.json").write_text(json.dumps(asdict(model.model_config), indent=2) + "\n")
    (directory / "stage.json").write_text(json.dumps({"stage": model.stage}) + "\n")


def load_weights(model, directory):
    directory = Path(directory)
    config = GMSAConfig(**json.loads((directory / "model.json").read_text()))
    if config != model.model_config:
        raise ValueError("checkpoint model configuration does not match")
    model.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True)


def checkpoint_stage(directory):
    return json.loads((Path(directory) / "stage.json").read_text())["stage"]
