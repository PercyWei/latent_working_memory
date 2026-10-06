from pathlib import Path

import pytest

from latent_working_memory.v3.config import (
    ModelConfig,
    ObjectiveConfig,
    TrainingConfig,
    load_experiment,
)


@pytest.mark.parametrize("filename", sorted(Path("configs/v3").glob("*.json")))
def test_presets_use_requested_model_and_memory_size(filename):
    config = load_experiment(filename)
    assert config.model.model_name_or_path == "Qwen/Qwen3-4B-Instruct-2507"
    assert config.model.memory_slots == 64
    assert config.training.swanlab_project is None


@pytest.mark.parametrize(
    "options",
    [
        {"method": "autocompressors", "stage": "qa"},
        {"method": "autocompressors", "stage": "lm", "ac_bptt_steps": 1},
        {"method": "information_loss", "stage": "policy", "eta": 0.1, "threshold_g": 0.1},
        {"threshold_i": float("nan")},
        {"rms_epsilon": 0.0},
        {"append_probability": 1.1},
        {"qa_prompt": "{question}: {answer}"},
        {"qa_batch_size": True},
    ],
)
def test_objective_rejects_invalid_experiment_contracts(options):
    with pytest.raises(ValueError):
        ObjectiveConfig(**options)


def test_model_rejects_module_string_in_place_of_list():
    with pytest.raises(ValueError, match="sequence"):
        ModelConfig(lora_target_modules="q_proj")


def test_tracking_requires_explicit_group():
    with pytest.raises(ValueError, match="group"):
        TrainingConfig("data", "output", swanlab_project="existing-project")


@pytest.mark.parametrize("name", ["max_train_samples", "max_dev_samples"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_sample_limits_require_positive_integers(name, value):
    with pytest.raises(ValueError, match=name):
        TrainingConfig("data", "output", **{name: value})


def test_sample_limits_default_to_full_splits():
    config = TrainingConfig("data", "output")
    assert config.max_train_samples is None
    assert config.max_dev_samples is None
