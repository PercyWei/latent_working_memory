from pathlib import Path

import pytest

from latent_working_memory.v3.config import (
    ExperimentConfig,
    ModelConfig,
    ObjectiveConfig,
    TrainingConfig,
    load_experiment,
)


@pytest.mark.parametrize("filename", sorted(Path("configs/v3").glob("*.json")))
def test_presets_use_requested_model_and_memory_size(filename):
    config = load_experiment(filename)
    assert config.model.model_name_or_path == str(Path.home() / "models/Qwen3-4B-Instruct-2507")
    assert config.model.memory_slots == 512
    assert config.model.gradient_checkpointing is True
    dynamic = config.objective.method in {"memory_change", "information_loss"}
    assert config.objective.append_slots == (32 if dynamic else 8)
    large_microbatch = filename.name == "dynamic_pretrain.json" or config.objective.method in {
        "icae_single",
        "icae_multi",
    }
    assert config.training.micro_batch_size_per_gpu == (8 if large_microbatch else 4)
    assert config.training.gradient_accumulation_steps == (1 if large_microbatch else 2)
    assert config.training.global_batch_size(2) == 16
    assert config.training.swanlab_project is None
    assert config.training.lm_target_tokens == 512
    assert config.objective.bptt_steps is None
    assert config.objective.icae_segment_ratio == 3
    if config.objective.stage in {"pretrain", "lm"}:
        assert config.training.pretrain_data_view == "multisegment_random_prefix"
        assert config.training.min_input_tokens == 1
        assert config.training.max_input_tokens == 8192
        assert config.training.lm_ratio == 0.5


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
        {"append_slots": 0},
        {"append_slots": True},
        {"append_slots": 1.5},
        {"method": "memory_change", "stage": "warmup", "bptt_steps": 0},
        {"method": "memory_change", "stage": "warmup", "bptt_steps": True},
        {"method": "memory_change", "stage": "policy", "bptt_steps": 1.5},
        {"method": "memory_change", "stage": "pretrain", "bptt_steps": 2},
        {"method": "icae_single", "stage": "qa", "bptt_steps": 2},
        {"method": "autocompressors", "stage": "lm", "bptt_steps": 2},
    ],
)
def test_objective_rejects_invalid_experiment_contracts(options):
    with pytest.raises(ValueError):
        ObjectiveConfig(**options)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_icae_segment_ratio_requires_a_positive_integer(value):
    with pytest.raises(ValueError, match="icae_segment_ratio"):
        ObjectiveConfig(method="icae_multi", icae_segment_ratio=value)


def test_icae_segment_ratio_defaults_to_three():
    assert ObjectiveConfig(method="icae_multi").icae_segment_ratio == 3
    assert ObjectiveConfig(method="icae_multi", icae_segment_ratio=2).icae_segment_ratio == 2


def test_model_rejects_module_string_in_place_of_list():
    with pytest.raises(ValueError, match="sequence"):
        ModelConfig(lora_target_modules="q_proj")


def test_dynamic_append_size_is_bounded_by_available_gist_embeddings():
    with pytest.raises(ValueError, match="append_slots"):
        ExperimentConfig(
            ModelConfig(memory_slots=4),
            ObjectiveConfig(append_slots=8),
            TrainingConfig("data", "output"),
        )
    config = ExperimentConfig(
        ModelConfig(memory_slots=64),
        ObjectiveConfig(append_slots=8),
        TrainingConfig("data", "output"),
    )
    assert config.to_dict()["objective"]["append_slots"] == 8


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_gradient_checkpointing_requires_boolean(value):
    with pytest.raises(ValueError, match="gradient_checkpointing"):
        ModelConfig(gradient_checkpointing=value)


def test_tracking_requires_explicit_group():
    with pytest.raises(ValueError, match="group"):
        TrainingConfig("data", "output", swanlab_project="existing-project")


@pytest.mark.parametrize(
    "options",
    [
        {"experiment_dir": "experiment"},
        {"experiment_id": "20261007-01"},
        {"experiment_dir": "experiment", "experiment_id": "../other"},
    ],
)
def test_experiment_identity_requires_paired_valid_fields(options):
    with pytest.raises(ValueError, match="experiment_"):
        TrainingConfig("data", "output", **options)


def test_stage_output_belongs_to_method_experiment(tmp_path):
    root = tmp_path / "memory-change-k64_20261007-01"
    training = TrainingConfig(
        "data", str(root / "warmup"), experiment_dir=str(root), experiment_id="20261007-01"
    )
    ExperimentConfig(ModelConfig(), ObjectiveConfig(stage="warmup"), training)
    with pytest.raises(ValueError, match="output_dir"):
        ExperimentConfig(ModelConfig(), ObjectiveConfig(stage="policy"), training)


@pytest.mark.parametrize(
    "name",
    [
        "max_train_samples",
        "max_dev_samples",
        "micro_batch_size_per_gpu",
        "gradient_accumulation_steps",
        "lm_target_tokens",
    ],
)
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_sample_limits_require_positive_integers(name, value):
    with pytest.raises(ValueError, match=name):
        TrainingConfig("data", "output", **{name: value})


def test_sample_limits_default_to_full_splits():
    config = TrainingConfig("data", "output")
    assert config.max_train_samples is None
    assert config.max_dev_samples is None


@pytest.mark.parametrize("method", ["memory_change", "information_loss"])
@pytest.mark.parametrize("stage", ["warmup", "policy"])
def test_dynamic_bptt_window_is_optional_and_counts_write_rounds(method, stage):
    assert ObjectiveConfig(method=method, stage=stage).bptt_steps is None
    assert ObjectiveConfig(method=method, stage=stage, bptt_steps=2).bptt_steps == 2


def test_presets_are_named_for_complete_method_flows():
    assert {path.name for path in Path("configs/v3").glob("*.json")} == {
        "dynamic_pretrain.json",
        "icae_single.json",
        "icae_multi.json",
        "autocompressors.json",
        "memory_change.json",
        "information_loss.json",
    }


def test_pretraining_data_view_requires_an_explicit_supported_format():
    assert TrainingConfig("data", "output").pretrain_data_view == "text_samples"
    assert (
        TrainingConfig(
            "data", "output", pretrain_data_view="multisegment_random_prefix"
        ).pretrain_data_view
        == "multisegment_random_prefix"
    )
    for view in (
        "auto",
        "reconstruction_single",
        "reconstruction_first_write",
        "multisegment_full",
        "multisegment_first_write",
    ):
        with pytest.raises(ValueError, match="pretrain_data_view"):
            TrainingConfig("data", "output", pretrain_data_view=view)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_lm_ratio_requires_a_finite_probability(value):
    with pytest.raises(ValueError, match="lm_ratio"):
        TrainingConfig("data", "output", lm_ratio=value)


@pytest.mark.parametrize("value", [0, 0.5, 1])
def test_lm_ratio_allows_ae_only_and_lm_only(value):
    assert TrainingConfig("data", "output", lm_ratio=value).lm_ratio == value


@pytest.mark.parametrize(
    "world_size,microbatch,accumulation,expected", [(2, 1, 4, 8), (2, 2, 2, 8), (4, 2, 3, 24)]
)
def test_global_batch_is_derived_from_devices_and_microbatches(
    world_size, microbatch, accumulation, expected
):
    config = TrainingConfig(
        "data",
        "output",
        micro_batch_size_per_gpu=microbatch,
        gradient_accumulation_steps=accumulation,
    )
    assert config.global_batch_size(world_size) == expected
