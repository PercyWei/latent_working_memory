import json
from dataclasses import replace
from pathlib import Path

import pytest

from latent_working_memory.v3.config import (
    ExperimentConfig,
    ModelConfig,
    ObjectiveConfig,
    TrainingConfig,
    load_experiment,
    load_preset,
    validate_stage_sequence,
)


@pytest.mark.parametrize("filename", sorted(Path("configs/v3").glob("*.json")))
def test_presets_use_requested_model_and_memory_size(filename):
    for config in load_preset(filename):
        if filename.name == "dynamic_pretrain.json":
            assert config.objective.method == "dynamic"
        assert "output_dir" not in json.loads(filename.read_text())["training"]
        assert config.training.output_dir is None
        assert config.training.init_checkpoint is None
        assert config.model.model_name_or_path == str(Path.home() / "models/Qwen3-4B-Instruct-2507")
        assert config.model.memory_slots == 512
        assert config.model.gradient_checkpointing is True
        dynamic = config.objective.method in {"memory_change", "information_loss"}
        assert config.objective.append_slots == (
            32 if dynamic and config.objective.stage != "pretrain" else 8
        )
        large_microbatch = filename.name == "dynamic_pretrain.json" or config.objective.method in {
            "icae_single",
            "icae_multi",
            "autocompressors",
        }
        assert config.training.micro_batch_size_per_gpu == (8 if large_microbatch else 4)
        assert config.training.gradient_accumulation_steps == (1 if large_microbatch else 2)
        assert config.training.global_batch_size(2) == 16
        assert config.training.save_total_limit == 2
        assert config.training.swanlab_project is None
        assert config.training.lm_target_tokens == 512
        assert config.training.max_qa_input_tokens == (
            None if filename.name in {"dynamic_pretrain.json", "autocompressors.json"} else 12288
        )
        assert config.objective.bptt_steps == (
            2 if config.objective.method == "autocompressors" else None
        )
        assert (config.objective.icae_min_segments, config.objective.icae_max_segments) == (3, 6)
        raw_objective = json.loads(filename.read_text())["objective"]
        ac_fields = {name for name in raw_objective if name.startswith("ac_")}
        assert ac_fields == (
            {"ac_num_segments"} if config.objective.method == "autocompressors" else set()
        )
        assert config.objective.ac_num_segments == 4
        if config.objective.stage == "pretrain":
            assert config.training.min_input_tokens == 1
            assert config.training.max_input_tokens == 12288
            assert config.training.lm_ratio == 0.5
        assert config.training.max_train_samples == (
            12800 if config.objective.stage == "pretrain" else None
        )


@pytest.mark.parametrize(
    "options",
    [
        {"method": "autocompressors", "stage": "qa"},
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
        {"method": "dynamic", "stage": "pretrain", "bptt_steps": 2},
        {"method": "dynamic", "stage": "warmup"},
        {"method": "dynamic", "stage": "policy"},
        {"method": "dynamic", "stage": "qa"},
        {"method": "dynamic", "stage": "lm"},
        {"method": "icae_single", "stage": "qa", "bptt_steps": 2},
        {"method": "icae_multi", "stage": "pretrain", "bptt_steps": 2},
    ],
)
def test_objective_rejects_invalid_experiment_contracts(options):
    with pytest.raises(ValueError):
        ObjectiveConfig(**options)


@pytest.mark.parametrize("name", ["icae_min_segments", "icae_max_segments"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_icae_segment_count_bounds_require_positive_integers(name, value):
    with pytest.raises(ValueError, match=name):
        ObjectiveConfig(method="icae_multi", **{name: value})


def test_icae_segment_count_defaults_to_three_through_six():
    config = ObjectiveConfig(method="icae_multi")
    assert (config.icae_min_segments, config.icae_max_segments) == (3, 6)
    fixed = ObjectiveConfig(method="icae_multi", icae_min_segments=2, icae_max_segments=2)
    assert fixed.icae_min_segments == fixed.icae_max_segments == 2
    with pytest.raises(ValueError, match="interval"):
        ObjectiveConfig(method="icae_multi", icae_min_segments=7, icae_max_segments=6)


@pytest.mark.parametrize("slots,segments", [(4, 6), (8, 16)])
def test_icae_multi_requires_a_positive_slot_allocation_for_every_segment(slots, segments):
    with pytest.raises(ValueError, match="must not exceed memory_slots"):
        ExperimentConfig(
            ModelConfig(memory_slots=slots),
            ObjectiveConfig(method="icae_multi", icae_max_segments=segments),
            TrainingConfig("data"),
        )


def test_icae_multi_slot_budget_does_not_need_to_be_divisible_by_segment_count():
    config = ExperimentConfig(
        ModelConfig(memory_slots=512), ObjectiveConfig(method="icae_multi"), TrainingConfig("data")
    )
    assert config.objective.icae_max_segments == 6


@pytest.mark.parametrize("name", ["ac_num_segments", "bptt_steps"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_autocompressors_counts_require_positive_integers(name, value):
    with pytest.raises(ValueError, match=name):
        ObjectiveConfig(method="autocompressors", stage="pretrain", **{name: value})


def test_autocompressors_runtime_default_is_full_bptt_and_preset_uses_two_steps():
    config = ObjectiveConfig(method="autocompressors", stage="pretrain")
    assert (config.ac_num_segments, config.bptt_steps) == (4, None)
    (preset,) = load_preset("configs/v3/autocompressors.json")
    assert preset.objective.bptt_steps == 2


@pytest.mark.parametrize("segments,bptt_steps", [(1, 1), (1, 2), (4, 7)])
def test_autocompressors_accepts_one_segment_and_bptt_windows_longer_than_the_document(
    segments, bptt_steps
):
    config = ExperimentConfig(
        ModelConfig(memory_slots=7),
        ObjectiveConfig(
            method="autocompressors",
            stage="pretrain",
            ac_num_segments=segments,
            bptt_steps=bptt_steps,
        ),
        TrainingConfig("data"),
    )
    assert (config.objective.ac_num_segments, config.objective.bptt_steps) == (
        segments,
        bptt_steps,
    )


def test_autocompressors_segment_count_does_not_need_to_be_divisible_by_bptt_window():
    config = ObjectiveConfig(
        method="autocompressors", stage="pretrain", ac_num_segments=5, bptt_steps=2
    )
    assert (config.ac_num_segments, config.bptt_steps) == (5, 2)


@pytest.mark.parametrize(
    "method,stage",
    [
        ("icae_single", "pretrain"),
        ("icae_single", "qa"),
        ("icae_multi", "pretrain"),
        ("icae_multi", "qa"),
        ("autocompressors", "pretrain"),
        ("dynamic", "pretrain"),
        ("memory_change", "pretrain"),
        ("memory_change", "warmup"),
        ("memory_change", "policy"),
        ("information_loss", "pretrain"),
        ("information_loss", "warmup"),
        ("information_loss", "policy"),
    ],
)
def test_full_bptt_is_valid_for_each_method_stage(method, stage):
    assert ObjectiveConfig(method=method, stage=stage, bptt_steps=None).bptt_steps is None


def test_removed_autocompressors_bptt_field_is_rejected(tmp_path):
    with pytest.raises(TypeError, match="ac_bptt_steps"):
        ObjectiveConfig(method="autocompressors", stage="pretrain", ac_bptt_steps=2)
    raw = json.loads(Path("configs/v3/autocompressors.json").read_text())
    raw["objective"]["ac_bptt_steps"] = raw["objective"].pop("bptt_steps")
    preset = tmp_path / "old-ac.json"
    preset.write_text(json.dumps(raw))
    with pytest.raises(TypeError, match="ac_bptt_steps"):
        load_preset(preset)


def test_autocompressors_allocates_a_positive_slot_count_to_every_segment():
    with pytest.raises(ValueError, match="ac_num_segments.*memory_slots"):
        ExperimentConfig(
            ModelConfig(memory_slots=3),
            ObjectiveConfig(method="autocompressors", stage="pretrain"),
            TrainingConfig("data"),
        )


def test_autocompressors_slot_budget_does_not_need_to_be_divisible_by_segment_count():
    config = ExperimentConfig(
        ModelConfig(memory_slots=7),
        ObjectiveConfig(method="autocompressors", stage="pretrain"),
        TrainingConfig("data"),
    )
    assert config.objective.ac_num_segments == 4


def test_model_rejects_module_string_in_place_of_list():
    with pytest.raises(ValueError, match="sequence"):
        ModelConfig(lora_target_modules="q_proj")


def test_dynamic_append_size_is_bounded_by_available_gist_embeddings():
    with pytest.raises(ValueError, match="append_slots"):
        ExperimentConfig(
            ModelConfig(memory_slots=4),
            ObjectiveConfig(method="memory_change", append_slots=8),
            TrainingConfig("data", "output"),
        )
    config = ExperimentConfig(
        ModelConfig(memory_slots=64),
        ObjectiveConfig(method="memory_change", append_slots=8),
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
    ExperimentConfig(
        ModelConfig(), ObjectiveConfig(method="memory_change", stage="warmup"), training
    )
    with pytest.raises(ValueError, match="output_dir"):
        ExperimentConfig(
            ModelConfig(), ObjectiveConfig(method="memory_change", stage="policy"), training
        )


def test_output_can_be_unresolved_only_without_a_method_directory():
    template = TrainingConfig("data")
    assert template.output_dir is None
    ExperimentConfig(ModelConfig(), ObjectiveConfig(), template)
    with pytest.raises(ValueError, match="output_dir"):
        ExperimentConfig(
            ModelConfig(),
            ObjectiveConfig(),
            replace(template, experiment_dir="experiment", experiment_id="unit-job"),
        )


@pytest.mark.parametrize(
    "name",
    [
        "max_train_samples",
        "max_dev_samples",
        "micro_batch_size_per_gpu",
        "gradient_accumulation_steps",
        "save_total_limit",
        "lm_target_tokens",
        "max_input_tokens",
        "max_qa_input_tokens",
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
    assert config.max_input_tokens is None
    assert config.max_qa_input_tokens is None


def test_pretraining_minimum_cannot_exceed_the_input_limit():
    with pytest.raises(ValueError, match="min_input_tokens"):
        TrainingConfig("data", min_input_tokens=2048, max_input_tokens=1024)


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


@pytest.mark.parametrize(
    "filename,stages",
    [
        ("icae_single.json", ("pretrain", "qa")),
        ("icae_multi.json", ("pretrain", "qa")),
        ("dynamic_pretrain.json", ("pretrain",)),
        ("autocompressors.json", ("pretrain",)),
        ("memory_change.json", ("warmup", "policy")),
        ("information_loss.json", ("warmup", "policy")),
    ],
)
def test_presets_expand_explicit_stage_sequences_without_changing_other_options(filename, stages):
    preset = Path("configs/v3") / filename
    raw = json.loads(preset.read_text())
    assert "stage" not in raw["objective"]
    configs = load_preset(preset)
    assert tuple(config.objective.stage for config in configs) == stages
    for config in configs:
        assert config.model == configs[0].model
        assert replace(
            config.training, max_train_samples=configs[0].training.max_train_samples
        ) == (configs[0].training)
        assert (
            config.training.max_train_samples
            == raw["training"]["stage_max_train_samples"][config.objective.stage]
        )
        assert replace(config.objective, stage=stages[0]) == configs[0].objective
        assert "stages" not in config.to_dict()["objective"]
        assert "stage_max_train_samples" not in config.to_dict()["training"]


def test_preset_expands_independent_stage_budgets(tmp_path):
    raw = json.loads(Path("configs/v3/icae_single.json").read_text())
    raw["training"]["stage_max_train_samples"] = {"pretrain": 37, "qa": None}
    path = tmp_path / "preset.json"
    path.write_text(json.dumps(raw))
    pretrain, qa = load_preset(path)
    assert pretrain.training.max_train_samples == 37
    assert qa.training.max_train_samples is None


@pytest.mark.parametrize(
    "limits",
    [
        {"pretrain": 4},
        {"pretrain": 4, "qa": 3, "policy": 2},
        {"pretrain": 0, "qa": 3},
        {"pretrain": True, "qa": 3},
        [4, 3],
    ],
)
def test_preset_rejects_missing_extra_or_invalid_stage_budgets(tmp_path, limits):
    raw = json.loads(Path("configs/v3/icae_single.json").read_text())
    raw["training"]["stage_max_train_samples"] = limits
    path = tmp_path / "preset.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="stage_max_train_samples|max_train_samples"):
        load_preset(path)


def test_preset_rejects_one_shared_training_budget(tmp_path):
    raw = json.loads(Path("configs/v3/icae_single.json").read_text())
    raw["training"]["max_train_samples"] = 7
    path = tmp_path / "preset.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="stage_max_train_samples"):
        load_preset(path)


@pytest.mark.parametrize("method", ["icae_single", "icae_multi"])
@pytest.mark.parametrize("stages", [["pretrain", "qa"], ["pretrain"], ["qa"]])
def test_icae_accepts_full_pipeline_and_single_stage_presets(method, stages):
    assert validate_stage_sequence(method, stages) == tuple(stages)


@pytest.mark.parametrize("method", ["memory_change", "information_loss"])
@pytest.mark.parametrize("stages", [["warmup", "policy"], ["warmup"], ["policy"], ["pretrain"]])
def test_dynamic_accepts_qa_pipeline_subsets_and_separate_shared_pretraining(method, stages):
    assert validate_stage_sequence(method, stages) == tuple(stages)


def test_shared_dynamic_pretraining_is_the_default_and_has_its_own_stage_sequence():
    config = ObjectiveConfig()
    assert (config.method, config.stage) == ("dynamic", "pretrain")
    assert validate_stage_sequence("dynamic", ["pretrain"]) == ("pretrain",)


@pytest.mark.parametrize("stage", ["pretrain", "lm"])
def test_stage_sequence_accepts_autocompressors_pretraining_and_legacy_lm(stage):
    assert validate_stage_sequence("autocompressors", (stage,)) == (stage,)


@pytest.mark.parametrize("stage", ["pretrain", "lm"])
@pytest.mark.parametrize("bptt_steps", [None, 1, 2])
def test_autocompressors_stage_keeps_its_original_value_and_bptt_setting(stage, bptt_steps):
    config = ObjectiveConfig(method="autocompressors", stage=stage, bptt_steps=bptt_steps)
    assert config.stage == stage
    assert config.bptt_steps == bptt_steps


def test_legacy_autocompressors_preset_keeps_lm_stage_and_budget(tmp_path):
    raw = json.loads(Path("configs/v3/autocompressors.json").read_text())
    raw["objective"]["stages"] = ["lm"]
    raw["training"]["stage_max_train_samples"] = {"lm": 11}
    path = tmp_path / "legacy-ac.json"
    path.write_text(json.dumps(raw))
    (config,) = load_preset(path)
    assert config.objective.stage == "lm"
    assert config.training.max_train_samples == 11


@pytest.mark.parametrize(
    "method,stages",
    [
        ("unknown", ["pretrain"]),
        ("dynamic", ["warmup", "policy"]),
        ("dynamic", ["pretrain", "warmup"]),
        ("icae_single", ["qa", "pretrain"]),
        ("icae_single", ["pretrain", "pretrain"]),
        ("icae_multi", ["pretrain", "policy"]),
        ("memory_change", ["policy", "warmup"]),
        ("memory_change", ["warmup", "warmup"]),
        ("memory_change", ["pretrain", "warmup"]),
        ("memory_change", ["pretrain", "policy"]),
        ("information_loss", ["pretrain", "warmup", "policy"]),
        ("information_loss", ["warmup", "qa", "policy"]),
        ("autocompressors", ["lm", "lm"]),
        ("autocompressors", ["pretrain", "lm"]),
        ("autocompressors", ["lm", "pretrain"]),
        ("icae_single", []),
        ("icae_single", ()),
        ("icae_single", "pretrain"),
        ("icae_single", {"pretrain"}),
        ("icae_single", None),
        ("icae_single", [None]),
    ],
)
def test_stage_sequence_rejects_invalid_and_mixed_pipeline_orders(method, stages):
    with pytest.raises(ValueError):
        validate_stage_sequence(method, stages)


@pytest.mark.parametrize("stages", [["pretrain"], ["qa"]])
def test_single_stage_preset_uses_requested_stage(tmp_path, stages):
    raw = json.loads(Path("configs/v3/icae_single.json").read_text())
    raw["objective"]["stages"] = stages
    raw["training"]["stage_max_train_samples"] = {
        stage: raw["training"]["stage_max_train_samples"][stage] for stage in stages
    }
    preset = tmp_path / "preset.json"
    preset.write_text(json.dumps(raw))
    (config,) = load_preset(preset)
    assert config.objective.stage == stages[0]


@pytest.mark.parametrize("stages", [[], "pretrain", ["qa", "pretrain"]])
def test_preset_validates_sequence_at_input_boundary(tmp_path, stages):
    raw = json.loads(Path("configs/v3/icae_single.json").read_text())
    raw["objective"]["stages"] = stages
    preset = tmp_path / "preset.json"
    preset.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="stages"):
        load_preset(preset)


@pytest.mark.parametrize("include_stages", [False, True])
def test_preset_rejects_singular_stage_and_mixed_stage_formats(tmp_path, include_stages):
    raw = json.loads(Path("configs/v3/icae_single.json").read_text())
    raw["objective"]["stage"] = "pretrain"
    if not include_stages:
        del raw["objective"]["stages"]
    preset = tmp_path / "preset.json"
    preset.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="requires stages"):
        load_preset(preset)


def test_preset_validates_each_expanded_objective(tmp_path):
    raw = json.loads(Path("configs/v3/icae_single.json").read_text())
    raw["objective"]["bptt_steps"] = 2
    preset = tmp_path / "preset.json"
    preset.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="bptt_steps"):
        load_preset(preset)


def test_runtime_loader_rejects_preset_and_round_trips_saved_stage_configs(tmp_path):
    preset = Path("configs/v3/icae_single.json")
    with pytest.raises(ValueError, match="use load_preset"):
        load_experiment(preset)
    for config in load_preset(preset):
        saved = tmp_path / f"{config.objective.stage}.json"
        saved.write_text(json.dumps(config.to_dict()))
        assert load_experiment(saved) == config


@pytest.mark.parametrize("method", ["dynamic", "memory_change", "information_loss"])
def test_runtime_loader_preserves_shared_pretraining_method_in_saved_records(tmp_path, method):
    (config,) = load_preset("configs/v3/dynamic_pretrain.json")
    config = replace(config, objective=replace(config.objective, method=method))
    saved = tmp_path / "pretrain.json"
    saved.write_text(json.dumps(config.to_dict()))

    assert load_experiment(saved) == config
    assert load_experiment(saved).objective.method == method


def test_runtime_loader_requires_an_explicit_stage(tmp_path):
    raw = load_preset("configs/v3/icae_single.json")[0].to_dict()
    del raw["objective"]["stage"]
    saved = tmp_path / "missing-stage.json"
    saved.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="explicit stage"):
        load_experiment(saved)


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
