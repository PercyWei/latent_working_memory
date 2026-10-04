from dataclasses import asdict
import json
import math

import pytest

from latent_working_memory.v4.config import ModelConfig, TrainingConfig, load_experiment


def test_canonical_config_resolves_paths_and_rejects_unknown_fields(tmp_path):
    model = ModelConfig("local-model", pending_size=4, recent_size=2)
    training = TrainingConfig("train.jsonl", "dev.jsonl", "output", max_seq_length=16)
    value = {"model": asdict(model), "training": asdict(training)}
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(value))
    loaded_model, loaded_training = load_experiment(path)
    assert loaded_model == model
    assert loaded_training.train_file == str(tmp_path / "train.jsonl")
    assert loaded_training.output_dir == str(tmp_path / "output")
    value["model"]["capacity_growth"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(TypeError, match="capacity_growth"):
        load_experiment(path)


@pytest.mark.parametrize(
    "values",
    [
        {"pending_size": 0},
        {"inner_steps": True},
        {"inner_lr": float("nan")},
        {"inner_lr": -0.1},
        {"query_mode": "online"},
        {"query_normalization": "learned"},
        {"read_query_norm": 0},
        {"read_query_norm": True},
        {"compression_query_norm": -1},
        {"compression_query_norm": float("inf")},
        {"backbone_dtype": "float16"},
    ],
)
def test_model_config_rejects_invalid_contract(values):
    with pytest.raises(ValueError):
        ModelConfig("tiny", **values)


def test_config_requires_post_write_target(tmp_path):
    path = tmp_path / "experiment.json"
    path.write_text(
        json.dumps(
            {
                "model": {"model_name_or_path": "tiny", "pending_size": 4, "recent_size": 2},
                "training": {
                    "train_file": "train.jsonl",
                    "dev_file": "dev.jsonl",
                    "output_dir": "out",
                    "max_seq_length": 7,
                },
            }
        )
    )
    with pytest.raises(ValueError, match="next-token"):
        load_experiment(path)


def test_query_norm_defaults_resolve_and_roundtrip():
    config = ModelConfig("tiny", query_dim=7)
    assert config.query_normalization == "none"
    assert config.read_query_norm == config.compression_query_norm == math.sqrt(7)
    assert ModelConfig(**json.loads(json.dumps(asdict(config)))) == config
    custom = ModelConfig(
        "tiny", query_normalization="fixed_norm", read_query_norm=2.5, compression_query_norm=4.0
    )
    assert custom.read_query_norm == 2.5
    assert custom.compression_query_norm == 4.0
