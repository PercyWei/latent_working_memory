from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from latent_working_memory.v1.config import (
    ExperimentConfig,
    load_config,
    write_resolved_config,
)


ROOT = Path(__file__).resolve().parents[2]
PILOT_CONFIG = ROOT / "configs" / "v1" / "pilot.json"


def test_pilot_config_is_complete_and_round_trips(tmp_path: Path) -> None:
    config = load_config(PILOT_CONFIG)
    assert config.framework_version == "v1"
    assert config.growth_actions == (0, 8, 16)
    assert config.update_chunk_cells == (1, 2, 4)

    output = tmp_path / "resolved.json"
    write_resolved_config(config, output)
    assert load_config(output) == config

    raw = json.loads(PILOT_CONFIG.read_text(encoding="utf-8"))
    assert set(raw) == set(config.to_dict())


def test_config_rejects_unknown_fields_and_other_versions() -> None:
    with pytest.raises(ValueError, match="Unknown configuration fields"):
        ExperimentConfig.from_mapping({"extra": True})
    with pytest.raises(ValueError, match="framework_version"):
        replace(ExperimentConfig(), framework_version="v2")
    with pytest.raises(ValueError, match="growth_actions"):
        replace(ExperimentConfig(), growth_actions=(0, 4, 8))


def test_config_rejects_inconsistent_dimensions_and_probabilities() -> None:
    with pytest.raises(ValueError, match="divisible"):
        replace(ExperimentConfig(), d_mem=10, num_heads=4)
    with pytest.raises(ValueError, match="sum to 1"):
        replace(ExperimentConfig(), exploration_probs=(0.5, 0.25, 0.05))
    with pytest.raises(ValueError, match="same base model"):
        replace(ExperimentConfig(), teacher_model_name_or_path="different-model")
    with pytest.raises(ValueError, match="same model revision"):
        replace(ExperimentConfig(), teacher_model_revision="different-revision")
    with pytest.raises(ValueError, match="k_init"):
        replace(ExperimentConfig(), k_init=24)
    with pytest.raises(ValueError, match="divisible by 8"):
        replace(ExperimentConfig(), k_limit=510)
