from dataclasses import replace
from pathlib import Path

import pytest

from latent_working_memory.v1.config import ExperimentConfig, load_config, write_resolved_config


def test_pilot_round_trip(tmp_path):
    config = load_config(Path(__file__).resolve().parents[2] / "configs/v1/pilot.json")
    assert config.pretrain_compression_ratios == (2, 4, 8)
    output = tmp_path / "resolved.json"
    write_resolved_config(config, output)
    assert load_config(output) == config


def test_invalid_config_fails_at_boundary():
    with pytest.raises(ValueError, match="Unknown"):
        ExperimentConfig.from_mapping({"cell_tokens": 64})
    for values in [
        dict(d_mem=10),
        dict(pretrain_k_min=8192),
        dict(max_input_tokens=4096),
        dict(ratio_weights_start=(0.1, 0.2, 0.3)),
        dict(batch_size=True),
        dict(growth_actions=(0, 4)),
        dict(ae_weight=0, lm_weight=0),
    ]:
        with pytest.raises(ValueError):
            replace(ExperimentConfig(), **values)
