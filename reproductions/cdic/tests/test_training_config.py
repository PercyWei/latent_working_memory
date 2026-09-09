from __future__ import annotations

import json
from pathlib import Path

from cdic_repro.config import RetrievedStateOrder
from cdic_repro.experiments.msc.train import load_training_config


def test_training_config_loads_paper_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "model": {
                    "model_path": "/models/llama",
                    "checkpoint_path": "/checkpoints/icae.pt",
                    "device": "cuda:0",
                    "devices": ["cuda:0", "cuda:1"],
                },
                "data": {"root": "/datasets/msc"},
                "retrieval": {},
                "training": {"output_dir": "/outputs/cdic"},
            }
        ),
        encoding="utf-8",
    )

    config = load_training_config(path)

    assert config.model.memory_size == 128
    assert config.model.devices == ("cuda:0", "cuda:1")
    assert config.model.gradient_checkpointing is True
    assert config.model.gradient_window_size == 1
    assert config.data.session_id == 4
    assert config.data.strict_pairs is False
    assert config.retrieval.threshold == 0.8
    assert config.retrieval.retrieved_state_order is RetrievedStateOrder.SCORE_DESC
    assert config.training.epochs == 2
    assert config.training.learning_rate == 2e-4
    assert len(config.fingerprint()) == 64
