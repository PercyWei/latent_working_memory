"""SwanLab 日志与原生 dev 面板的分组、统计口径、运行关联。"""

import json
import socket

from latent_working_memory.v2.pretrain.tracking import (
    development_panels,
    development_scalars,
    panel_style,
    reconstruction_run,
    log_training,
    log_development,
    log_final_evaluation,
)


def metrics():
    values = {
        "trajectories": 2,
        "all/ae_tokens": 64,
        "all/lm_tokens": 32,
        "generation/final_round_exact_match": 0.5,
        "generation/samples": 2,
    }
    for task in ("ae", "lm"):
        for step in (1, 2, 3):
            values[f"round/{step}/{task}_nll"] = float(step)
        values[f"trajectory_{task}"] = 2.0
        values[f"one_shot_{task}"] = 2.5
        values[f"final_minus_one_shot_{task}"] = 0.5
    return values


def test_native_panels_register_only_intended_metrics():
    panels = development_panels()
    keys = {axis["key"] for panel in panels.values() for axis in panel["config"]["yAxis"]}
    scalars = development_scalars(metrics())
    assert set(scalars).issubset(keys)
    assert "dev/ae/nll/compression-4" not in scalars
    assert not any("tokens" in key or "ppl" in key for key in scalars)
    assert len(panels) == 3
    for panel in panels.values():
        assert panel["config"]["xAxis"]["key"] == "step"
        assert len(panel["config"]["yAxis"]) <= 8
        assert len(panel_style(panel, "test", "run", "#2459A6")) == len(panel["config"]["yAxis"])


def test_offline_run_records_namespaces_resources_tables_and_identity(tmp_path, monkeypatch):
    def reject_network(*args, **kwargs):
        raise AssertionError("offline logging must not access the network")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    config = {
        "model": {"compression": "mean"},
        "training": {
            "learning_rate": 1e-4,
            "objective": "ae",
            "warmup_epochs": 1,
            "multiround_epochs": 2,
        },
    }
    output = tmp_path / "qwen3-4b_pooling_ae-warmup_20260916"
    output.mkdir()
    captured = []
    with reconstruction_run(
        output,
        config,
        "offline",
        "latent-working-memory-v2",
        "qwen3-4b_pooling_reconstruction_20260916",
        ("study:reconstruction",),
    ) as run:
        original = run.log

        def capture(values, step):
            captured.append((values, step))
            return original(values, step=step)

        monkeypatch.setattr(run, "log", capture)
        record = {
            "step": 2,
            "loss": 1.0,
            "ae": 1.0,
            "lm": None,
            "grad_norm": 0.2,
            "global_epoch": 1,
            "seconds": 2.0,
            "source_tokens": 1000,
            "peak_memory_bytes": 2 * 1024**3,
        }
        log_training(
            run,
            record,
            {"source_tokens": 2000, "target_tokens": 6000, "sample_visits": 16},
            1e-4,
            0.25,
        )
        values, step = captured[-1]
        assert step == 2 and values["resources/source_tokens_per_second"] == 500
        assert values["resources/peak_memory_gib"] == 2
        assert values["progress/sample_visits"] == 16
        assert values["train/learning_rate"] == 1e-4
        assert "train/lm/nll" not in values
        log_development(run, metrics(), 2)
        values, step = captured[-1]
        assert step == 2 and "dev/details" in values
        assert "dev/ae/nll/compression-1" in values
        assert "dev/ae/nll/compression-4" not in values
        records = [
            {
                "document_id": "x",
                "depth": 3,
                "generation": {
                    "prediction": "red",
                    "reference": "red",
                    "final_round_exact_match": True,
                },
            }
        ]
        log_final_evaluation(run, metrics(), records, 2)
        values, step = captured[-1]
        assert step == 2
        assert set(values) == {
            "evaluation/ae/nll",
            "evaluation/lm/nll",
            "evaluation/ae/final_compression_em",
            "evaluation/details",
            "evaluation/examples/page-1",
        }
    identity = json.loads((output / "swanlab.json").read_text())
    assert identity["project"] == "latent-working-memory-v2"
    assert identity["group"] == "qwen3-4b_pooling_reconstruction_20260916"
    assert identity["job_type"] == "train"
    assert identity["tags"] == [
        "data:fineweb",
        "method:v2-pooling",
        "scope:main",
        "study:reconstruction",
    ]
    assert not run.alive
