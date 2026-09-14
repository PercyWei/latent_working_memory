from __future__ import annotations

import json
import socket
from contextlib import contextmanager

from latent_working_memory.v1 import tracking as common_tracking
from latent_working_memory.v1.pretrain import tracking as pretrain_tracking

from latent_working_memory.v1.pretrain.tracking import log_evaluation, log_training
from latent_working_memory.v1.pretrain.tracking import pretraining_run


def test_offline_metrics_match_local_reports_and_close_run(tmp_path, monkeypatch):
    def reject_network(*args, **kwargs):
        raise AssertionError("offline tracking must not access the network")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    calls = []
    with pretraining_run(
        tmp_path, {}, mode="offline", group="lwm-pretrain-test", tags=("study:pretraining",)
    ) as run:
        original_log = run.log

        def capture(data, step):
            calls.append((data, step))
            original_log(data, step=step)

        monkeypatch.setattr(run, "log", capture)
        record = {
            "step": 3,
            "loss": 5.0,
            "gradient_norm": 1.0,
            "seconds": 2.0,
            "input_tokens_per_second": 6.0,
            "peak_memory_bytes": 2 * 1024**3,
            "distinct_documents": 2,
            "document_visits": 6,
            "input_length_bounds": [32, 128, 512, 1024],
            "samples": [
                {
                    "ae_nll": 1.0,
                    "lm_nll": 2.0,
                    "input_tokens": 4,
                    "continuation_tokens": 3,
                    "length_bucket": 32,
                    "capacity": 2,
                    "effective_ratio": 2.0,
                },
                {
                    "ae_nll": 3.0,
                    "lm_nll": 4.0,
                    "input_tokens": 8,
                    "continuation_tokens": 4,
                    "length_bucket": 32,
                    "capacity": 2,
                    "effective_ratio": 4.0,
                },
            ],
        }
        log_training(run, record, 36, 48)
        logged, step = calls[-1]
        assert step == 3
        assert logged["train/ae_nll"] == 2.0 and logged["train/lm_nll"] == 3.0
        assert logged["batch/capacity_mean"] == 2.0
        assert logged["resources/peak_memory_gib"] == 2.0
        assert logged["progress/input_tokens"] == 36
        groups = {
            f"all/{task}/{condition}": {"nll": nll, "ppl": 10.0, "token_accuracy": 0.5}
            for task in ("ae", "continuation")
            for condition, nll in (("memory", 1.0), ("no_memory", 3.0), ("wrong_memory", 2.0))
        }
        groups["all/continuation/full_context"] = {"nll": 0.5, "ppl": 1.65}
        groups["length_ratio/32/2/ae/memory"] = {
            "nll": 1.0,
            "bleu_4": 90.0,
            "correct_prefix_ratio": 0.8,
            "generated_reads": 110,
        }
        comparisons = {
            f"all/{task}": {"gain_vs_no_memory": 2.0, "gain_vs_wrong_memory": 1.0}
            for task in ("ae", "continuation")
        }
        comparisons["all/continuation"]["nll_gap_to_full_context"] = 0.5
        comparisons["length_ratio/32/2/continuation"] = {"gain_vs_wrong_memory": 0.4}
        path = tmp_path / "dev-step-000003.jsonl"
        path.write_text(
            (
                json.dumps(
                    {
                        "condition": "memory",
                        "reference": "A.",
                        "prediction": "A.",
                        "input_tokens": 2,
                        "capacity": 1,
                        "sequence_match": True,
                        "correct_prefix_ratio": 1.0,
                    }
                )
                + "\n"
            )
            * 110
        )
        rows = [json.loads(line) | {"episode_id": str(i)}
                for i, line in enumerate(path.read_text().splitlines())]
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        metrics = {"groups": groups, "comparisons": comparisons, "training_input_tokens": 36}
        path.with_suffix(".json").write_text(json.dumps(metrics))
        log_evaluation(run, {"semantic": metrics}, {"semantic": path}, 3)
        logged, step = calls[-1]
        assert step == 3 and logged["progress/input_tokens"] == 36
        assert "dev/overview/ae/nll/semantic/memory" in logged
        assert "dev/overview/ae/nll" not in logged
        assert "dev/overview/summary" in logged and "dev/overview/details" in logged
        assert len(logged["dev/overview/examples"]) == 100
        assert len(logged["dev/overview/examples/page_2"]) == 10
        assert all(key.startswith("dev/overview/") or key == "progress/input_tokens" for key in logged)
    assert not run.alive
    assert json.loads((tmp_path / "swanlab.json").read_text())["mode"] == "offline"
    identity = json.loads((tmp_path / "swanlab.json").read_text())
    assert identity["project"] == "latent-working-memory-v1"
    assert identity["group"] == "lwm-pretrain-test"
    assert identity["job_type"] == "train"
    assert identity["tags"] == [
        "data:fineweb",
        "method:latent-working-memory",
        "scope:main",
        "study:pretraining",
    ]


def test_common_run_does_not_interpret_stage_configuration(tmp_path, monkeypatch):
    calls = []

    class Session:
        id = "run-id"
        url = "https://swanlab.example/@workspace/project/runs/run-id"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def initialize(**kwargs):
        calls.append(kwargs)
        return Session()

    monkeypatch.setattr(common_tracking.swanlab, "init", initialize)
    # These field names used to implicitly activate pretraining tags and panels.
    config = {"data_preparation": {}, "evaluation_preparations": {"custom": {}}}
    with common_tracking.swanlab_run(
        tmp_path, config, mode="online", group="common-test", tags=("data:custom",)
    ) as run:
        assert run.id == "run-id"
    assert calls[0]["config"] == config
    assert calls[0]["project"] == "latent-working-memory-v1"
    assert calls[0]["tags"] == ["data:custom", "method:latent-working-memory", "scope:main"]


def test_pretraining_session_owns_source_tags_and_training_panels(tmp_path, monkeypatch):
    sessions, panels = [], []
    active_run = object()

    @contextmanager
    def session(output_dir, config, mode, project, **kwargs):
        sessions.append(kwargs)
        yield active_run if mode != "disabled" else None

    monkeypatch.setattr(pretrain_tracking, "swanlab_run", session)
    monkeypatch.setattr(
        pretrain_tracking,
        "configure_development_panels",
        lambda run, sources, mode: panels.append((run, sources, mode)),
    )
    config = {
        "data_preparation": {"source_weights": {"semantic": 0.5, "random": 0.5}},
        "evaluation_preparations": {"semantic": "first", "random": "second"},
    }
    for job_type in ("train", "evaluate"):
        with pretrain_tracking.pretraining_run(
            tmp_path, config, mode="online", group="pretrain-test", job_type=job_type
        ) as run:
            assert run is active_run
        assert set(sessions[-1]["fixed_tags"]) == {
            "scope:main", "method:latent-working-memory", "data:fineweb",
            "data:semantic", "data:random",
        }
    assert panels == [(active_run, ["semantic", "random"], "online")]
    with pretrain_tracking.pretraining_run(tmp_path, {"data_preparation": {}}) as run:
        assert run is None
    assert len(panels) == 1
