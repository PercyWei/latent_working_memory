from __future__ import annotations

import json
import socket

from latent_working_memory.v1.tracking import log_evaluation, log_training, swanlab_run


def test_offline_metrics_match_local_reports_and_close_run(tmp_path, monkeypatch):
    def reject_network(*args, **kwargs):
        raise AssertionError("offline tracking must not access the network")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    calls = []
    with swanlab_run(
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
        path = tmp_path / "evaluation.jsonl"
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
        log_evaluation(
            run,
            {
                "groups": groups,
                "comparisons": comparisons,
                "training_input_tokens": 36,
            },
            path,
            3,
            "test",
        )
        logged, step = calls[-1]
        assert step == 3
        assert logged["test/ae/gain_vs_no_memory"] == 2.0
        assert logged["test/continuation/gain_vs_wrong_memory"] == 1.0
        assert logged["test/continuation/nll_gap_to_full_context"] == 0.5
        assert logged["test_by_length_ratio/32/2/ae/memory/bleu_4"] == 90.0
        assert logged["test_by_length_ratio/32/2/ae/memory/correct_prefix_ratio"] == 0.8
        assert logged["test_by_length_ratio/32/2/continuation/gain_vs_wrong_memory"] == 0.4
        assert logged["progress/input_tokens"] == 36
        assert len(logged["test/reconstruction"]) == 100
        assert len(logged["test/reconstruction/page_2"]) == 10
        assert "dev/ae/memory/nll" not in logged
    assert not run.alive
    assert json.loads((tmp_path / "swanlab.json").read_text())["mode"] == "offline"
    identity = json.loads((tmp_path / "swanlab.json").read_text())
    assert identity["group"] == "lwm-pretrain-test"
    assert identity["job_type"] == "train"
    assert identity["tags"] == [
        "data:fineweb",
        "method:latent-working-memory",
        "scope:main",
        "study:pretraining",
    ]
