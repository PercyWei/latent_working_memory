import json

import pytest
import swanlab

from latent_working_memory.v1.publish_reports import main


def test_static_publication_records_checkpoint_and_blocks_partial_republish(tmp_path, monkeypatch):
    steps = []
    original_log = swanlab.Run.log

    def log(run, data, step=None):
        steps.append(step)
        return original_log(run, data, step=step)

    monkeypatch.setattr(swanlab.Run, "log", log)
    report = tmp_path / "test-step-020000.json"
    report.write_text(json.dumps({"step": 20000}))
    record = {
        "episode_id": "one",
        "document_id": "one",
        "task": "ae",
        "condition": "memory",
        "input_tokens": 64,
        "capacity": 32,
        "effective_ratio": 2,
        "target_tokens": 64,
        "nll_sum": 128,
        "eos_nll": 2,
    }
    report.with_suffix(".jsonl").write_text(json.dumps(record) + "\n")
    manifest = tmp_path / "reports.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "training_source": "semantic",
                    "evaluation_source": "semantic",
                    "report": str(report),
                }
            ]
        )
    )
    evaluation, comparison = (
        tmp_path / "pretrain-semantic-157k-eval-20260912",
        tmp_path / "pretrain-157k-compare-20260912",
    )
    args = [
        "--reports",
        str(manifest),
        "--output-dir",
        str(comparison),
        "--evaluation-output",
        "semantic",
        str(evaluation),
        "--swanlab-project",
        "static-report-test",
        "--swanlab-group",
        "static-report-test",
    ]
    main(args)
    assert json.loads((evaluation / "reports.json").read_text())[0]["checkpoint_step"] == 20000
    assert steps == [0, 0]
    fresh_comparison = tmp_path / "new-compare"
    args[args.index(str(comparison))] = str(fresh_comparison)
    with pytest.raises(ValueError, match="already published"):
        main(args)
    assert not fresh_comparison.exists()
    assert steps == [0, 0]


def test_append_uses_training_identity_config_and_checkpoint_step(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from latent_working_memory.v1.tracking import append_evaluation_reports

    training = tmp_path / "train"
    training.mkdir()
    identity = {"id": "train-id", "project": "project", "job_type": "train",
                "mode": "online", "url": "https://swanlab.cn/@owner/project/runs/train-id"}
    (training / "swanlab.json").write_text(json.dumps(identity))
    report = tmp_path / "test-step-020000.json"
    report.write_text(json.dumps({"split": "test", "step": 20000,
                                 "groups": {"all/ae/memory": {"nll": 2.0}},
                                 "comparisons": {}, "protocol": {"generation": "greedy"}}))
    report.with_suffix(".jsonl").write_text("")
    entries = [{"training_source": "ae-only", "evaluation_source": "semantic", "report": str(report)}]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(entries))
    remote = SimpleNamespace(name="cloud-training-name", state="FINISHED",
                             profile={"config": {"lr": {"value": 0.1, "sort": 0}}})
    lookups, calls, logs = [], [], []

    def lookup(path):
        lookups.append(path)
        return remote

    class Run:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def log(self, data, step):
            logs.append((data, step))

    def init(**kwargs):
        calls.append(kwargs)
        return Run()

    monkeypatch.setattr(swanlab, "Api", lambda: SimpleNamespace(run=lookup))
    monkeypatch.setattr(swanlab, "init", init)
    main(["--reports", str(manifest), "--training-run", "ae-only", str(training),
          "--swanlab-mode", "online", "--swanlab-project", "project", "--swanlab-group", "group"])
    assert lookups == ["owner/project/train-id"]
    assert calls[0]["id"] == "train-id" and calls[0]["resume"] == "must"
    assert calls[0]["config"] == {"lr": 0.1}
    assert calls[0]["name"] == "cloud-training-name"
    assert "job_type" not in calls[0] and "tags" not in calls[0] and "group" not in calls[0]
    values, step = logs[0]
    assert step == 20000
    assert values["evaluation/test/semantic/ae/memory/nll"] == 2.0
    assert all(key.startswith("evaluation/test/") for key in values)
    assert json.loads((training / "swanlab.json").read_text()) == identity
    assert (training / "evaluation-publications/test-step-020000.json").exists()
    with pytest.raises(ValueError, match="already appended"):
        append_evaluation_reports(training, entries)
    assert len(calls) == 1
    # A different checkpoint is allowed, but never while training is active.
    report.write_text(report.read_text().replace('20000', '21000'))
    remote.state = "RUNNING"
    with pytest.raises(ValueError, match="finished training"):
        append_evaluation_reports(training, entries)
    assert len(calls) == 1
    remote.state = "FINISHED"
    other = tmp_path / "other.json"
    other.write_text(report.read_text().replace('21000', '22000'))
    with pytest.raises(ValueError, match="one split/checkpoint"):
        append_evaluation_reports(training, entries + [{"evaluation_source": "random", "report": str(other)}])
    assert len(calls) == 1
