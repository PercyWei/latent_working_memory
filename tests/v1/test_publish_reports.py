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
        "episode_id": "one", "document_id": "one", "task": "ae",
        "condition": "memory", "input_tokens": 64, "capacity": 32,
        "effective_ratio": 2, "target_tokens": 64, "nll_sum": 128, "eos_nll": 2,
    }
    report.with_suffix(".jsonl").write_text(json.dumps(record) + "\n")
    manifest = tmp_path / "reports.json"
    manifest.write_text(json.dumps([
        {"training_source": "semantic", "evaluation_source": "semantic", "report": str(report)}
    ]))
    evaluation, comparison = tmp_path / "evaluate", tmp_path / "compare"
    args = [
        "--reports", str(manifest), "--output-dir", str(comparison),
        "--evaluation-output", "semantic", str(evaluation),
        "--swanlab-project", "static-report-test", "--swanlab-group", "static-report-test",
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
