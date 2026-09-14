import json
from types import SimpleNamespace

from latent_working_memory.v1.pretrain import objective_comparison as objective
import pytest

from latent_working_memory.v1 import experiment_execution
from latent_working_memory.v1.pretrain.data_comparison import run_series


def test_data_comparison_commands_and_plan_files(tmp_path, monkeypatch):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"sources": {"semantic": "unused", "random": "unused"}}))
    spec = json.loads(open("configs/experiments/pretrain-data-comparison-2048.json").read())
    spec["data_selection"] = str(selection)
    commands = []

    def execute(argv, **kwargs):
        commands.append((argv, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(experiment_execution.subprocess, "run", execute)
    monkeypatch.setattr(experiment_execution.subprocess, "check_output", lambda *a, **k: "test")
    output = tmp_path / "experiment"
    run_series(spec, output)
    training = [c for c, _ in commands if "latent_working_memory.v1.pretrain.train" in c]
    assert [c[c.index("--data-run") + 1] for c in training] == ["semantic", "random", "mixed"]
    evaluations = [(c, gpu) for c, gpu in commands if "latent_working_memory.v1.pretrain.evaluate" in c]
    assert len(evaluations) == 6
    assert {gpu for _, gpu in evaluations} == {"4", "5"}
    assert all(c[c.index("--swanlab-mode") + 1] == "disabled" for c, _ in evaluations)
    publication = commands[-1][0]
    assert publication.count("--training-run") == 3
    assert "--evaluation-output" not in publication
    assert json.loads((output / "plan/status.json").read_text())["status"] == "complete"
    expected = {"series.json", "data-selection.json", "status.json", "commands.json", "reports.json"}
    expected |= {f"{r['data_run']}-train.log" for r in spec["runs"]}
    expected |= {f"{r['data_run']}-test-{s}.log" for r in spec["runs"] for s in ("semantic", "random")}
    expected.add("publish-comparison.log")
    assert {p.name for p in (output / "plan").iterdir()} == expected


def test_execution_rejects_gpu_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_execution.subprocess, "check_output", lambda *a, **k: "test")
    execution = experiment_execution.ExperimentExecution(tmp_path, [4, 5])
    with pytest.raises(ValueError, match="disjoint"):
        execution.stage([("a", ["unused"], [4, 5]), ("b", ["unused"], [4, 5])])


def test_objective_launcher_can_start_with_independent_warmup(tmp_path, monkeypatch):
    spec = json.loads(open("configs/experiments/pretrain-objective-comparison-128.json").read())
    spec["runs"] = [spec["runs"][2], *spec["runs"][:2]]
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"sources": {"semantic": "unused", "random": "unused"}}))
    note = tmp_path / "note.md"
    note.write_text("实验记录\n")
    spec.update(data_selection=str(selection), note=str(note))
    commands = []

    def execute(argv, **kwargs):
        commands.append(argv)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(experiment_execution.subprocess, "run", execute)
    monkeypatch.setattr(experiment_execution.subprocess, "check_output", lambda *a, **k: "test")
    monkeypatch.setattr(objective, "summarize", lambda *a: "结果\n")
    output = tmp_path / "experiment"
    objective.run_series(spec, output)
    training = [c for c in commands if "latent_working_memory.v1.pretrain.train" in c]
    assert len(training) == 3
    assert all(c[c.index("--data-run") + 1] == "mixed" for c in training)
    assert training[0][training[0].index("--config") + 1] == spec["runs"][0]["config"]
    assert all("--fork-from" not in command and "--resume" not in command for command in training)
    evaluations = [c for c in commands if "latent_working_memory.v1.pretrain.evaluate" in c]
    assert len(evaluations) == 6
    assert all(c[c.index("--prefix-tokens") + 1:c.index("--prefix-tokens") + 4] == ["1", "8", "32"] for c in evaluations)
    assert commands[-1].count("--training-run") == 3
    expected = {"series.json", "data-selection.json", "status.json", "commands.json", "reports.json", "results.md", "publish-comparison.log"}
    expected |= {f"{r['label']}-train.log" for r in spec["runs"]}
    expected |= {f"{r['label']}-test-{s}.log" for r in spec["runs"] for s in ("semantic", "random")}
    assert {p.name for p in (output / "plan").iterdir()} == expected
