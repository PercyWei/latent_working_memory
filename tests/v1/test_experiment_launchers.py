import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from latent_working_memory.v1 import experiment_execution
from latent_working_memory.v1.pretrain import experiment as pretrain
from latent_working_memory.v1.dynamic import experiment as dynamic


PRETRAIN = Path("configs/v1/pretrain")
DYNAMIC = Path("configs/v1/dynamic")


def test_pretrain_and_squad_experiments_have_independent_valid_configs():
    pretraining = sorted(PRETRAIN.glob("*/experiment.json"))
    dynamics = sorted(DYNAMIC.glob("*_squad/experiment.json"))
    assert len(pretraining) == 8 and len(dynamics) == 3
    for path, loader in [(p, pretrain.load_pretrain_experiment) for p in pretraining] + [
        (p, dynamic.load_dynamic_experiment) for p in dynamics
    ]:
        spec = loader(path)
        assert "group" not in spec and "runs" not in spec
        assert Path(spec["model"]).parent == path.parent.resolve()
        assert Path(spec["selection"]).parent == path.parent.resolve()
        expected_gpus = [6, 7] if path.parent.name.endswith("_epoch64k") else [4, 5]
        assert spec["gpus"] == expected_gpus
    qwen = json.loads((PRETRAIN / "qwen2.5-3b-instruct_mixed-2048/selection.json").read_text())
    assert qwen["evaluation"]["samples_per_source"] == {"dev": None, "test": None}
    assert qwen["training"]["source_schedule"][0]["weights"] == {"semantic": 0.5, "random": 0.5}


def test_pretrain_runs_warmup_first_and_appends_test_to_each_training_run(tmp_path, monkeypatch):
    commands = []
    monkeypatch.delenv("LWM_ALLOWED_PHYSICAL_GPUS", raising=False)

    def execute(argv, **kwargs):
        commands.append((argv, kwargs["env"]))
        if "latent_working_memory.v1.pretrain.train" in argv:
            output = Path(argv[argv.index("--output-dir") + 1])
            output.mkdir(parents=True)
            (output / "training-result.json").write_text(
                json.dumps(
                    {
                        "completed_steps": 120,
                        "complete": True,
                        "final_checkpoint": str(output / "checkpoints/pretrain-step-000120.pt"),
                    }
                )
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        experiment_execution.subprocess, "check_output", lambda *a, **k: "test-commit"
    )
    monkeypatch.setattr(experiment_execution.subprocess, "run", execute)
    paths = [
        PRETRAIN / f"llama-{name}_mixed-128/experiment.json"
        for name in ("ae-warmup", "ae-only", "joint")
    ]
    output = tmp_path / "series"
    pretrain.main(
        [
            "--experiments",
            *map(str, paths),
            "--output-dir",
            str(output),
            "--swanlab-mode",
            "online",
            "--swanlab-group",
            "objectives",
            "--swanlab-tag",
            "study:pretrain-objective-comparison",
        ]
    )
    training = [c for c, _ in commands if "latent_working_memory.v1.pretrain.train" in c]
    evaluations = [c for c, _ in commands if "latent_working_memory.v1.pretrain.evaluate" in c]
    assert len(training) == len(evaluations) == 3
    assert "ae-warmup" in training[0][training[0].index("--config") + 1]
    assert all(
        "--resume" not in c and "--fork-from" not in c and "--data-run" not in c for c in training
    )
    for train, evaluation in zip(training, evaluations, strict=True):
        directory = train[train.index("--output-dir") + 1]
        assert evaluation[evaluation.index("--training-run") + 1] == directory
        assert (
            evaluation[evaluation.index("--training-result") + 1]
            == directory + "/training-result.json"
        )
        assert train[train.index("--swanlab-group") + 1] == "objectives"
        assert Path(train[train.index("--config") + 1]).is_file()
    assert all("LWM_ALLOWED_PHYSICAL_GPUS" not in env for _, env in commands)
    assert {env["CUDA_VISIBLE_DEVICES"] for _, env in commands} == {"4", "4,5"}
    assert json.loads((output / "plan/status.json").read_text())["status"] == "complete"


def test_dynamic_plan_has_explicit_pretrain_dependency_and_separate_evaluation(
    tmp_path, monkeypatch
):
    def no_execution(*a, **k):
        raise AssertionError("plan-only must not execute commands")

    monkeypatch.setattr(
        experiment_execution.subprocess, "check_output", lambda *a, **k: "test-commit"
    )
    monkeypatch.setattr(experiment_execution.subprocess, "run", no_execution)
    output = tmp_path / "plan"
    paths = sorted(DYNAMIC.glob("*_squad/experiment.json"))
    dynamic.main(
        [
            "--experiments",
            *map(str, paths),
            "--output-dir",
            str(output),
            "--swanlab-group",
            "bptt",
            "--plan-only",
        ]
    )
    commands = json.loads((output / "plan/commands.json").read_text())
    assert len(commands) == 9
    for i in range(0, 9, 3):
        prepare, train, test = [c["argv"] for c in commands[i : i + 3]]
        assert "latent_working_memory.v1.dynamic.prepare" in prepare
        assert train[train.index("--checkpoint") + 1].endswith("pretrain-step-020000.pt")
        assert test[test.index("--checkpoint") + 1].endswith("dynamic-step-000750.pt")
        assert (
            prepare[prepare.index("--output-dir") + 1] + "/evaluation-sets.json"
            == train[train.index("--evaluation-sets") + 1]
        )
    assert json.loads((output / "plan/status.json").read_text())["status"] == "planned"


def test_experiment_rejects_invalid_gpu_or_duplicate_names_before_writes(tmp_path, monkeypatch):
    path = PRETRAIN / "llama-mixed-2048/experiment.json"
    monkeypatch.setenv("LWM_ALLOWED_PHYSICAL_GPUS", "6,7")
    with pytest.raises(ValueError, match="allowed physical"):
        pretrain.load_pretrain_experiment(path)
    monkeypatch.delenv("LWM_ALLOWED_PHYSICAL_GPUS")
    with pytest.raises(ValueError, match="distinct names"):
        pretrain.main(
            ["--experiments", str(path), str(path), "--output-dir", str(tmp_path / "out")]
        )
    assert not (tmp_path / "out").exists()


def test_execution_rejects_gpu_overlap(tmp_path):
    execution = experiment_execution.ExperimentExecution(tmp_path, [4, 5])
    with pytest.raises(ValueError, match="disjoint"):
        execution.stage([("a", ["unused"], [4, 5]), ("b", ["unused"], [4, 5])])


@pytest.mark.parametrize(
    "name,limit,gpus",
    [
        ("qwen2.5-3b-instruct_mixed-2048", 32000, [4, 5]),
        ("qwen2.5-3b-instruct_mixed-2048_epoch64k", 64000, [6, 7]),
    ],
)
def test_qwen_plan_passes_epoch_sample_cap(tmp_path, name, limit, gpus):
    output = tmp_path / "qwen"
    pretrain.main(
        [
            "--experiments",
            str(PRETRAIN / name / "experiment.json"),
            "--output-dir",
            str(output),
            "--plan-only",
        ]
    )
    commands = json.loads((output / "plan/commands.json").read_text())
    training = commands[0]["argv"]
    assert training[training.index("--max-samples-per-epoch") + 1] == str(limit)
    assert commands[0]["CUDA_VISIBLE_DEVICES"] == gpus
    assert commands[1]["CUDA_VISIBLE_DEVICES"] == gpus[:1]
    assert training[training.index("--tokenizer-workers") + 1] == "4"
    assert training[training.index("--tokenization-batch-size") + 1] == "256"
    assert training[training.index("--prefetch-batches") + 1] == "2"
    evaluation = commands[1]["argv"]
    assert evaluation[evaluation.index("--tokenizer-workers") + 1] == "4"
