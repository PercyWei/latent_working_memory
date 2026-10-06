"""验证任务编排和 shell 契约；子进程替身不运行 GPU、模型或云端请求。"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from latent_working_memory.v3 import gpu_job
from latent_working_memory.v3.config import load_experiment


def arguments(tmp_path, *options):
    return gpu_job.parse_args(
        [
            "--run-id",
            "unit-job",
            "--output-root",
            str(tmp_path / "outputs"),
            "--pretrain-data",
            str(tmp_path / "pretraining data"),
            "--long-pretrain-data",
            str(tmp_path / "long pretraining data"),
            "--qa-data",
            str(tmp_path / "qa data"),
            *options,
        ]
    )


def prepare_dataset_entries(args):
    # 这里只测试编排入口的文件要求；内容解析由真正的训练/评估测试覆盖。
    for directory in (args.pretrain_data, args.long_pretrain_data, args.qa_data):
        directory.mkdir()
        for split in ("train", "dev", "test"):
            (directory / f"{split}.jsonl").write_text("")
    (args.qa_data / "preparation.json").write_text("{}")


def value(command, flag):
    return command[command.index(flag) + 1]


class Commands:
    def __init__(self, fail_at=None, omit_checkpoint=False):
        self.calls = []
        self.fail_at = fail_at
        self.omit_checkpoint = omit_checkpoint

    def __call__(self, command, environment, log_path):
        call = {"command": command, "environment": dict(environment), "log_path": log_path}
        self.calls.append(call)
        if self.fail_at == len(self.calls):
            raise subprocess.CalledProcessError(7, command)
        if "latent_working_memory.v3.train" in command:
            config = load_experiment(value(command, "--config"))
            call["config"] = config
            output = Path(config.training.output_dir)
            (output / "checkpoints").mkdir(parents=True)
            steps = (
                int(value(command, "--stop-after-steps")) if "--stop-after-steps" in command else 80
            )
            checkpoint = output / "checkpoints" / f"step-{steps:06d}.pt"
            if not self.omit_checkpoint:
                # 编排器只传递此文件，不读取权重；这不是可加载的模型 checkpoint。
                checkpoint.write_bytes(b"opaque orchestration checkpoint fixture")
            (output / "training-result.json").write_text(
                json.dumps(
                    {
                        "complete": "--stop-after-steps" not in command,
                        "completed_steps": steps,
                        "total_steps": 80,
                        "completed_epochs": 1 if steps == 80 else 0,
                        "sample_visits": steps * config.training.global_batch_size,
                        "stop_after_steps": steps if steps != 80 else None,
                        "checkpoint": str(checkpoint),
                    }
                )
            )
            call["checkpoint"] = checkpoint
        elif "latent_working_memory.v3.evaluate" in command:
            output = Path(value(command, "--output-dir"))
            output.mkdir(parents=True)
            (output / "summary.json").write_text(
                json.dumps({"split": value(command, "--split"), "trajectories": 2})
            )
        else:
            assert "latent_working_memory.v3.compare" in command


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("smoke", (16, 4, 2, 2, "dev")),
        ("pilot", (256, 32, 20, 16, "dev")),
        ("full", (None, None, None, None, "test")),
    ],
)
def test_modes_apply_bounded_or_complete_training_and_evaluation(tmp_path, mode, expected):
    args = arguments(tmp_path, "--mode", mode)
    directory, level, jobs = gpu_job.build_jobs(args)
    assert (
        tuple(
            level[name]
            for name in (
                "train_samples",
                "dev_samples",
                "max_steps",
                "eval_trajectories",
                "eval_split",
            )
        )
        == expected
    )
    assert directory.name == f"capacity-{mode}_unit-job"
    for job in jobs:
        assert job.config.training.max_train_samples == expected[0]
        assert job.config.training.max_dev_samples == expected[1]
        assert job.config.training.init_checkpoint is None
        assert job.config.training.swanlab_project == "latent-working-memory-v3"
        assert job.config.training.tags == (f"study:{'main' if mode == 'full' else mode}",)
        command = gpu_job.training_command(job, 2, level["max_steps"])
        assert "--nproc_per_node=2" in command
        assert ("--stop-after-steps" in command) == (mode != "full")
    evaluation = gpu_job.evaluation_command(args, level, Path("final.pt"), Path("evaluation"))
    assert ("--max-trajectories" in evaluation) == (mode != "full")
    assert value(evaluation, "--split") == expected[-1]
    assert "--log-to-swanlab" in evaluation


def test_dynamic_methods_share_only_pretraining_and_have_independent_warmup_policy(tmp_path):
    _, _, jobs = gpu_job.build_jobs(arguments(tmp_path))
    assert [job.key for job in jobs] == [
        "dynamic-pretrain",
        "memory-change-warmup",
        "memory-change-policy",
        "information-loss-warmup",
        "information-loss-policy",
    ]
    assert [job.initialize_from for job in jobs] == [
        None,
        "dynamic-pretrain",
        "memory-change-warmup",
        "dynamic-pretrain",
        "information-loss-warmup",
    ]
    assert [job.evaluate for job in jobs] == [False, False, True, False, True]
    assert len({job.config.training.output_dir for job in jobs}) == 5
    assert len({job.config.training.seed for job in jobs}) == 1
    assert len({job.config.training.group for job in jobs}) == 1


def test_all_methods_build_a_complete_topologically_ordered_stage_graph(tmp_path):
    args = arguments(tmp_path, "--method", "all", "--mode", "full")
    _, _, jobs = gpu_job.build_jobs(args)
    assert len(jobs) == 10
    assert {
        (job.config.objective.method, job.config.objective.stage) for job in jobs if job.evaluate
    } == {
        ("icae_single", "qa"),
        ("icae_multi", "qa"),
        ("autocompressors", "lm"),
        ("memory_change", "policy"),
        ("information_loss", "policy"),
    }
    seen = set()
    for job in jobs:
        assert job.initialize_from is None or job.initialize_from in seen
        seen.add(job.key)
        if job.config.objective.stage in {"qa", "warmup", "policy"}:
            assert Path(job.config.training.dataset_dir) == args.qa_data
        elif job.config.objective.method in {"memory_change", "information_loss"}:
            assert Path(job.config.training.dataset_dir) == args.pretrain_data
        else:
            assert Path(job.config.training.dataset_dir) == args.long_pretrain_data


def test_explicit_overrides_and_zero_remove_profile_limits(tmp_path):
    args = arguments(
        tmp_path,
        "--method",
        "memory_change",
        "--stage",
        "policy",
        "--init-checkpoint",
        "parent checkpoint.pt",
        "--train-samples",
        "0",
        "--dev-samples",
        "6",
        "--max-steps",
        "0",
        "--eval-trajectories",
        "0",
        "--eval-split",
        "test",
        "--epochs",
        "3",
        "--global-batch-size",
        "4",
        "--qa-batch-size",
        "2",
        "--threshold-i",
        "0.25",
        "--eval-every",
        "2",
        "--save-every",
        "3",
        "--tracking",
        "disabled",
        "--group",
        "chosen-group",
        "--model-path",
        "models/local Qwen3",
    )
    _, level, jobs = gpu_job.build_jobs(args)
    assert len(jobs) == 1
    job = jobs[0]
    assert job.initialize_from is None and job.evaluate
    assert job.config.training.init_checkpoint == str(Path("parent checkpoint.pt").resolve())
    assert job.config.training.max_train_samples is None
    assert job.config.training.max_dev_samples == 6
    assert job.config.training.epochs == 3
    assert job.config.training.global_batch_size == 4
    assert job.config.training.group == "chosen-group"
    assert job.config.training.swanlab_project is None
    assert job.config.objective.qa_batch_size == 2
    assert job.config.objective.threshold_i == 0.25
    assert job.config.model.model_name_or_path == "models/local Qwen3"
    assert level["max_steps"] is None and level["eval_trajectories"] is None
    assert level["eval_every"] == 2 and level["save_every"] == 3
    assert "--log-to-swanlab" not in gpu_job.evaluation_command(args, level, Path("a"), Path("b"))


@pytest.mark.parametrize(
    "options",
    [
        ["--gpus", "2"],
        ["--gpus", "0,0"],
        ["--train-samples", "-1"],
        ["--epochs", "0"],
        ["--run-id", "../escape"],
        ["--stage", "warmup"],
        ["--method", "memory_change", "--stage", "policy"],
        ["--init-checkpoint", "parent.pt"],
    ],
)
def test_invalid_gpu_scope_or_stage_arguments_are_rejected(tmp_path, options):
    with pytest.raises(SystemExit):
        arguments(tmp_path, *options)


@pytest.mark.parametrize("gpus", ["0,1", "1"])
def test_execution_writes_resolved_configs_and_chains_actual_checkpoints(
    tmp_path, monkeypatch, gpus
):
    args = arguments(tmp_path, "--gpus", gpus)
    prepare_dataset_entries(args)
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)

    result = gpu_job.run_job(args)

    assert result["status"] == "finished"
    training = [call for call in commands.calls if "config" in call]
    evaluations = [
        call for call in commands.calls if "latent_working_memory.v3.evaluate" in call["command"]
    ]
    assert len(training) == 5 and len(evaluations) == 2 and len(commands.calls) == 8
    assert training[0]["config"].training.init_checkpoint is None
    assert training[1]["config"].training.init_checkpoint == str(training[0]["checkpoint"])
    assert training[2]["config"].training.init_checkpoint == str(training[1]["checkpoint"])
    assert training[3]["config"].training.init_checkpoint == str(training[0]["checkpoint"])
    assert training[4]["config"].training.init_checkpoint == str(training[3]["checkpoint"])
    for call in training:
        assert call["environment"]["CUDA_VISIBLE_DEVICES"] == gpus
        assert f"--nproc_per_node={len(gpus.split(','))}" in call["command"]
        assert value(call["command"], "--stop-after-steps") == "2"
        assert (
            json.loads(Path(value(call["command"], "--config")).read_text())
            == call["config"].to_dict()
        )
    for call, trained in zip(evaluations, (training[2], training[4]), strict=True):
        assert value(call["command"], "--checkpoint") == str(trained["checkpoint"])
        assert call["environment"]["CUDA_VISIBLE_DEVICES"] == gpus.split(",")[0]
        assert value(call["command"], "--split") == "dev"
        assert value(call["command"], "--max-trajectories") == "2"
        assert "--log-to-swanlab" in call["command"]
    assert "latent_working_memory.v3.compare" in commands.calls[-1]["command"]
    assert all(str(path) in commands.calls[-1]["command"] for path in result["summaries"])
    directory, _, _ = gpu_job.build_jobs(args)
    assert json.loads((directory / "plan/result.json").read_text()) == result
    with pytest.raises(ValueError, match="already exists"):
        gpu_job.run_job(args)


def test_full_execution_starts_a_new_unbounded_run_without_smoke_checkpoint(tmp_path, monkeypatch):
    smoke = arguments(tmp_path, "--method", "memory_change", "--stage", "pretrain")
    prepare_dataset_entries(smoke)
    commands = Commands()
    monkeypatch.setattr(gpu_job, "execute", commands)
    short_result = gpu_job.run_job(smoke)
    full = arguments(tmp_path, "--mode", "full", "--method", "memory_change", "--stage", "pretrain")
    full_result = gpu_job.run_job(full)
    assert set(short_result["checkpoints"].values()).isdisjoint(full_result["checkpoints"].values())
    full_call = commands.calls[-1]
    assert "--stop-after-steps" not in full_call["command"]
    assert full_call["config"].training.init_checkpoint is None
    assert full_call["config"].training.max_train_samples is None
    assert full_call["config"].training.max_dev_samples is None


@pytest.mark.parametrize("fail_at", [1, 4])
def test_failed_training_or_evaluation_stops_the_remaining_graph(tmp_path, monkeypatch, fail_at):
    args = arguments(tmp_path)
    prepare_dataset_entries(args)
    commands = Commands(fail_at=fail_at)
    monkeypatch.setattr(gpu_job, "execute", commands)
    with pytest.raises(subprocess.CalledProcessError):
        gpu_job.run_job(args)
    assert len(commands.calls) == fail_at
    directory, _, _ = gpu_job.build_jobs(args)
    failure = json.loads((directory / "plan/result.json").read_text())
    assert failure["status"] == "failed"
    assert len(failure["checkpoints"]) == (0 if fail_at == 1 else 3)
    assert not failure["summaries"]


def test_training_must_save_the_reported_checkpoint_before_dependent_stage(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    prepare_dataset_entries(args)
    commands = Commands(omit_checkpoint=True)
    monkeypatch.setattr(gpu_job, "execute", commands)
    with pytest.raises(FileNotFoundError, match="reported checkpoint"):
        gpu_job.run_job(args)
    assert len(commands.calls) == 1


def test_dry_run_does_not_access_datasets_or_execute_or_create_artifacts(
    tmp_path, monkeypatch, capsys
):
    args = arguments(tmp_path, "--dry-run", "--method", "all")

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not execute commands or check dataset files")

    monkeypatch.setattr(gpu_job, "execute", forbidden)
    monkeypatch.setattr(Path, "is_file", forbidden)
    plan = gpu_job.run_job(args)
    assert len(plan["jobs"]) == 10
    assert json.loads(capsys.readouterr().out) == plan
    assert list(tmp_path.iterdir()) == []


def test_planning_imports_no_model_or_tracking_runtime(tmp_path):
    code = (
        "import sys; from latent_working_memory.v3.gpu_job import main; "
        f"main(['--dry-run', '--output-root', {str(tmp_path)!r}]); "
        "assert not {'torch', 'transformers', 'swanlab'} & sys.modules.keys()"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("exit_code", [0, 7])
def test_shell_changes_to_spaced_repo_preserves_arguments_and_returns_process_status(
    tmp_path, exit_code
):
    script = Path("src/latent_working_memory/v3/scripts/run_gpu.sh").resolve()
    repository = tmp_path / "repository with spaces"
    repository.mkdir()
    binaries = tmp_path / "fake bin"
    binaries.mkdir()
    fake_uv = binaries / "uv"
    fake_uv.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({'cwd': os.getcwd(), 'args': sys.argv[1:], "
        "'unbuffered': os.environ['PYTHONUNBUFFERED'], "
        "'tokenizers': os.environ['TOKENIZERS_PARALLELISM']}))\n"
        f"sys.exit({exit_code})\n"
    )
    fake_uv.chmod(0o755)
    options = [
        "--mode",
        "smoke",
        "--model-path",
        "models/name with spaces",
        "--group",
        "literal $name; echo unchanged",
    ]
    environment = dict(
        os.environ, PATH=f"{binaries}:{os.environ['PATH']}", LWM_REPO_DIR=str(repository)
    )
    result = subprocess.run(
        ["bash", str(script), *options], env=environment, capture_output=True, text=True
    )
    assert result.returncode == exit_code
    recorded = json.loads(result.stdout)
    assert recorded["cwd"] == str(repository)
    assert recorded["args"] == [
        "run",
        "--frozen",
        "python",
        "-m",
        "latent_working_memory.v3.gpu_job",
        *options,
    ]
    assert recorded["unbuffered"] == "1"
    assert recorded["tokenizers"] == "false"
    assert list(repository.iterdir()) == []


def test_execute_propagates_real_child_failure_and_keeps_the_log(tmp_path):
    log = tmp_path / "child.log"
    command = [sys.executable, "-c", "print('visible failure'); raise SystemExit(3)"]
    with pytest.raises(subprocess.CalledProcessError) as error:
        gpu_job.execute(command, dict(os.environ), log)
    assert error.value.returncode == 3
    assert log.read_text().strip() == "visible failure"
