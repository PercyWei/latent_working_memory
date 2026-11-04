"""训练生命周期、目录覆盖及独立于评估计划的训练恢复，无需 GPU。"""

from dataclasses import asdict, replace
from datetime import datetime
from io import StringIO
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest
import torch

from latent_working_memory.v4_1.config import ExperimentConfig, TrainingConfig, load_config
from latent_working_memory.v4_1.execution import run_lock
from latent_working_memory.v4_1.plans import load_training_plan, save_training_plan
from latent_working_memory.v4_1.training import run, train
from latent_working_memory.v4_1.training.engine import training_steps


def write_checkpoint(config, step, summary=True):
    directory = Path(config.training.output_dir)
    actual = replace(config, training=replace(
        config.training, max_steps=training_steps(config.training, 16, 16),
    ))
    path = directory / "checkpoints" / f"global_step_{step}" / "state.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": asdict(actual), "model": {}, "optimizer": {},
        "scheduler": {"last_epoch": step}, "cursor": {"step": step},
        "rng": [None, None], "query_rng": [None, None],
        "binding": {"train": {"episodes": 16}},
    }, path)
    (directory / "config.json").write_text(json.dumps(asdict(actual)), encoding="utf-8")
    result = {"step": step, "checkpoint": str(path), "output_dir": str(directory)}
    if summary:
        (directory / "summary.json").write_text(json.dumps(result), encoding="utf-8")
    return path


def configuration(tmp_path):
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    return ExperimentConfig(training=TrainingConfig(
        output_root=str(tmp_path), dataset_dir=str(data), epochs=3, max_steps=None,
        warmup_steps=0, micro_batch_size=1, tracking="disabled",
    ))


@pytest.mark.parametrize(("inner", "label"), [
    ("am", "am"), ("output_only_mass_calibrated", "oomc"),
])
@pytest.mark.parametrize("outer", ["am", "distill"])
def test_run_directory_distinguishes_inner_and_outer_objectives(tmp_path, monkeypatch, inner, label, outer):
    monkeypatch.setattr(train, "datetime", SimpleNamespace(
        now=lambda zone: datetime(2026, 11, 4, 9, 40, 59, tzinfo=zone),
    ))
    config = configuration(tmp_path)
    config = replace(config, model=replace(config.model, inner_objective=inner, inner_steps=1),
                     training=replace(config.training, outer_write_obj=outer))
    assert run._training_output_dir(config, None) == (
        tmp_path / f"m256-d512_p1024-r256_inner-s1-{label}_outer-{outer}_20261104-0940/train"
    )


def test_resume_uses_saved_directory_without_renaming(tmp_path):
    config = configuration(tmp_path)
    original = tmp_path / "am-recent_m256-d512_p1024-r256_inner3_original/train"
    checkpoint = {"config": {"training": {"output_dir": str(original)}}}
    assert run._training_output_dir(config, checkpoint) == original


def test_training_process_lifecycle_overwrite_and_independent_evaluation(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source_config = configuration(tmp_path / "artifacts")
    source.write_text(json.dumps(asdict(source_config)), encoding="utf-8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(train, "datetime", SimpleNamespace(
        now=lambda zone: datetime(2026, 10, 29, 21, 0, 1, tzinfo=zone),
    ))
    processes, stopped = [], []
    mode = "success"

    class Process:
        def __init__(self, command, env, start_new_session, **options):
            assert start_new_session
            assert command[1:4] == ["-m", "torch.distributed.run", "--standalone"]
            assert "--nproc-per-node=2" in command
            assert command[command.index("--module") + 1] == "latent_working_memory.v4_1.training.train"
            assert "--overwrite" not in command
            assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
            assert env["PYTHONUNBUFFERED"] == "1" and env["TOKENIZERS_PARALLELISM"] == "false"
            self.plan = Path(command[command.index("--plan-dir") + 1])
            self.config = load_training_plan(self.plan)
            self.directory = Path(self.config.training.output_dir)
            status = json.loads((self.directory / "status.json").read_text())
            assert status["status"] == "running" and status["command"] == command
            assert sorted(path.name for path in self.plan.iterdir()) == ["config.json", "train.json"]
            assert not (self.directory / "config.json").exists()
            self.stdout = StringIO("training worker output\n")
            self.pid, self.returncode = 100 + len(processes), None
            processes.append(self)

        def wait(self, timeout=None):
            if self.returncode is not None:
                return self.returncode
            if mode == "interrupt":
                raise KeyboardInterrupt
            if mode == "terminate":
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            if mode == "success":
                write_checkpoint(self.config, 1)
                write_checkpoint(self.config, 3)
            self.returncode = 0 if mode == "success" else 7
            return self.returncode

        def poll(self):
            return self.returncode

    def stop(pid, signum):
        assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
        stopped.append((pid, signum))
        next(process for process in processes if process.pid == pid).returncode = -signum

    def forbidden(*args, **kwargs):
        raise AssertionError("training must not start an evaluation")

    monkeypatch.setattr(run.subprocess, "Popen", Process)
    monkeypatch.setattr(run.os, "killpg", stop)
    monkeypatch.setattr(run, "run_evaluations", forbidden)
    result = run.run_job(run.parse_args(["--config", str(source)]))
    process = processes[0]
    assert result["step"] == 3 and Path(result["checkpoint"]).parent.name == "global_step_3"
    assert process.config.training.run_id == "m256-d512_p1024-r256_inner-s3-am_outer-am_20261029-2100"
    assert process.directory == Path(source_config.training.output_root) / process.config.training.run_id / "train"
    assert process.config.training.swanlab_project == "latent-working-memory-v4.1"
    assert process.config.training.group == Path(result["output_dir"]).parent.name
    requested = json.loads((process.plan / "train.json").read_text())
    assert requested["max_steps"] is None and "overwrite" not in requested
    retired = {"train_only", "test_data", "max_test_samples", "max_new_tokens", "eval_id"}
    assert not retired.intersection(requested)
    assert load_config(process.directory / "config.json").training.max_steps == 3
    assert not (process.directory.parent / "eval").exists()
    assert "training worker output" in (process.directory / "train.log").read_text()
    assert json.loads((process.directory / "status.json").read_text())["status"] == "finished"

    for mode, code in (("failure", 7), ("interrupt", 130), ("terminate", 143)):
        with pytest.raises(SystemExit) as error:
            run.main(["--config", str(source), "--output-root", str(tmp_path / mode)])
        assert error.value.code == code
        failed = processes[-1]
        status = json.loads((failed.directory / "status.json").read_text())
        assert status["status"] == ("failed" if mode == "failure" else "stopped")
        assert status["return_code"] == code and "error" in status
        assert failed.plan.is_dir() and "training worker output" in (failed.directory / "train.log").read_text()
    assert stopped == [(process.pid, signal.SIGTERM) for process in processes[-2:]]

    mode = "success"
    run_directory = process.directory.parent
    old_result = run_directory / "eval/manual/test/obsolete.json"
    old_result.parent.mkdir(parents=True)
    old_result.write_text("old result")
    old_plan = process.plan / "eval-manual/test.json"
    old_plan.parent.mkdir(parents=True)
    old_plan.write_text("old plan")
    old_checkpoint = process.directory / "checkpoints/global_step_77/state.pt"
    old_checkpoint.parent.mkdir(parents=True)
    old_checkpoint.write_text("old checkpoint")
    cache_marker = Path(source_config.training.teacher_cache_dir) / "keep.txt"
    cache_marker.parent.mkdir(parents=True)
    cache_marker.write_text("shared teacher")
    neighbor = tmp_path / "artifacts/another-run/keep.txt"
    neighbor.parent.mkdir(parents=True)
    neighbor.write_text("another run")
    launch_count = len(processes)
    with pytest.raises(ValueError, match="run directory"):
        run.run_job(run.parse_args(["--config", str(source)]))
    with pytest.raises(FileNotFoundError, match="training data directory"):
        run.run_job(run.parse_args([
            "--config", str(source), "--overwrite", "--train-data", str(tmp_path / "missing"),
        ]))
    assert old_result.exists() and old_plan.exists() and len(processes) == launch_count
    with run_lock(run_directory):
        with pytest.raises(ValueError, match="already running"):
            run.run_job(run.parse_args(["--config", str(source), "--overwrite"]))
    assert old_result.exists() and old_plan.exists()
    overwritten = run.run_job(run.parse_args(["--config", str(source), "--overwrite"]))
    assert overwritten == result and len(processes) == launch_count + 1
    assert not old_result.exists() and not old_plan.exists() and not old_checkpoint.exists()
    assert cache_marker.read_text() == "shared teacher" and neighbor.read_text() == "another run"

    custom = run.run_job(run.parse_args([
        "--config", str(source), "--output-root", str(tmp_path / "custom"),
        "--run-id", "example", "--group", "explicit-series",
    ]))
    assert Path(custom["output_dir"]) == tmp_path / "custom/example/train"
    assert load_config(Path(custom["output_dir"]) / "config.json").training.run_id == "example"
    assert load_config(Path(custom["output_dir"]) / "config.json").training.group == "explicit-series"

    def training_forbidden(*args, **kwargs):
        raise AssertionError("evaluate-only must not enter the training path")

    only_evaluations, evaluation_result = [], object()

    def evaluate_only(args):
        only_evaluations.append(args)
        return evaluation_result

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    for name in ("_visible_devices", "_execute", "load_config", "load_training_plan", "read_checkpoint"):
        monkeypatch.setattr(run, name, training_forbidden)
    monkeypatch.setattr(run, "run_evaluations", evaluate_only)
    assert run.main(["--evaluate-only", "--eval-run", "saved",
                     "--checkpoint", "global_step_1"]) is evaluation_result
    assert only_evaluations[-1].checkpoint == ["global_step_1"]
    assert run.main(["--evaluate-only", "--eval-run", "saved", "--checkpoint",
                     "global_step_1", "global_step_2", "--overwrite"]) is evaluation_result
    assert only_evaluations[-1].checkpoint == ["global_step_1", "global_step_2"]
    assert only_evaluations[-1].overwrite
    assert run.main(["--evaluate-only", "--eval-run", "saved"]) is evaluation_result
    assert only_evaluations[-1].eval_run == "saved"
    assert only_evaluations[-1].checkpoint is None
    assert only_evaluations[-1].eval_task == "test"
    assert only_evaluations[-1].eval_id is only_evaluations[-1].output_root is None
    assert run.main([
        "--evaluate-only", "--resume", "--eval-run", "saved", "--eval-id", "continued",
        "--eval-task", "ruler", "--checkpoint", "global_step_1", "global_step_17",
        "--config", "missing.json",
        "--eval-data", "ignored-data", "--lengths", "65536", "--max-new-tokens", "1",
        "--max-test-samples", "1", "--epochs", "99", "--eval-batch-size-per-gpu", "3",
    ]) is evaluation_result
    resumed = only_evaluations[-1]
    assert resumed.resume and resumed.eval_run == "saved"
    assert resumed.checkpoint == ["global_step_1", "global_step_17"]
    assert resumed.eval_task == "ruler" and resumed.eval_batch_size_per_gpu == 3
    assert not any(hasattr(resumed, name) for name in (
        "config", "eval_data", "lengths", "max_new_tokens", "max_test_samples", "epochs",
    ))


def test_resume_uses_training_state_only_and_cli_keeps_modes_separate(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    config = configuration(tmp_path / "artifacts")
    config = replace(config, training=replace(config.training, run_id="saved", group="saved-series"))
    config = replace(config, training=replace(
        config.training, output_dir=str(run._training_output_dir(config, None)),
    ))
    plan_dir = save_training_plan(config)
    first = write_checkpoint(config, 1, summary=False)
    original_plan = (plan_dir / "train.json").read_bytes()
    test_plan = plan_dir / "eval-original-test/test.json"
    test_plan.parent.mkdir()
    test_plan.write_text("obsolete evaluation plan")
    test_plan.unlink()
    launched = []

    def execute(command, environment, log_path):
        launched.append(command)
        prepared = load_training_plan(command[command.index("--plan-dir") + 1])
        if "--resume" in command:
            assert command[command.index("--checkpoint") + 1] == str(first)
        log_path.write_text("worker\n")
        write_checkpoint(prepared, 3)

    def forbidden(*args, **kwargs):
        raise AssertionError("resume must not read or execute an evaluation plan")

    monkeypatch.setattr(run, "_execute", execute)
    monkeypatch.setattr(run, "run_evaluations", forbidden)
    args = run.parse_args([
        "--resume", "--checkpoint", str(first), "--config", str(tmp_path / "missing.json"),
        "--train-data", str(tmp_path / "missing-data"), "--output-root", str(tmp_path / "ignored"),
        "--run-id", "ignored", "--epochs", "999", "--micro-batch-size-per-gpu", "99",
        "--eval-run", "ignored/train", "--eval-task", "ruler", "--lengths", "65536",
        "--eval-batch-size-per-gpu", "99", "--tracking", "online",
    ])
    assert not any(hasattr(args, name) for name in (
        "config", "train_data", "output_root", "run_id", "epochs", "micro_batch_size_per_gpu",
        "eval_task", "lengths", "eval_batch_size_per_gpu", "tracking",
    ))
    assert args.eval_run is None
    ignored = capsys.readouterr().err
    assert "[resume] ignored options" in ignored and "--config" in ignored and "--epochs" in ignored
    result = run.run_job(args)
    assert len(launched) == 1
    assert load_config(Path(result["output_dir"]) / "config.json").training.group == "saved-series"
    assert (plan_dir / "train.json").read_bytes() == original_plan

    final = Path(result["checkpoint"])
    summary = Path(result["output_dir"]) / "summary.json"
    summary.unlink()
    status_path = summary.parent / "status.json"
    previous_status = json.loads(status_path.read_text())
    status_path.write_text(json.dumps({
        **previous_status, "status": "stopped", "return_code": 143,
        "error": "controller interrupted after final checkpoint",
    }))
    assert run.run_job(run.parse_args(["--resume", "--checkpoint", str(final)])) == result
    repaired = json.loads(status_path.read_text())
    assert repaired["status"] == "finished" and repaired["return_code"] == 0 and "error" not in repaired
    assert all(repaired[key] == previous_status[key] for key in (
        "command", "gpus", "started_at", "finished_at",
    ))
    assert json.loads(summary.read_text()) == result and len(launched) == 1
    test_plan.write_text("not even valid JSON")
    assert run.run_job(run.parse_args(["--resume", "--checkpoint", str(final)])) == result
    with run_lock(summary.parent.parent):
        with pytest.raises(ValueError, match="already running"):
            run.run_job(run.parse_args(["--resume", "--checkpoint", str(final)]))

    changed_plan = json.loads(original_plan)
    changed_plan["learning_rate"] *= 2
    (plan_dir / "train.json").write_text(json.dumps(changed_plan))
    with pytest.raises(ValueError, match="saved training plan"):
        run.run_job(run.parse_args(["--resume", "--checkpoint", str(final)]))
    (plan_dir / "train.json").write_bytes(original_plan)
    with pytest.raises(ValueError, match="latest complete checkpoint"):
        run.run_job(run.parse_args(["--resume", "--checkpoint", str(first)]))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="original number of training processes"):
        run.run_job(run.parse_args(["--resume", "--checkpoint", str(final)]))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    # 新产物也只绑定训练计划，完成恢复不再次启动 worker。
    source = tmp_path / "fresh.json"
    source.write_text(json.dumps(asdict(configuration(tmp_path / "fresh"))), encoding="utf-8")
    fresh = run.run_job(run.parse_args(["--config", str(source)]))
    assert len(launched) == 2 and not list((Path(fresh["output_dir"]).parent / "plan").glob("eval-*"))
    assert run.run_job(run.parse_args(["--resume", "--checkpoint", fresh["checkpoint"]])) == fresh
    assert len(launched) == 2

    benchmark = run.parse_args([
        "--evaluate-only", "--eval-run", "saved", "--checkpoint", "global_step_1", "--overwrite",
        "--eval-task", "ruler", "--eval-data", "ruler-data", "--lengths", "8192", "16384",
        "--output-root", "other-artifacts",
    ])
    assert benchmark.eval_task == "ruler" and benchmark.lengths == [8192, 16384] and benchmark.overwrite
    assert benchmark.output_root == "other-artifacts" and not hasattr(benchmark, "max_test_samples")
    for argv in (
        ["--train-only"],
        ["--evaluate-only"],
        ["--evaluate-only", "--checkpoint", "global_step_1"],
        ["--evaluate-only", "--resume", "--checkpoint", "saved"],
        ["--evaluate-only", "--resume", "--checkpoint", "saved", "--eval-id", "existing"],
        ["--evaluate-only", "--resume", "--eval-run", "saved"],
        ["--evaluate-only", "--resume", "--eval-run", "saved", "--eval-id", "existing"],
        ["--resume", "--checkpoint", "first", "second"],
        ["--resume", "--checkpoint", "saved", "--overwrite"],
        ["--eval-run", "saved"],
        ["--checkpoint", "saved"],
        ["--evaluate-only", "--eval-run", "saved", "--config", "config.json"],
        ["--evaluate-only", "--eval-run", "saved", "--train-data", "train-data"],
        ["--evaluate-only", "--eval-run", "saved", "--epochs", "4"],
        ["--evaluate-only", "--checkpoint", "saved", "--test-data", "test", "--benchmark-data", "benchmark"],
        ["--evaluate-only", "--checkpoint", "saved", "--eval-type", "benchmark"],
        ["--evaluate-only", "--eval-run", "saved", "--eval-task", "ruler", "--max-test-samples", "null"],
        ["--evaluate-only", "--eval-run", "saved", "--eval-task", "ruler", "--max-new-tokens", "8"],
        ["--evaluate-only", "--eval-run", "saved", "--eval-task", "locomo", "--lengths", "8192"],
        ["--eval-task", "locomo"],
        ["--eval-data", "test"],
        ["--evaluate-only", "--checkpoint", "saved", "--eval-name", "test"],
        ["--evaluate-only", "--resume", "--overwrite", "--eval-id", "same", "--checkpoint", "saved"],
        ["--max-test-samples", "null"],
        ["--max-new-tokens", "8"],
        ["--eval-id", "manual"],
    ):
        with pytest.raises(SystemExit) as error:
            run.parse_args(argv)
        assert error.value.code == 2
