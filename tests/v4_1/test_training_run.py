"""永久计划、训练进程生命周期及最终测试的恢复。"""

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
from latent_working_memory.v4_1.plans import (
    load_training_plan, prepare_evaluation_plan, save_training_plan,
)
from latent_working_memory.v4_1.training import run
from latent_working_memory.v4_1.training.engine import training_steps


def write_checkpoint(config, step, summary=True):
    directory = Path(config.training.output_dir)
    actual = replace(config, training=replace(
        config.training, max_steps=training_steps(config.training, 16, 16),
    ))
    path = directory / "checkpoints" / f"global_step_{step}" / "state.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": asdict(actual), "model": {}, "optimizer": {}, "scheduler": {},
        "cursor": {"step": step}, "rng": [None, None], "query_rng": [None, None],
        "binding": {"train": {"episodes": 16}},
    }, path)
    (directory / "config.json").write_text(json.dumps(asdict(actual)), encoding="utf-8")
    result = {"step": step, "checkpoint": str(path), "output_dir": str(directory)}
    if summary:
        (directory / "summary.json").write_text(json.dumps(result), encoding="utf-8")
    return path


def configuration(tmp_path):
    return ExperimentConfig(training=TrainingConfig(
        output_root=str(tmp_path), dataset_dir=str(tmp_path / "data"),
        epochs=3, max_steps=None, warmup_steps=0, tracking="disabled",
    ))


def test_final_checkpoint_test_follows_process_exit_and_failure_stops_pipeline(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_text(json.dumps(asdict(configuration(tmp_path / "artifacts"))), encoding="utf-8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(run, "datetime", SimpleNamespace(
        now=lambda zone: datetime(2026, 10, 29, 21, 0, 1, tzinfo=zone),
    ))
    processes, evaluated, stopped = [], [], []
    mode = "success"

    class Process:
        def __init__(self, command, env, start_new_session, **options):
            assert start_new_session
            assert command[1:4] == ["-m", "torch.distributed.run", "--standalone"]
            assert "--nproc-per-node=2" in command
            assert command[command.index("--module") + 1] == "latent_working_memory.v4_1.training.train"
            assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
            assert env["PYTHONUNBUFFERED"] == "1" and env["TOKENIZERS_PARALLELISM"] == "false"
            self.plan = Path(command[command.index("--plan-dir") + 1])
            self.config = load_training_plan(self.plan)
            self.directory = Path(self.config.training.output_dir)
            status = json.loads((self.directory / "status.json").read_text())
            assert status["status"] == "running" and status["command"] == command
            assert self.plan.is_dir() and (self.plan / f"eval-{self.config.training.eval_id}/test.json").is_file()
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

    def evaluate(args):
        assert processes[-1].returncode == 0
        assert processes[-1].plan.is_dir()
        assert not hasattr(args, "test_data") and not hasattr(args, "max_new_tokens")
        evaluated.append(args)

    monkeypatch.setattr(run.subprocess, "Popen", Process)
    monkeypatch.setattr(run.os, "killpg", stop)
    monkeypatch.setattr(run, "run_evaluations", evaluate)
    result = run.run_job(run.parse_args([
        "--config", str(source), "--test-data", str(tmp_path / "test"),
        "--max-test-samples", "2", "--max-new-tokens", "7",
    ]))
    process = processes[0]
    plan = json.loads((process.plan / "eval-auto/test.json").read_text())
    assert result["step"] == 3 and evaluated[0].checkpoint == [result["checkpoint"]]
    assert Path(result["checkpoint"]).parent.name == "global_step_3"
    assert process.config.training.run_id == "20261029-210001"
    assert process.config.training.eval_id == evaluated[0].eval_id == "auto"
    assert process.config.training.swanlab_project == "latent-working-memory-v4.1"
    assert process.config.training.group == Path(result["output_dir"]).parent.name
    assert plan["data"] == str(tmp_path / "test")
    assert plan["max_samples"] == 2 and plan["max_new_tokens"] == 7
    assert json.loads((process.plan / "train.json").read_text())["max_steps"] is None
    assert load_config(process.directory / "config.json").training.max_steps == 3
    assert json.loads(source.read_text())["training"]["eval_id"] is None
    assert "training worker output" in (process.directory / "train.log").read_text()
    assert json.loads((process.directory / "status.json").read_text())["status"] == "finished"

    for mode, code in (("failure", 7), ("interrupt", 130), ("terminate", 143)):
        with pytest.raises(SystemExit) as error:
            run.main(["--config", str(source), "--output-root", str(tmp_path / mode)])
        assert error.value.code == code and len(evaluated) == 1
        failed = processes[-1]
        status = json.loads((failed.directory / "status.json").read_text())
        assert status["status"] == ("failed" if mode == "failure" else "stopped")
        assert status["return_code"] == code and "error" in status
        assert failed.plan.is_dir() and "training worker output" in (failed.directory / "train.log").read_text()
    assert stopped == [(process.pid, signal.SIGTERM) for process in processes[-2:]]

    mode = "success"
    custom = run.run_job(run.parse_args([
        "--config", str(source), "--output-root", str(tmp_path / "custom"),
        "--group", "explicit-series", "--eval-id", "explicit-test",
    ]))
    saved = load_config(Path(custom["output_dir"]) / "config.json")
    assert saved.training.group == "explicit-series"
    assert saved.training.eval_id == evaluated[-1].eval_id == "explicit-test"

    def forbidden(*args, **kwargs):
        raise AssertionError("evaluate-only must not enter the training path")

    only_evaluations, evaluation_result = [], object()

    def evaluate_only(args):
        only_evaluations.append(args)
        return evaluation_result

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    for name in ("_visible_devices", "_execute", "load_config", "load_training_plan", "read_checkpoint"):
        monkeypatch.setattr(run, name, forbidden)
    monkeypatch.setattr(run, "run_evaluations", evaluate_only)
    assert run.main(["--evaluate-only", "--checkpoint", "step1"]) is evaluation_result
    assert only_evaluations[-1].checkpoint == ["step1"]
    assert run.main(["--evaluate-only", "--checkpoint", "step1", "step2"]) is evaluation_result
    assert only_evaluations[-1].checkpoint == ["step1", "step2"]
    assert run.main(["--evaluate-only", "--eval-run", "saved/train"]) is evaluation_result
    assert only_evaluations[-1].eval_run == "saved/train"
    assert only_evaluations[-1].eval_type == "test" and only_evaluations[-1].benchmark is None
    assert only_evaluations[-1].eval_id is only_evaluations[-1].output_root is None
    assert not hasattr(only_evaluations[-1], "test_data")


def test_train_only_and_resume_retain_test_settings_and_recover_final_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    config = configuration(tmp_path / "artifacts")
    config = replace(config, training=replace(
        config.training, run_id="saved", eval_id="original-test", test_data=str(tmp_path / "test"),
        max_test_samples=2, max_new_tokens=7, group="saved-series",
    ))
    config = replace(config, training=replace(
        config.training, output_dir=str(run._training_output_dir(config, None)),
    ))
    plan_dir = save_training_plan(config)
    prepare_evaluation_plan(Path(config.training.output_dir).parent, config, "original-test", "test",
                            {"data": config.training.test_data, "max_samples": 2, "max_new_tokens": 7})
    original_plan = (plan_dir / "train.json").read_bytes()
    first = write_checkpoint(config, 1, summary=False)
    launched, evaluated = [], []

    def execute(command, environment, log_path):
        launched.append(command)
        prepared = load_training_plan(command[command.index("--plan-dir") + 1])
        if "--resume" in command:
            assert command[command.index("--checkpoint") + 1] == str(first)
        else:
            assert prepared.training.train_only
        log_path.write_text("worker\n")
        write_checkpoint(prepared, 3)

    monkeypatch.setattr(run, "_execute", execute)
    monkeypatch.setattr(run, "run_evaluations", evaluated.append)
    result = run.run_job(run.parse_args(["--resume", "--checkpoint", str(first)]))
    assert len(launched) == len(evaluated) == 1
    assert evaluated[0].eval_id == "original-test" and not hasattr(evaluated[0], "test_data")
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
    # 自动 test 失败不重写训练的成功状态，重试只调用评估。
    def failed_test(args):
        raise RuntimeError("test failed")

    monkeypatch.setattr(run, "run_evaluations", failed_test)
    with pytest.raises(RuntimeError, match="test failed"):
        run.run_job(run.parse_args(["--resume", "--checkpoint", str(final)]))
    repaired = json.loads(status_path.read_text())
    assert repaired["status"] == "finished" and repaired["return_code"] == 0
    assert "error" not in repaired
    assert all(repaired[key] == previous_status[key] for key in (
        "command", "gpus", "started_at", "finished_at",
    ))
    monkeypatch.setattr(run, "run_evaluations", evaluated.append)
    assert run.run_job(run.parse_args(["--resume", "--checkpoint", str(final)])) == result
    assert json.loads(summary.read_text()) == result
    assert len(launched) == 1 and len(evaluated) == 2
    assert evaluated[-1].eval_id == evaluated[0].eval_id

    changed_plan = json.loads(original_plan)
    changed_plan["learning_rate"] *= 2
    (plan_dir / "train.json").write_text(json.dumps(changed_plan))
    with pytest.raises(ValueError, match="saved training plan"):
        run.run_job(run.parse_args(["--resume", "--checkpoint", str(final)]))
    (plan_dir / "train.json").write_bytes(original_plan)
    test_plan = plan_dir / "eval-original-test/test.json"
    original_test = test_plan.read_bytes()
    test_plan.unlink()
    with pytest.raises(FileNotFoundError):
        run.run_job(run.parse_args(["--resume", "--checkpoint", str(final)]))
    changed_test = json.loads(original_test)
    changed_test["data"] = str(tmp_path / "other-test")
    test_plan.write_text(json.dumps(changed_test))
    with pytest.raises(ValueError, match="plan"):
        run.run_job(run.parse_args(["--resume", "--checkpoint", str(final)]))
    assert len(launched) == 1
    test_plan.write_bytes(original_test)
    with pytest.raises(ValueError, match="latest complete checkpoint"):
        run.run_job(run.parse_args(["--resume", "--checkpoint", str(first)]))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="original number of training processes"):
        run.run_job(run.parse_args(["--resume", "--checkpoint", str(final)]))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    source = tmp_path / "train-only.json"
    source.write_text(json.dumps(asdict(configuration(tmp_path / "only"))), encoding="utf-8")
    only = run.run_job(run.parse_args(["--config", str(source), "--train-only"]))
    assert len(launched) == 2 and len(evaluated) == 2
    assert not (Path(only["output_dir"]).parent / "plan/eval-auto").exists()
    assert run.run_job(run.parse_args(["--resume", "--checkpoint", only["checkpoint"]])) == only
    assert len(launched) == 2 and len(evaluated) == 2

    benchmark = run.parse_args([
        "--evaluate-only", "--checkpoint", "saved", "--eval-type", "benchmark",
        "--benchmark", "ruler", "--benchmark-data", "ruler-data", "--lengths", "8192", "16384",
        "--output-root", "other-artifacts",
    ])
    assert benchmark.benchmark == "ruler" and benchmark.lengths == [8192, 16384]
    assert benchmark.output_root == "other-artifacts" and not hasattr(benchmark, "max_test_samples")
    for argv in (
        ["--evaluate-only"],
        ["--evaluate-only", "--train-only", "--checkpoint", "saved"],
        ["--evaluate-only", "--resume", "--checkpoint", "saved"],
        ["--resume", "--checkpoint", "first", "second"],
        ["--eval-run", "saved/train"],
        ["--checkpoint", "saved"],
        ["--evaluate-only", "--checkpoint", "saved", "--eval-run", "saved/train"],
        ["--evaluate-only", "--checkpoint", "saved", "--config", "config.json"],
        ["--evaluate-only", "--checkpoint", "saved", "--train-data", "train-data"],
        ["--evaluate-only", "--checkpoint", "saved", "--epochs", "4"],
        ["--evaluate-only", "--checkpoint", "saved", "--test-data", "test", "--benchmark-data", "benchmark"],
        ["--evaluate-only", "--checkpoint", "saved", "--eval-type", "benchmark"],
        ["--evaluate-only", "--checkpoint", "saved", "--eval-type", "benchmark", "--benchmark", "ruler", "--max-test-samples", "null"],
        ["--evaluate-only", "--checkpoint", "saved", "--eval-type", "benchmark", "--benchmark", "ruler", "--max-new-tokens", "8"],
        ["--evaluate-only", "--checkpoint", "saved", "--eval-type", "benchmark", "--benchmark", "locomo", "--lengths", "8192"],
        ["--benchmark", "locomo"],
        ["--train-only", "--test-data", "test"],
    ):
        with pytest.raises(SystemExit) as error:
            run.parse_args(argv)
        assert error.value.code == 2
