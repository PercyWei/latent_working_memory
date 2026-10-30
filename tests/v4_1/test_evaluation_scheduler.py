"""checkpoint 串行、样本多卡分片、缓存和失败重试，无需 GPU。"""

from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import threading
from types import SimpleNamespace

import pytest
import torch

from latent_working_memory.v4_1.config import ExperimentConfig, TrainingConfig
from latent_working_memory.v4_1.evaluation import evaluate, run


def checkpoints(directory, steps):
    config = ExperimentConfig(training=TrainingConfig(
        dataset_dir=str(directory.parent.parent / "training-data"),
        output_root=str(directory.parent.parent), output_dir=str(directory),
        max_steps=20, warmup_steps=0, tracking="disabled",
    ))
    paths = []
    for step in steps:
        path = directory / "checkpoints" / f"global_step_{step}" / "state.pt"
        path.parent.mkdir(parents=True)
        torch.save({
            "config": asdict(config), "model": {}, "optimizer": {}, "scheduler": {},
            "cursor": {"step": step}, "rng": [], "query_rng": [], "binding": {},
        }, path)
        paths.append(path)
    return paths


def arguments(tmp_path, paths, **overrides):
    values = {
        "checkpoint": paths, "eval_run": None, "eval_id": None,
        "eval_type": "test", "benchmark": None, "output_root": None,
        **overrides,
    }
    option = "test_data" if values["eval_type"] == "test" else "benchmark_data"
    values.setdefault(option, str(tmp_path / "data"))
    return SimpleNamespace(**values)


def forbidden(*args, **kwargs):
    raise AssertionError("cached scheduling must not load a model, use CUDA, or launch a worker")


def test_cached_checkpoints_sort_and_preserve_plan_without_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "datetime", SimpleNamespace(
        now=lambda zone: datetime(2026, 10, 29, 20, 58, 12, tzinfo=zone),
    ))
    directory = tmp_path / "artifacts/method_01/train"
    paths = checkpoints(directory, (10, 2))
    (directory / "checkpoints/global_step_19").mkdir()
    plan_path = directory.parent / "plan/eval-20261029-205812/test.json"
    checked = []

    def cached(checkpoint, data, output, **settings):
        assert (output.parents[3] / "plan" / f"eval-{settings['eval_id']}"
                / f"{settings['task']}.json").is_file()
        step = int(checkpoint.parent.name.removeprefix("global_step_"))
        checked.append((step, output))
        return {"step": step}

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(run, "load_cached_evaluation", cached)
    monkeypatch.setattr(run, "_compute", forbidden)
    monkeypatch.setattr(run, "_visible_devices", forbidden)
    monkeypatch.setattr(evaluate, "OnlineMemoryLM", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    output = run.run_evaluations(arguments(tmp_path, paths))
    assert output == directory.parent / "eval/20261029-205812/test"
    plan = json.loads(plan_path.read_text())
    assert plan["eval_id"] == "20261029-205812"
    assert plan["training_run"] == str(directory) and plan["max_new_tokens"] == 128
    assert not (output / "plan.json").exists()
    assert checked == [(step, output / f"global_step_{step}") for step in (2, 10)]
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "finished" and status["progress"]["finished"] == 2
    assert status["selected_checkpoints"] == ["global_step_2", "global_step_10"]
    assert all(row["compute_status"] == "cached" for row in status["checkpoints"].values())

    # eval-run 扫描完整 checkpoint；计划独立于本次 checkpoint 选择。
    resumed = arguments(tmp_path, None, eval_run=str(directory), eval_id="20261029-205812")
    del resumed.test_data
    run.run_evaluations(resumed)
    assert json.loads(plan_path.read_text())["data"] == str(tmp_path / "data")
    assert run.run_evaluations(arguments(tmp_path, paths, max_new_tokens=128)) == output
    selected = arguments(tmp_path, [paths[1]], eval_id="20261029-205812")
    run.run_evaluations(selected)
    status = json.loads((output / "status.json").read_text())
    assert status["progress"]["total"] == 1 and len(status["checkpoints"]) == 2
    assert status["selected_checkpoints"] == ["global_step_2"]
    with pytest.raises(ValueError, match="plan settings differ"):
        run.run_evaluations(arguments(tmp_path, paths, max_test_samples=1))
    other = checkpoints(tmp_path / "artifacts/other-method_02/train", (3,))
    with pytest.raises(ValueError, match="same training run"):
        run.run_evaluations(arguments(tmp_path, [paths[0], other[0]]))
    with pytest.raises(ValueError, match="duplicate"):
        run.run_evaluations(arguments(tmp_path, [paths[0], paths[0]]))
    copied = tmp_path / "copied/checkpoints/global_step_10/state.pt"
    copied.parent.mkdir(parents=True)
    copied.write_bytes(paths[0].read_bytes())
    with pytest.raises(ValueError, match="path differs from its saved training directory"):
        run.run_evaluations(arguments(tmp_path, [copied]))

    for options, message in (
        ({"eval_id": "bad\\id"}, "eval-id must be a nonempty directory name"),
        ({"benchmark_data": "benchmark"}, "require --eval-type benchmark"),
        ({"benchmark": "locomo"}, "require --eval-type benchmark"),
        ({"lengths": [8192]}, "require --eval-type benchmark"),
        ({"eval_type": "benchmark"}, "requires --benchmark"),
        ({"eval_type": "benchmark", "benchmark": "locomo", "test_data": "test"}, "require --eval-type test"),
        ({"eval_type": "benchmark", "benchmark": "ruler", "max_test_samples": None}, "require --eval-type test"),
        ({"eval_type": "benchmark", "benchmark": "ruler", "max_new_tokens": 9}, "require --eval-type test"),
        ({"eval_type": "benchmark", "benchmark": "locomo", "lengths": [8192]}, "requires --benchmark ruler"),
    ):
        with pytest.raises(ValueError, match=message):
            run.run_evaluations(arguments(tmp_path, paths, **options))

    benchmark = arguments(tmp_path, paths, eval_type="benchmark", benchmark="ruler", lengths=[16384, 8192])
    benchmark_output = run.run_evaluations(benchmark)
    benchmark_plan = directory.parent / "plan/eval-20261029-205812/ruler.json"
    assert json.loads(benchmark_plan.read_text())["lengths"] == [8192, 16384]
    expanded = arguments(tmp_path, paths, eval_type="benchmark", benchmark="ruler",
                         eval_id="20261029-205812", lengths=[32768, 16384, 8192])
    assert run.run_evaluations(expanded) == benchmark_output
    assert json.loads(benchmark_plan.read_text())["lengths"] == [8192, 16384, 32768]
    with pytest.raises(ValueError, match="expand|superset"):
        run.run_evaluations(benchmark)

    monkeypatch.setattr(run, "datetime", SimpleNamespace(
        now=lambda zone: datetime(2026, 10, 29, 20, 58, 13, tzinfo=zone),
    ))
    assert run.run_evaluations(arguments(tmp_path, paths)) == directory.parent / "eval/20261029-205813/test"
    assert run.run_evaluations(resumed) == output
    first = arguments(tmp_path, paths, eval_id="new-without-data")
    del first.test_data
    with pytest.raises(ValueError, match="test-data"):
        run.run_evaluations(first)
    assert not (directory.parent / "plan/eval-new-without-data/test.json").exists()

    parsed = []
    monkeypatch.setattr(run, "run_evaluations", parsed.append)
    run.main(["--checkpoint", str(paths[0]), "--test-data", "test", "--max-test-samples", "null"])
    assert parsed[-1].eval_type == "test" and parsed[-1].max_test_samples is None
    run.main(["--checkpoint", str(paths[0]), "--eval-type", "benchmark", "--benchmark", "locomo", "--benchmark-data", "benchmark"])
    assert parsed[-1].benchmark == "locomo" and parsed[-1].benchmark_data == "benchmark"
    for option in ("--task", "--data", "--max-samples"):
        with pytest.raises(SystemExit):
            run.main(["--checkpoint", str(paths[0]), option, "invalid"])


def test_checkpoints_are_serial_and_samples_use_all_gpus_with_retry(tmp_path, monkeypatch):
    paths = checkpoints(tmp_path / "artifacts/method_01/train", (12, 5, 2, 10, 7))
    active, launched, processes, killed, completed = {}, [], [], [], {}
    prepared, merged = [], []
    mutex, barrier = threading.Lock(), threading.Barrier(2)
    started = threading.Event()
    phase = "first"

    def sample_ids(output):
        return [f"{output.name}/sample-{index}" for index in range(6)]

    def prepare(checkpoint, data, output, num_workers, **settings):
        output = Path(output)
        assert all(step is None for step in active.values())
        identifiers = sample_ids(output)
        done = completed.setdefault(str(output), set())
        pending = [value for value in identifiers if value not in done]
        count = min(num_workers, len(pending))
        work = {
            "total_samples": len(identifiers), "completed_samples": len(done),
            "pending_samples": len(pending),
            "workers": [pending[rank::count] for rank in range(count)],
        }
        (output / "work.json").write_text(json.dumps(work))
        prepared.append(int(output.name.removeprefix("global_step_")))
        return work

    def finalize(checkpoint, data, output, **settings):
        output = Path(output)
        assert all(step is None for step in active.values())
        if len(completed[str(output)]) != 6:
            raise run.IncompleteEvaluationError(len(completed[str(output)]), 6)
        summary = {"step": int(output.name.removeprefix("global_step_"))}
        (output / "summary.json").write_text(json.dumps(summary))
        merged.append(summary["step"])
        return summary

    class Process:
        def __init__(self, command, env, **options):
            self.gpu = env["CUDA_VISIBLE_DEVICES"]
            self.output = Path(command[command.index("--output-dir") + 1])
            self.step = int(self.output.name.removeprefix("global_step_"))
            self.rank = int(command[command.index("--worker-rank") + 1])
            self.identifiers = json.loads((self.output / "work.json").read_text())["workers"][self.rank]
            self.returncode = None
            self.stopped = threading.Event()
            assert options["start_new_session"] and "--no-publish" not in command
            assert command[command.index("--max-samples") + 1] == "3"
            assert command[command.index("--max-new-tokens") + 1] == "9"
            plan = self.output.parents[3] / "plan" / f"eval-{self.output.parent.parent.name}" / "test.json"
            assert plan.is_file()
            with mutex:
                self.pid = 100 + len(processes)
                assert active.get(self.gpu) is None
                assert all(step in {None, self.step} for step in active.values())
                active[self.gpu] = self.step
                assert not set(self.identifiers) & completed[str(self.output)]
                launched.append((self.gpu, self.step, self.rank, tuple(self.identifiers)))
                processes.append(self)
            self.stdout = self.lines()

        def lines(self):
            if phase in {"first", "single"}:
                barrier.wait(timeout=10)
            if phase == "interrupt":
                started.set()
                assert self.stopped.wait(timeout=10)
            yield "completed\n"

        def wait(self, timeout=None):
            with mutex:
                active[self.gpu] = None
                if self.returncode is None:
                    self.returncode = 3 if phase == "first" and self.step == 7 and self.rank == 1 else 0
                if self.returncode == 0:
                    completed[str(self.output)].update(self.identifiers)
                elif self.returncode == 3:
                    completed[str(self.output)].add(self.identifiers[0])
            return self.returncode

        def poll(self):
            return self.returncode

    def stop(pid, signum):
        process = next(value for value in processes if value.pid == pid)
        killed.append(pid)
        process.returncode = -signum
        process.stopped.set()

    def cached(checkpoint, data, output, **settings):
        if checkpoint.parent.name == "global_step_5":
            return {"step": 5}
        path = output / "summary.json"
        return json.loads(path.read_text()) if path.exists() else None

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(run, "load_cached_evaluation", cached)
    monkeypatch.setattr(run, "prepare_sample_work", prepare)
    monkeypatch.setattr(run, "finalize_sample_work", finalize)
    monkeypatch.setattr(run.subprocess, "Popen", Process)
    monkeypatch.setattr(run.os, "killpg", stop)
    monkeypatch.setattr(evaluate, "OnlineMemoryLM", forbidden)
    args = arguments(tmp_path, paths, output_root=str(tmp_path / "evaluation"), eval_id="comparison",
                     max_test_samples=3, max_new_tokens=9)
    output = tmp_path / "evaluation/method_01/eval/comparison/test"
    with pytest.raises(RuntimeError, match="1/5 checkpoints"):
        run.run_evaluations(args)
    status = json.loads((output / "status.json").read_text())
    assert prepared == [2, 7, 10, 12] and merged == [2, 10, 12]
    assert status["status"] == "failed" and status["progress"]["finished"] == 4
    failed = status["checkpoints"]["global_step_7"]
    assert failed["return_code"] == 3 and "4/6 samples completed" in failed["error"]
    assert failed["samples"]["completed_samples"] == 4 and failed["samples"]["pending_samples"] == 2
    assert [value[1] for value in launched] == [2, 2, 7, 7, 10, 10, 12, 12]
    assert {gpu for gpu, _, _, _ in launched} == {"0", "1"}
    for step in (2, 7, 10, 12):
        log = (output / f"global_step_{step}/eval.log").read_text()
        assert "rank=0" in log and "rank=1" in log and "completed\n" in log
    assert all(step is None for step in active.values()) and not killed

    # 改为单卡重试，完整 checkpoint 不重跑，失败 worker 已提交的样本也不重复。
    phase = "retry"
    launched.clear()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert run.run_evaluations(args) == output
    assert len(launched) == 1 and launched[0][:3] == ("1", 7, 0)
    assert len(launched[0][3]) == 2
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "finished" and status["progress"]["finished"] == 5
    assert status["checkpoints"]["global_step_7"]["samples"]["completed_samples"] == 6

    # 单 checkpoint 同样启动双卡；全样本已完成但尚未合并时可不提供 GPU。
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    phase = "single"
    launched.clear()
    single = arguments(tmp_path, [paths[2]], output_root=str(tmp_path / "evaluation"),
                       eval_id="single", max_test_samples=3, max_new_tokens=9)
    single_output = run.run_evaluations(single) / "global_step_2"
    assert len(launched) == 2 and {value[0] for value in launched} == {"0", "1"}
    (single_output / "summary.json").unlink()
    launched.clear()
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    monkeypatch.setattr(run, "_visible_devices", forbidden)
    run.run_evaluations(single)
    assert not launched and (single_output / "summary.json").is_file()
    monkeypatch.undo()

    # 中断时只停止当前 checkpoint 的全部分片，后续 checkpoint 不启动。
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(run, "load_cached_evaluation", cached)
    monkeypatch.setattr(run, "prepare_sample_work", prepare)
    monkeypatch.setattr(run, "finalize_sample_work", finalize)
    monkeypatch.setattr(run.subprocess, "Popen", Process)
    monkeypatch.setattr(run.os, "killpg", stop)
    phase = "interrupt"
    original = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    def interrupt():
        if started.wait(timeout=10):
            os.kill(os.getpid(), signal.SIGTERM)
    interrupter = threading.Thread(target=interrupt)
    interrupter.start()
    args.eval_id = "interrupted"
    try:
        with pytest.raises(KeyboardInterrupt, match="SIGTERM"):
            run.run_evaluations(args)
    finally:
        interrupter.join(timeout=15)
    status = json.loads((output.parent.parent / "interrupted/test/status.json").read_text())
    assert status["status"] == "stopped" and status["progress"]["stopped"] == 4
    assert status["progress"]["finished"] == 1 and killed
    assert all(process.poll() is not None for process in processes)
    assert all(signal.getsignal(signum) == handler for signum, handler in original.items())
