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
from latent_working_memory.v4_1.execution import evaluation_lock, run_lock


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
        "checkpoint": [path.parent.name for path in paths] if paths is not None else None,
        "eval_run": str(paths[0].parents[3]) if paths else None, "eval_id": None,
        "eval_task": "test", "eval_data": str(tmp_path / "data"), "output_root": None,
        "resume": False, "overwrite": False,
        **overrides,
    }
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
    checked, requested = [], []

    def cached(checkpoint, data, output, **settings):
        assert (output.parents[3] / "plan" / f"eval-{settings['eval_id']}"
                / f"{settings['task']}.json").is_file()
        step = int(checkpoint.parent.name.removeprefix("global_step_"))
        checked.append((step, output))
        requested.append((data, settings))
        return {"step": step}

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(run, "load_cached_evaluation", cached)
    monkeypatch.setattr(run, "_compute", forbidden)
    monkeypatch.setattr(run, "_visible_devices", forbidden)
    monkeypatch.setattr(evaluate, "OnlineMemoryLM", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    with pytest.raises(FileNotFoundError):
        run.run_evaluations(arguments(
            tmp_path, paths, eval_run=str(directory), eval_id="train-path-is-not-run-root",
        ))
    assert not (directory / "plan/eval-train-path-is-not-run-root/test.json").exists()
    output = run.run_evaluations(arguments(tmp_path, paths))
    assert output == directory.parent / "eval/20261029-205812/test"
    plan = json.loads(plan_path.read_text())
    assert plan["eval_id"] == "20261029-205812"
    assert plan["training_run"] == str(directory) and plan["max_new_tokens"] == 128
    assert plan["checkpoints"] == [str(path.resolve()) for path in (paths[1], paths[0])]
    original_plan = plan_path.read_bytes()
    assert not (output / "plan.json").exists()
    assert checked == [(step, output / f"global_step_{step}") for step in (2, 10)]
    status = json.loads((output / "status.json").read_text())
    assert status["eval_run"] == str(directory.parent)
    assert status["status"] == "finished" and status["progress"]["finished"] == 2
    assert status["selected_checkpoints"] == ["global_step_2", "global_step_10"]
    assert all(row["compute_status"] == "cached" for row in status["checkpoints"].values())
    with pytest.raises(ValueError, match="plan already exists"):
        run.run_evaluations(arguments(tmp_path, paths, eval_id="20261029-205812"))
    with pytest.raises(ValueError, match="plan does not exist"):
        run.run_evaluations(arguments(
            tmp_path, None, eval_run=str(directory.parent), eval_id="missing", resume=True,
        ))

    # 恢复时省略 checkpoint 沿用原列表，数据、预算及长度覆盖仍不生效。
    added = checkpoints(directory, (17,))[0]
    resumed = arguments(
        tmp_path, None, eval_run=str(directory.parent),
        eval_id="20261029-205812", resume=True, eval_data=str(tmp_path / "ignored-data"),
        max_test_samples=1, max_new_tokens=9, lengths=[8192], eval_batch_size_per_gpu=3,
    )
    run.run_evaluations(resumed)
    assert checked[-2:] == [(step, output / f"global_step_{step}") for step in (2, 10)]
    assert all(step != 17 for step, _ in checked)
    assert not (output / "global_step_17").exists()
    assert requested[-1][0] == str(tmp_path / "data")
    assert requested[-1][1]["max_samples"] is None
    assert requested[-1][1]["max_new_tokens"] == 128
    assert requested[-1][1]["lengths"] is None
    assert plan_path.read_bytes() == original_plan
    assert run.run_evaluations(arguments(
        tmp_path, paths, eval_run=str(directory.parent), eval_id="20261029-205812", resume=True,
        max_new_tokens=128,
    )) == output
    selected = arguments(tmp_path, [paths[1]], eval_run=str(directory.parent),
                         eval_id="20261029-205812", resume=True)
    run.run_evaluations(selected)
    status = json.loads((output / "status.json").read_text())
    assert status["progress"]["total"] == 1 and len(status["checkpoints"]) == 2
    assert status["selected_checkpoints"] == ["global_step_2"]
    assert checked[-1] == (2, output / "global_step_2")
    assert plan_path.read_bytes() == original_plan
    # 本次未选择的旧 checkpoint 已删除时，仍可只恢复其他 checkpoint。
    missing_checkpoint = paths[0].with_suffix(".saved")
    paths[0].rename(missing_checkpoint)
    try:
        assert run.run_evaluations(selected) == output
        assert plan_path.read_bytes() == original_plan
        with pytest.raises(FileNotFoundError):
            run.run_evaluations(arguments(
                tmp_path, [paths[0]], eval_id="20261029-205812", resume=True,
            ))
        assert plan_path.read_bytes() == original_plan
    finally:
        missing_checkpoint.rename(paths[0])
    assert plan_path.read_bytes() == original_plan

    # 新选择必须先通过原实验配置校验，失败不能扩写计划或创建结果。
    incompatible = checkpoints(directory, (18,))[0]
    state = torch.load(incompatible, weights_only=False)
    state["config"]["seed"] += 1
    torch.save(state, incompatible)
    with pytest.raises(ValueError, match="settings differ|configuration"):
        run.run_evaluations(arguments(
            tmp_path, [incompatible], eval_id="20261029-205812", resume=True,
        ))
    assert plan_path.read_bytes() == original_plan
    assert not (output / "global_step_18").exists()
    incompatible.unlink()

    # 指定新项与旧项只调度本次选择，计划保存按 step 排序去重后的并集。
    extended = arguments(tmp_path, [added, paths[1]], eval_id="20261029-205812", resume=True)
    assert run.run_evaluations(extended) == output
    status = json.loads((output / "status.json").read_text())
    assert status["selected_checkpoints"] == ["global_step_2", "global_step_17"]
    assert status["progress"]["total"] == 2 and len(status["checkpoints"]) == 3
    extended_plan = {
        **plan, "checkpoints": [str(path.resolve()) for path in (paths[1], paths[0], added)],
    }
    assert json.loads(plan_path.read_text()) == extended_plan
    extended_bytes = plan_path.read_bytes()
    assert run.run_evaluations(resumed) == output
    assert checked[-3:] == [(step, output / f"global_step_{step}") for step in (2, 10, 17)]
    assert plan_path.read_bytes() == extended_bytes
    with pytest.raises(ValueError, match="duplicate"):
        run.run_evaluations(arguments(tmp_path, [paths[0], paths[0]]))
    for name in (str(paths[0]), "other/global_step_3", "global_step_2/state.pt"):
        with pytest.raises(ValueError, match="checkpoint.*name|global_step"):
            run.run_evaluations(arguments(tmp_path, paths, checkpoint=[name]))
        with pytest.raises(ValueError, match="checkpoint.*name|global_step"):
            run.run_evaluations(arguments(
                tmp_path, paths, eval_id="20261029-205812", resume=True, checkpoint=[name],
            ))
        assert plan_path.read_bytes() == extended_bytes

    for options, message in (
        ({"eval_id": "bad\\id"}, "eval-id must be a nonempty directory name"),
        ({"eval_task": "benchmark"}, "eval-task"),
        ({"lengths": [8192]}, "requires --eval-task ruler"),
        ({"eval_task": "ruler", "max_test_samples": None}, "require --eval-task test"),
        ({"eval_task": "ruler", "max_new_tokens": 9}, "require --eval-task test"),
        ({"eval_task": "locomo", "lengths": [8192]}, "requires --eval-task ruler"),
        ({"resume": True, "eval_run": str(directory.parent)}, "resume.*eval-id"),
        ({"eval_run": None}, "requires --eval-run"),
        ({"resume": True, "eval_run": None, "eval_id": "20261029-205812"}, "requires --eval-run"),
        ({"resume": True, "overwrite": True, "eval_id": "20261029-205812"}, "mutually exclusive"),
    ):
        with pytest.raises(ValueError, match=message):
            run.run_evaluations(arguments(tmp_path, paths, **options))
    missing_task = arguments(
        tmp_path, None, eval_run=str(directory.parent), eval_id="20261029-205812", resume=True,
    )
    del missing_task.eval_task
    with pytest.raises(ValueError, match="resume.*eval-task"):
        run.run_evaluations(missing_task)

    benchmark = arguments(tmp_path, paths, eval_task="ruler", lengths=[16384, 8192])
    benchmark_output = run.run_evaluations(benchmark)
    benchmark_plan = directory.parent / "plan/eval-20261029-205812/ruler.json"
    assert json.loads(benchmark_plan.read_text())["lengths"] == [8192, 16384]
    original_benchmark_plan = benchmark_plan.read_bytes()
    for lengths in ([32768, 16384, 8192], [8192]):
        fixed = arguments(
            tmp_path, paths, eval_run=str(directory.parent), eval_task="ruler", resume=True,
            eval_id="20261029-205812", lengths=lengths,
            max_test_samples=7, max_new_tokens=99, eval_data=str(tmp_path / "ignored-data"),
        )
        assert run.run_evaluations(fixed) == benchmark_output
        assert benchmark_plan.read_bytes() == original_benchmark_plan
        assert requested[-1][1]["lengths"] == [8192, 16384]
        assert requested[-1][1]["max_samples"] is None
        assert requested[-1][1]["max_new_tokens"] is None

    monkeypatch.setattr(run, "datetime", SimpleNamespace(
        now=lambda zone: datetime(2026, 10, 29, 20, 58, 13, tzinfo=zone),
    ))
    all_output = run.run_evaluations(arguments(tmp_path, None, eval_run=str(directory.parent)))
    assert all_output == directory.parent / "eval/20261029-205813/test"
    all_plan = json.loads((directory.parent / "plan/eval-20261029-205813/test.json").read_text())
    assert all_plan["checkpoints"] == [str(path.resolve()) for path in (
        paths[1], paths[0], directory / "checkpoints/global_step_17/state.pt",
    )]
    assert run.run_evaluations(resumed) == output
    first = arguments(tmp_path, paths, eval_id="new-without-data")
    del first.eval_data
    with pytest.raises(ValueError, match="eval-data"):
        run.run_evaluations(first)
    assert not (directory.parent / "plan/eval-new-without-data/test.json").exists()
    orphan = directory.parent / "eval/orphan/test"
    orphan.mkdir(parents=True)
    (orphan / "keep").write_text("result without plan")
    with pytest.raises(ValueError, match="results already exist"):
        run.run_evaluations(arguments(tmp_path, paths, eval_id="orphan"))
    with pytest.raises(ValueError, match="plan does not exist"):
        run.run_evaluations(arguments(
            tmp_path, None, eval_run=str(directory.parent), eval_id="orphan", resume=True,
        ))
    assert (orphan / "keep").read_text() == "result without plan"
    assert not (directory.parent / "plan/eval-orphan/test.json").exists()

    # 历史任务缺少 checkpoint 范围时补入当前全部完整项，本次选择仍独立生效。
    historical_plans = []
    current_checkpoints = [str(path.resolve()) for path in (paths[1], paths[0], added)]
    for task, settings, selected_paths in (
        ("test", plan, None),
        ("ruler", json.loads(benchmark_plan.read_text()), [paths[1]]),
    ):
        eval_id = f"historical-{task}"
        historical = {key: value for key, value in settings.items() if key != "checkpoints"}
        historical["eval_id"] = eval_id
        historical_path = directory.parent / "plan" / f"eval-{eval_id}" / f"{task}.json"
        historical_path.parent.mkdir(parents=True)
        historical_path.write_text(json.dumps(historical), encoding="utf-8")
        before = len(checked)
        task_output = run.run_evaluations(arguments(
            tmp_path, selected_paths, eval_run=str(directory.parent), eval_id=eval_id,
            eval_task=task, resume=True, eval_data=str(tmp_path / "ignored-data"),
            max_test_samples=1, max_new_tokens=9, lengths=[32768],
        ))
        selected_steps = (2, 10, 17) if selected_paths is None else (2,)
        assert checked[before:] == [
            (step, task_output / f"global_step_{step}") for step in selected_steps
        ]
        assert json.loads(historical_path.read_text()) == {
            **historical, "checkpoints": current_checkpoints,
        }
        historical_plans.append((task, eval_id, historical_path, historical_path.read_bytes()))

    checkpoints(directory, (20,))
    for task, eval_id, historical_path, original in historical_plans:
        before = len(checked)
        task_output = run.run_evaluations(arguments(
            tmp_path, None, eval_run=str(directory.parent), eval_id=eval_id,
            eval_task=task, resume=True,
        ))
        assert checked[before:] == [
            (step, task_output / f"global_step_{step}") for step in (2, 10, 17)
        ]
        assert historical_path.read_bytes() == original
    parsed = []
    monkeypatch.setattr(run, "run_evaluations", parsed.append)
    run.main(["--eval-run", str(directory.parent), "--checkpoint", paths[0].parent.name,
              "--eval-data", "test", "--max-test-samples", "null", "--overwrite"])
    assert parsed[-1].eval_task == "test" and parsed[-1].max_test_samples is None
    assert parsed[-1].overwrite
    run.main(["--eval-run", str(directory.parent), "--eval-task", "locomo", "--eval-data", "benchmark"])
    assert parsed[-1].eval_task == "locomo" and parsed[-1].eval_data == "benchmark"
    run.main(["--eval-run", str(directory.parent), "--eval-task", "test", "--eval-id", "chosen",
              "--resume", "--checkpoint", "global_step_2"])
    assert parsed[-1].resume and not parsed[-1].overwrite
    assert parsed[-1].eval_run == str(directory.parent)
    assert parsed[-1].checkpoint == ["global_step_2"]
    with pytest.raises(SystemExit):
        run.main(["--eval-run", str(directory.parent), "--eval-id", "chosen", "--resume"])
    for option in ("--task", "--data", "--max-samples", "--eval-type", "--eval-name", "--benchmark",
                   "--test-data", "--benchmark-data"):
        with pytest.raises(SystemExit):
            run.main(["--eval-run", str(directory.parent), option, "invalid"])
    with pytest.raises(SystemExit):
        run.main(["--eval-run", str(directory.parent), "--resume", "--overwrite"])


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
    assert status["eval_run"] == str(paths[0].parents[3])
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
    plan_path = output.parents[2] / "plan/eval-comparison/test.json"
    original_plan = plan_path.read_bytes()
    assert json.loads(original_plan)["checkpoints"] == [str(path.resolve()) for path in (
        paths[2], paths[1], paths[4], paths[3], paths[0],
    )]

    # 改为单卡重试，完整 checkpoint 不重跑，失败 worker 已提交的样本也不重复。
    phase = "retry"
    launched.clear()
    args.resume = True
    args.eval_run = str(paths[0].parents[3])
    args.checkpoint = None
    args.eval_data = str(tmp_path / "ignored-data")
    args.max_test_samples = 1
    args.max_new_tokens = 99
    args.lengths = [8192]
    args.eval_batch_size_per_gpu = 3
    checkpoints(paths[0].parents[2], (17,))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert run.run_evaluations(args) == output
    assert len(launched) == 1 and launched[0][:3] == ("1", 7, 0)
    assert len(launched[0][3]) == 2
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "finished" and status["progress"]["finished"] == 5
    assert status["checkpoints"]["global_step_7"]["samples"]["completed_samples"] == 6
    command = status["checkpoints"]["global_step_7"]["command"]
    assert status["eval_batch_size_per_gpu"] == 3
    assert command[command.index("--eval-batch-size-per-gpu") + 1] == "3"
    assert status["selected_checkpoints"] == [
        f"global_step_{step}" for step in (2, 5, 7, 10, 12)
    ]
    assert not (output / "global_step_17").exists()
    assert plan_path.read_bytes() == original_plan

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
    single.resume = True
    single.eval_run = str(paths[0].parents[3])
    single.checkpoint = None
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
    interrupted_args = arguments(
        tmp_path, paths, output_root=str(tmp_path / "evaluation"), eval_id="interrupted",
        max_test_samples=3, max_new_tokens=9,
    )
    try:
        with pytest.raises(KeyboardInterrupt, match="SIGTERM"):
            run.run_evaluations(interrupted_args)
    finally:
        interrupter.join(timeout=15)
    status = json.loads((output.parent.parent / "interrupted/test/status.json").read_text())
    assert status["status"] == "stopped" and status["progress"]["stopped"] == 4
    assert status["progress"]["finished"] == 1 and killed
    assert all(process.poll() is not None for process in processes)
    assert all(signal.getsignal(signum) == handler for signum, handler in original.items())


def test_overwrite_rebuilds_only_current_task_and_keeps_validation_before_deletion(tmp_path, monkeypatch):
    directory = tmp_path / "artifacts/method_01/train"
    paths = checkpoints(directory, (10, 2))
    (tmp_path / "data").mkdir()
    computed = []

    def cached(checkpoint, data, output, **settings):
        path = output / "summary.json"
        return json.loads(path.read_text()) if path.exists() else None

    def compute(results, summary, status_path, settings):
        pending = [result for result in results if result["status"] == "pending"]
        computed.append((settings["task"], [result["step"] for result in pending]))
        for result in pending:
            output = Path(result["output_dir"])
            (output / "summary.json").write_text(json.dumps({"settings": settings}))
            (output / "old-result").write_text("completed")
            result.update(status="finished", return_code=0)

    monkeypatch.setattr(run, "load_cached_evaluation", cached)
    monkeypatch.setattr(run, "_compute", compute)
    monkeypatch.setattr(run, "_visible_devices", forbidden)
    test = arguments(tmp_path, paths, eval_id="auto")
    test_output = run.run_evaluations(test)
    test_plan = directory.parent / "plan/eval-auto/test.json"
    test_settings = test_plan.read_bytes()
    ruler = arguments(tmp_path, paths, eval_id="auto", eval_task="ruler",
                      lengths=[8192, 16384])
    ruler_output = run.run_evaluations(ruler)
    ruler_plan = directory.parent / "plan/eval-auto/ruler.json"
    old_settings = ruler_plan.read_bytes()

    # 相同 eval-id 下其他 task 独立；缺少新任务所需数据时，旧计划和结果不动。
    missing_data = arguments(tmp_path, paths, eval_id="auto", overwrite=True,
                             eval_task="ruler", lengths=[8192])
    del missing_data.eval_data
    with pytest.raises(ValueError, match="eval-data"):
        run.run_evaluations(missing_data)
    assert ruler_plan.read_bytes() == old_settings
    assert (ruler_output / "global_step_2/old-result").is_file()
    with pytest.raises(FileNotFoundError):
        run.run_evaluations(arguments(tmp_path, paths, eval_id="auto", overwrite=True,
                                      eval_task="ruler", eval_data=str(tmp_path / "missing-data")))
    assert ruler_plan.read_bytes() == old_settings
    assert (ruler_output / "global_step_2/old-result").is_file()
    resumed = arguments(
        tmp_path, None, eval_run=str(directory.parent), eval_id="auto",
        eval_task="ruler", resume=True, lengths=[8192],
        eval_data=str(tmp_path / "ignored-data"), max_test_samples=1, max_new_tokens=9,
    )
    before = len(computed)
    assert run.run_evaluations(resumed) == ruler_output
    assert len(computed) == before
    assert ruler_plan.read_bytes() == old_settings

    # 覆盖完整 task，一次清理后只调度本次选择，允许重建不同数据与更短长度集合。
    new_data = str(tmp_path / "new-benchmark-data")
    Path(new_data).mkdir()
    overwrite = arguments(tmp_path, [paths[0]], eval_id="auto", overwrite=True,
                          eval_task="ruler", eval_data=new_data,
                          lengths=[8192])
    assert run.run_evaluations(overwrite) == ruler_output
    assert computed == [("test", [2, 10]), ("ruler", [2, 10]), ("ruler", [10])]
    assert not (ruler_output / "global_step_2").exists()
    settings = json.loads(ruler_plan.read_text())
    assert settings["data"] == new_data and settings["lengths"] == [8192]
    assert settings["checkpoints"] == [str(paths[0].resolve())]
    status = json.loads((ruler_output / "status.json").read_text())
    assert set(status["checkpoints"]) == {"global_step_10"}
    assert test_plan.read_bytes() == test_settings
    assert (test_output / "global_step_2/old-result").is_file()

    # auto 是普通标识；覆盖 test 可改预算，不影响同 ID 的 ruler。
    ruler_settings = ruler_plan.read_bytes()
    run.run_evaluations(arguments(tmp_path, paths, eval_id="auto", overwrite=True,
                                  max_new_tokens=9))
    assert computed[-1] == ("test", [2, 10])
    assert json.loads(test_plan.read_text())["max_new_tokens"] == 9
    assert all((test_output / f"global_step_{step}/summary.json").is_file() for step in (2, 10))
    assert ruler_plan.read_bytes() == ruler_settings
    assert (ruler_output / "global_step_10/old-result").is_file()
    reuse = arguments(tmp_path, None, eval_run=str(directory.parent), eval_id="auto", resume=True)
    del reuse.eval_data
    before = len(computed)
    original_test_plan = test_plan.read_bytes()
    run.run_evaluations(reuse)
    assert len(computed) == before
    assert test_plan.read_bytes() == original_test_plan
    with pytest.raises(ValueError, match="plan does not exist"):
        run.run_evaluations(arguments(
            tmp_path, None, eval_run=str(directory.parent), eval_id="auto", eval_task="locomo", resume=True,
        ))
    assert test_plan.read_bytes() == original_test_plan
    assert ruler_plan.read_bytes() == ruler_settings


def test_overwrite_respects_source_target_and_task_locks(tmp_path, monkeypatch):
    directory = tmp_path / "artifacts/method_01/train"
    paths = checkpoints(directory, (2,))
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(run, "load_cached_evaluation", lambda *args, **kwargs: {})
    monkeypatch.setattr(run, "_compute", forbidden)
    source = directory.parent
    target = tmp_path / "evaluation/method_01"
    args = arguments(tmp_path, paths, output_root=str(target.parent), eval_id="same", overwrite=True)
    output = run.run_evaluations(args)
    assert json.loads((output / "status.json").read_text())["eval_run"] == str(source)
    marker = output / "keep-until-replaced"
    marker.write_text("old result")
    for protected in (source, target):
        with run_lock(protected, exclusive=True):
            with pytest.raises(ValueError):
                run.run_evaluations(args)
        assert marker.read_text() == "old result" and paths[0].is_file()
    with evaluation_lock(target, "same", "test"):
        with pytest.raises(ValueError):
            run.run_evaluations(args)
    assert marker.is_file()

    # 同 ID 的其他任务可并行；稳定锁不会随当前 task 结果目录一起删除。
    with evaluation_lock(target, "same", "ruler"):
        assert run.run_evaluations(args) == output
    assert not marker.exists()
    assert paths[0].is_file()
    assert not (source / "eval").exists()
