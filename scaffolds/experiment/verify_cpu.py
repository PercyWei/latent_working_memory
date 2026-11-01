"""用真实 worker、信号中断和恢复验证模板；不向训练入口增加测试专用路径。"""

import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

from .template.execution.state import evaluation_lock, read_checkpoint, run_lock


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            assert_equal(first, second)
    else:
        assert left == right


def verify(directory):
    directory.mkdir(parents=True, exist_ok=True)
    module = f"{__package__}.template.run"
    prefix = [sys.executable, "-m", module, "--device", "cpu"]
    logs = directory / "verification-logs"
    logs.mkdir()

    def execute(name, options, interrupt_when=None, expected_code=0):
        with (logs / f"{name}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([*prefix, *options], stdout=log, stderr=subprocess.STDOUT)
            try:
                if interrupt_when is None:
                    code = process.wait(timeout=180)
                    assert code == expected_code, f"{name} exited {code}; see {log.name}"
                else:
                    deadline = time.monotonic() + 90
                    while not interrupt_when():
                        assert process.poll() is None, (
                            f"{name} finished before interruption; see {log.name}"
                        )
                        if time.monotonic() > deadline:
                            raise TimeoutError(f"{name}: interruption point not reached")
                        time.sleep(0.005)
                    process.send_signal(signal.SIGTERM)
                    assert process.wait(timeout=30) == 130, (
                        f"{name}: expected stopped exit; see {log.name}"
                    )
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                    process.wait(timeout=30)
        print(f"PASS {name}", flush=True)

    def run_path(name):
        return directory / f"mean-regression-h8_{name}"

    def training(name, steps=64):
        return [
            "--output-root",
            str(directory),
            "--run-id",
            name,
            "--cpu-processes",
            "2",
            "--epochs",
            "null",
            "--max-steps",
            str(steps),
            "--micro-batch-size-per-gpu",
            "1",
            "--gradient-accumulation-steps",
            "2",
            "--max-train-samples",
            "5",
            "--save-every",
            "1",
            "--save-total-limit",
            "3",
        ]

    def latest(run):
        return max(
            (run / "train/checkpoints").glob("global_step_*/state.pt"),
            key=lambda path: int(path.parent.name.rsplit("_", 1)[1]),
        )

    execute("training", training("baseline"))
    full = run_path("baseline")
    assert read_json(full / "train/status.json")["status"] == "finished"
    assert not (full / "eval").exists()
    assert not list((full / "plan").glob("eval-*"))

    execute("overwrite-seed", training("overwrite", steps=4))
    overwritten = run_path("overwrite")
    marker = overwritten / "old-result.txt"
    marker.write_text("old")
    shared = directory / "teacher"
    shared.mkdir()
    (shared / "keep.txt").write_text("shared")
    execute("reject-existing-run", training("overwrite", steps=2), expected_code=1)
    assert marker.read_text() == "old"
    with run_lock(overwritten):
        execute(
            "reject-active-run-overwrite",
            [*training("overwrite", steps=2), "--overwrite"],
            expected_code=1,
        )
    assert marker.exists()
    execute("overwrite-training", [*training("overwrite", steps=2), "--overwrite"])
    assert not marker.exists()
    assert read_json(overwritten / "train/summary.json")["step"] == 2
    assert not (overwritten / "train/checkpoints/global_step_4").exists()
    assert (shared / "keep.txt").read_text() == "shared"

    resumed = run_path("resumed")
    execute(
        "interrupt-training",
        training("resumed"),
        lambda: any((resumed / "train/checkpoints").glob("global_step_[3-9]*/state.pt")),
    )
    assert read_json(resumed / "train/status.json")["status"] == "stopped"
    checkpoint = latest(resumed)
    saved_step = read_checkpoint(checkpoint)["cursor"]["step"]
    assert 0 < saved_step < 64
    # 模拟中断留下的末尾片段；恢复只回退 checkpoint 后的记录。
    with (resumed / "train/metrics.jsonl").open("ab") as stream:
        stream.write(b'{"incomplete":')
    # 旧 checkpoint 的已退役编排字段不会再要求自动 test 计划存在。
    retired = {
        "train_only": False,
        "test_data": "/old/test",
        "max_test_samples": 3,
        "max_new_tokens": 128,
        "eval_id": "auto",
    }
    for path in (resumed / "plan/train.json", resumed / "train/config.json"):
        payload = read_json(path)
        (payload["training"] if "training" in payload else payload).update(retired)
        path.write_text(json.dumps(payload))
    payload = torch.load(checkpoint, weights_only=False)
    payload["config"]["training"].update(retired)
    torch.save(payload, checkpoint)
    execute(
        "reject-resume-overwrite",
        ["--resume", "--checkpoint", str(checkpoint), "--overwrite"],
        expected_code=2,
    )
    execute(
        "resume-training", ["--resume", "--checkpoint", str(checkpoint), "--cpu-processes", "2"]
    )
    first, second = read_checkpoint(latest(full)), read_checkpoint(latest(resumed))
    for field in ("model", "optimizer", "scheduler", "cursor", "rng", "method_rng", "binding"):
        assert_equal(first[field], second[field])
    rows = [read_json(full / "train/config.json"), read_json(resumed / "train/config.json")]
    assert rows[0]["training"]["max_steps"] == rows[1]["training"]["max_steps"] == 64
    for run in (full, resumed):
        metrics = [
            json.loads(line) for line in (run / "train/metrics.jsonl").read_text().splitlines()
        ]
        assert [row["step"] for row in metrics] == list(range(1, 65))
        assert [row["train/samples"] for row in metrics[:2]] == [4, 1]  # rank 1 的尾批为空。
    print(
        "PASS uninterrupted/resumed model, optimizer, scheduler, cursor and RNG are identical",
        flush=True,
    )

    data = Path(__file__).parent / "template/datasets"
    execute(
        "evaluate-only",
        [
            "--evaluate-only",
            "--eval-run",
            str(full),
            "--eval-id",
            "auto",
            "--test-data",
            str(data),
            "--cpu-processes",
            "2",
        ],
    )
    statuses = read_json(full / "eval/auto/test/status.json")["checkpoints"]
    assert len(statuses) == 3 and all(row["status"] == "finished" for row in statuses.values())
    auto = full / "eval/auto/test"
    before = (auto / "global_step_64/summary.json").stat().st_mtime_ns
    execute(
        "resume-completed", ["--resume", "--checkpoint", str(latest(full)), "--cpu-processes", "2"]
    )
    assert (auto / "global_step_64/summary.json").stat().st_mtime_ns == before

    # 同一个 ID 下，覆盖当前 task 的全部 checkpoints，其余 task 和计划保留。
    other = full / "eval/auto/locomo/keep.txt"
    other.parent.mkdir(parents=True)
    other.write_text("other task")
    other_plan = full / "plan/eval-auto/locomo.json"
    other_plan.write_text('{"task": "locomo"}')
    selected = ["--evaluate-only", "--checkpoint", str(latest(full)), "--eval-id", "auto"]
    execute(
        "reject-eval-changed-settings",
        [*selected, "--test-data", str(data), "--max-test-samples", "2"],
        expected_code=1,
    )
    execute("reject-overwrite-missing-data", [*selected, "--overwrite"], expected_code=1)
    execute(
        "reject-overwrite-absent-directory",
        [*selected, "--overwrite", "--test-data", str(directory / "absent")],
        expected_code=1,
    )
    assert (auto / "global_step_64/summary.json").stat().st_mtime_ns == before
    with evaluation_lock(full, "auto", "test"):
        execute(
            "reject-active-task-overwrite",
            [*selected, "--overwrite", "--test-data", str(data)],
            expected_code=1,
        )
    execute(
        "overwrite-evaluation",
        [
            *selected,
            "--overwrite",
            "--test-data",
            str(data),
            "--max-test-samples",
            "2",
            "--max-new-tokens",
            "64",
            "--cpu-processes",
            "2",
        ],
    )
    assert [path.name for path in auto.glob("global_step_*")] == ["global_step_64"]
    assert read_json(auto / "global_step_64/summary.json")["metrics"]["samples"] == 2
    assert other.read_text() == "other task" and read_json(other_plan) == {"task": "locomo"}
    before = (auto / "global_step_64/summary.json").stat().st_mtime_ns
    execute("reuse-overwritten-evaluation", [*selected, "--cpu-processes", "1"])
    assert (auto / "global_step_64/summary.json").stat().st_mtime_ns == before
    # 删除最终评估计划后，训练恢复仍独立成立。
    (full / "plan/eval-auto/test.json").unlink()
    execute(
        "resume-without-evaluation-plan",
        ["--resume", "--checkpoint", str(latest(full)), "--cpu-processes", "2"],
    )

    # 较多小样本仅用于留出真实中断窗口，每条仍是一次小型 CPU 前向。
    retry_data = directory / "retry-dataset"
    retry_data.mkdir()
    with (retry_data / "test.jsonl").open("w") as stream:
        for index in range(1024):
            stream.write(
                json.dumps({"id": f"retry-{index}", "values": [1.0, 2.0], "target": 1.5}) + "\n"
            )
    retry_options = [
        "--evaluate-only",
        "--checkpoint",
        str(latest(full)),
        "--eval-id",
        "retry",
        "--test-data",
        str(retry_data),
    ]
    retry = full / "eval/retry/test/global_step_64"
    execute(
        "interrupt-evaluation",
        [*retry_options, "--cpu-processes", "2"],
        lambda: any(path.stat().st_size > 0 for path in (retry / "progress").glob("*.jsonl")),
    )
    committed = {}
    for path in (retry / "progress").glob("*.jsonl"):
        for line in path.read_bytes().split(b"\n")[:-1]:
            entry = json.loads(line)
            committed[entry["record"]["id"]] = entry
    assert 0 < len(committed) < 1024
    with (retry / "progress/worker_0.jsonl").open("ab") as stream:
        stream.write(b'{"incomplete":')
    execute("retry-evaluation", [*retry_options, "--cpu-processes", "1"])
    work = read_json(retry / "work.json")
    assert work["completed_samples"] == len(committed)
    assert work["pending_samples"] == 1024 - len(committed)
    entries = {}
    for path in (retry / "progress").glob("*.jsonl"):
        for line in path.read_text().splitlines():
            entry = json.loads(line)
            entries[entry["record"]["id"]] = entry
    assert len(entries) == 1024
    for sample_id, entry in committed.items():
        assert entries[sample_id] == entry
    before = (retry / "summary.json").stat().st_mtime_ns
    execute("reuse-evaluation", [*retry_options, "--cpu-processes", "2"])
    assert (retry / "summary.json").stat().st_mtime_ns == before
    print(
        f"PASS three modes, overwrite, distributed tail/empty rank and interruption/cache recovery: {directory}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, help="保存验证产物；省略时使用临时目录并自动清理"
    )
    args = parser.parse_args()
    if args.output_root:
        verify(args.output_root.expanduser().resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="experiment-template-") as directory:
            verify(Path(directory).resolve())


if __name__ == "__main__":
    main()
