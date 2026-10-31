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

from .template.execution.state import read_checkpoint


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

    def execute(name, options, interrupt_when=None):
        with (logs / f"{name}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([*prefix, *options], stdout=log, stderr=subprocess.STDOUT)
            try:
                if interrupt_when is None:
                    code = process.wait(timeout=180)
                    assert code == 0, f"{name} exited {code}; see {log.name}"
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

    def training(name, steps=64, train_only=False):
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
            *(["--train-only"] if train_only else []),
        ]

    def latest(run):
        return max(
            (run / "train/checkpoints").glob("global_step_*/state.pt"),
            key=lambda path: int(path.parent.name.rsplit("_", 1)[1]),
        )

    execute("full", training("full"))
    full = run_path("full")
    assert read_json(full / "train/status.json")["status"] == "finished"
    auto = full / "eval/auto/test"
    assert [path.name for path in auto.glob("global_step_*")] == ["global_step_64"]
    assert read_json(auto / "status.json")["status"] == "finished"

    execute("train-only", training("train-only", steps=4, train_only=True))
    train_only = run_path("train-only")
    assert not (train_only / "plan/eval-auto").exists()
    assert not (train_only / "eval").exists()

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
    assert_equal(
        read_json(auto / "global_step_64/summary.json")["metrics"],
        read_json(resumed / "eval/auto/test/global_step_64/summary.json")["metrics"],
    )
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
            "independent",
            "--test-data",
            str(data),
            "--cpu-processes",
            "2",
        ],
    )
    statuses = read_json(full / "eval/independent/test/status.json")["checkpoints"]
    assert len(statuses) == 3 and all(row["status"] == "finished" for row in statuses.values())
    before = (auto / "global_step_64/summary.json").stat().st_mtime_ns
    execute(
        "resume-completed", ["--resume", "--checkpoint", str(latest(full)), "--cpu-processes", "2"]
    )
    assert (auto / "global_step_64/summary.json").stat().st_mtime_ns == before

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
        f"PASS four modes, distributed tail/empty rank, training/evaluation interruption and cache reuse: {directory}",
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
