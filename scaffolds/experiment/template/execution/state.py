"""原子文件、checkpoint 与逐题进度；对象构造、评分和分布式通信由训练与评估模块负责。"""

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import shutil
import time

import torch


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_details(path, records):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def checkpoint_file(path):
    path = Path(path).expanduser().resolve()
    return path / "state.pt" if path.is_dir() else path


def read_checkpoint(path):
    # 仅用于自己生成、可信的训练 checkpoint；不是外部权重导入接口。
    state = torch.load(checkpoint_file(path), map_location="cpu", weights_only=False)
    if set(state) != {
        "config",
        "model",
        "optimizer",
        "scheduler",
        "cursor",
        "rng",
        "method_rng",
        "binding",
    }:
        raise ValueError("checkpoint fields differ from the execution contract")
    return state


def save_checkpoint(output_dir, state, keep):
    """Rank 0 在完整 optimizer step 后写入；调用方已汇集各 rank 的 RNG。"""
    directory = Path(output_dir) / "checkpoints" / f"global_step_{state['cursor']['step']}"
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "state.tmp"
    torch.save(state, temporary)
    temporary.replace(directory / "state.pt")
    complete = sorted(
        directory.parent.glob("global_step_*/state.pt"),
        key=lambda path: int(path.parent.name.rsplit("_", 1)[1]),
    )
    for path in complete[:-keep]:
        shutil.rmtree(path.parent)
    return directory / "state.pt"


def rewind_metrics(path, step):
    """保留 checkpoint 已提交的连续步数；未换行的末行视为未提交。"""
    kept = []
    for raw in Path(path).read_bytes().split(b"\n")[:-1]:
        row = json.loads(raw)
        if type(row["step"]) is not int:
            raise ValueError("training metric step must be an integer")
        if row["step"] <= step:
            if row["step"] != len(kept) + 1 or "train/loss" not in row:
                raise ValueError("training metrics are not consecutive committed steps")
            kept.append(row)
    if len(kept) != step:
        raise ValueError("training metrics do not cover the checkpoint step")
    write_details(path, kept)


@contextmanager
def scheduler_lock(directory):
    """同一任务仅一个调度器；锁覆盖分工、worker 执行与结果提交。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".scheduler.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"another scheduler is running in {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def prepare_work(output_dir, request, sample_ids, num_workers):
    """固定任务集合的参考路径；调用方持有 scheduler_lock，当前没有 worker。"""
    if num_workers < 1 or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("workers must be positive and sample ids must be unique")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    request_path = directory / "eval.json"
    if request_path.exists():
        saved = json.loads(request_path.read_text(encoding="utf-8"))
        if saved != request:
            raise ValueError("evaluation identity changed; use a new eval-id")
    elif (directory / "summary.json").exists() or any(
        (directory / "progress").glob("worker_*.jsonl")
    ):
        raise ValueError("existing evaluation records are missing their request identity")
    write_json(request_path, request)
    entries = read_progress(directory, sample_ids)
    pending = [sample_id for sample_id in sample_ids if sample_id not in entries]
    count = min(num_workers, len(pending))
    work = {
        "total_samples": len(sample_ids),
        "completed_samples": len(entries),
        "pending_samples": len(pending),
        "workers": [pending[rank::count] for rank in range(count)],
        "started_at": time.time(),
    }
    write_json(directory / "work.json", work)
    return work, entries


def read_progress(output_dir, sample_ids):
    """读取已提交样本，仅修剪中断留下的末尾不完整行。"""
    entries = {}
    expected = set(sample_ids)
    for path in sorted((Path(output_dir) / "progress").glob("worker_*.jsonl")):
        data = path.read_bytes()
        committed = data.rfind(b"\n") + 1
        if committed != len(data):
            with path.open("r+b") as stream:
                stream.truncate(committed)
        for raw in data[:committed].splitlines():
            entry = json.loads(raw)
            sample_id = entry["record"]["id"]
            if sample_id not in expected or sample_id in entries:
                raise ValueError(f"unexpected or duplicate completed sample: {sample_id}")
            entries[sample_id] = entry
    return entries


def completed_result(output_dir, request, sample_ids):
    directory = Path(output_dir)
    path = directory / "summary.json"
    if not path.exists():
        return None
    if json.loads((directory / "eval.json").read_text(encoding="utf-8")) != request:
        raise ValueError("completed evaluation identity differs")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if {key: summary[key] for key in request} != request:
        raise ValueError("completed summary identity differs")
    records = [
        json.loads(line)
        for line in (directory / "details.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if [row["id"] for row in records] != sample_ids:
        raise ValueError("completed details differ from the selected sample order")
    return summary


def commit_results(output_dir, request, sample_ids, entries, metrics, resources):
    """汇总值由任务适配器从 entries 计算；summary 最后提交，作为完成标记。"""
    if len(set(sample_ids)) != len(sample_ids) or set(entries) != set(sample_ids):
        raise ValueError("evaluation samples are incomplete or contain unexpected ids")
    records = [entries[sample_id]["record"] for sample_id in sample_ids]
    if [row["id"] for row in records] != sample_ids:
        raise ValueError("progress ids differ from their records")
    summary = {**request, "metrics": metrics, "resources": resources}
    write_details(Path(output_dir) / "details.jsonl", records)
    write_json(Path(output_dir) / "summary.json", summary)
    return summary
