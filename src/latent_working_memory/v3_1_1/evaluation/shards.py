"""按样本提交评估进度，并从完整逐题记录汇总结果。"""

from collections import Counter
import json
from pathlib import Path
import time

from benchmarks.locomo.engine import shard_conversations
from benchmarks.comparison import comparison_data
from ..io import write_json
from .diagnostics import summarize_outputs
from .records import (
    check_evaluation, completed_evaluation, length_request, read_details,
    prepare_evaluation, write_details,
)


class IncompleteEvaluationError(RuntimeError):
    def __init__(self, completed_samples, total_samples):
        self.completed_samples = completed_samples
        self.total_samples = total_samples
        super().__init__(f"evaluation incomplete: {completed_samples}/{total_samples} samples completed")


def load_progress(output_dir, request, samples, repair=False):
    """只有调度器准备新一轮时裁掉未提交末行；完整坏行和重复样本直接报错。"""
    output_dir = Path(output_dir)
    expected = {sample.id for sample in samples}
    entries = {}
    for path in sorted((output_dir / "progress").glob("worker_*.jsonl")):
        data = path.read_bytes()
        committed = data.rfind(b"\n") + 1
        if repair and committed != len(data):
            with path.open("r+b") as stream:
                stream.truncate(committed)
        for line in data[:committed].splitlines():
            entry = json.loads(line)
            sample_id = entry["record"]["id"]
            if sample_id not in expected:
                raise ValueError(f"unexpected sample in evaluation progress: {sample_id}")
            if sample_id in entries:
                raise ValueError(f"duplicate sample in evaluation progress: {sample_id}")
            entries[sample_id] = entry
    if request["task"] == "ruler":
        for length in request["protocol"]["lengths"]:
            directory = output_dir / "lengths" / str(length)
            if not (directory / "summary.json").exists():
                continue
            current = length_request(request, length)
            check_evaluation(directory, current)
            summary = completed_evaluation(directory, current)
            for row in read_details(directory / "details.jsonl"):
                sample_id = row["id"]
                if sample_id not in expected:
                    raise ValueError(f"unexpected sample in completed RULER length: {sample_id}")
                if sample_id in entries:
                    if entries[sample_id]["record"] != row:
                        raise ValueError(f"completed RULER sample differs from progress: {sample_id}")
                else:
                    entries[sample_id] = {
                        "record": row, "device": summary["resources"]["device"],
                        "peak_memory_bytes": summary["resources"]["peak_memory_bytes"],
                    }
    return entries


def write_work(output_dir, samples, entries, num_workers, task):
    if type(num_workers) is not int or num_workers < 1:
        raise ValueError("num_workers must be a positive integer")
    pending = [sample.id for sample in samples if sample.id not in entries]
    if task == "locomo":
        remaining = [sample for sample in samples if sample.id not in entries]
        workers = [[sample.id for sample in shard] for shard in shard_conversations(remaining, num_workers)]
    else:
        count = min(num_workers, len(pending))
        workers = [pending[rank::count] for rank in range(count)]
    work = {
        "total_samples": len(samples), "completed_samples": len(entries),
        "pending_samples": len(pending),
        "workers": workers,
        "started_at": time.time(),
    }
    write_json(Path(output_dir) / "work.json", work)
    return work


def _summary(request, entries, wall_seconds, task_module):
    records = [entry["record"] for entry in entries]
    metrics = task_module.summarize(records)
    documents = (list({row["metadata"]["conversation_id"]: row for row in records}.values())
                 if request["task"] == "locomo" else records)
    if "qa_nll" in metrics:
        metrics["objective"] = sum(row["objective"] for row in documents) / len(documents)
    peaks = [entry["peak_memory_bytes"] for entry in entries if entry["peak_memory_bytes"] is not None]
    return {
        **request, "metrics": {
            **metrics,
            "blocks": sum(row["blocks"] for row in documents) / len(documents),
            "memory_slots": sum(row["memory_slots"] for row in documents) / len(documents),
            "recent_tokens": sum(row["recent_tokens"] for row in documents) / len(documents),
            "pending_tokens": sum(row["pending_tokens"] for row in documents) / len(documents),
            "retained_tokens": sum(row["retained_tokens"] for row in documents) / len(documents),
            "retention_ratio": sum(row["retention_ratio"] for row in documents) / len(documents),
            "prefix_retention_ratio": sum(row["prefix_retention_ratio"] for row in documents) / len(documents),
            "memory_cost": sum(row["memory_cost"] for row in documents) / len(documents),
        }, "outputs": summarize_outputs(records),
        "stop_reasons": dict(Counter(answer["stop_reason"] for row in records
                                     for answer in row.get("qas", [row]))),
        "resources": {
            "device": ",".join(sorted({entry["device"] for entry in entries})),
            "seconds": sum(row["seconds"] for row in records),
            "seconds_scope": "sum of completed-sample attributed compute and scoring times; compression and NLL amortized across batch samples, decode attributed to active samples; excludes model loading",
            "wall_seconds": wall_seconds,
            "wall_timing_scope": "current attempt from work allocation to merge; includes model loading; cached samples are not rerun",
            "generated_tokens": sum(row["generated_tokens"] for row in records),
            "generation_seconds": sum(row["generation_seconds"] for row in records),
            "generation_timing_scope": "batched compression, question prefill and autoregressive decode; phases attributed to participating samples, decode only to active samples; memory shared by all QA in one document",
            "nll_seconds": sum(row.get("nll_seconds", 0.0) for row in records),
            "peak_memory_bytes": max(peaks) if peaks else None,
            "peak_memory_scope": "worker peak observed during a batch; shared by all documents or benchmark samples in that batch",
            "batch_sizes": sorted({row["eval_batch_size"] for row in records}),
        },
    }


def _write_results(directory, entries, summary, write_boundary):
    records = [entry["record"] for entry in entries]
    if write_boundary:
        boundaries = [entry["boundary"] for entry in entries if "boundary" in entry]
        documents = (len({row["metadata"]["conversation_id"] for row in records})
                     if summary["task"] == "locomo" else len(records))
        # 从没有诊断的旧进度恢复时，不把局部覆盖的数据发布成整组评估的分布。
        path = directory / "boundary.jsonl"
        if len(boundaries) == documents:
            write_details(path, boundaries)
        else:
            path.unlink(missing_ok=True)
    write_details(directory / "details.jsonl", records)
    write_json(directory / "summary.json", summary)


def commit_results(output_dir, request, samples, entries, task_module):
    """先提交齐全的 RULER 长度；所有样本齐全后才提交根目录完成标记。"""
    output_dir = Path(output_dir)
    work = json.loads((output_dir / "work.json").read_text(encoding="utf-8"))
    wall_seconds = max(0.0, time.time() - work["started_at"])
    if request["task"] == "ruler":
        for length in request["protocol"]["lengths"]:
            selected = [sample for sample in samples if sample.metadata["length"] == length]
            if not all(sample.id in entries for sample in selected):
                continue
            directory = output_dir / "lengths" / str(length)
            directory.mkdir(parents=True, exist_ok=True)
            current = length_request(request, length)
            prepare_evaluation(directory, current)
            if completed_evaluation(directory, current) is not None:
                continue
            ordered = [entries[sample.id] for sample in selected]
            summary = _summary(current, ordered, wall_seconds, task_module)
            summary["comparison_data"] = comparison_data(selected, "ruler")
            _write_results(directory, ordered, summary, write_boundary=True)
    if len(entries) != len(samples):
        raise IncompleteEvaluationError(len(entries), len(samples))
    ordered = [entries[sample.id] for sample in samples]
    summary = _summary(request, ordered, wall_seconds, task_module)
    if request["task"] != "test":
        summary["comparison_data"] = comparison_data(samples, request["task"])
    _write_results(output_dir, ordered, summary, write_boundary=request["task"] != "ruler")
    return summary
