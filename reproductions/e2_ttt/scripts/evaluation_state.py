"""E²-TTT 固定评估协议、全局样本记录及原生生成缓存的 CPU 读写。"""

from contextlib import contextmanager
from datetime import datetime
import fcntl
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

from sqlitedict import SqliteDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RULER_DATA = PROJECT_ROOT / "data/benchmarks/ruler-v1/reproductions/e2-ttt-1.3b"
SQUAD_DATA = PROJECT_ROOT / "data/benchmarks/based-squad"
SWDE_DATA = PROJECT_ROOT / "data/benchmarks/based-swde"
LENGTHS = (512, 1024, 2048, 4096, 8192, 16384)
BENCHMARKS = ("ruler", "squad_completion", "swde")
RULER_TASKS = ("niah_single_1", "niah_single_2")
RECALL_COUNTS = {"squad_completion": 2984, "swde": 1111}


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_record(entrypoint, inference_mode):
    def git(*command):
        return subprocess.run(["git", *command], cwd=PROJECT_ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    status = git("status", "--porcelain=v1").splitlines()
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(status), "git_status": status,
            "entrypoint": str(Path(entrypoint).resolve().relative_to(PROJECT_ROOT)),
            "inference_mode": inference_mode,
            "implementation_path": "reproductions/e2_ttt/" + (
                "optimized" if inference_mode == "optimized" else "upstream/e2_ttt")}


def runtime_versions():
    return {key: version(package) for key, package in (
        ("torch", "torch"), ("transformers", "transformers"),
        ("datasets", "datasets"), ("e2_ttt", "e2-ttt"),
    )}


def build_protocol(checkpoint, inference_mode, benchmarks, lengths, versions):
    checkpoint = Path(checkpoint).expanduser().resolve()
    paths = {"ruler": RULER_DATA,
             "squad_completion": SQUAD_DATA / "data/validation-00000-of-00001.parquet",
             "swde": SWDE_DATA / "data/validation-00000-of-00001.parquet"}
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_config": json.loads((checkpoint / "config.json").read_text()),
        "inference_mode": inference_mode, "benchmarks": list(benchmarks),
        "ruler_lengths": list(lengths) if "ruler" in benchmarks else [],
        "data": {name: str(paths[name]) for name in benchmarks},
        "ruler_tasks": list(RULER_TASKS), "ruler_samples_per_task_length": 500,
        "ruler_generation_tokens": 128, "recall_total_token_limit": 2048,
        "recall_input_token_limit": 2000, "recall_generation_tokens": 48,
        "recall_truncation": "left", "ruler_truncation": "forbidden",
        "prompt_format": "base; RULER input + answer_prefix once",
        "num_fewshot": 0, "dtype": "bfloat16", "bootstrap_iters": 0,
        "seeds": {"random": 0, "numpy": 1234, "torch": 1234, "fewshot": 1234},
        "versions": versions,
    }


def task_specs(protocol):
    specs = {}
    if "ruler" in protocol["benchmarks"]:
        for length in protocol["ruler_lengths"]:
            for name in RULER_TASKS:
                specs[f"{name}_{length}"] = {
                    "prefix": f"ruler/{length}/{name}", "benchmark": "ruler",
                    "length": length, "task_name": name, "samples": 500, "metric": str(length),
                }
    for name, count in RECALL_COUNTS.items():
        if name in protocol["benchmarks"]:
            specs[name] = {"prefix": name, "benchmark": name, "samples": count, "metric": "contains"}
    return specs


def sample_index(protocol):
    return {f"{spec['prefix']}/{index}": (task, index, spec)
            for task, spec in task_specs(protocol).items() for index in range(spec["samples"])}


def validate_sample_ids(ids, expected):
    if (not isinstance(ids, list) or any(not isinstance(item, str) for item in ids)
            or len(ids) != len(set(ids)) or not set(ids) <= expected.keys()):
        raise ValueError("Assigned sample IDs must be a unique JSON list within the evaluation protocol")


def worker_directories(directory):
    return sorted(directory.glob("attempts/attempt-*/workers/worker-*"))


def collect_records(directory, protocol):
    expected, records, sources = sample_index(protocol), {}, []
    for folder in worker_directories(directory):
        plan = json.loads((folder / "plan.json").read_text())
        if plan["protocol"] != protocol:
            raise ValueError(f"Worker evaluation protocol differs: {folder}")
        validate_sample_ids(plan["sample_ids"], expected)
        assigned, ids, ignored_tail = set(plan["sample_ids"]), [], False
        path = folder / "predictions.jsonl"
        if path.exists():
            with path.open("rb") as stream:
                for number, line in enumerate(stream, 1):
                    try:
                        record = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        if line.endswith(b"\n"):
                            raise ValueError(f"Invalid committed prediction: {path}:{number}") from None
                        ignored_tail = True
                        break
                    key = record["id"]
                    if key not in assigned:
                        raise ValueError(f"Prediction was not assigned: {key} ({path})")
                    task, index, spec = expected[key]
                    if (record["task"] != task or type(record["source_doc_id"]) is not int
                            or record["source_doc_id"] != index or record["doc"]["source_doc_id"] != index
                            or record["filter"] != "none" or record["metrics"] != [spec["metric"]]):
                        raise ValueError(f"Invalid native sample identity or metric: {key} ({path})")
                    score = record[spec["metric"]]
                    if not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError(f"Invalid native per-document score: {key} ({path})")
                    if key in records:
                        raise ValueError(f"Duplicate original prediction: {key} ({path})")
                    records[key] = record
                    ids.append(key)
        sources.append({"directory": str(folder.relative_to(directory)), "sample_ids": ids,
                        "samples": len(ids), "ignored_incomplete_tail": ignored_tail})
    return records, sources


def merge_generation_caches(directories, output):
    """原生 key/value 原样复制；历史顺序优先，仅用于尚未落盘样本的恢复。"""
    sources = []
    with SqliteDict(str(output), flag="n") as merged:
        for directory in directories:
            path = directory / "generation_cache.sqlite"
            if not path.exists():
                continue
            added = conflicts = 0
            with SqliteDict(str(path), flag="r") as source:
                for key, value in source.items():
                    if key not in merged:
                        merged[key] = value
                        added += 1
                    elif merged[key] != value:
                        conflicts += 1
            sources.append({"path": str(path), "added": added, "different_existing_responses": conflicts})
        merged.commit()
    return sources


def merge_results(directory, protocol, records, sources):
    expected, specs = sample_index(protocol), task_specs(protocol)
    if records.keys() != expected.keys():
        raise ValueError("Predictions do not cover the complete planned evaluation")
    metrics = {}
    for task, spec in specs.items():
        scores = [records[f"{spec['prefix']}/{index}"][spec["metric"]] for index in range(spec["samples"])]
        result = {"samples": len(scores), "accuracy_percent": 100 * sum(scores) / len(scores)}
        if spec["benchmark"] == "ruler":
            metrics.setdefault("ruler", {}).setdefault(str(spec["length"]), {})[spec["task_name"]] = result
        else:
            metrics[task] = result
    with (directory / "predictions.jsonl").open("w") as stream:
        for key in expected:
            stream.write(json.dumps(records[key], ensure_ascii=False) + "\n")
    write_json(directory / "manifest.json", {"parts": sources})
    summary = {"checkpoint": protocol["checkpoint"], "inference_mode": protocol["inference_mode"],
               "completed": True, "samples": len(records), "metrics": metrics, "manifest": "manifest.json"}
    write_json(directory / "summary.json", summary)
    return summary


@contextmanager
def evaluation_lock(directory):
    with (directory / ".evaluation.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another E²-TTT evaluation holds the output lock: {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def reject_live_workers(directory):
    for path in sorted(directory.glob("attempts/attempt-*/execution.json")):
        execution = json.loads(path.read_text())
        for worker in execution["workers"]:
            if worker["status"] == "running":
                try:
                    os.kill(worker["pid"], 0)
                except ProcessLookupError:
                    continue
                raise RuntimeError(f"A recorded E²-TTT worker is still alive: {worker['pid']} ({path})")
