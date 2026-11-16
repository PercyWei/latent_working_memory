"""LaCT 跨执行轮次的 CPU 样本调度与原始记录读取。"""

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path

from batching import parse_batch_size_by_length
from records import (
    BATCHING, LENGTHS, PROTOCOL_FIELDS, build_protocol, load_samples, protocol_identity,
    validate_prediction,
)


ROOT = Path(__file__).resolve().parents[3]


def add_lact_arguments(parser):
    parser.add_argument("--inference-mode", choices=("original", "incremental"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-size-by-length", nargs="+", default=[], metavar="LENGTH=BATCH")
    parser.add_argument("--ruler-data", type=Path)
    parser.add_argument("--lengths", type=int, nargs="+", choices=LENGTHS)


def resolve_protocol_arguments(args, saved=None):
    defaults = saved if saved is not None else {
        "inference_mode": "original", "lengths": list(LENGTHS),
        "ruler_data": str(ROOT / "data/benchmarks/ruler-v1/reproductions/lact-760m"),
    }
    for field in ("inference_mode", "lengths", "ruler_data"):
        if getattr(args, field) is None:
            setattr(args, field, defaults[field])
    args.ruler_data = Path(args.ruler_data).expanduser().resolve()


def execution_parameters(args, devices):
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not args.lengths or len(set(args.lengths)) != len(args.lengths):
        raise ValueError("--lengths must be nonempty and unique")
    mapping = parse_batch_size_by_length(args.batch_size_by_length, args.lengths)
    effective = {str(length): mapping.get(str(length), args.batch_size) for length in args.lengths}
    if args.inference_mode == "original" and any(value > 1 for value in effective.values()):
        raise ValueError("original inference requires effective batch size 1 at every selected length")
    return {
        "devices": devices, "num_shards": len(devices), "batch_size": args.batch_size,
        "batch_size_by_length": mapping, "effective_batch_size_by_length": effective,
        "batching": BATCHING,
    }


def worker_arguments(args):
    arguments = [
        "--inference-mode", args.inference_mode, "--ruler-data", str(args.ruler_data),
        "--lengths", *map(str, args.lengths), "--batch-size", str(args.batch_size),
    ]
    if args.batch_size_by_length:
        arguments.extend(["--batch-size-by-length", *args.batch_size_by_length])
    return arguments


@contextmanager
def evaluation_lock(directory):
    with (directory / ".evaluation.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another LaCT evaluation holds the output lock: {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def reject_live_workers(directory):
    paths = [directory / "execution.json", directory / "launcher_execution.json"]
    paths.extend(sorted(directory.glob("attempts/attempt-*/execution.json")))
    for path in paths:
        if not path.is_file():
            continue
        execution = json.loads(path.read_text())
        attempts = execution["attempts"] if "attempts" in execution else [execution]
        for attempt in attempts:
            for worker in attempt.get("workers", []):
                if worker["status"] != "running":
                    continue
                try:
                    os.kill(worker["pid"], 0)
                except ProcessLookupError:
                    continue
                raise RuntimeError(f"A recorded evaluation worker is still alive: PID {worker['pid']} ({path})")


def check_protocol(plan, protocol, path):
    for field in PROTOCOL_FIELDS:
        if plan[field] != protocol[field]:
            raise ValueError(f"Frozen LaCT evaluation setting differs: {field} ({path})")


def existing_protocol(directory):
    path = directory / "evaluation_plan.json"
    if path.is_file():
        return protocol_identity(json.loads(path.read_text()))
    path = directory / "plan.json"
    plan = json.loads(path.read_text())
    if "worker_arguments" not in plan:
        return protocol_identity(plan)
    # 旧双卡根计划只保存命令；从原命令恢复科学设置，不冻结设备或batch。
    parser = argparse.ArgumentParser()
    add_lact_arguments(parser)
    args = parser.parse_args(plan["worker_arguments"])
    resolve_protocol_arguments(args)
    data = args.ruler_data.expanduser()
    if not data.is_absolute():
        data = ROOT / data
    samples = load_samples({"ruler_data": str(data), "lengths": args.lengths})
    protocol = build_protocol(plan["checkpoint"], data, args.lengths, args.inference_mode, samples)
    protocol["checkpoint_config"] = plan["checkpoint_config"]
    return protocol


def raw_directories(directory):
    candidates = [directory]
    candidates.extend(sorted(directory.glob("shards/shard-*")))
    candidates.extend(sorted(directory.glob("attempts/attempt-*/shards/shard-*")))
    return [path for path in candidates if (path / "predictions.jsonl").is_file()]


def collect_records(directory, protocol, expected, directories=None):
    records, sources = {}, []
    for path in raw_directories(directory) if directories is None else directories:
        plan_path, execution_path = path / "plan.json", path / "execution.json"
        plan = json.loads(plan_path.read_text())
        check_protocol(plan, protocol, plan_path)
        execution = json.loads(execution_path.read_text())
        attempts = execution["attempts"]
        if "assigned_sample_ids" in plan:
            ids = plan["assigned_sample_ids"]
            if len(ids) != len(set(ids)) or not set(ids) <= expected.keys():
                raise ValueError(f"Invalid assigned sample IDs: {plan_path}")
            assigned = {sample_id: expected[sample_id] for sample_id in ids}
        else:
            assigned = {
                sample_id: sample for sample_id, sample in expected.items()
                if sample[2] % plan["num_shards"] == plan["shard_index"]
            }
        seen = []
        ignored_tail = False
        with (path / "predictions.jsonl").open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    if line.endswith(b"\n"):
                        raise ValueError(f"Invalid committed JSON record: {path}:{line_number}") from None
                    ignored_tail = True
                    break
                validate_prediction(record, assigned, plan["shard_index"])
                attempt = record["execution_attempt"]
                if not 0 <= attempt < len(attempts) or attempts[attempt]["execution_attempt"] != attempt:
                    raise ValueError(f"Unknown execution attempt: {record['id']} ({path})")
                if record["generated_tokens"] != len(record["token_ids"]):
                    raise ValueError(f"Generated token count differs: {record['id']} ({path})")
                sample_id = record["id"]
                if sample_id in records:
                    raise ValueError(f"Duplicate original prediction ID: {sample_id} ({path})")
                records[sample_id] = record
                seen.append(sample_id)
        sources.append({
            "directory": str(path), "plan_path": str(plan_path), "execution_path": str(execution_path),
            "source": plan["source"], "num_shards": plan["num_shards"], "shard_index": plan["shard_index"],
            "sample_ids": seen, "samples": len(seen), "ignored_incomplete_tail": ignored_tail,
            "batch_size": plan["batch_size"], "batch_size_by_length": plan.get("batch_size_by_length", {}),
            "batching": plan["batching"],
            "cuda_visible_devices": plan["cuda_visible_devices"],
            "seconds": execution["seconds"],
            "peak_cuda_memory_allocated_bytes": execution["peak_cuda_memory_allocated_bytes"],
        })
    return records, sources
