"""单 checkpoint 评估身份、原子提交与 RULER 长度增量。"""

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path

from latent_working_memory.benchmarks.data import DEFAULT_RULER_LENGTHS
from latent_working_memory.benchmarks.protocol import data_identity


def normalize_lengths(lengths):
    values = tuple(DEFAULT_RULER_LENGTHS if lengths is None else lengths)
    if not values or any(type(value) is not int or value < 1 for value in values):
        raise ValueError("RULER lengths must be positive integers")
    if len(set(values)) != len(values):
        raise ValueError("RULER lengths must be unique")
    return tuple(sorted(values))


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


def read_details(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


@contextmanager
def evaluation_lock(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def ruler_data_identity(samples, data, lengths):
    return {
        **data_identity(samples, data, "ruler"),
        "by_length": [
            {"length": length, **data_identity(
                [sample for sample in samples if sample.metadata["length"] == length], data, "ruler",
            )}
            for length in lengths
        ],
    }


def length_request(request, length):
    identity = next(row for row in request["data"]["by_length"] if row["length"] == length)
    return {
        **request,
        "data": {**{key: value for key, value in identity.items() if key != "length"},
                 "by_length": [identity]},
        "protocol": {**request["protocol"], "lengths": [length]},
    }


def validate_request(saved, requested):
    """固定计算设置必须一致；RULER 只允许扩展长度，旧长度数据保持不变。"""
    if requested["task"] != "ruler":
        if saved != requested:
            changed = sorted(key for key in saved.keys() | requested.keys()
                             if saved.get(key) != requested.get(key))
            raise ValueError(f"evaluation settings differ: {', '.join(changed)}")
        return
    saved_fixed = {key: value for key, value in saved.items() if key not in {"data", "protocol"}}
    requested_fixed = {key: value for key, value in requested.items() if key not in {"data", "protocol"}}
    if saved_fixed != requested_fixed:
        raise ValueError("evaluation checkpoint, model or run settings differ")
    previous = saved["protocol"]["lengths"]
    if not set(previous).issubset(requested["protocol"]["lengths"]):
        raise ValueError("RULER lengths may only stay the same or expand within an evaluation")
    if (
        {key: value for key, value in saved["protocol"].items() if key != "lengths"}
        != {key: value for key, value in requested["protocol"].items() if key != "lengths"}
    ):
        raise ValueError("evaluation protocol differs")
    identities = {row["length"]: row for row in requested["data"]["by_length"]}
    if any(row != identities[row["length"]] for row in saved["data"]["by_length"]):
        raise ValueError("evaluation data differs at an existing RULER length")


def check_evaluation(output_dir, request):
    path = output_dir / "eval.json"
    if path.exists():
        validate_request(json.loads(path.read_text(encoding="utf-8")), request)
    elif (output_dir / "summary.json").exists():
        raise ValueError(f"{output_dir}: completed evaluation is missing eval.json")


def prepare_evaluation(output_dir, request):
    check_evaluation(output_dir, request)
    write_json(output_dir / "eval.json", request)


def completed_evaluation(output_dir, request):
    path = output_dir / "summary.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    validate_request({key: summary[key] for key in request}, request)
    if not (output_dir / "details.jsonl").is_file():
        raise FileNotFoundError(output_dir / "details.jsonl")
    if request["task"] == "ruler" and summary["protocol"]["lengths"] != request["protocol"]["lengths"]:
        return None
    return summary
