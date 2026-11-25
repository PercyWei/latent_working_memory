"""LaCT 评估的 CPU 数据契约、原生评分与结果记录。"""

from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from evaluation_runtime import write_json as write_json
from benchmarks.vendor.ruler import postprocess_pred, string_match_all
from provenance import UPSTREAM_REVISION


ROOT = Path(__file__).resolve().parents[3]
RULER_DATA = ROOT / "data/benchmarks/ruler-v1/reproductions/lact-760m"


TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multiquery",
    "niah_multivalue",
)


LENGTHS = (4096, 8192, 16384, 32768)


GENERATION_BUDGET = 128


SAMPLES_PER_TASK_LENGTH = 500


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d %H:%M:%S Asia/Shanghai")


def read_samples(path):
    samples = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            prompt = row["input"]
            prefix = row["answer_prefix"]
            references = row["outputs"]
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"{path}:{line_number}: input must be nonempty text")
            if not isinstance(prefix, str):
                raise ValueError(f"{path}:{line_number}: answer_prefix must be text")
            if not isinstance(references, list) or not references:
                raise ValueError(f"{path}:{line_number}: outputs must be a nonempty list")
            if any(not isinstance(value, str) or not value for value in references):
                raise ValueError(f"{path}:{line_number}: outputs must contain nonempty strings")
            samples.append(row)
    if len(samples) != SAMPLES_PER_TASK_LENGTH:
        raise ValueError(f"{path}: expected {SAMPLES_PER_TASK_LENGTH} samples, got {len(samples)}")
    return samples


def validate_prediction(record, expected, shard_index):
    sample_id = record["id"]
    if sample_id not in expected:
        raise ValueError(f"Prediction is outside the planned shard: {sample_id}")
    length, task, position, row = expected[sample_id]
    if (
        record["length"] != length
        or record["task"] != task
        or record["sample_position"] != position
        or record["source_index"] != row["index"]
        or record["references"] != row["outputs"]
        or record["shard_index"] != shard_index
    ):
        raise ValueError(f"Saved prediction does not match the current sample: {sample_id}")
    # 重新计分时直接使用原始 prediction；已保存的 score 不是汇总依据。
    postprocess_pred(record["prediction"])


def build_summary(plan, samples, records, output):
    shard_index, num_shards = plan["shard_index"], plan["num_shards"]
    assigned = set(plan["assigned_sample_ids"]) if "assigned_sample_ids" in plan else None
    positions = {
        key: [
            position
            for position in range(len(rows))
            if (
                f"ruler/{key[0]}/{key[1]}/{position}" in assigned
                if assigned is not None
                else shard_index is None or position % num_shards == shard_index
            )
        ]
        for key, rows in samples.items()
    }
    expected_ids = {
        f"ruler/{length}/{task}/{position}"
        for (length, task), selected in positions.items()
        for position in selected
    }
    if records.keys() != expected_ids:
        raise ValueError("Predictions do not exactly cover the planned samples")
    group_scores = {}
    for (length, task), rows in samples.items():
        scores = [
            string_match_all(
                [postprocess_pred(records[f"ruler/{length}/{task}/{position}"]["prediction"])],
                [rows[position]["outputs"]],
            )
            for position in positions[(length, task)]
        ]
        group_scores[(length, task)] = sum(scores) / len(scores)
    by_length = []
    for length in plan["lengths"]:
        means = [group_scores[(length, task)] for task in TASKS]
        by_length.append(
            {
                "length": length,
                "tasks": {
                    task: {
                        "samples": len(positions[(length, task)]),
                        "accuracy": round(group_scores[(length, task)], 2),
                    }
                    for task in TASKS
                },
                "s_niah_average": round(sum(means[:3]) / 3, 2),
                "multi_niah_average": round(sum(means[3:]) / 3, 2),
                "six_task_average": round(sum(means) / len(means), 2),
            }
        )
    return {
        "completed_at": timestamp(),
        "benchmark": "ruler",
        "plan_path": str(output / "plan.json"),
        "execution_path": str(output / "execution.json"),
        "checkpoint": plan["checkpoint"],
        "batch_size": plan["batch_size"],
        "batching": plan["batching"],
        "num_shards": num_shards,
        "shard_index": shard_index,
        "samples": len(records),
        "score_unit": "percent",
        "task_length_macro_average": round(sum(group_scores.values()) / len(group_scores), 2),
        "by_length": by_length,
        "generation": plan["generation"],
    }


PROTOCOL_FIELDS = (
    "method", "checkpoint", "checkpoint_config", "ruler_data", "lengths", "tasks",
    "samples_per_task_length", "inference_mode", "prompt", "generation",
    "parameter_dtype", "compute_autocast_dtype", "upstream_revision", "scoring",
)
BATCHING = "Descending actual token length across tasks within each RULER length; adjacent batches; left padding to batch maximum"


def load_samples(plan):
    return {
        (length, task): read_samples(
            RULER_DATA / str(length) / task / "test.jsonl"
        )
        for length in plan["lengths"]
        for task in TASKS
    }


def sample_index(samples):
    return {
        f"ruler/{length}/{task}/{position}": (length, task, position, row)
        for (length, task), rows in samples.items()
        for position, row in enumerate(rows)
    }


def build_protocol(checkpoint, lengths, inference_mode, samples):
    checkpoint = Path(checkpoint).expanduser().resolve()
    return {
        "method": "lact",
        "checkpoint": str(checkpoint),
        "checkpoint_config": json.loads((checkpoint / "config.json").read_text()),
        "ruler_data": str(RULER_DATA),
        "lengths": list(lengths),
        "tasks": list(TASKS),
        "samples_per_task_length": [
            {"length": length, "task": task, "samples": len(rows)}
            for (length, task), rows in samples.items()
        ],
        "inference_mode": inference_mode,
        "prompt": "base; input + answer_prefix; tokenizer default special tokens; no truncation",
        "generation": {
            "do_sample": False, "num_beams": 1, "max_new_tokens": GENERATION_BUDGET,
            "use_cache": inference_mode == "optimized", "logits_to_keep": 1,
            "recompute_complete_prefix_per_token": inference_mode == "original",
        },
        "parameter_dtype": "bfloat16",
        "compute_autocast_dtype": None,
        "upstream_revision": UPSTREAM_REVISION,
        "scoring": "NVIDIA RULER string_match_all; case-insensitive reference recall, percent",
    }


def protocol_identity(plan):
    if plan["inference_mode"] not in ("original", "optimized"):
        raise ValueError(f"Unknown LaCT inference mode: {plan['inference_mode']}")
    if plan["ruler_data"] != str(RULER_DATA):
        raise ValueError(f"Saved LaCT RULER path differs from the fixed data path: {plan['ruler_data']}")
    return {key: plan[key] for key in PROTOCOL_FIELDS}
