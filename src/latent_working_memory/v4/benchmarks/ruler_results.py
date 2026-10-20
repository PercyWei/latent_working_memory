"""按 checkpoint × length 提交 RULER 结果，并重建供看板读取的完整汇总。"""

from collections import defaultdict
import json
from pathlib import Path

from latent_working_memory.benchmarks.protocol import data_identity
from latent_working_memory.benchmarks.scoring import summarize_scores
from latent_working_memory.v4.evaluation_records import matching_evaluation_config


def ruler_data_identity(samples, data_dir):
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample.metadata["length"]].append(sample)
    return {
        **data_identity(samples, data_dir, "ruler"),
        "by_length": [
            {
                "length": length,
                **{
                    name: value
                    for name, value in data_identity(group, data_dir, "ruler").items()
                    if name not in {"directory", "source_commit"}
                },
            }
            for length, group in sorted(grouped.items())
        ],
    }


def length_config(config, lengths, samples):
    lengths = sorted(lengths)
    return {
        **config,
        "protocol": {**config["protocol"], "lengths": lengths},
        "data": ruler_data_identity(
            [sample for sample in samples if sample.metadata["length"] in lengths],
            config["data"]["directory"],
        ),
    }


def load_length_summaries(output_dir, config):
    summaries = {}
    for length in config["protocol"]["lengths"]:
        path = Path(output_dir) / "lengths" / str(length) / "summary.json"
        if not path.is_file():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        matching_evaluation_config(summary, config, path)
        if summary["protocol"]["lengths"] != [length]:
            raise ValueError(f"RULER length summary does not match its directory: {path}")
        if summary["checkpoint"] != config["checkpoint"]:
            raise ValueError(f"RULER result belongs to a different checkpoint: {path}")
        summaries[length] = summary
    return summaries


def merge_length_summaries(output_dir, config, summaries, history, samples):
    """只合并原子提交完成的长度；新长度失败时保留已有完整结果。"""
    if not summaries:
        return None
    output_dir = Path(output_dir)
    lengths = sorted(summaries)
    temporary = output_dir / "details.jsonl.tmp"
    records = []
    with temporary.open("w", encoding="utf-8") as stream:
        for length in lengths:
            path = output_dir / "lengths" / str(length) / "details.jsonl"
            with path.open(encoding="utf-8") as source:
                for line in source:
                    records.append(json.loads(line))
                    stream.write(line)
    first = summaries[lengths[0]]
    resource_names = ("evaluation_seconds", "generation_seconds", "write_seconds")
    resources = {
        name: sum(summary["resources"][name] for summary in summaries.values())
        for name in resource_names
    }
    resources.update(
        {name: first["resources"][name] for name in ("scope", "write_measurement")}
    )
    for name in ("peak_memory_allocated_bytes", "peak_memory_reserved_bytes"):
        values = [
            summary["resources"][name]
            for summary in summaries.values() if name in summary["resources"]
        ]
        if values:
            resources[name] = max(values)
    summary = {
        **length_config(config, lengths, samples),
        "memory": first["memory"],
        "samples": len(records),
        "metrics": summarize_scores("ruler", records),
        **{
            name: sum(value[name] for value in summaries.values())
            for name in ("input_tokens", "generated_tokens", "write_events")
        },
        "resources": resources,
    }
    temporary.replace(output_dir / "details.jsonl")
    history.save_summary(summary)
    return summary
