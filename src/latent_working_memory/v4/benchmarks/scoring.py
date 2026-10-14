"""调用官方评分公式；逐题分数使用 0–1，RULER 仅在任务汇总时取官方精度。"""

from collections import defaultdict

from latent_working_memory.v4.benchmarks.data import LOCOMO_CATEGORIES, RULER_TASKS
from latent_working_memory.v4.benchmarks.vendor.locomo import f1, f1_score
from latent_working_memory.v4.benchmarks.vendor.ruler import (
    postprocess_pred,
    string_match_all,
    string_match_part,
)


def score_sample(benchmark, sample, prediction):
    if benchmark == "ruler":
        prediction = postprocess_pred(prediction)
        metric = string_match_part if sample.metadata["task"].startswith("qa_") else string_match_all
        return metric([prediction], [sample.references]) / 100
    if benchmark == "locomo":
        reference = sample.references[0]
        category = sample.metadata["category"]
        if category == 3:
            reference = reference.split(";")[0].strip()
        return float(f1(prediction, reference) if category == 1 else f1_score(prediction, reference))
    raise ValueError("benchmark must be ruler or locomo")


def summarize_scores(benchmark, records):
    """records 为评估器写出的记录；每条包含 score 与原样保留的 metadata。"""
    if benchmark == "ruler":
        grouped = defaultdict(list)
        for row in records:
            metadata = row["metadata"]
            grouped[(metadata["length"], metadata["task"])].append(row["score"])
        lengths = sorted({length for length, _ in grouped})
        groups, by_length = [], []
        for length in lengths:
            length_groups = []
            for task in RULER_TASKS:
                values = grouped[(length, task)]
                if not values:
                    raise ValueError(f"RULER summary is missing {task} at length {length}")
                group = {
                    "task": task,
                    "length": length,
                    "samples": len(values),
                    "score": round(sum(values) / len(values) * 100, 2) / 100,
                }
                groups.append(group)
                length_groups.append(group)
            by_length.append(
                {
                    "length": length,
                    "samples": sum(group["samples"] for group in length_groups),
                    "score": sum(group["score"] for group in length_groups) / len(RULER_TASKS),
                }
            )
        return {
            "score": sum(row["score"] for row in by_length) / len(by_length),
            "metric": "accuracy",
            "samples": len(records),
            "groups": groups,
            "by_length": by_length,
        }
    if benchmark == "locomo":
        grouped = defaultdict(list)
        for row in records:
            grouped[row["metadata"]["category"]].append(row["score"])
        return {
            "score": sum(row["score"] for row in records) / len(records),
            "metric": "f1",
            "samples": len(records),
            "groups": [
                {
                    "category": category,
                    "category_name": LOCOMO_CATEGORIES[category],
                    "samples": len(values),
                    "score": sum(values) / len(values),
                }
                for category, values in sorted(grouped.items())
            ],
        }
    raise ValueError("benchmark must be ruler or locomo")
