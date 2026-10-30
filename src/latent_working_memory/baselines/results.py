"""跨方法的质量比较；不参与缓存复用或恢复校验。"""

import json
from pathlib import Path


def _quality_conditions(summary):
    names = ["input", "decoding", "generation_budget", "reference"]
    names.extend(["answer_prefix"] if summary["benchmark"] == "ruler"
                 else ["categories", "history_order"])
    return {
        "benchmark": summary["benchmark"],
        "base_model": Path(summary["model"]["model_name_or_path"]).name,
        "metric": summary["metrics"]["metric"],
        "protocol": {name: summary["protocol"][name] for name in names},
    }


def _data_matches(actual, expected, length=None):
    """新结果按内容比较；既有结果仅在原指纹能证明一致的范围内使用。"""
    if "comparison_data" in actual and "comparison_data" in expected:
        left = next(row for row in actual["comparison_data"] if row["length"] == length)
        right = next(row for row in expected["comparison_data"] if row["length"] == length)
        return left == right
    names = ("samples", "fingerprint")
    if all(actual["data"][name] == expected["data"][name] for name in names):
        return True
    if length is None:
        return False
    identities = []
    for summary in (actual, expected):
        data = summary["data"]
        if "by_length" in data:
            data = next(row for row in data["by_length"] if row["length"] == length)
        elif summary["protocol"]["lengths"] != [length]:
            return False
        identities.append({name: data[name] for name in names})
    return identities[0] == identities[1]


def _ruler_scope(summary, lengths):
    by_length = [row for row in summary["metrics"]["by_length"] if row["length"] in lengths]
    groups = [row for row in summary["metrics"]["groups"] if row["length"] in lengths]
    samples = sum(row["samples"] for row in by_length)
    return {
        **summary, "samples": samples,
        "protocol": {**summary["protocol"], "lengths": sorted(lengths)},
        "metrics": {**summary["metrics"], "samples": samples, "groups": groups,
                    "by_length": by_length,
                    "score": sum(row["score"] for row in by_length) / len(by_length)},
        "resources": {**summary["resources"], "lengths": summary["protocol"]["lengths"],
                      "samples": summary["metrics"]["samples"]},
    }


def _benchmark_view(summary, benchmark):
    if summary.get("task") == benchmark:
        return {**summary, "benchmark": benchmark, "samples": summary["metrics"]["samples"],
                "model": {**summary["model"], "attention_implementation": "eager"}}
    if summary["benchmark"] != benchmark:
        raise ValueError(f"expected {benchmark} results")
    return summary


def load_baseline_results(results, benchmark, summaries):
    """按标签顺序加载本次完整展示集，不读取或保存基线绑定。"""
    if not results:
        return {}
    if benchmark not in {"ruler", "locomo"}:
        raise ValueError("baseline results apply only to RULER or LoCoMo")
    methods = {Path(value["training_run"]).name for value in summaries}
    baselines = {}
    labels = set()
    for label, directory in results:
        if not isinstance(label, str) or not label.strip():
            raise ValueError("baseline labels must be nonempty strings")
        if label in methods:
            raise ValueError(f"baseline labels cannot use the main method identity: {label}")
        if label in labels:
            raise ValueError(f"duplicate baseline label: {label}")
        labels.add(label)
        directory = Path(directory).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"baseline results directory does not exist: {directory}")
        path = directory / "summary.json"
        if not path.is_file():
            path = directory / benchmark / "summary.json"
        if not path.is_file():
            print(
                f"[evaluation] baseline {label!r}: no completed {benchmark} results at {path}; "
                "skipping this comparison",
                flush=True,
            )
            continue
        summary = _benchmark_view(json.loads(path.read_text(encoding="utf-8")), benchmark)
        expected = _quality_conditions(summary)
        for value in summaries:
            actual = _quality_conditions(value)
            if actual != expected:
                changed = sorted(
                    name for name in actual.keys() | expected.keys()
                    if actual.get(name) != expected.get(name)
                )
                raise ValueError(f"quality comparison differs at {path}: {', '.join(changed)}")
        if benchmark == "ruler":
            lengths = set(summary["protocol"]["lengths"])
            common = lengths & {length for value in summaries for length in value["protocol"]["lengths"]}
            if not common:
                raise ValueError(f"no common RULER lengths at {path}")
            for value in summaries:
                for length in common & set(value["protocol"]["lengths"]):
                    if not _data_matches(summary, value, length):
                        raise ValueError(f"cannot verify identical RULER samples at length {length}: {path}")
            summary = _ruler_scope(summary, common)
        elif any(not _data_matches(summary, value) for value in summaries):
            raise ValueError(f"cannot verify identical LoCoMo samples: {path}")
        baselines[label] = summary
    return baselines
