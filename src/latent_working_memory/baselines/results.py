"""加载本次报告指定的 full-context 结果并校验比较协议。"""

import json
from pathlib import Path

from latent_working_memory.benchmarks.protocol import comparison_config


def load_baseline_results(results, benchmark, summaries):
    """按标签顺序加载本次完整展示集，不读取或保存基线绑定。"""
    if not results:
        return {}
    if benchmark not in {"ruler", "locomo"}:
        raise ValueError("baseline results apply only to RULER or LoCoMo")
    methods = {value.get("method", "attention-matching") for value in summaries} or {
        "attention-matching"
    }
    comparisons = []
    for value in summaries:
        comparison = comparison_config(value)
        if value["benchmark"] == "ruler":
            comparison["data"].pop("by_length", None)
        comparisons.append(comparison)
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
        if directory.name != "eval" or directory.parent.parent.name != "full-context":
            raise ValueError("baseline-results must point to full-context/<run-id>/eval/")
        if not directory.is_dir():
            raise FileNotFoundError(f"baseline results directory does not exist: {directory}")
        path = directory / benchmark / "summary.json"
        if not path.is_file():
            print(
                f"[evaluation] baseline {label!r}: no completed {benchmark} results at {path}; "
                "skipping this comparison",
                flush=True,
            )
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary["method"] != "full-context" or summary["benchmark"] != benchmark:
            raise ValueError(f"{path}: expected full-context results for {benchmark}")
        expected = comparison_config(summary)
        if not comparisons:
            comparisons.append(expected)
        for actual in comparisons:
            if actual != expected:
                changed = sorted(
                    name for name in actual.keys() | expected.keys()
                    if actual.get(name) != expected.get(name)
                )
                raise ValueError(f"baseline comparison differs at {path}: {', '.join(changed)}")
        baselines[label] = summary
    return baselines
