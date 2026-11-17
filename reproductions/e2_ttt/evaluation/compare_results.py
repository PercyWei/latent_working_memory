#!/usr/bin/env python3
"""从 E²-TTT 评估目录生成与论文 1.3B 模型对比的离线图表。"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reporting.comparison import write_comparison


TASKS = {"niah_single_1": "S-NIAH-1", "niah_single_2": "S-NIAH-2"}
BENCHMARKS = {"squad_completion": "SQuAD completion", "swde": "SWDE"}


def build_comparison(directory):
    plan = json.loads((directory / "evaluation_plan.json").read_text(encoding="utf-8"))
    protocol = plan["protocol"]
    config = protocol["checkpoint_config"]
    paper = json.loads(Path(__file__).with_name("paper_reference.json").read_text(encoding="utf-8"))
    if (config["model_type"] not in paper["rows"]
            or config["hidden_size"] != 2048 or config["num_hidden_layers"] != 24):
        raise ValueError("当前论文参照用于公开 1.3B MLP／SwiGLU 模型")
    reference = paper["rows"][config["model_type"]]
    completed = {}
    for benchmark in ("ruler", *BENCHMARKS):
        path = directory / benchmark / "summary.json"
        if benchmark not in protocol["benchmarks"] or not path.is_file():
            continue
        child = json.loads(path.read_text(encoding="utf-8"))
        if child["completed"]:
            if child["checkpoint"] != protocol["checkpoint"]:
                raise ValueError(f"{benchmark} checkpoint 与计划不同")
            completed[benchmark] = child
    if "ruler" in protocol["benchmarks"]:
        if set(protocol["ruler_tasks"]) != set(TASKS):
            raise ValueError("RULER 对比要求 S-NIAH-1 和 S-NIAH-2")
        selected = set(protocol["ruler_lengths"])
        if not selected <= set(paper["lengths"]):
            raise ValueError("RULER 长度超出论文的 512、1K、2K、4K、8K、16K 范围")
        lengths = [length for length in paper["lengths"] if length in selected]
    else:
        lengths = paper["lengths"]
    ruler = completed["ruler"]["metrics"]["ruler"] if "ruler" in completed else None
    if ruler is not None:
        if set(ruler) != {str(length) for length in lengths}:
            raise ValueError("RULER 汇总长度与评估计划不同")
        total = 0
        for length in lengths:
            if set(ruler[str(length)]) != set(TASKS):
                raise ValueError("RULER 汇总任务与评估计划不同")
            total += sum(result["samples"] for result in ruler[str(length)].values())
        if total != completed["ruler"]["samples"]:
            raise ValueError("RULER 汇总样本数与任务样本数之和不同")
    tasks, benchmarks = [], []
    for key, label in TASKS.items():
        rows = []
        for length in lengths:
            result = ruler[str(length)][key] if ruler is not None else None
            rows.append({
                "length": length, "label": str(length) if length < 1024 else f"{length // 1024}K",
                "ours": result["accuracy_percent"] if result is not None else None,
                "paper_e2": reference[key][paper["lengths"].index(length)],
                "samples": result["samples"] if result is not None else None,
            })
        tasks.append({"key": key, "label": label, "rows": rows})
    rows = []
    for index, length in enumerate(lengths):
        rows.append({
            "length": length, "label": tasks[0]["rows"][index]["label"],
            "ours": sum(task["rows"][index]["ours"] for task in tasks) / len(TASKS) if ruler is not None else None,
            "paper_e2": sum(task["rows"][index]["paper_e2"] for task in tasks) / len(TASKS),
        })
    groups = [{"key": "s_niah_average", "label": "S-NIAH-1／2 算术平均", "rows": rows}]
    for benchmark, label in BENCHMARKS.items():
        result = None
        if benchmark in completed:
            child = completed[benchmark]
            result = child["metrics"][benchmark]
            if result["samples"] != child["samples"]:
                raise ValueError(f"{benchmark} 汇总样本数与任务样本数不同")
        benchmarks.append({
            "label": label, "ours": result["accuracy_percent"] if result is not None else None,
            "paper_e2": reference[benchmark], "samples": result["samples"] if result is not None else None,
        })
    data = {
        "title": "E²-TTT 复现与论文对比",
        "model_label": f"{Path(protocol['checkpoint']).name} · {reference['label']}",
        "references": [{"key": "paper_e2", "label": "论文 " + reference["label"]}],
        "groups": groups, "tasks": tasks, "benchmarks": benchmarks,
        "task_note": "RULER 分组均值为 S-NIAH-1、S-NIAH-2 的算术平均；论文参照由原文单任务数值计算。",
    }
    return write_comparison(directory, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True, help="<output-root>/eval/<output-id> 运行目录")
    args = parser.parse_args()
    print(build_comparison(args.eval_dir.expanduser().resolve()))


if __name__ == "__main__":
    main()
