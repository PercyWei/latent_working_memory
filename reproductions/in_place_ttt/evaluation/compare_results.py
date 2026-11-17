#!/usr/bin/env python3
"""从已完成的 RULER 结果生成 In-Place TTT 论文对比图表。"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reporting.comparison import write_comparison


METHOD_ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "niah_single_1": "S-NIAH-1",
    "niah_single_2": "S-NIAH-2",
    "niah_single_3": "S-NIAH-3",
    "niah_multikey_1": "MK-NIAH-1",
    "niah_multikey_2": "MK-NIAH-2",
    "niah_multikey_3": "MK-NIAH-3",
    "niah_multivalue": "MV-NIAH",
    "niah_multiquery": "MQ-NIAH",
    "vt": "VT",
    "cwe": "CWE",
    "fwe": "FWE",
    "qa_1": "QA-1",
    "qa_2": "QA-2",
}


def build_comparison(directory):
    ruler_directory = directory / "ruler"
    summary = json.loads((ruler_directory / "summary.json").read_text(encoding="utf-8"))
    protocol = json.loads((ruler_directory / "evaluation_plan.json").read_text(encoding="utf-8"))["protocol"]
    execution = json.loads((ruler_directory / "launcher_execution.json").read_text(encoding="utf-8"))
    if not summary["completed"] or execution["attempts"][-1]["status"] != "completed":
        raise ValueError("只为已完成的评估生成报告")
    if summary["benchmark"] != "ruler" or summary["score_unit"] != "percent":
        raise ValueError("报告要求 RULER 百分制 summary")
    config = protocol["checkpoint_config"]
    expected = {
        "model_type": "qwen3", "hidden_size": 2560,
        "num_hidden_layers": 36, "use_sliding_window": False,
    }
    if any(config[key] != value for key, value in expected.items()) or any(
        kind != "full_attention" for kind in config["layer_types"]
    ):
        raise ValueError("当前论文参照用于完整 attention 的 Qwen3-4B TTT／baseline 配置")
    paper = json.loads((METHOD_ROOT / "evaluation/paper_reference.json").read_text(encoding="utf-8"))
    variant = "ttt" if config["ttt_mode"] else "baseline"
    reference = paper["rows"][variant]
    lengths = sorted(protocol["lengths"])
    if set(map(int, summary["lengths"])) != set(lengths) or not set(lengths) <= set(paper["lengths"]):
        raise ValueError("汇总长度必须与评估计划一致，且属于论文的 4K 至 256K 长度")
    if set(protocol["tasks"]) != set(TASKS) or any(
        set(summary["lengths"][str(length)]["tasks"]) != set(TASKS) for length in lengths
    ):
        raise ValueError("任务集合与 13 项 RULER 评估不一致")
    samples_per_task = protocol["samples_per_task"]
    if summary["samples"] != len(lengths) * len(TASKS) * samples_per_task:
        raise ValueError("汇总样本数与评估计划不同")

    groups = [{
        "key": "ruler_average", "label": "RULER 13 项任务平均",
        "rows": [{
            "length": length, "label": f"{length // 1024}K",
            "ours": summary["lengths"][str(length)]["score"],
            "paper_inplace": reference["ruler_average"][paper["lengths"].index(length)],
        } for length in lengths],
    }]
    tasks = [{
        "key": key, "label": label,
        "rows": [{
            "length": length, "label": f"{length // 1024}K",
            "ours": summary["lengths"][str(length)]["tasks"][key]["score"],
            "paper_inplace": None, "samples": samples_per_task,
        } for length in lengths],
    } for key, label in TASKS.items()]
    return write_comparison(directory, {
        "title": "In-Place TTT 复现与论文对比",
        "model_label": f"{Path(summary['checkpoint']).name} · Qwen3-4B／{'TTT' if config['ttt_mode'] else 'baseline'}",
        "references": [{"key": "paper_inplace", "label": reference["label"]}],
        "groups": groups, "tasks": tasks, "benchmarks": [],
        "task_note": "论文仅报告 13 项任务的平均分，未提供逐任务结果。",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True, help="<output-root>/eval/<output-id> 运行目录")
    args = parser.parse_args()
    print(build_comparison(args.eval_dir.expanduser().resolve()))


if __name__ == "__main__":
    main()
