#!/usr/bin/env python3
"""从已有评估结果生成 LaCT Momentum／Muon 760M 的离线论文对比图表。"""

import argparse
import json
from pathlib import Path


METHOD_ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "niah_single_1": "S-NIAH-1",
    "niah_single_2": "S-NIAH-2",
    "niah_single_3": "S-NIAH-3",
    "niah_multikey_1": "MK-NIAH-1",
    "niah_multiquery": "MQ-NIAH",
    "niah_multivalue": "MV-NIAH",
}
GROUPS = {
    "s_niah_average": "S-NIAH-1/2/3 平均",
    "multi_niah_average": "MK/MQ/MV-NIAH 平均",
}


def build_comparison(directory):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (directory / "evaluation_plan.json").is_file():
        plan = json.loads((directory / "evaluation_plan.json").read_text(encoding="utf-8"))
        execution = json.loads((directory / "launcher_execution.json").read_text(encoding="utf-8"))
    else:
        plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
        execution = json.loads((directory / "execution.json").read_text(encoding="utf-8"))
        if "worker_arguments" in plan:
            plan = json.loads((directory / "merged/plan.json").read_text(encoding="utf-8"))
    if execution["attempts"][-1]["status"] != "completed":
        raise ValueError("只为已完成的评估生成报告")
    if summary["benchmark"] != "ruler" or summary["score_unit"] != "percent":
        raise ValueError("报告要求 RULER 百分制 summary")
    if summary["num_shards"] != 1 and summary["shard_index"] is not None:
        raise ValueError("请提供完整评估根目录，而非某一卡的分片目录")
    config = plan["checkpoint_config"]
    expected = {
        "num_hidden_layers": 24, "hidden_size": 1536,
        "lact_chunk_size": 2048, "use_momentum": True,
    }
    if any(config[key] != value for key, value in expected.items()):
        raise ValueError("当前论文参照用于 24 层、hidden size 1536、chunk 2048 的公开 760M 模型")
    # Muon 公开权重同时启用 Momentum；按 use_muon 区分论文参考。
    if config["use_muon"]:
        if config["num_lact_heads"] != 4 or not config["ttt_nope"]:
            raise ValueError("Muon 论文对照支持公开四头 NoPE 的 PreNorm／PostNorm 760M 配置")
        references = [{"key": "paper_muon_four_heads", "label": "论文 Muon 4 heads"}]
        variant = "Muon／NoPE／" + ("PreNorm" if config["ttt_prenorm"] else "PostNorm")
    else:
        if config["num_lact_heads"] != 2 or config["ttt_nope"] or not config["ttt_prenorm"]:
            raise ValueError("Momentum 论文对照支持公开两头 RoPE／PreNorm 760M 配置")
        references = [
            {"key": "paper_two_heads", "label": "论文 2 heads"},
            {"key": "paper_four_heads", "label": "论文 4 heads"},
        ]
        variant = "Momentum／RoPE／PreNorm"
    paper = json.loads(
        (METHOD_ROOT / "evaluation/ruler_paper_reference.json").read_text(encoding="utf-8")
    )
    reference_rows = {reference["key"]: paper["rows"][reference["key"]] for reference in references}
    lengths = paper["lengths"]
    if [row["length"] for row in summary["by_length"]] != lengths:
        raise ValueError("完整对比要求 4K、8K、16K、32K 四个长度")
    total = 0
    for row in summary["by_length"]:
        if set(row["tasks"]) != set(TASKS):
            raise ValueError("任务集合与六项 RULER 评估不一致")
        total += sum(item["samples"] for item in row["tasks"].values())
    if total != summary["samples"]:
        raise ValueError("汇总样本数与各任务样本数之和不同")

    def comparison_rows(key):
        rows = []
        for index, item in enumerate(summary["by_length"]):
            row = {
                "length": item["length"], "label": f"{item['length'] // 1024}K",
                "ours": item["tasks"][key]["accuracy"] if key in TASKS else item[key],
                **{name: reference[key][index] for name, reference in reference_rows.items()},
            }
            if key in TASKS:
                row["samples"] = item["tasks"][key]["samples"]
            rows.append(row)
        return rows

    data = {
        "model_label": (
            f"{Path(summary['checkpoint']).name} · {variant}"
            f"（760M，实际 {config['num_lact_heads']} heads）"
        ),
        "references": references,
        "groups": [{"key": key, "label": label, "rows": comparison_rows(key)} for key, label in GROUPS.items()],
        "tasks": [{"key": key, "label": label, "rows": comparison_rows(key)} for key, label in TASKS.items()],
    }
    template = (METHOD_ROOT / "evaluation/comparison_template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    output = directory / "comparison.html"
    output.write_text(template.replace("__REPORT_DATA__", payload), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    args = parser.parse_args()
    print(build_comparison(args.eval_dir.expanduser().resolve()))


if __name__ == "__main__":
    main()
