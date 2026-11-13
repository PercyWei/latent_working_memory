#!/usr/bin/env python3
"""从已有评估结果生成 LaCT Momentum 760M 的离线对比图表。"""

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
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    execution = json.loads((directory / "execution.json").read_text(encoding="utf-8"))
    # 双卡根目录保存 launcher plan；完整评估配置由合并目录保存。
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
        "num_lact_heads": 2, "num_hidden_layers": 24, "hidden_size": 1536,
        "use_momentum": True, "use_muon": False, "ttt_prenorm": True,
        "ttt_nope": False,
    }
    if any(config[key] != value for key, value in expected.items()):
        raise ValueError("当前论文参照仅用于公开两头 Momentum/RoPE/PreNorm 760M")
    paper = json.loads(
        (METHOD_ROOT / "configs/ruler_paper_reference.json").read_text(encoding="utf-8")
    )
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
                **{name: reference[key][index] for name, reference in paper["rows"].items()},
            }
            if key in TASKS:
                row["samples"] = item["tasks"][key]["samples"]
            rows.append(row)
        return rows

    data = {
        "model_label": Path(summary["checkpoint"]).name,
        "groups": [{"key": key, "label": label, "rows": comparison_rows(key)} for key, label in GROUPS.items()],
        "tasks": [{"key": key, "label": label, "rows": comparison_rows(key)} for key, label in TASKS.items()],
    }
    template = (METHOD_ROOT / "scripts/comparison_template.html").read_text(encoding="utf-8")
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
