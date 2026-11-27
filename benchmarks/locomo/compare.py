"""比较指定 LoCoMo 运行的 F1 和逐题回答。"""

import argparse
import json
from pathlib import Path

import streamlit as st

from benchmarks.data import LOCOMO_CATEGORIES, LOCOMO_COMMIT
from benchmarks.dashboard_data import load_details


def load_run(label, directory):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    metrics = summary["metrics"]
    if summary["data"]["source_commit"] != LOCOMO_COMMIT or metrics["metric"] != "f1":
        raise ValueError(f"{label}：需要 LoCoMo 官方 F1 评估结果。")
    path = directory / "details.jsonl"
    rows = load_details(path, path.stat().st_mtime_ns)
    records = {row["id"]: row for row in rows}
    if not rows or len(records) != len(rows):
        raise ValueError(f"{label}：逐题结果为空或题号重复。")
    if len(rows) != metrics["samples"] or len(rows) != summary["data"]["samples"]:
        raise ValueError(f"{label}：逐题结果与汇总的题数不一致。")
    categories = {group["category"] for group in metrics["groups"]}
    if not categories or not categories.issubset(LOCOMO_CATEGORIES):
        raise ValueError(f"{label}：题型必须属于 LoCoMo 类别 1–4。")
    return {"label": label, "directory": directory, "summary": summary, "records": records}


def validate_runs(runs):
    reference = runs[0]
    expected_data = reference["summary"]["data"]
    expected_groups = {
        group["category"]: group["samples"] for group in reference["summary"]["metrics"]["groups"]
    }
    for run in runs[1:]:
        data = run["summary"]["data"]
        if any(data[key] != expected_data[key] for key in ("source_commit", "samples", "fingerprint")):
            raise ValueError(f"{run['label']} 与 {reference['label']} 的评估数据不一致。")
        groups = {group["category"]: group["samples"] for group in run["summary"]["metrics"]["groups"]}
        if groups != expected_groups or run["records"].keys() != reference["records"].keys():
            raise ValueError(f"{run['label']} 与 {reference['label']} 的题目范围不一致。")
        for sample_id, expected in reference["records"].items():
            actual = run["records"][sample_id]
            if actual["references"] != expected["references"] or any(
                actual["metadata"][key] != expected["metadata"][key]
                for key in ("question", "category", "conversation_id")
            ):
                raise ValueError(f"{run['label']} 的题目 {sample_id} 与参考运行不一致。")


def show_comparison(runs, all_labels, colors=None, show_chart=True):
    st.subheader("F1 对比")
    categories = sorted(group["category"] for group in runs[0]["summary"]["metrics"]["groups"])
    scopes = ["总体", *(LOCOMO_CATEGORIES[category] for category in categories)]
    chart, table = [], []
    for run in runs:
        metrics = run["summary"]["metrics"]
        groups = {group["category"]: group["score"] for group in metrics["groups"]}
        values = [metrics["score"], *(groups[category] for category in categories)]
        if show_chart:
            chart.extend({"运行": run["label"], "评估范围": scope, "F1 (%)": score * 100}
                         for scope, score in zip(scopes, values, strict=True))
        table.append({"运行": run["label"], "题数": metrics["samples"], **{
            scope: score * 100 for scope, score in zip(scopes, values, strict=True)
        }})
    if show_chart:
        color_scale = {"domain": all_labels, "scheme": "tableau20"} if colors is None else {
            "domain": all_labels, "range": [colors[label] for label in all_labels],
        }
        st.vega_lite_chart(chart, {
            "mark": "bar",
            "encoding": {
                "x": {"field": "评估范围", "type": "nominal", "sort": scopes,
                      "axis": {"labelAngle": 0}},
                "xOffset": {"field": "运行", "type": "nominal", "sort": all_labels},
                "y": {"field": "F1 (%)", "type": "quantitative", "scale": {"domain": [0, 100]}},
                "color": {"field": "运行", "type": "nominal",
                          "legend": {"values": all_labels},
                          "scale": color_scale},
                "tooltip": [
                    {"field": "运行", "type": "nominal"},
                    {"field": "评估范围", "type": "nominal"},
                    {"field": "F1 (%)", "type": "quantitative", "format": ".2f"},
                ],
            },
            "height": 360,
        }, width="stretch")
    st.dataframe(table, hide_index=True, column_config={
        scope: st.column_config.NumberColumn(f"{scope} F1 (%)", format="%.2f") for scope in scopes
    })
    st.caption("总体及各题型 F1 均按题平均。")
    category_counts = [
        {"题型": group["category_name"], "题数": group["samples"]}
        for group in runs[0]["summary"]["metrics"]["groups"]
    ]
    if show_chart:
        with st.expander("各题型题数"):
            st.dataframe(category_counts, hide_index=True)
    else:
        st.caption("各题型题数　" + " · ".join(f"{row['题型']} {row['题数']} 题" for row in category_counts))


def show_answer_comparison(runs):
    st.subheader("逐题回答对照")
    records = runs[0]["records"]
    categories = sorted({row["metadata"]["category"] for row in records.values()})
    conversations = list(dict.fromkeys(row["metadata"]["conversation_id"] for row in records.values()))
    left, right = st.columns(2)
    category = left.selectbox(
        "题型", [None, *categories],
        format_func=lambda value: "全部" if value is None else LOCOMO_CATEGORIES[value],
        key="compare-category",
    )
    conversation = right.selectbox(
        "对话", [None, *conversations],
        format_func=lambda value: "全部" if value is None else value,
        key="compare-conversation",
    )
    sample_ids = [sample_id for sample_id, row in records.items()
                  if (category is None or row["metadata"]["category"] == category)
                  and (conversation is None or row["metadata"]["conversation_id"] == conversation)]
    if not sample_ids:
        st.info("当前筛选条件下没有题目。")
        return
    sample_id = st.selectbox(
        "题目", sample_ids,
        format_func=lambda value: f"{value} · {records[value]['metadata']['question']}",
        key=f"compare-question:{category}:{conversation}",
    )
    reference = records[sample_id]
    st.text(reference["metadata"]["question"])
    st.caption("参考答案")
    st.code("\n".join(reference["references"]), language=None, wrap_lines=True)
    columns = st.columns(min(3, len(runs)))
    for index, run in enumerate(runs):
        row = run["records"][sample_id]
        with columns[index % len(columns)].container(border=True):
            st.markdown(f"**{run['label']}**")
            st.caption(f"F1 {row['score']:.2%}")
            st.code(row["prediction"], language=None, wrap_lines=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs="+", action="extend", required=True,
                        metavar="LABEL DIRECTORY",
                        help="连续传入一组或多组标签和目录；目录直接包含 summary.json、details.jsonl")
    args = parser.parse_args(argv)
    if len(args.run) % 2:
        parser.error("--run 必须按“标签 目录”成对传入。")
    paths = {}
    for label, directory in zip(args.run[::2], args.run[1::2], strict=True):
        if not label.strip() or label in paths:
            parser.error("运行标签必须非空且不重复。")
        paths[label] = Path(directory).expanduser().resolve()

    st.set_page_config(page_title="LoCoMo · 运行比较", layout="wide")
    st.title("LoCoMo · 运行比较")
    st.caption("选择运行，比较官方 F1 与同题回答。")
    labels = list(paths)
    selected = st.sidebar.multiselect("参与比较的运行", labels, default=labels, key="compare-runs")
    if not selected:
        st.info("请选择需要展示的运行。")
        return
    try:
        runs = [load_run(label, paths[label]) for label in labels if label in selected]
        validate_runs(runs)
    except (OSError, ValueError, KeyError) as error:
        st.error(f"无法比较这些结果：{error}")
        return
    with st.expander("结果目录"):
        st.dataframe([{"运行": run["label"], "结果目录": str(run["directory"])} for run in runs],
                     hide_index=True)
    show_comparison(runs, labels)
    show_answer_comparison(runs)


if __name__ == "__main__":
    main()
