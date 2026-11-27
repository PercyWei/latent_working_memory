"""按原始问题 ID 对照 LongMemEval-S 和 Oracle 的准确率与逐题回答。"""

import argparse
from colorsys import hls_to_rgb
import json
from pathlib import Path

import streamlit as st

from benchmarks.dashboard_data import load_details
from benchmarks.longmemeval.data import QUESTION_TYPES


SETTINGS = {"longmemeval_s_cleaned": "S", "longmemeval_oracle": "Oracle"}


def load_run(label, directory):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary["benchmark"] not in SETTINGS or summary["metrics"]["metric"] != "accuracy":
        raise ValueError(f"{label} 需要 LongMemEval 的评分结果")
    path = directory / "details.jsonl"
    records = {row["question_id"]: row for row in load_details(path, path.stat().st_mtime_ns)}
    if len(records) != summary["metrics"]["samples"] or not records:
        raise ValueError(f"{label} 的逐题记录与已评分题数不一致")
    return {"label": label, "directory": directory, "summary": summary, "records": records}


def load_runs(runs):
    """读取各设置的结果，核对同一设置的数据身份和各运行的题目对齐关系。"""
    loaded = [load_run(label, directory) for label, directory in runs]
    first = loaded[0]["records"]
    fingerprints = {}
    for run in loaded:
        summary, records = run["summary"], run["records"]
        fingerprint = summary["data"]["fingerprint"]
        if fingerprints.setdefault(summary["benchmark"], fingerprint) != fingerprint:
            raise ValueError(f"{run['label']} 的同设置数据与其他运行不一致")
        if first.keys() != records.keys() or any(
            row["reference"] != first[qid]["reference"] or
            row["metadata"]["question"] != first[qid]["metadata"]["question"] or
            row["metadata"]["question_type"] != first[qid]["metadata"]["question_type"]
            for qid, row in records.items()
        ):
            raise ValueError(f"{run['label']} 的题目或参考答案与其他运行不一致")
    return loaded


def show_scores(runs, colors=None, show_chart=True):
    chart, palette, groups, table = [], {}, {}, []
    for run in runs:
        label, summary = run["label"], run["summary"]
        metrics = summary["metrics"]
        task = SETTINGS[summary["benchmark"]]
        judge = summary["judge"]["model"]
        if show_chart:
            if colors is None:
                identity = (summary["method"], json.dumps(summary["settings"], sort_keys=True), judge)
                group = groups.setdefault(identity, len(groups))
                hue = (0.60 + group * 0.618) % 1
                rgb = hls_to_rgb(hue, 0.42 if task == "S" else 0.66, 0.60)
                palette[label] = "#" + "".join(f"{round(v * 255):02x}" for v in rgb)
            else:
                palette[label] = colors[label]
            for scope, score in (("总体", metrics["score"]), ("题型宏平均", metrics["task_averaged_accuracy"]),
                                 ("证据不足", metrics["abstention"]["score"])):
                if score is not None:
                    chart.append({"运行": label, "设置": task, "指标": scope, "准确率": score})
        entry = {"运行": label, "设置": task, "方法": summary["method"], "裁判": judge, "题数": metrics["samples"],
                 "证据不足题数": metrics["abstention"]["samples"]}
        for item in metrics["groups"]:
            entry[item["question_type"]] = item["score"]
        table.append(entry)
    if show_chart:
        st.vega_lite_chart(chart, {
            "mark": "bar",
            "encoding": {
                "x": {"field": "指标", "type": "nominal", "sort": ["总体", "题型宏平均", "证据不足"],
                      "axis": {"labelAngle": 0}},
                "xOffset": {"field": "运行", "sort": list(palette)},
                "y": {"field": "准确率", "type": "quantitative", "scale": {"domain": [0, 1]},
                      "axis": {"format": ".0%"}},
                "color": {"field": "运行", "type": "nominal",
                          "legend": {"values": list(palette)},
                          "scale": {"domain": list(palette), "range": list(palette.values())}},
                "tooltip": [{"field": key} for key in ("运行", "设置", "指标")] +
                           [{"field": "准确率", "type": "quantitative", "format": ".2%"}],
            },
        }, width="stretch")
    st.subheader("分题型准确率")
    st.dataframe(table, hide_index=True, column_config={
        kind: st.column_config.NumberColumn(format="percent", step=0.0001) for kind in QUESTION_TYPES
    })


def show_answers(runs):
    st.subheader("逐题回答对照")
    records = runs[0]["records"]
    category = st.selectbox("题型", [None, *QUESTION_TYPES], format_func=lambda value: value or "全部")
    abstention = st.checkbox("仅看证据不足题")
    ids = [qid for qid, row in records.items() if (category is None or row["metadata"]["question_type"] == category)
           and (not abstention or "_abs" in qid)]
    if not ids:
        st.info("当前筛选条件下没有题目。")
        return
    qid = st.selectbox("题目", ids, format_func=lambda q: f"{q} · {records[q]['metadata']['question']}")
    st.text(records[qid]["metadata"]["question"])
    st.caption("参考答案")
    st.code(str(records[qid]["reference"]), language=None, wrap_lines=True)
    cols = st.columns(min(len(runs), 3))
    for i, run in enumerate(runs):
        row = run["records"][qid]
        with cols[i % len(cols)].container(border=True):
            st.markdown(f"**{run['label']}**")
            st.caption(f"提问日期 {row['metadata']['question_date']} · "
                       f"裁判判定{'正确' if row['autoeval_label']['label'] else '错误'}")
            st.code(row["hypothesis"], language=None, wrap_lines=True)


def show_runs(runs):
    """展示一组（运行标签，已评分设置目录），供公共与各方法看板共用。"""
    try:
        runs = load_runs(runs)
    except (OSError, KeyError, ValueError) as error:
        st.error(f"无法读取评估结果。{error}")
        return
    show_scores(runs)
    with st.expander("运行配置"):
        for run in runs:
            st.markdown(f"**{run['label']}**")
            st.json({"目录": str(run["directory"]), "模型配置": run["summary"]["settings"],
                     "评分配置": run["summary"]["judge"]}, expanded=False)
    show_answers(runs)


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
    st.set_page_config(page_title="LongMemEval · 运行比较", layout="wide")
    st.title("LongMemEval · 运行比较")
    selected = st.sidebar.multiselect("参与比较的运行", paths.keys(), default=list(paths))
    if not selected:
        st.info("请选择需要展示的运行。")
        return
    show_runs([(label, paths[label]) for label in selected])


if __name__ == "__main__":
    main()
