"""按原始问题 ID 对照 LongMemEval-S 和 Oracle 的准确率与逐题回答。"""

from colorsys import hls_to_rgb
import json

import streamlit as st

from benchmarks.comparison_dashboard import run_comparison
from benchmarks.dashboard_data import load_details
from benchmarks.longmemeval.data import QUESTION_TYPES


SETTINGS = {"longmemeval_s_cleaned": "S", "longmemeval_oracle": "Oracle"}
QUESTION_TYPE_LABELS = {
    "single-session-user": "单会话·用户",
    "single-session-preference": "单会话·偏好",
    "single-session-assistant": "单会话·助手",
    "multi-session": "跨会话",
    "temporal-reasoning": "时间推理",
    "knowledge-update": "知识更新",
}


def comparison_scores(metrics):
    """为公共看板和 checkpoint 看板提供相同顺序与标签的主图指标。"""
    groups = {group["question_type"]: group["score"] for group in metrics["groups"]}
    return {"总体准确率": metrics["score"],
            **{QUESTION_TYPE_LABELS[kind]: groups[kind] for kind in QUESTION_TYPES},
            "不可回答": metrics["abstention"]["score"]}


def load_run(label, directory):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary["benchmark"] not in SETTINGS or summary["metrics"]["metric"] != "accuracy":
        raise ValueError(f"{label} 需要 LongMemEval 的评分结果")
    path = directory / "details.jsonl"
    rows = load_details(path, path.stat().st_mtime_ns)
    records = {row["question_id"]: row for row in rows}
    if not rows or len(records) != len(rows) or any(not qid.strip() for qid in records):
        raise ValueError(f"{label} 的逐题结果为空或题号重复")
    if len(rows) != summary["metrics"]["samples"] or len(rows) != summary["data"]["samples"]:
        raise ValueError(f"{label} 的逐题记录与已评分题数不一致")
    return {"label": label, "directory": directory, "summary": summary, "records": records}


def load_runs(runs):
    """读取各设置的结果，核对同一设置的数据身份和各运行的题目对齐关系。"""
    loaded = [load_run(label, directory) for label, directory in runs]
    first = loaded[0]["records"]
    scope_data, scope_records = {}, {}
    for run in loaded:
        summary, records = run["summary"], run["records"]
        scope = summary["benchmark"]
        data = {key: summary["data"][key] for key in
                ("source_commit", "benchmark", "filename", "samples", "fingerprint")}
        if data["benchmark"] != scope or scope_data.setdefault(scope, data) != data:
            raise ValueError(f"{run['label']} 的同设置数据与其他运行不一致")
        same_scope = scope_records.setdefault(scope, records)
        if first.keys() != records.keys() or any(
            row["reference"] != first[qid]["reference"] or
            row["metadata"]["question"] != first[qid]["metadata"]["question"] or
            row["metadata"]["question_type"] != first[qid]["metadata"]["question_type"]
            for qid, row in records.items()
        ):
            raise ValueError(f"{run['label']} 的题目或参考答案与其他运行不一致")
        if any(row["metadata"]["question_date"] != same_scope[qid]["metadata"]["question_date"]
               for qid, row in records.items()):
            raise ValueError(f"{run['label']} 的同设置提问日期与其他运行不一致")
    return loaded


def show_scores(runs, colors=None, show_chart=True):
    chart, palette, groups, table = [], {}, {}, []
    for run in runs:
        label, summary = run["label"], run["summary"]
        metrics = summary["metrics"]
        task = SETTINGS[summary["benchmark"]]
        judge = summary["judge"]["model"]
        scores = comparison_scores(metrics)
        if show_chart:
            if colors is None:
                identity = (summary["method"], json.dumps(summary["settings"], sort_keys=True), judge)
                group = groups.setdefault(identity, len(groups))
                hue = (0.60 + group * 0.618) % 1
                rgb = hls_to_rgb(hue, 0.42 if task == "S" else 0.66, 0.60)
                palette[label] = "#" + "".join(f"{round(v * 255):02x}" for v in rgb)
            else:
                palette[label] = colors[label]
            for scope, score in scores.items():
                if score is not None:
                    chart.append({"运行": label, "设置": task, "指标": scope, "准确率": score})
        entry = {"运行": label, "设置": task, "方法": summary["method"], "裁判": judge, "题数": metrics["samples"],
                 "不可回答题数": metrics["abstention"]["samples"], **scores,
                 "题型宏平均": metrics["task_averaged_accuracy"]}
        table.append(entry)
    if show_chart:
        st.vega_lite_chart(chart, {
            "mark": "bar",
            "encoding": {
                "x": {"field": "指标", "type": "nominal", "sort": list(scores),
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
    st.subheader("准确率统计")
    st.dataframe(table, hide_index=True, column_config={
        name: st.column_config.NumberColumn(format="percent", step=0.0001)
        for name in (*scores, "题型宏平均")
    })
    st.caption("不可回答题属于六类题型中的子集，也计入所属题型及总体准确率。")


def show_answers(runs):
    st.subheader("逐题回答对照")
    records = runs[0]["records"]
    category = st.selectbox("题型", [None, *QUESTION_TYPES],
                            format_func=lambda value: "全部" if value is None else QUESTION_TYPE_LABELS[value])
    abstention = st.checkbox("仅看不可回答题")
    ids = [qid for qid, row in records.items() if (category is None or row["metadata"]["question_type"] == category)
           and (not abstention or "_abs" in qid)]
    if not ids:
        st.info("当前筛选条件下没有题目。")
        return
    qid = st.selectbox("题目", ids, format_func=lambda q: f"{q} · {records[q]['metadata']['question']}")
    st.text(records[qid]["metadata"]["question"])
    st.caption("参考答案")
    st.code(str(records[qid]["reference"]), language=None, wrap_lines=True)
    for start in range(0, len(runs), 3):
        batch = runs[start:start + 3]
        for column, run in zip(st.columns(len(batch)), batch, strict=True):
            row = run["records"][qid]
            with column.container(border=True):
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
    run_comparison(
        "LongMemEval", load_runs, {key: f"LongMemEval-{value}" for key, value in SETTINGS.items()},
        lambda metrics: {"准确率": comparison_scores(metrics)},
        lambda runs: show_scores(runs, show_chart=False), show_answers, argv,
    )


if __name__ == "__main__":
    main()
