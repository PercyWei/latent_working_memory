"""展示同一 QASPER 划分中各运行的 Answer F1、Judge Accuracy 和逐题回答。"""

from functools import partial
import json

import streamlit as st

from benchmarks.comparison_dashboard import run_comparison, show_judge_accuracy_chart
from benchmarks.dashboard_data import (
    QASPER_ANSWER_TYPE_LABELS as ANSWER_TYPE_LABELS,
    load_details,
    load_judge_results,
    show_judge_answer,
    show_judge_statistics,
)
from benchmarks.qasper.data import SPLIT_FILES
from benchmarks.vendor.qasper import ANSWER_TYPES


TASK_LABELS = {f"qasper/{split}": f"QASPER · {split}" for split in SPLIT_FILES}


def comparison_scores(metrics):
    """统一主图指标顺序；没有题目的答案类型不绘制分数。"""
    groups = {group["answer_type"]: group for group in metrics["groups"]}
    return {"总体 Answer F1": metrics["score"], **{
        f"{ANSWER_TYPE_LABELS[kind]} F1": groups[kind]["score"] if groups[kind]["samples"] else None
        for kind in ANSWER_TYPES
    }}


def load_runs(runs):
    """读取一组（展示标签，已评分的划分目录），按同一划分的 question_id 对齐。"""
    loaded = []
    identity = None
    for label, directory in runs:
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if summary["benchmark"] != "qasper" or summary["metrics"]["metric"] != "answer_f1":
            raise ValueError(f"{label} 需要 QASPER Answer F1 结果。")
        data = summary["data"]
        if summary["split"] not in SPLIT_FILES or data["split"] != summary["split"]:
            raise ValueError(f"{label} 需要 QASPER test 的评分结果。")
        source = (summary["split"], data["filename"], data["source_release"], data["fingerprint"],
                  summary["scoring"]["source_commit"])
        if identity is not None and source != identity:
            raise ValueError("参与比较的运行需要使用相同的 QASPER 划分、数据和评分器。")
        identity = source
        path = directory / "details.jsonl"
        rows = load_details(path, path.stat().st_mtime_ns)
        records = {row["question_id"]: row for row in rows}
        if not records or len(records) != len(rows) or any(not qid for qid in records):
            raise ValueError(f"{label} 的逐题结果包含空题号、重复题号或没有题目。")
        if len(records) != summary["metrics"]["samples"] or len(records) != data["samples"]:
            raise ValueError(f"{label} 的逐题结果没有覆盖完整的已评分题目集合。")
        if loaded:
            first = loaded[0]["records"]
            if records.keys() != first.keys() or any(
                any(row[field] != first[qid][field] for field in ("paper_id", "question", "references"))
                for qid, row in records.items()
            ):
                raise ValueError(f"{label} 的题目、所属论文或参考答案与其他运行不一致。")
        judge = load_judge_results(directory, "qasper", records.keys())
        loaded.append({"label": label, "directory": directory, "summary": summary, "records": records,
                       "judge": judge})
    return loaded


def show_scores(runs, show_chart=True):
    chart, totals, groups = [], [], []
    labels = [run["label"] for run in runs]
    for run in runs:
        label, summary = run["label"], run["summary"]
        metrics = summary["metrics"]
        scores = comparison_scores(metrics)
        if show_chart:
            chart.extend({"运行": label, "指标": metric, "Answer F1": score}
                         for metric, score in scores.items() if score is not None)
        totals.append({"运行": label, "划分": summary["split"], "题数": metrics["samples"],
                       "Answer F1": metrics["score"]})
        by_type = {item["answer_type"]: item for item in metrics["groups"]}
        groups.extend({"运行": label, "答案类型": ANSWER_TYPE_LABELS[kind], "题数": by_type[kind]["samples"],
                       "Answer F1": scores[f"{ANSWER_TYPE_LABELS[kind]} F1"]} for kind in ANSWER_TYPES)
    if show_chart:
        st.subheader("Answer F1 对比")
        st.vega_lite_chart(chart, {
            "encoding": {
                "x": {"field": "指标", "type": "nominal", "sort": list(scores),
                      "axis": {"title": None, "labelAngle": 0}},
                "xOffset": {"field": "运行", "type": "nominal", "sort": labels},
                "y": {"field": "Answer F1", "type": "quantitative", "scale": {"domain": [0, 1]},
                      "axis": {"format": ".0%"}},
                "tooltip": [{"field": "运行"}, {"field": "指标"},
                            {"field": "Answer F1", "type": "quantitative", "format": ".2%"}],
            },
            "layer": [
                {"mark": "bar", "encoding": {
                    "color": {"field": "运行", "type": "nominal",
                              "scale": {"domain": labels, "scheme": "tableau20"},
                              "legend": {"orient": "top", "direction": "vertical", "title": None,
                                         "values": labels, "labelLimit": 0}},
                }},
                {"mark": {"type": "text", "dy": -7, "fontSize": 11},
                 "encoding": {"text": {"field": "Answer F1", "type": "quantitative", "format": ".1%"}}},
            ],
            "height": 360,
        }, width="stretch")
    f1_format = {"Answer F1": st.column_config.NumberColumn(format="percent")}
    for title, rows in (("总体表现", totals), ("分答案类型表现", groups)):
        st.subheader(title)
        st.dataframe(rows, hide_index=True, column_config=f1_format)
    st.caption("每题按最高 F1 对应的人工标注归入一种答案类型；不同运行的各类题数可能不同。无题目的类型不显示分数。")
    if show_chart:
        show_judge_accuracy_chart(runs, color_labels=labels)
    show_judge_statistics(runs)


def show_answers(runs):
    st.subheader("逐题回答")
    questions = runs[0]["records"]
    papers = list(dict.fromkeys(row["paper_id"] for row in questions.values()))
    paper_id = st.selectbox("论文", papers)
    question_ids = [qid for qid, row in questions.items() if row["paper_id"] == paper_id]
    qid = st.selectbox("问题", question_ids, format_func=lambda q: questions[q]["question"])
    row = questions[qid]
    st.text(row["question"])
    st.caption(f"题号 {qid} · 人工参考答案")
    st.code("\n".join(ref["answer"] for ref in row["references"]), language=None, wrap_lines=True)
    for start in range(0, len(runs), 3):
        row_runs = runs[start:start + 3]
        for column, run in zip(st.columns(len(row_runs)), row_runs):
            answer = run["records"][qid]
            with column.container(border=True):
                st.markdown(f"**{run['label']}**")
                st.caption(f"{ANSWER_TYPE_LABELS[answer['answer_type']]} · Answer F1 {answer['score']:.2%}")
                st.code(answer["predicted_answer"], language=None, wrap_lines=True)
                show_judge_answer(run.get("judge"), qid)


def show_runs(runs):
    try:
        loaded = load_runs(runs)
    except (OSError, KeyError, ValueError) as error:
        st.error(str(error))
        return
    show_scores(loaded)
    show_answers(loaded)


def main(argv=None):
    run_comparison("QASPER", load_runs, {"qasper": "QASPER · test"},
                   lambda metrics: {"Answer F1": comparison_scores(metrics)},
                   partial(show_scores, show_chart=False), show_answers, argv)


if __name__ == "__main__":
    main()
