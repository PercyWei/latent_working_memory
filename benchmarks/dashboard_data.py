"""读取看板产物并展示公共的裁判统计与逐题判定。"""

from hashlib import sha256
import json

import streamlit as st

from benchmarks.data import LOCOMO_CATEGORIES


QASPER_ANSWER_TYPE_LABELS = {
    "extractive": "抽取", "abstractive": "自由文本", "boolean": "是非", "none": "不可回答",
}


@st.cache_data(show_spinner=False)
def load_details(path, modified_ns):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


@st.cache_data(show_spinner=False)
def details_digest(path, modified_ns):
    return sha256(path.read_bytes()).hexdigest()


def load_judge_results(directory, benchmark, question_ids):
    """读取已评分裁判结果，并确认它关联当前预测与题目集合。"""
    path = directory / "judge_summary.json"
    if not path.is_file():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    details = directory / "details.jsonl"
    if summary["details_sha256"] != details_digest(details, details.stat().st_mtime_ns):
        st.warning(f"{directory} 的逐题结果已变更，已跳过对应的 Judge Accuracy。")
        return None
    if summary["benchmark"] != benchmark or summary["metrics"]["metric"] != "judge_accuracy":
        raise ValueError(f"{path} 的 benchmark 或评分指标不一致。")
    details = directory / "judge_details.jsonl"
    rows = load_details(details, details.stat().st_mtime_ns)
    records = {row["question_id"]: row for row in rows}
    if (len(records) != len(rows) or records.keys() != question_ids
            or len(rows) != summary["metrics"]["samples"]):
        raise ValueError(f"{details} 的题号与评估结果或裁判汇总不一致。")
    return {"summary": summary, "records": records}


def judge_config(judge):
    """返回可比较的评分配置，去掉该运行的产物身份与分数。"""
    return {key: value for key, value in judge["summary"].items()
            if key not in ("data_fingerprint", "details_sha256", "metrics")}


def judge_groups(judge):
    """按产物顺序返回总体及各组裁判得分。"""
    summary = judge["summary"]
    metrics = summary["metrics"]
    benchmark = summary["benchmark"]
    yield "总体", metrics
    for group in metrics["groups"]:
        if benchmark == "locomo":
            label = LOCOMO_CATEGORIES[group["category"]]
        elif benchmark == "qasper":
            label = QASPER_ANSWER_TYPE_LABELS[group["answer_type"]]
        else:
            label = f"{group['hops']}-hop"
        yield label, group


def show_judge_statistics(runs):
    completed = [run for run in runs if run.get("judge")]
    if not completed:
        return
    rows = [
        {"运行": run["label"], "裁判": run["judge"]["summary"]["model"],
         "评估范围": scope, "题数": group["samples"], "Judge Accuracy": group["score"]}
        for run in completed
        for scope, group in judge_groups(run["judge"])
    ]
    st.subheader("Judge Accuracy 统计")
    st.dataframe(rows, hide_index=True,
                 column_config={"Judge Accuracy": st.column_config.NumberColumn(format="percent")})
    if completed[0]["judge"]["summary"]["benchmark"] == "qasper":
        st.caption("答案类型沿用每题 F1 最高的人工标注。裁判独立比较全部人工参考答案。")


def show_judge_answer(judge, question_id):
    if judge is None:
        return
    verdict = judge["records"][question_id]
    label = "正确" if verdict["correct"] else "错误"
    st.caption(f"裁判判定 {label} · {judge['summary']['model']}")
    st.code(verdict["reason"], language=None, wrap_lines=True)
