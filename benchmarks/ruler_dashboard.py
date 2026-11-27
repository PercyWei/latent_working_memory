"""对照同一批 RULER 题目的长度、子任务得分和逐题回答。"""

import json

import streamlit as st

from benchmarks.dashboard_data import load_details


def load_runs(runs):
    """读取已评分的运行，核对数据身份并按题号对齐。"""
    loaded = []
    expected_identity = None
    for label, directory in runs:
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        metrics, data = summary["metrics"], summary["data"]
        if summary["benchmark"] != "ruler" or metrics["metric"] != "accuracy":
            raise ValueError(f"{label} 需要 RULER 准确率评估结果。")
        identity = (data["source_commit"], data["samples"], summary["comparison_data"])
        if loaded and identity != expected_identity:
            raise ValueError(f"{label} 与 {loaded[0]['label']} 的 RULER 长度范围或评估数据不一致。")
        expected_identity = identity

        path = directory / "details.jsonl"
        rows = load_details(path, path.stat().st_mtime_ns)
        records = {row["id"]: row for row in rows}
        if not rows or len(records) != len(rows):
            raise ValueError(f"{label} 的 RULER 逐题记录为空或题号重复。")
        if not (len(rows) == metrics["samples"] == data["samples"] == summary["samples"]):
            raise ValueError(f"{label} 的 RULER 逐题记录与汇总题数不一致。")
        if loaded:
            reference = loaded[0]["records"]
            if records.keys() != reference.keys():
                raise ValueError(f"{label} 与 {loaded[0]['label']} 的 RULER 题号范围不一致。")
            for qid, row in records.items():
                if (row["references"] != reference[qid]["references"]
                        or row["metadata"] != reference[qid]["metadata"]):
                    raise ValueError(f"{label} 的 RULER 题目 {qid} 与参考运行的数据不一致。")
        loaded.append({"label": label, "directory": directory, "summary": summary, "records": records})
    return loaded


def comparison_scores(metrics):
    lengths = sorted(metrics["by_length"], key=lambda row: row["length"])
    return {"总体准确率": metrics["score"],
            **{f"{row['length']:,} tokens": row["score"] for row in lengths}}


def show_scores(runs):
    st.subheader("总体与各长度准确率")
    rows = []
    for run in runs:
        metrics = run["summary"]["metrics"]
        rows.append({"运行": run["label"], "范围": "总体", "题数": metrics["samples"], "准确率": metrics["score"]})
        for group in sorted(metrics["by_length"], key=lambda row: row["length"]):
            rows.append({"运行": run["label"], "范围": f"{group['length']:,} tokens",
                         "题数": group["samples"], "准确率": group["score"]})
    score_format = {"准确率": st.column_config.NumberColumn(format="percent")}
    st.dataframe(rows, hide_index=True, column_config=score_format)
    st.caption("各长度准确率按子任务等权平均，总体准确率按长度等权平均。")

    st.subheader("分子任务准确率")
    lengths = sorted(group["length"] for group in runs[0]["summary"]["metrics"]["by_length"])
    length = st.selectbox("长度档（token）", lengths, key="ruler-scores-length")
    st.dataframe([
        {"运行": run["label"], "子任务": group["task"], "题数": group["samples"], "准确率": group["score"]}
        for run in runs for group in run["summary"]["metrics"]["groups"] if group["length"] == length
    ], hide_index=True, column_config=score_format)


def show_answers(runs):
    st.subheader("逐题回答对照")
    reference = runs[0]["records"]
    metrics = runs[0]["summary"]["metrics"]
    lengths = sorted(group["length"] for group in metrics["by_length"])
    length = st.selectbox("长度档（token）", lengths, key="ruler-answers-length")
    tasks = [group["task"] for group in metrics["groups"] if group["length"] == length]
    task = st.selectbox("子任务", [None, *tasks], format_func=lambda value: value or "全部",
                        key=f"ruler-answers-task:{length}")
    ids = [qid for qid, row in reference.items() if row["metadata"]["length"] == length
           and (task is None or row["metadata"]["task"] == task)]
    qid = st.selectbox("题号", ids, key=f"ruler-answers-id:{length}:{task}")
    st.caption("参考答案")
    st.code("\n".join(reference[qid]["references"]), language=None, wrap_lines=True)
    columns = st.columns(min(3, len(runs)))
    for index, run in enumerate(runs):
        row = run["records"][qid]
        with columns[index % len(columns)].container(border=True):
            st.markdown(f"**{run['label']}**")
            st.caption(f"得分 {row['score']:.2%}")
            st.code(row["prediction"], language=None, wrap_lines=True)
