"""独立查看原始模型的 LoCoMo 和 RULER 评估结果。"""

import argparse
import json
from pathlib import Path

import streamlit as st

from benchmarks.locomo.dashboard import load_details, show_answers, show_scores


TASK_LABELS = {"locomo": "LoCoMo", "ruler": "RULER"}


def score_table(groups, field, label, metric):
    rows = [
        {label: group[field], "题数": group["samples"], metric: group["score"] * 100}
        for group in groups
    ]
    st.dataframe(rows, hide_index=True, column_config={
        metric: st.column_config.ProgressColumn(format="%.2f%%", min_value=0, max_value=100),
    })


def show_results(directory, task):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary["method"] != "full-context" or summary["benchmark"] != task:
        st.error(f"该目录不是 Full-context 的 {TASK_LABELS[task]} 评估结果。")
        return
    metrics = summary["metrics"]
    if task == "locomo":
        st.caption(f"LoCoMo · {Path(summary['settings']['model']['model_name_or_path']).name}")
        show_scores(metrics)
        show_answers(directory, metrics)
        return
    metric = "Accuracy"
    left, right = st.columns(2)
    left.metric(f"总体 {metric}", f"{metrics['score']:.2%}")
    right.metric("题数", f"{metrics['samples']:,}")

    groups = metrics["groups"]
    score_table(metrics["by_length"], "length", "长度档", metric)
    length = st.selectbox("长度档", [group["length"] for group in metrics["by_length"]],
                          key=f"{directory}:length")
    groups = [group for group in groups if group["length"] == length]
    field, label = "task", "子任务"
    score_table(groups, field, label, metric)

    st.subheader("逐题结果")
    group_name = st.selectbox(label, [None, *(group[field] for group in groups)],
                              format_func=lambda value: value or "全部", key=f"{directory}:{length}:group")
    path = directory / "details.jsonl"
    records = [
        row for row in load_details(path, path.stat().st_mtime_ns)
        if row["metadata"]["length"] == length
        and (group_name is None or row["metadata"][field] == group_name)
    ]
    selected = st.selectbox("题目", range(len(records)), format_func=lambda index: records[index]["id"],
                            key=f"{directory}:{length}:{group_name}:sample")
    row = records[selected]
    st.caption(f"{metric} {row['score']:.2%}")
    prediction, references = st.columns(2)
    prediction.markdown("**模型回答**")
    prediction.code(row["prediction"], language=None, wrap_lines=True)
    references.markdown("**参考答案**")
    references.code("\n".join(row["references"]), language=None, wrap_lines=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="评估运行目录，其下包含 locomo/ 或 ruler/ 结果")
    directory = parser.parse_args(argv).output_dir.expanduser().resolve()
    if not directory.is_dir():
        parser.error(f"--output-dir 必须指向已有的评估运行目录：{directory}")
    st.set_page_config(page_title="原始模型基线", layout="wide")
    st.title(f"原始模型基线 · {directory.name}")
    tasks = [task for task in TASK_LABELS if (directory / task / "summary.json").is_file()]
    if not tasks:
        st.info(f"在 {directory} 下未找到 locomo/summary.json 或 ruler/summary.json。")
        return
    task = st.selectbox("评估任务", tasks, format_func=TASK_LABELS.__getitem__)
    show_results(directory / task, task)


if __name__ == "__main__":
    main()
