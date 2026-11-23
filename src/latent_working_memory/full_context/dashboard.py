"""独立查看原始模型的 LoCoMo 和 RULER 评估结果。"""

import argparse
import json
from pathlib import Path

import streamlit as st


TASK_LABELS = {"locomo": "LoCoMo", "ruler": "RULER"}


@st.cache_data(show_spinner=False)
def load_details(path, modified_ns):
    """完成结果被替换时按文件修改时间刷新逐题记录。"""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


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
    metrics = summary["metrics"]
    metric = "F1" if task == "locomo" else "Accuracy"
    left, right = st.columns(2)
    left.metric(f"总体 {metric}", f"{metrics['score']:.2%}")
    right.metric("题数", f"{metrics['samples']:,}")

    groups = metrics["groups"]
    length = None
    if task == "locomo":
        field, label = "category_name", "题型"
    else:
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
        if (length is None or row["metadata"]["length"] == length)
        and (group_name is None or row["metadata"][field] == group_name)
    ]
    def sample_label(index):
        row = records[index]
        if task == "locomo":
            return f"{row['id']} · {row['metadata']['question']}"
        return row["id"]

    selected = st.selectbox("题目", range(len(records)), format_func=sample_label,
                            key=f"{directory}:{length}:{group_name}:sample")
    row = records[selected]
    if task == "locomo":
        st.text(row["metadata"]["question"])
    st.caption(f"{metric} {row['score']:.2%}")
    prediction, references = st.columns(2)
    prediction.markdown("**模型回答**")
    prediction.code(row["prediction"], language=None, wrap_lines=True)
    references.markdown("**参考答案**")
    references.code("\n".join(row["references"]), language=None, wrap_lines=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="单个模型的评估结果目录，其下按 locomo/ 和 ruler/ 保存结果")
    directory = parser.parse_args(argv).output_dir.expanduser().resolve()
    if not directory.is_dir():
        parser.error(f"--output-dir 必须指向已有的模型结果目录：{directory}")
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
