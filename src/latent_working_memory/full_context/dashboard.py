"""独立查看原始模型的 LoCoMo、LongMemEval、QASPER 和 RULER 评估结果。"""

import argparse
import json
from pathlib import Path

import streamlit as st

from benchmarks.locomo.dashboard import load_details, show_answers, show_scores
from benchmarks.longmemeval.dashboard import SETTINGS as LONGMEMEVAL_LABELS, show_runs
from benchmarks.qasper.dashboard import TASK_LABELS as QASPER_LABELS, show_runs as show_qasper


TASK_LABELS = {"locomo": "LoCoMo", "ruler": "RULER",
               **{task: f"LongMemEval-{name}" for task, name in LONGMEMEVAL_LABELS.items()},
               **QASPER_LABELS}


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
    benchmark, _, split = task.partition("/")
    if (summary["method"] != "full-context" or summary["benchmark"] != benchmark
            or (split and summary["split"] != split)):
        st.error(f"该目录不是 Full-context 的 {TASK_LABELS[task]} 评估结果。")
        return
    if task in QASPER_LABELS:
        show_qasper([(TASK_LABELS[task], directory)])
        return
    if task in LONGMEMEVAL_LABELS:
        show_runs([(TASK_LABELS[task], directory)])
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
                        help="模型评估目录 <output_root>/eval/<model_name>，包含已评分的 output_id 子目录")
    directory = parser.parse_args(argv).output_dir.expanduser().resolve()
    if not directory.is_dir():
        parser.error(f"--output-dir 必须指向已有的模型评估目录：{directory}")
    st.set_page_config(page_title="原始模型基线", layout="wide")
    st.title(f"原始模型基线 · {directory.name}")
    grouped = {}
    try:
        for path in sorted(directory.glob("*/summary.json")):
            summary = json.loads(path.read_text(encoding="utf-8"))
            task = summary["benchmark"]
            if task == "qasper":
                task = f"qasper/{summary['split']}"
            if summary["method"] != "full-context" or task not in TASK_LABELS:
                raise ValueError(f"{path.parent} 不是支持的 Full-context 评估结果。")
            grouped.setdefault(task, []).append(path.parent)
    except (OSError, KeyError, ValueError) as error:
        st.error(f"无法读取评估结果。{error}")
        return
    tasks = [task for task in TASK_LABELS if task in grouped]
    if not tasks:
        st.info(f"在 {directory} 下未找到已汇总的评估结果。")
        return
    task = st.selectbox("评估任务", tasks, format_func=TASK_LABELS.__getitem__)
    run_directory = st.selectbox(
        "评估运行（output_id）", grouped[task], format_func=lambda path: path.name,
        key=f"{directory}:{task}:run",
    )
    show_results(run_directory, task)


if __name__ == "__main__":
    main()
