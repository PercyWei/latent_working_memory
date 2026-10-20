"""从单次 run 的本地评估结果重建 v4 看板，不加载模型或连接实验追踪服务。"""

import argparse
from itertools import islice
import json
import math
from pathlib import Path

import streamlit as st

from latent_working_memory.baselines.results import load_baseline_results
from latent_working_memory.v4.evaluation_records import load_completed_evaluations
from latent_working_memory.v4.reporting import evaluation_charts, ruler_task_chart


TASKS = ("test", "ruler", "locomo")
TASK_LABELS = {"test": "Test", "ruler": "RULER", "locomo": "LoCoMo"}
CHART_LABELS = {
    "evaluation/quality": "NLL / EM / F1",
    "evaluation/nll": "NLL",
    "evaluation/em_f1": "EM / F1",
    "evaluation/write_loss": "Attention Matching loss",
    "evaluation/ruler": "RULER 训练进度",
    "evaluation/ruler_lengths": "RULER 长度总览",
    "evaluation/locomo": "LoCoMo",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="单次运行目录，如 artifacts/v4/<run-id>")
    parser.add_argument(
        "--baseline-results",
        nargs="*",
        default=[],
        metavar="LABEL_OR_PATH",
        help="按“标签 full-context/<run-id>/eval/”成对传入，仅本次展示使用",
    )
    args = parser.parse_args()
    args.run_dir = args.run_dir.expanduser().resolve()
    if not args.run_dir.is_dir():
        parser.error(f"--run-dir 必须是已有的运行目录：{args.run_dir}")
    if len(args.baseline_results) % 2:
        parser.error("--baseline-results 必须按“标签 结果目录”成对传入")
    args.baseline_results = list(zip(args.baseline_results[::2], args.baseline_results[1::2]))
    labels = [label for label, _ in args.baseline_results]
    if any(not label.strip() for label in labels) or len(labels) != len(set(labels)):
        parser.error("baseline 标签必须非空且不能重复")
    return args


def discover_methods(run_dir):
    """只枚举完成标记；读取及协议校验在选中任务时进行。"""
    methods = {}
    for directory in sorted(run_dir.glob("eval/*/*")):
        if directory.is_dir() and any(
            next((directory / task).glob("global_step_*/summary.json"), None) is not None
            for task in TASKS
        ):
            methods.setdefault(directory.parent.name, {})[directory.name] = directory
    return methods


def _baseline_selection(results, task, summaries, key):
    if not results:
        return {}
    selected = st.multiselect(
        "Baseline",
        [label for label, _ in results],
        default=[label for label, _ in results],
        key=f"{key}:baselines",
    )
    selected_results = [(label, path) for label, path in results if label in selected]
    try:
        baselines = load_baseline_results(selected_results, task, summaries)
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.error(f"Baseline 无法比较：{error}。请取消该项后重试。")
        return {}
    for label, directory in selected_results:
        if label not in baselines:
            st.warning(f"Baseline {label} 缺少 {TASK_LABELS[task]} 结果，已跳过：{directory}")
    return baselines


def _summary_rows(task, summaries, baselines):
    rows = []
    for label, summary in baselines.items():
        metrics = summary["metrics"]
        row = {
            "方法": label,
            "累计 step": None,
            "样本": summary["samples"],
            f"{metrics['metric']} (%)": 100 * metrics["score"],
        }
        if task == "ruler":
            row["评估长度"] = ", ".join(
                str(group["length"])
                for group in sorted(metrics["by_length"], key=lambda group: group["length"])
            )
        rows.append(row)
    for summary in summaries:
        metrics = summary["metrics"]
        row = {
            "方法": Path(summary["training_run"]).name,
            "累计 step": summary["checkpoint"]["step"],
            "样本": summary["samples"],
        }
        if task == "test":
            row.update(
                {
                    "NLL": metrics["nll"],
                    "EM": metrics["em"],
                    "F1": metrics["f1"],
                    "Loss before write": metrics["write_loss_before"],
                    "Loss after write": metrics["write_loss"],
                }
            )
        else:
            row[f"{metrics['metric']} (%)"] = 100 * metrics["score"]
            if task == "ruler":
                row["评估长度"] = ", ".join(
                    str(group["length"])
                    for group in sorted(metrics["by_length"], key=lambda group: group["length"])
                )
        rows.append(row)
    return rows


def _details(records, key):
    if not st.toggle("查看逐样本记录", key=f"{key}:details"):
        return
    checkpoints = {
        summary["checkpoint"]["step"]: (directory, summary) for directory, summary in records
    }
    step = st.selectbox(
        "记录 checkpoint",
        list(checkpoints),
        index=len(checkpoints) - 1,
        format_func=lambda value: f"global_step_{value}",
        key=f"{key}:details-checkpoint",
    )
    directory, summary = checkpoints[step]
    path = directory / "details.jsonl"
    if not path.is_file():
        st.warning(f"未找到逐样本记录：{path}")
        return
    page_size = 20
    pages = max(1, math.ceil(summary["samples"] / page_size))
    page = st.number_input(
        "页码",
        min_value=1,
        max_value=pages,
        value=1,
        step=1,
        key=f"{key}:{step}:details-page",
    )
    start = (page - 1) * page_size
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in islice(stream, start, start + page_size)]
    st.caption(f"第 {page} / {pages} 页，每页 {page_size} 条，共 {summary['samples']} 条 · {path}")
    st.json(rows, expanded=1)


def _render_task(directory, task, baseline_results):
    records = load_completed_evaluations(directory / task)
    if not records:
        st.info(f"暂无已完成的 {TASK_LABELS[task]} 评估。")
        return
    key = f"{directory}:{task}"
    if st.toggle("全部 checkpoint", value=True, key=f"{key}:all-checkpoints"):
        selected = records
    else:
        steps = [summary["checkpoint"]["step"] for _, summary in records]
        selected_steps = st.multiselect(
            "Checkpoint",
            steps,
            default=steps,
            format_func=lambda value: f"global_step_{value}",
            key=f"{key}:checkpoints",
        )
        selected = [
            (path, summary)
            for path, summary in records
            if summary["checkpoint"]["step"] in selected_steps
        ]
    if not selected:
        st.info("请选择至少一个 checkpoint。")
        return
    summaries = [summary for _, summary in selected]
    baselines = (
        _baseline_selection(baseline_results, task, summaries, key) if task != "test" else {}
    )
    st.dataframe(_summary_rows(task, summaries, baselines), hide_index=True)
    ruler_length = None
    if task == "ruler" and len(summaries) > 1:
        lengths = sorted(
            {group["length"] for summary in summaries for group in summary["metrics"]["by_length"]}
        )
        ruler_length = st.selectbox(
            "训练进度长度",
            [None, *lengths],
            format_func=lambda value: "Overall" if value is None else f"{value:,} tokens",
            key=f"{key}:progress-length",
        )
    for name, chart in evaluation_charts(summaries, baselines, ruler_length=ruler_length).items():
        st.subheader(CHART_LABELS[name])
        st.echarts_chart(chart, height=460, theme=None, key=f"{key}:{name}")
    if task == "ruler":
        summary = summaries[0]
        if len(summaries) > 1:
            checkpoints = {summary["checkpoint"]["step"]: summary for summary in summaries}
            step = st.selectbox(
                "明细 checkpoint",
                list(checkpoints),
                index=len(checkpoints) - 1,
                format_func=lambda value: f"global_step_{value}",
                key=f"{key}:task-checkpoint",
            )
            summary = checkpoints[step]
            for name, chart in evaluation_charts([summary], baselines).items():
                st.subheader(CHART_LABELS[name])
                st.echarts_chart(
                    chart,
                    height=460,
                    theme=None,
                    key=f"{key}:{step}:{name}",
                )
        lengths = sorted(group["length"] for group in summary["metrics"]["by_length"])
        length = st.selectbox(
            "任务明细长度",
            lengths,
            format_func=lambda value: f"{value:,} tokens",
            key=f"{key}:{summary['checkpoint']['step']}:task-length",
        )
        st.subheader(f"RULER 任务明细 · {length:,} tokens")
        st.echarts_chart(
            ruler_task_chart(summary, length, baselines),
            height=460,
            theme=None,
            key=f"{key}:{summary['checkpoint']['step']}:tasks:{length}",
        )
    with st.expander("评估配置与统计"):
        step = st.selectbox(
            "汇总 checkpoint",
            [summary["checkpoint"]["step"] for summary in summaries],
            index=len(summaries) - 1,
            format_func=lambda value: f"global_step_{value}",
            key=f"{key}:summary-checkpoint",
        )
        st.json(
            next(summary for summary in summaries if summary["checkpoint"]["step"] == step),
            expanded=1,
        )
    _details(selected, key)


def main():
    args = parse_args()
    st.set_page_config(page_title="v4 评估", layout="wide")
    st.title("v4 评估结果")
    run_dir = args.run_dir
    run_id = run_dir.name
    st.sidebar.caption(str(run_dir))
    st.sidebar.button("刷新结果")
    methods = discover_methods(run_dir)
    if not methods:
        st.info(
            f"未找到已完成评估：{run_dir}/eval/*/<eval-id>/"
            "<test|ruler|locomo>/global_step_*/summary.json"
        )
        return
    method = st.sidebar.selectbox("方法", list(methods), key=f"{run_id}:method")
    evaluations = methods[method]
    default_eval_id = f"{run_id}-eval"
    eval_ids = list(evaluations)
    eval_id = st.sidebar.selectbox(
        "Eval ID",
        eval_ids,
        index=eval_ids.index(default_eval_id) if default_eval_id in evaluations else 0,
        key=f"{run_id}:{method}:eval-id",
    )
    directory = evaluations[eval_id]
    st.caption(f"{run_id} / {method} / {eval_id} · {directory}")
    tabs = st.tabs(
        [TASK_LABELS[task] for task in TASKS],
        key=f"{directory}:task",
        on_change="rerun",
    )
    for task, tab in zip(TASKS, tabs):
        if not tab.open:
            continue
        with tab:
            try:
                _render_task(directory, task, args.baseline_results)
            except (OSError, ValueError, KeyError, TypeError) as error:
                st.error(f"{TASK_LABELS[task]} 结果无法展示：{error}")
                st.exception(error)


if __name__ == "__main__":
    main()
