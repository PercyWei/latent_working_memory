"""按任务比较同一 checkpoint 的全部已评分压缩设置。"""

import argparse
from colorsys import hls_to_rgb
import json
from pathlib import Path

import streamlit as st

from benchmarks.locomo import compare as locomo
from benchmarks.longmemeval import dashboard as longmemeval
from benchmarks.qasper import dashboard as qasper


TASK_LABELS = {
    "locomo": "LoCoMo",
    **{task: f"LongMemEval-{label}" for task, label in longmemeval.SETTINGS.items()},
    **qasper.TASK_LABELS,
}


def run_dashboard(method, title, render_memory, argv=None):
    """展示所有任务概览，再通过任务及题目选择器对照逐题回答。"""
    parser = argparse.ArgumentParser(description=f"比较 {title} checkpoint 的全部评估设置。")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="包含已评分的压缩设置与任务子目录的 checkpoint 目录")
    directory = parser.parse_args(argv).output_dir.expanduser().resolve()
    if not directory.is_dir():
        parser.error(f"--output-dir 必须指向已有的 checkpoint 目录：{directory}")
    st.set_page_config(page_title=title, layout="wide")
    st.title(f"{title} · {directory.name}")

    grouped = {}
    try:
        for path in sorted(directory.glob("*/summary.json")):
            summary = json.loads(path.read_text(encoding="utf-8"))
            task = summary["benchmark"]
            if task == "qasper":
                task = f"qasper/{summary['split']}"
            if summary["method"] != method or task not in TASK_LABELS:
                raise ValueError(f"{path.parent} 中的方法或任务与 {title} 不一致。")
            label = path.parent.name
            grouped.setdefault(task, []).append((label, path.parent))
        if not grouped:
            st.info(f"在 {directory} 下未找到已评分的评估结果。")
            return

        task_runs = {}
        for task in TASK_LABELS:
            if task not in grouped:
                continue
            if task == "locomo":
                runs = [locomo.load_run(label, path) for label, path in grouped[task]]
                locomo.validate_runs(runs)
            elif task in qasper.TASK_LABELS:
                runs = qasper.load_runs(grouped[task])
            else:
                runs = longmemeval.load_runs(grouped[task])
            task_runs[task] = runs
    except (OSError, KeyError, ValueError) as error:
        st.error(f"无法比较评估结果。{error}")
        return

    parameters = {
        "icae": ("fixed_mem_size", "mean_compression_rate"),
        "autocompressors": ("segment_length",),
        "sdcc": ("compress_ratio_scale",),
    }[method]
    compression_settings = sorted({
        tuple(run["summary"]["settings"][name] for name in parameters)
        for runs in task_runs.values() for run in runs
    })
    setting_indices = {setting: index for index, setting in enumerate(compression_settings)}
    result_count = sum(len(runs) for runs in task_runs.values())
    st.caption(f"共 {len(compression_settings)} 种压缩设置、{result_count} 份结果、{len(task_runs)} 个任务；"
               "各任务展示已评分的全部运行。")
    for task, runs in task_runs.items():
        runs.sort(key=lambda run: (
            tuple(run["summary"]["settings"][name] for name in parameters), run["label"],
        ))
        st.header(TASK_LABELS[task])
        colors = {}
        for run in runs:
            setting = tuple(run["summary"]["settings"][name] for name in parameters)
            index = setting_indices[setting]
            hue = (0.60 + index * 0.618) % 1
            rgb = hls_to_rgb(hue, 0.66 if task == "longmemeval_oracle" else 0.42, 0.60)
            colors[run["label"]] = "#" + "".join(f"{round(value * 255):02x}" for value in rgb)
        if task == "locomo":
            locomo.show_comparison(runs, [run["label"] for run in runs], colors)
            render_memory(runs)
        elif task in qasper.TASK_LABELS:
            qasper.show_scores(runs)
        else:
            longmemeval.show_scores(runs, colors)

    st.header("逐题结果")
    task = st.selectbox("评估任务", list(task_runs), format_func=TASK_LABELS.__getitem__,
                        key="checkpoint-task")
    runs = task_runs[task]
    with st.expander("评估配置"):
        for run in runs:
            st.markdown(f"**{run['label']}**")
            config = {"目录": str(run["directory"]), "模型配置": run["summary"]["settings"]}
            if task in longmemeval.SETTINGS:
                config["评分配置"] = run["summary"]["judge"]
            st.json(config, expanded=False)
    if task == "locomo":
        locomo.show_answer_comparison(runs)
    elif task in qasper.TASK_LABELS:
        qasper.show_answers(runs)
    else:
        longmemeval.show_answers(runs)
