"""按任务比较一个或多个 checkpoint 的全部已评分压缩设置。"""

import argparse
from colorsys import hls_to_rgb
import json
from pathlib import Path
import re

import streamlit as st

from benchmarks.locomo import compare as locomo
from benchmarks.longmemeval import dashboard as longmemeval
from benchmarks.qasper import dashboard as qasper


TASK_LABELS = {
    "locomo": "LoCoMo",
    **{task: f"LongMemEval-{label}" for task, label in longmemeval.SETTINGS.items()},
    **qasper.TASK_LABELS,
}


def natural_key(value):
    """名称中的数字按数值排序，使 checkpoint-20 排在 checkpoint-100 前。"""
    parts = re.split(r"([0-9]+)", value)
    return tuple(int(part) if index % 2 else part.casefold()
                 for index, part in enumerate(parts)), value


def run_dashboard(method, title, render_memory, argv=None):
    """展示所有任务概览，再通过任务及题目选择器对照逐题回答。"""
    parser = argparse.ArgumentParser(description=f"比较 {title} checkpoint 的全部评估设置。")
    parser.add_argument("--output-dir", type=Path, nargs="+", action="extend", required=True,
                        metavar="DIRECTORY", help="一个或多个包含已评分结果的 checkpoint 目录")
    directories = list(dict.fromkeys(path.expanduser().resolve()
                                    for path in parser.parse_args(argv).output_dir))
    for directory in directories:
        if not directory.is_dir():
            parser.error(f"--output-dir 必须指向已有的 checkpoint 目录：{directory}")
    directories.sort(key=lambda path: natural_key(path.name))
    checkpoint_names = [directory.name for directory in directories]
    if len(set(checkpoint_names)) != len(checkpoint_names):
        parser.error("参与比较的 checkpoint 目录名须各不相同。")
    st.set_page_config(page_title=title, layout="wide")
    multiple = len(directories) > 1
    st.title(f"{title} · {'checkpoint 比较' if multiple else checkpoint_names[0]}")
    selected = st.sidebar.multiselect("Checkpoint", checkpoint_names, default=checkpoint_names,
                                      key="checkpoint-directories") if multiple else checkpoint_names
    if not selected:
        st.info("请选择需要比较的 checkpoint。")
        return

    parameters = {
        "icae": ("fixed_mem_size", "mean_compression_rate"),
        "autocompressors": ("segment_length",),
        "sdcc": ("compress_ratio_scale",),
    }[method]
    compression_settings = set()
    grouped = {}
    try:
        for directory in directories:
            for path in directory.glob("*/summary.json"):
                summary = json.loads(path.read_text(encoding="utf-8"))
                task = summary["benchmark"]
                if task == "qasper":
                    task = f"qasper/{summary['split']}"
                if summary["method"] != method or task not in TASK_LABELS:
                    raise ValueError(f"{path.parent} 中的方法或任务与 {title} 不一致。")
                compression_settings.add(tuple(summary["settings"][name] for name in parameters))
                if directory.name in selected:
                    label = f"{directory.name}/{path.parent.name}" if multiple else path.parent.name
                    grouped.setdefault(task, []).append((label, path.parent))
        if not grouped:
            st.info("所选 checkpoint 下未找到已评分的评估结果。")
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

    setting_indices = {setting: index for index, setting in enumerate(sorted(compression_settings))}
    selected_settings = {
        tuple(run["summary"]["settings"][name] for name in parameters)
        for runs in task_runs.values() for run in runs
    }
    checkpoint_indices = {name: index for index, name in enumerate(checkpoint_names)}
    result_count = sum(len(runs) for runs in task_runs.values())
    st.caption(f"已选 {len(selected)} 个 checkpoint，共 {len(selected_settings)} 种压缩设置、"
               f"{result_count} 份结果、{len(task_runs)} 个任务；"
               "各任务展示已评分的全部运行。")
    for task, runs in task_runs.items():
        runs.sort(key=lambda run: (
            tuple(run["summary"]["settings"][name] for name in parameters),
            natural_key(run["directory"].parent.name), natural_key(run["directory"].name),
        ))
        st.header(TASK_LABELS[task])
        colors = {}
        for run in runs:
            setting = tuple(run["summary"]["settings"][name] for name in parameters)
            index = setting_indices[setting]
            hue = (0.60 + index * 0.618) % 1
            if multiple:
                checkpoint_index = checkpoint_indices[run["directory"].parent.name]
                lightness = 0.34 + 0.24 * checkpoint_index / (len(directories) - 1)
                if task == "longmemeval_oracle":
                    lightness += 0.10
            else:
                lightness = 0.66 if task == "longmemeval_oracle" else 0.42
            rgb = hls_to_rgb(hue, lightness, 0.60)
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
