"""按任务选择并比较模型或 checkpoint 目录下的已评分运行。"""

import argparse
from colorsys import hls_to_rgb
import json
from pathlib import Path
import re

import streamlit as st

from benchmarks import ruler_dashboard as ruler
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


def select_runs(available, method, task):
    """按模型或 checkpoint 全选或逐个勾选运行，沿用传入的比较顺序。"""
    def toggle_checkpoint(parent_key, child_keys):
        for key in child_keys:
            st.session_state[key] = st.session_state[parent_key]

    by_checkpoint = {}
    for run in available:
        directory = run["directory"]
        by_checkpoint.setdefault(directory.parent, []).append(directory)

    selected = set()
    for checkpoint in sorted(by_checkpoint, key=lambda path: natural_key(path.name)):
        directories = by_checkpoint[checkpoint]
        parent_key = f"checkpoint-all:{method}:{task}:{checkpoint}"
        child_keys = [f"checkpoint-run:{method}:{task}:{path}" for path in directories]
        count = sum(st.session_state.setdefault(key, True) for key in child_keys)
        all_selected = count == len(directories)
        if st.session_state.get(parent_key) != all_selected:
            st.session_state[parent_key] = all_selected

        with st.container(horizontal=True, vertical_alignment="center"):
            st.checkbox(checkpoint.name, key=parent_key, on_change=toggle_checkpoint,
                        args=(parent_key, child_keys))
            st.caption(f"已选 {count}/{len(directories)}", width="content")
        with st.columns([1, 30], gap="small")[1]:
            for directory, key in zip(directories, child_keys, strict=True):
                if st.checkbox(directory.name, key=key):
                    selected.add(directory)

    return [run for run in available if run["directory"] in selected]


def show_comparison_chart(runs, colors):
    """将所选运行放在同一张核心指标柱图中。"""
    task = runs[0]["summary"]["benchmark"]
    chart = []
    labels = [run["label"] for run in runs]
    for run in runs:
        metrics = run["summary"]["metrics"]
        if task == "locomo":
            scores = {"总体 F1": metrics["score"], **{
                f"{locomo.LOCOMO_CATEGORIES[group['category']]} F1": group["score"]
                for group in sorted(metrics["groups"], key=lambda group: group["category"])
            }}
        elif task == "qasper":
            scores = qasper.comparison_scores(metrics)
        elif task == "ruler":
            scores = ruler.comparison_scores(metrics)
        else:
            scores = longmemeval.comparison_scores(metrics)
        chart.extend({"运行": run["label"], "指标": metric, "得分": score}
                     for metric, score in scores.items() if score is not None)
    st.vega_lite_chart(chart, {
        "encoding": {
            "x": {"field": "指标", "type": "nominal", "sort": list(scores),
                  "axis": {"title": None, "labelAngle": 0}},
            "xOffset": {"field": "运行", "type": "nominal", "sort": labels},
            "y": {"field": "得分", "type": "quantitative", "scale": {"domain": [0, 1]},
                  "axis": {"format": ".0%"}},
            "tooltip": [{"field": "运行"}, {"field": "指标"},
                        {"field": "得分", "type": "quantitative", "format": ".2%"}],
        },
        "layer": [
            {"mark": "bar", "encoding": {
                "color": {"field": "运行", "type": "nominal",
                          "scale": {"domain": labels, "range": [colors[label] for label in labels]},
                          "legend": {"orient": "top", "direction": "vertical", "title": None,
                                     "values": labels, "labelLimit": 0}},
            }},
            {"mark": {"type": "text", "dy": -7, "fontSize": 11},
             "encoding": {"text": {"field": "得分", "type": "quantitative", "format": ".1%"}}},
        ],
        "height": 360,
    }, width="stretch")


def run_dashboard(method, title, render_memory=None, argv=None):
    """按任务选择运行，联动展示核心指标、统计表和逐题回答。"""
    full_context = method == "full-context"
    group_label = "模型" if full_context else "checkpoint"
    task_labels = {**TASK_LABELS, "ruler": "RULER"} if full_context else TASK_LABELS
    parser = argparse.ArgumentParser(description=f"比较 {title} {group_label} 下的全部评估运行。")
    parser.add_argument("--output-dir", type=Path, nargs="+", action="extend", required=True,
                        metavar="DIRECTORY", help=f"一个或多个包含已评分结果的 {group_label} 目录")
    directories = list(dict.fromkeys(path.expanduser().resolve()
                                    for path in parser.parse_args(argv).output_dir))
    for directory in directories:
        if not directory.is_dir():
            parser.error(f"--output-dir 必须指向已有的 {group_label} 目录：{directory}")
    directories.sort(key=lambda path: natural_key(path.name))
    group_names = [directory.name for directory in directories]
    if len(set(group_names)) != len(group_names):
        parser.error(f"参与比较的 {group_label} 目录名须各不相同。")
    st.set_page_config(page_title=title, layout="wide")
    multiple = len(directories) > 1
    st.title(f"{title} · {group_label + ' 比较' if multiple else group_names[0]}")

    parameters = {
        "icae": ("fixed_mem_size", "mean_compression_rate"),
        "autocompressors": ("segment_length",),
        "sdcc": ("compress_ratio_scale",),
        "full-context": (),
    }[method]
    settings = set()
    grouped = {}
    try:
        for directory in directories:
            for path in directory.glob("*/summary.json"):
                summary = json.loads(path.read_text(encoding="utf-8"))
                task = summary["benchmark"]
                if task == "qasper":
                    task = f"qasper/{summary['split']}"
                if summary["method"] != method or task not in task_labels:
                    raise ValueError(f"{path.parent} 中的方法或任务与 {title} 不一致。")
                setting = ((path.parent.name,) if full_context else
                           tuple(summary["settings"][name] for name in parameters))
                settings.add(setting)
                label = f"{directory.name}/{path.parent.name}" if multiple else path.parent.name
                grouped.setdefault(task, []).append({
                    "label": label, "directory": path.parent, "summary": summary, "setting": setting,
                })
        if not grouped:
            st.info(f"输入的 {group_label} 目录下未找到已评分的评估结果。")
            return
    except (OSError, KeyError, ValueError) as error:
        st.error(f"无法读取评估结果。{error}")
        return

    ordered_settings = (sorted(settings, key=lambda value: natural_key(value[0])) if full_context
                        else sorted(settings))
    setting_count = len(ordered_settings)
    setting_lightness = {
        setting: 0.50 if setting_count == 1 else 0.28 + 0.48 * index / (setting_count - 1)
        for index, setting in enumerate(ordered_settings)
    }
    group_indices = {name: index for index, name in enumerate(group_names)}
    st.caption(f"各任务可按 {group_label} 全选运行，也可逐项勾选；图表、统计表和逐题回答同步更新。")
    task_runs = {}
    for task in task_labels:
        if task not in grouped:
            continue
        available = sorted(grouped[task], key=lambda run: (
            natural_key(run["directory"].parent.name),
            natural_key(run["setting"][0]) if full_context else run["setting"],
            natural_key(run["directory"].name),
        ))
        with st.container(border=True):
            st.header(task_labels[task])
            chosen = select_runs(available, method, task)
            paths = [(run["label"], run["directory"]) for run in chosen]
            if not paths:
                st.info("请选择该任务中需要比较的运行。")
                continue
            try:
                if task == "locomo":
                    runs = [locomo.load_run(label, path) for label, path in paths]
                    locomo.validate_runs(runs)
                elif task in qasper.TASK_LABELS:
                    runs = qasper.load_runs(paths)
                elif task == "ruler":
                    runs = ruler.load_runs(paths)
                else:
                    runs = longmemeval.load_runs(paths)
            except (OSError, KeyError, ValueError) as error:
                st.error(f"无法比较所选运行。{error}")
                continue
            task_runs[task] = runs
            st.caption(f"已选 {len(runs)} / {len(available)} 份结果。")
            colors = {}
            for run in runs:
                setting = ((run["directory"].name,) if full_context else
                           tuple(run["summary"]["settings"][name] for name in parameters))
                group_index = group_indices[run["directory"].parent.name]
                hue = (0.60 + group_index * 0.618) % 1
                rgb = hls_to_rgb(hue, setting_lightness[setting], 0.65)
                colors[run["label"]] = "#" + "".join(f"{round(value * 255):02x}" for value in rgb)
            show_comparison_chart(runs, colors)
            with st.expander("详细统计"):
                if task == "locomo":
                    locomo.show_comparison(runs, [run["label"] for run in runs], show_chart=False)
                elif task in qasper.TASK_LABELS:
                    qasper.show_scores(runs, show_chart=False)
                elif task == "ruler":
                    ruler.show_scores(runs)
                else:
                    longmemeval.show_scores(runs, show_chart=False)
            if task == "locomo" and render_memory is not None:
                render_memory(runs)

    if not task_runs:
        return
    with st.container(border=True):
        st.header("逐题结果")
        task = st.selectbox("评估任务", list(task_runs), format_func=task_labels.__getitem__,
                            key="checkpoint-task")
        runs = task_runs[task]
        with st.expander("评估配置"):
            for run in runs:
                st.markdown(f"**{run['label']}**")
                summary = run["summary"]
                if task == "ruler":
                    config = {"目录": str(run["directory"]), "模型配置": summary["model"],
                              "评估协议": summary["protocol"], "执行统计": summary["resources"]}
                else:
                    config = {"目录": str(run["directory"]), "模型配置": summary["settings"]}
                if task in longmemeval.SETTINGS:
                    config["评分配置"] = run["summary"]["judge"]
                st.json(config, expanded=False)
        if task == "locomo":
            locomo.show_answer_comparison(runs)
        elif task in qasper.TASK_LABELS:
            qasper.show_answers(runs)
        elif task == "ruler":
            ruler.show_answers(runs)
        else:
            longmemeval.show_answers(runs)
