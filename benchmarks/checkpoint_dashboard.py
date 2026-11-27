"""按任务比较一个或多个 checkpoint 的全部已评分压缩设置。"""

import argparse
from collections import Counter
from colorsys import hls_to_rgb
from html import escape
from itertools import groupby
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


def show_setting_scores(runs, parameters, colors):
    """按压缩设置展示条形表，让 checkpoint 名称与核心指标始终可见。"""
    method = runs[0]["summary"]["method"]
    task = runs[0]["summary"]["benchmark"]
    prefixes = {"icae": ("mem", "cr"), "autocompressors": ("seg",), "sdcc": ("scale",)}[method]
    st.caption("同一压缩设置下逐行比较 checkpoint；条形长度统一对应 0–100%。")
    for setting, setting_runs in groupby(
        runs, key=lambda run: tuple(run["summary"]["settings"][name] for name in parameters),
    ):
        setting_runs = list(setting_runs)
        setting_label = "_".join(prefix + str(value).removesuffix(".0")
                                 for prefix, value in zip(prefixes, setting, strict=True))
        st.subheader(setting_label)
        counts = Counter(run["directory"].parent.name for run in setting_runs)
        rows = []
        for run in setting_runs:
            metrics = run["summary"]["metrics"]
            if task == "locomo":
                scores = {"总体 F1": metrics["score"], **{
                    f"{locomo.LOCOMO_CATEGORIES[group['category']]} F1": group["score"]
                    for group in sorted(metrics["groups"], key=lambda group: group["category"])
                }}
            elif task == "qasper":
                scores = {"Answer F1": metrics["score"]}
            else:
                scores = {"总体准确率": metrics["score"],
                          "题型宏平均": metrics["task_averaged_accuracy"],
                          "证据不足": metrics["abstention"]["score"]}
            checkpoint = run["directory"].parent.name
            name = escape(checkpoint)
            if counts[checkpoint] > 1:
                name += f'<div class="output-id">{escape(run["directory"].name)}</div>'
            cells = [f'<th scope="row">{name}</th>']
            for score in scores.values():
                if score is None:
                    cells.append('<td aria-label="无适用样本">—</td>')
                else:
                    cells.append(
                        f'<td><span>{score:.2%}</span><div class="score-track" aria-hidden="true">'
                        f'<div style="width:{score * 100:.4f}%;background:{colors[run["label"]]}">'
                        '</div></div></td>'
                    )
            rows.append('<tr>' + ''.join(cells) + '</tr>')
        headers = ''.join(f'<th scope="col">{escape(scope)}</th>' for scope in scores)
        min_width = max(520, 220 + 120 * len(scores))
        st.html(f"""
            <style>
                .checkpoint-scores {{ overflow-x: auto; margin-bottom: 1rem; }}
                .checkpoint-scores table {{ width: 100%; table-layout: fixed; border-collapse: collapse; }}
                .checkpoint-scores th, .checkpoint-scores td {{
                    padding: 12px; text-align: left; vertical-align: middle;
                    border-bottom: 1px solid rgba(128, 128, 128, .2); overflow-wrap: anywhere;
                }}
                .checkpoint-scores th:first-child {{ width: 28%; }}
                .checkpoint-scores tbody th {{ font-weight: 500; }}
                .checkpoint-scores td {{ font-variant-numeric: tabular-nums; }}
                .checkpoint-scores .output-id {{ font-size: .85em; opacity: .75; margin-top: 4px; }}
                .checkpoint-scores .score-track {{
                    height: 9px; background: rgba(128, 128, 128, .15); border-radius: 3px; margin-top: 6px;
                }}
                .checkpoint-scores .score-track > div {{ height: 100%; border-radius: 3px; }}
            </style>
            <div class="checkpoint-scores">
                <table style="min-width:{min_width}px">
                    <thead><tr><th scope="col">Checkpoint</th>{headers}</tr></thead>
                    <tbody>{''.join(rows)}</tbody>
                </table>
            </div>
        """)


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
        show_setting_scores(runs, parameters, colors)
        with st.expander("详细统计"):
            if task == "locomo":
                locomo.show_comparison(runs, [run["label"] for run in runs], show_chart=False)
            elif task in qasper.TASK_LABELS:
                qasper.show_scores(runs)
            else:
                longmemeval.show_scores(runs, show_chart=False)
        if task == "locomo":
            render_memory(runs)

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
