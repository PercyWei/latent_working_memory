"""按任务选择并比较一个或多个 checkpoint 的已评分压缩设置。"""

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


def show_comparison_chart(runs, colors):
    """将所选 checkpoint 和压缩设置放在同一张核心指标柱图中。"""
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
            scores = {"Answer F1": metrics["score"]}
        else:
            scores = {"总体准确率": metrics["score"],
                      "题型宏平均": metrics["task_averaged_accuracy"],
                      "证据不足": metrics["abstention"]["score"]}
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


def run_dashboard(method, title, render_memory, argv=None):
    """按任务选择运行，联动展示核心指标、统计表和逐题回答。"""
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
                    grouped.setdefault(task, []).append({
                        "label": label, "directory": path.parent, "summary": summary,
                    })
        if not grouped:
            st.info("所选 checkpoint 下未找到已评分的评估结果。")
            return
    except (OSError, KeyError, ValueError) as error:
        st.error(f"无法读取评估结果。{error}")
        return

    setting_indices = {setting: index for index, setting in enumerate(sorted(compression_settings))}
    checkpoint_indices = {name: index for index, name in enumerate(checkpoint_names)}
    st.caption("各任务可独立选择参与比较的运行；图表、统计表和逐题回答随选择同步更新。")
    task_runs = {}
    for task in TASK_LABELS:
        if task not in grouped:
            continue
        available = sorted(grouped[task], key=lambda run: (
            tuple(run["summary"]["settings"][name] for name in parameters),
            natural_key(run["directory"].parent.name), natural_key(run["directory"].name),
        ))
        st.header(TASK_LABELS[task])
        labels = [run["label"] for run in available]
        selected_labels = st.multiselect(
            "参与比较的运行", labels, default=labels, key=f"checkpoint-runs:{method}:{task}",
            help="可自由组合不同 checkpoint 和压缩设置；每个 output_id 对应一次独立评估。",
        )
        paths = [(run["label"], run["directory"]) for run in available if run["label"] in selected_labels]
        if not paths:
            st.info("请选择该任务中需要比较的运行。")
            continue
        try:
            if task == "locomo":
                runs = [locomo.load_run(label, path) for label, path in paths]
                locomo.validate_runs(runs)
            elif task in qasper.TASK_LABELS:
                runs = qasper.load_runs(paths)
            else:
                runs = longmemeval.load_runs(paths)
        except (OSError, KeyError, ValueError) as error:
            st.error(f"无法比较所选运行。{error}")
            continue
        task_runs[task] = runs
        st.caption(f"已选 {len(runs)} / {len(available)} 份结果。")
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
        show_comparison_chart(runs, colors)
        with st.expander("详细统计"):
            if task == "locomo":
                locomo.show_comparison(runs, [run["label"] for run in runs], show_chart=False)
            elif task in qasper.TASK_LABELS:
                qasper.show_scores(runs)
            else:
                longmemeval.show_scores(runs, show_chart=False)
        if task == "locomo":
            render_memory(runs)

    if not task_runs:
        return
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
