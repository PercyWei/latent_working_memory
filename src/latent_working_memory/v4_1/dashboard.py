"""读取并比较多个 run 的本地评估结果，不加载模型或连接实验追踪服务。"""

import argparse
from colorsys import hsv_to_rgb
from hashlib import sha256
from itertools import islice
import json
import math
from pathlib import Path

from pyecharts import charts
import streamlit as st

from latent_working_memory.baselines.results import load_baseline_results, validate_quality_comparison
from latent_working_memory.benchmarks.data import RULER_TASKS
from latent_working_memory.benchmarks.protocol import protocol
from latent_working_memory.v4.reporting import RULER_TASK_LABELS, ruler_overall
from latent_working_memory.v4_1.evaluation.diagnostics import output_flags, visible_text
from latent_working_memory.v4_1.evaluation.records import completed_evaluation


TASKS = ("test", "ruler", "locomo")
TASK_LABELS = {"test": "Test", "ruler": "RULER", "locomo": "LoCoMo"}
CHART_LABELS = {
    "evaluation/nll": "NLL",
    "evaluation/em_f1": "EM / F1",
    "evaluation/ruler": "RULER 训练进度",
    "evaluation/ruler_lengths": "RULER 长度总览",
    "evaluation/locomo": "LoCoMo",
    "evaluation/locomo_groups": "LoCoMo 分组对照",
}
OUTPUT_LABELS = {
    "blank": "空白答案", "whitespace_repeat": "连续空白 ≥12 字符",
    "abnormal_character": "异常字符输出", "budget_exhausted": "生成触顶",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard-run-dir", type=Path, nargs="+", required=True,
                        help="一个或多个运行根目录，仅需保留 eval/ 目录结构")
    parser.add_argument("--dashboard-eval-id", nargs="+",
                        help="启动时选中的评估标识；一个值供所有运行共用，或按运行顺序逐一传入")
    parser.add_argument("--dashboard-baselines", nargs="*", default=[], metavar="LABEL_OR_PATH",
                        help="按“标签 评估结果目录”成对传入，仅本次展示使用")
    args = parser.parse_args(argv)
    args.dashboard_run_dir = [path.expanduser().resolve() for path in args.dashboard_run_dir]
    for path in args.dashboard_run_dir:
        if not path.is_dir():
            parser.error(f"--dashboard-run-dir 必须是已有的运行目录：{path}")
    if len(set(args.dashboard_run_dir)) != len(args.dashboard_run_dir):
        parser.error("--dashboard-run-dir 不能重复")
    if args.dashboard_eval_id is None:
        args.dashboard_eval_id = [None] * len(args.dashboard_run_dir)
    elif len(args.dashboard_eval_id) == 1:
        args.dashboard_eval_id *= len(args.dashboard_run_dir)
    elif len(args.dashboard_eval_id) != len(args.dashboard_run_dir):
        parser.error("--dashboard-eval-id 必须传入一个共用值，或与运行目录数量相同的值")
    if len(args.dashboard_baselines) % 2:
        parser.error("--dashboard-baselines 必须按“标签 结果目录”成对传入")
    args.dashboard_baselines = list(zip(args.dashboard_baselines[::2], args.dashboard_baselines[1::2]))
    labels = [label for label, _ in args.dashboard_baselines]
    if any(not label.strip() for label in labels) or len(labels) != len(set(labels)):
        parser.error("baseline 标签必须非空且不能重复")
    return args


def discover_evaluations(run_dir):
    return {
        directory.name: directory
        for directory in sorted((run_dir / "eval").glob("*"))
        if directory.is_dir() and any(
            next((directory / task).glob("global_step_*/summary.json"), None) is not None
            for task in TASKS
        )
    }


def load_completed_evaluations(directory):
    """校验本地结果身份；同计划的不同 checkpoint 可以覆盖不同 RULER 长度。"""
    directory = Path(directory)
    records, expected, lengths = [], None, {}
    for path in directory.glob("global_step_*/summary.json"):
        request = json.loads((path.parent / "eval.json").read_text(encoding="utf-8"))
        summary = completed_evaluation(path.parent, request)
        if summary is None:
            continue
        step = summary["checkpoint_step"]
        if (
            type(step) is not int or step < 1 or path.parent.name != f"global_step_{step}"
            or summary["eval_id"] != directory.parent.name or summary["task"] != directory.name
            or Path(summary["checkpoint"]).parent.name != path.parent.name
        ):
            raise ValueError(f"summary does not match its checkpoint, eval-id and task: {path}")
        fixed = {name: summary[name] for name in (
            "eval_id", "task", "training_run", "model", "model_window", "seed",
        )}
        fixed["protocol"] = {
            key: value for key, value in summary["protocol"].items() if key != "lengths"
        }
        if summary["task"] == "ruler":
            fixed["data"] = {key: value for key, value in summary["data"].items()
                             if key not in {"samples", "fingerprint", "by_length"}}
            for row in summary["data"]["by_length"]:
                if row["length"] in lengths and lengths[row["length"]] != row:
                    raise ValueError(f"RULER data differs at length {row['length']}: {path}")
                lengths[row["length"]] = row
        else:
            fixed["data"] = summary["data"]
        if expected is not None and fixed != expected:
            raise ValueError(f"checkpoint evaluation settings differ: {path}")
        expected = fixed
        records.append((path.parent, summary))
    return sorted(records, key=lambda pair: pair[1]["checkpoint_step"])


def _benchmark_view(summary):
    """将真实公共评估条件交给共用基线校验与绘图，保留原始结果格式。"""
    task = summary["task"]
    common_protocol = protocol(task, summary["protocol"].get("lengths", ()), "")
    return {
        **summary,
        "benchmark": task,
        "checkpoint": {"step": summary["checkpoint_step"]},
        "training_run": str(Path(summary["training_run"]["directory"]).parent),
        "samples": summary["metrics"]["samples"],
        # v4.1 的 backbone 和正式读取均使用显式 eager attention。
        "model": {**summary["model"], "attention_implementation": "eager"},
        "protocol": {key: summary["protocol"][key] for key in common_protocol},
    }


def _baseline_selection(results, task, summaries, key):
    if not results:
        return {}
    selected = st.multiselect("Baseline", [label for label, _ in results],
                              default=[label for label, _ in results], key=f"{key}:baselines")
    requested = [(label, path) for label, path in results if label in selected]
    baselines = {}
    for label, path in requested:
        try:
            current = load_baseline_results([(label, path)], task, summaries)
        except (OSError, ValueError, KeyError, TypeError) as error:
            st.warning(f"Baseline {label} 无法比较：{error}")
            continue
        if label not in current:
            st.warning(f"Baseline {label} 缺少 {TASK_LABELS[task]} 结果：{path}")
        baselines.update(current)
    return baselines


def _method_color(label, index=None, count=1):
    hue = int.from_bytes(sha256(label.encode("utf-8")).digest()[:8], "big") / 2**64
    channels = hsv_to_rgb(hue, 0.7, 0.65)
    fraction = 0 if index is None else 0.55 * (1 - (index - 1) / max(1, count - 1))
    return "#" + "".join(f"{round(255 * (value + (1 - value) * fraction)):02x}" for value in channels)


def _quality_value(summary, task, metric, ruler_lengths):
    if task == "test":
        return summary["metrics"][metric]
    if task == "ruler":
        if metric is None:
            return ruler_overall(summary, ruler_lengths)
        return next((row["score"] for row in summary["metrics"]["by_length"]
                     if row["length"] == metric), None)
    if metric is None:
        return summary["metrics"]["score"]
    return next((row["score"] for row in summary["metrics"]["groups"]
                 if row["category"] == metric), None)


def _progress_charts(methods, baselines, task, ruler_lengths, ruler_length):
    summaries = [summary for records in methods.values() for _, summary in records]
    steps = sorted({summary["checkpoint_step"] for summary in summaries})
    if task == "test":
        panels = (("nll", (("nll", "NLL"),), "NLL", None),
                  ("em_f1", (("em", "EM"), ("f1", "F1")), "EM / F1", 1))
    elif task == "ruler":
        panels = (("ruler", ((ruler_length, "Mean"),), "Mean accuracy (%)", 100),)
    else:
        groups = {row["category"]: row["category_name"] for summary in summaries
                  for row in summary["metrics"]["groups"]}
        panels = (("locomo", ((None, "总体"), *sorted(groups.items())),
                   f"{summaries[0]['metrics']['metric']} (%)", 100),)
    charts_by_name = {}
    for name, metrics, unit, maximum in panels:
        chart = charts.Line().add_xaxis(steps)
        series = [(label, [result], True) for label, result in baselines.items()]
        series.extend((label, [summary for _, summary in records], False)
                      for label, records in methods.items())
        for label, values, baseline in series:
            for index, (metric, metric_label) in enumerate(metrics):
                points = {value["checkpoint_step"]: value for value in values} if not baseline else {}
                measurements = []
                for step in steps:
                    summary = values[0] if baseline else points.get(step)
                    score = (_quality_value(summary, task, metric, ruler_lengths)
                             if summary is not None else None)
                    measurements.append(score * (1 if task == "test" else 100)
                                        if score is not None else None)
                color = _method_color(label, index if index else None, len(metrics))
                legend = f"{label} / {metric_label}" if len(metrics) > 1 else label
                chart.add_yaxis(
                    legend, measurements, color=color, is_symbol_show=not baseline,
                    symbol_size=7, is_connect_nones=True, label_opts={"show": False},
                    linestyle_opts={"color": color, "type": "dashed" if baseline else "solid"},
                    itemstyle_opts={"color": color},
                )
        chart.set_global_opts(
            tooltip_opts={"trigger": "axis"}, legend_opts={"type": "scroll"},
            xaxis_opts={"name": "累计 step", "type": "value", "minInterval": 1},
            yaxis_opts={"name": unit, "min": 0, **({"max": maximum} if maximum is not None else {})},
        )
        charts_by_name[f"evaluation/{name}"] = chart
    return charts_by_name


def _checkpoint_charts(points, baselines, task, ruler_length=None):
    series = [(label, label, value) for label, value in baselines.items()]
    series.extend((label, f"{label} / step {value['checkpoint_step']}", value)
                  for label, value in points.items())
    if task == "ruler":
        lengths = sorted({row["length"] for _, _, value in series for row in value["metrics"]["by_length"]})
        lengths_chart = charts.Line().add_xaxis([f"{length / 1024:g}K" for length in lengths])
        tasks_chart = charts.Bar().add_xaxis([RULER_TASK_LABELS[task] for task in RULER_TASKS])
        for label, legend, value in series:
            color = _method_color(label)
            scores = {row["length"]: 100 * row["score"] for row in value["metrics"]["by_length"]}
            lengths_chart.add_yaxis(
                legend, [scores.get(length) for length in lengths], color=color,
                is_symbol_show=True, symbol_size=7, label_opts={"show": False},
                linestyle_opts={"color": color}, itemstyle_opts={"color": color},
            )
            scores = {row["task"]: 100 * row["score"] for row in value["metrics"]["groups"]
                      if row["length"] == ruler_length}
            tasks_chart.add_yaxis(legend, [scores.get(task) for task in RULER_TASKS],
                                 label_opts={"show": False}, itemstyle_opts={"color": color})
        lengths_chart.set_global_opts(
            tooltip_opts={"trigger": "axis"}, legend_opts={"type": "scroll"},
            xaxis_opts={"name": "Context length", "axisLabel": {"interval": 0}},
            yaxis_opts={"name": "Mean accuracy (%)", "min": 0, "max": 100},
        )
        tasks_chart.set_global_opts(
            tooltip_opts={"trigger": "axis"}, legend_opts={"type": "scroll"},
            xaxis_opts={"axisLabel": {"interval": 0}},
            yaxis_opts={"name": "Accuracy (%)", "min": 0, "max": 100},
        )
        return {"evaluation/ruler_lengths": lengths_chart, "ruler_tasks": tasks_chart}
    groups = {row["category"]: row["category_name"] for _, _, value in series
              for row in value["metrics"]["groups"]}
    groups = sorted(groups.items())
    chart = charts.Bar().add_xaxis([*[name for _, name in groups], "总体"])
    for label, legend, value in series:
        scores = {row["category"]: 100 * row["score"] for row in value["metrics"]["groups"]}
        chart.add_yaxis(legend, [scores.get(category) for category, _ in groups]
                        + [100 * value["metrics"]["score"]],
                        label_opts={"show": False}, itemstyle_opts={"color": _method_color(label)})
    chart.set_global_opts(tooltip_opts={"trigger": "axis"}, legend_opts={"type": "scroll"},
                          yaxis_opts={"name": f"{next(iter(points.values()))['metrics']['metric']} (%)",
                                      "min": 0, "max": 100})
    return {"evaluation/locomo_groups": chart}


def _summary_rows(methods, baselines, ruler_lengths):
    summaries = [summary for records in methods.values() for _, summary in records]
    rows = []
    if summaries[0]["task"] == "ruler":
        results = [(label, None, result) for label, result in baselines.items()]
        results.extend((label, value["checkpoint_step"], value)
                       for label, records in methods.items() for _, value in records)
        for label, step, result in results:
            by_length = {row["length"]: row for row in result["metrics"]["by_length"]}
            score = ruler_overall(result, ruler_lengths)
            missing = [str(value) for value in ruler_lengths if value not in by_length]
            rows.append({
                "方法": label, "累计 step": step,
                "样本": sum(by_length[value]["samples"] for value in ruler_lengths) if not missing else None,
                "accuracy (%)": 100 * score if score is not None else None,
                "比较长度": ", ".join(map(str, ruler_lengths)),
                "评估长度": ", ".join(map(str, result["protocol"]["lengths"])),
                "覆盖状态": "完整" if not missing else f"缺少长度：{', '.join(missing)}",
            })
        return rows
    rows.extend({"方法": label, "累计 step": None, "样本": result["samples"],
                 f"{result['metrics']['metric']} (%)": 100 * result["metrics"]["score"]}
                for label, result in baselines.items())
    for label, records in methods.items():
        for _, summary in records:
            metrics = summary["metrics"]
            row = {"方法": label, "累计 step": summary["checkpoint_step"], "样本": metrics["samples"]}
            if summary["task"] == "test":
                row.update({name.upper(): metrics[name] for name in ("nll", "em", "f1")})
            else:
                row[f"{metrics['metric']} (%)"] = 100 * metrics["score"]
            rows.append(row)
    return rows


def _resource_rows(methods, baselines):
    results = [(label, None, result) for label, result in baselines.items()]
    results.extend((label, value["checkpoint_step"], value)
                   for label, records in methods.items() for _, value in records)
    rows = []
    for label, step, result in results:
        resources, model = result["resources"], result["model"]
        peak = resources.get("peak_memory_bytes", resources.get("peak_memory_allocated_bytes"))
        rows.append({
            "方法": label, "累计 step": step,
            "测量样本": resources.get("samples", result["metrics"]["samples"]),
            "测量长度": ", ".join(map(str, resources.get("lengths", result["protocol"].get("lengths", [])))),
            "精度": model["backbone_dtype"],
            "Attention 实现": model.get("attention_implementation", "eager"),
            "设备记录": resources.get("device"),
            "生成路径累计耗时（秒）": resources["generation_seconds"],
            "峰值 allocated（GiB）": peak / 1024 ** 3 if peak is not None else None,
            "本次调度墙钟（秒）": resources.get("wall_seconds"),
            "基线评估循环耗时（秒）": resources.get("evaluation_seconds"),
        })
    return rows


def _ruler_prompt(data_dir, record):
    metadata = record["metadata"]
    path = Path(data_dir).expanduser() / str(metadata["length"]) / metadata["task"] / "test.jsonl"
    # ID 的末段是源文件行号；官方 index 可能重复，并非样本序号。
    position = int(record["id"].rsplit("/", 1)[1])
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index == position:
                source = json.loads(line)
                if source["index"] != metadata["index"] or source["outputs"] != record["references"]:
                    raise ValueError(f"{path}:{position + 1} 与评估记录不一致，请选择原评估数据")
                return source["input"]
    raise ValueError(f"{path} 中不存在第 {position + 1} 行")


def _details(methods, key):
    if not st.toggle("查看逐样本记录", key=f"{key}:details"):
        return
    label = st.selectbox("记录运行", list(methods), key=f"{key}:details-run")
    records = methods[label]
    key = f"{key}:{label}"
    checkpoints = {summary["checkpoint_step"]: (path, summary) for path, summary in records}
    step = st.selectbox("记录 checkpoint", list(checkpoints), index=len(checkpoints) - 1,
                        format_func=lambda value: f"global_step_{value}", key=f"{key}:details-step")
    directory, summary = checkpoints[step]
    ruler = summary["task"] == "ruler"
    length, task = None, None
    if ruler:
        with (directory / "details.jsonl").open(encoding="utf-8") as stream:
            details = [json.loads(line) for line in stream]
        length_column, task_column = st.columns(2)
        with length_column:
            length = st.selectbox(
                "RULER 长度", [None, *sorted({row["metadata"]["length"] for row in details})],
                format_func=lambda value: "全部长度" if value is None else f"{value:,} tokens",
                key=f"{key}:{step}:details-length",
            )
        details = [row for row in details if length is None or row["metadata"]["length"] == length]
        tasks = {row["metadata"]["task"] for row in details}
        with task_column:
            task = st.selectbox(
                "RULER 子任务", [None, *(value for value in RULER_TASKS if value in tasks)],
                format_func=lambda value: "全部子任务" if value is None else value,
                key=f"{key}:{step}:{length}:details-task",
            )
        details = [row for row in details if task is None or row["metadata"]["task"] == task]
        data_dir = st.text_input("RULER 数据目录", value=summary["data"]["directory"],
                                 key=f"{key}:ruler-data-directory")
    count, page_size = summary["metrics"]["samples"], 20
    if ruler:
        count = len(details)
    pages = max(1, math.ceil(count / page_size))
    page = st.number_input("页码", min_value=1, max_value=pages, value=1, step=1,
                           key=f"{key}:{step}:{length}:{task}:details-page")
    start = (page - 1) * page_size
    if ruler:
        rows = details[start:start + page_size]
    else:
        with (directory / "details.jsonl").open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in islice(stream, start, start + page_size)]
    st.caption(f"第 {page} / {pages} 页，每页 {page_size} 条，共 {count} 条")
    for row in rows:
        with st.container(border=True):
            st.markdown(f"**{row['id']}**")
            quality = (f"EM {row['em']:.3f} · F1 {row['f1']:.3f}" if summary["task"] == "test"
                       else f"{summary['metrics']['metric']} {row['score']:.3f}")
            st.caption(f"{quality} · {row['prompt_tokens']} prompt tokens · "
                       f"{row['generated_tokens']}/{row['max_new_tokens']} generated tokens · "
                       f"{row['stop_reason']}")
            if row["question"]:
                st.write(row["question"])
            if ruler:
                with st.expander("完整输入题面", key=f"{key}:{step}:{row['id']}:prompt",
                                 on_change="rerun") as prompt_panel:
                    if prompt_panel.open:
                        try:
                            prompt = _ruler_prompt(data_dir, row)
                        except (OSError, ValueError) as error:
                            st.warning(f"题面读取失败：{error}")
                        else:
                            st.code(prompt, language=None, wrap_lines=True, height=350)
            st.write("参考答案：" + " / ".join(row["references"]))
            flags = " · ".join(OUTPUT_LABELS[name] for name, value in output_flags(row).items() if value)
            if flags:
                st.warning(flags)
            st.code(visible_text(row["prediction"]), language=None, wrap_lines=True)
            with st.expander("原始记录与 token IDs"):
                st.json(row, expanded=1)


def _render_task(directories, task, baseline_results):
    methods = {}
    for label, directory in directories.items():
        records = load_completed_evaluations(directory / task)
        if records:
            methods[label] = records
        else:
            st.info(f"{label} 暂无已完成的 {TASK_LABELS[task]} 评估。")
    if not methods:
        st.info(f"暂无已完成的 {TASK_LABELS[task]} 评估。")
        return
    key = f"{list(directories.values())}:{task}"
    if not st.toggle("全部 checkpoint", value=True, key=f"{key}:all-checkpoints"):
        choices = [(label, value["checkpoint_step"]) for label, records in methods.items() for _, value in records]
        chosen = st.multiselect("Checkpoint", choices, default=choices,
                                format_func=lambda value: f"{value[0]} / global_step_{value[1]}",
                                key=f"{key}:steps")
        methods = {label: [(path, value) for path, value in records
                           if (label, value["checkpoint_step"]) in chosen]
                   for label, records in methods.items()}
        methods = {label: records for label, records in methods.items() if records}
    if not methods:
        st.info("请选择至少一个 checkpoint。")
        return
    summaries = [summary for records in methods.values() for _, summary in records]
    views = [_benchmark_view(summary) for summary in summaries] if task != "test" else summaries
    validate_quality_comparison(views, task)
    for label, _ in baseline_results:
        if label in methods:
            st.warning(f"Baseline {label} 无法比较：标签与运行名称相同。")
    baseline_results = [(label, path) for label, path in baseline_results if label not in methods]
    baselines = _baseline_selection(baseline_results, task, views, key) if task != "test" else {}
    ruler_lengths, ruler_length = (), None
    if task == "ruler":
        scopes = [set(value["protocol"]["lengths"]) for value in summaries]
        lengths = sorted(set.union(*scopes))
        common = sorted(set.intersection(*scopes))
        ruler_lengths = st.multiselect("Overall 比较长度", lengths, default=common or lengths,
                                       format_func=lambda value: f"{value:,} tokens",
                                       key=f"{key}:overall-lengths")
        if not ruler_lengths:
            st.info("请选择至少一个 Overall 比较长度。")
            return
        st.caption("Overall 按统一的比较长度重算；未覆盖全部所选长度的 checkpoint 或基线不显示总体分数。")
        ruler_length = st.selectbox("训练进度长度", [None, *lengths],
                                    format_func=lambda value: "Overall" if value is None else f"{value:,} tokens",
                                    key=f"{key}:progress-length")
    st.subheader("质量指标")
    st.dataframe(_summary_rows(methods, baselines, ruler_lengths), hide_index=True)
    panels = _progress_charts(methods, baselines, task, ruler_lengths, ruler_length)
    for name, chart in panels.items():
        st.subheader(CHART_LABELS[name])
        st.echarts_chart(chart, height=460, theme=None, key=f"{key}:{name}")
    st.subheader("明细 checkpoint")
    points = {}
    for label, records in methods.items():
        steps = [value["checkpoint_step"] for _, value in records]
        step = st.selectbox(label, steps, index=len(steps) - 1,
                            format_func=lambda value: f"global_step_{value}", key=f"{key}:{label}:summary-step")
        points[label] = next(value for _, value in records if value["checkpoint_step"] == step)
    if task != "test":
        length = None
        if task == "ruler":
            lengths = sorted({value for summary in points.values() for value in summary["protocol"]["lengths"]})
            length = st.selectbox("任务明细长度", lengths,
                                  format_func=lambda value: f"{value:,} tokens", key=f"{key}:task-length")
        for name, chart in _checkpoint_charts(points, baselines, task, length).items():
            st.subheader(f"RULER 任务明细 · {length:,} tokens" if name == "ruler_tasks" else CHART_LABELS[name])
            st.echarts_chart(chart, height=460, theme=None, key=f"{key}:{name}")
    st.subheader("资源测量与执行设置")
    st.dataframe(_resource_rows(methods, baselines), hide_index=True)
    st.caption(
        "资源保留各运行的原始测量范围，未按质量共同范围裁剪。生成路径含 prefill、方法自身的压缩／写入和 decode；"
        "累计耗时与多卡墙钟不同。本次调度墙钟包含模型加载，缓存重试只统计本次执行，"
        "不能作为完整运行速度。硬件与批量记录不完整时，仅展示测量值，不计算加速比。"
    )
    st.subheader("输出诊断")
    st.dataframe([{"方法": label, "累计 step": value["checkpoint_step"],
                   **{OUTPUT_LABELS[name]: count for name, count in value["outputs"].items()}}
                  for label, records in methods.items() for _, value in records], hide_index=True)
    st.caption("字符标记仅用于展示，不改变保存的原文或评分；连续空白按至少 12 个字符统计。")
    with st.expander("评估配置与统计"):
        st.json(points, expanded=1)
    _details(methods, key)


def main():
    args = parse_args()
    st.set_page_config(page_title="v4.1 评估", layout="wide")
    st.title("v4.1 评估结果")
    st.sidebar.button("刷新结果")
    names = [path.name for path in args.dashboard_run_dir]
    runs = {path.name if names.count(path.name) == 1 else str(path): (path, eval_id)
            for path, eval_id in zip(args.dashboard_run_dir, args.dashboard_eval_id)}
    labels = st.sidebar.multiselect("运行", list(runs), default=list(runs))
    directories = {}
    for label in labels:
        run_dir, initial_eval_id = runs[label]
        st.sidebar.caption(str(run_dir))
        evaluations = discover_evaluations(run_dir)
        if initial_eval_id is not None and initial_eval_id not in evaluations:
            st.error(f"指定评估没有已完成结果：{run_dir}/eval/{initial_eval_id}")
            continue
        if not evaluations:
            st.info(f"未找到已完成评估：{run_dir}/eval/<eval-id>/<task>/global_step_*/summary.json")
            continue
        eval_ids = list(evaluations)
        eval_id = st.sidebar.selectbox(
            f"Eval ID · {label}", eval_ids,
            index=eval_ids.index(initial_eval_id) if initial_eval_id is not None else 0,
            key=f"{run_dir}:{initial_eval_id}:eval-id",
        )
        directories[label] = evaluations[eval_id]
    if not directories:
        st.info("请选择至少一个有完整评估结果的运行。")
        return
    for label, directory in directories.items():
        st.caption(f"{label} / {directory.name} · {directory}")
    tabs = st.tabs([TASK_LABELS[task] for task in TASKS], key=f"{list(directories.values())}:task", on_change="rerun")
    for task, tab in zip(TASKS, tabs):
        if not tab.open:
            continue
        with tab:
            try:
                _render_task(directories, task, args.dashboard_baselines)
            except (OSError, ValueError, KeyError, TypeError) as error:
                st.error(f"{TASK_LABELS[task]} 结果无法展示：{error}")
                st.exception(error)


if __name__ == "__main__":
    main()
