"""读取单次 run 的本地评估结果，不加载模型或连接实验追踪服务。"""

import argparse
from itertools import islice
import json
import math
from pathlib import Path

from pyecharts import charts
import streamlit as st

from latent_working_memory.baselines.results import load_baseline_results
from latent_working_memory.benchmarks.protocol import protocol
from latent_working_memory.v4.reporting import evaluation_charts, ruler_overall, ruler_task_chart
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
}
OUTPUT_LABELS = {
    "blank": "空白答案", "whitespace_repeat": "连续空白 ≥12 字符",
    "abnormal_character": "异常字符输出", "budget_exhausted": "生成触顶",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard-run-dir", type=Path, required=True,
                        help="单次运行根目录，仅需保留 eval/ 目录结构")
    parser.add_argument("--dashboard-baselines", nargs="*", default=[], metavar="LABEL_OR_PATH",
                        help="按“标签 评估结果目录”成对传入，仅本次展示使用")
    args = parser.parse_args(argv)
    args.dashboard_run_dir = args.dashboard_run_dir.expanduser().resolve()
    if not args.dashboard_run_dir.is_dir():
        parser.error(f"--dashboard-run-dir 必须是已有的运行目录：{args.dashboard_run_dir}")
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


def _test_charts(summaries):
    steps = [summary["checkpoint_step"] for summary in summaries]
    panels = {}
    for name, metrics in (("nll", ("nll",)), ("em_f1", ("em", "f1"))):
        chart = charts.Line().add_xaxis(steps)
        for metric in metrics:
            chart.add_yaxis(metric.upper(), [summary["metrics"][metric] for summary in summaries],
                            is_symbol_show=True, symbol_size=8, label_opts={"show": False})
        chart.set_global_opts(
            tooltip_opts={"trigger": "axis"},
            xaxis_opts={"name": "累计 step", "type": "value", "minInterval": 1},
            yaxis_opts={"name": name.upper(), "min": 0,
                        **({"max": 1} if name == "em_f1" else {})},
        )
        panels[f"evaluation/{name}"] = chart
    return panels


def _summary_rows(summaries, baselines):
    rows = []
    if summaries[0]["task"] == "ruler":
        scopes = sorted({tuple(sorted(value["protocol"]["lengths"])) for value in summaries})
        for label, result in baselines.items():
            by_length = {row["length"]: row for row in result["metrics"]["by_length"]}
            for lengths in scopes:
                score = ruler_overall(result, lengths)
                missing = [str(value) for value in lengths if value not in by_length]
                rows.append({
                    "方法": label, "累计 step": None,
                    "样本": sum(by_length[value]["samples"] for value in lengths) if not missing else None,
                    "accuracy (%)": 100 * score if score is not None else None,
                    "评估长度": ", ".join(map(str, lengths)),
                    "覆盖状态": "完整" if not missing else f"缺少长度：{', '.join(missing)}",
                })
    else:
        rows.extend({"方法": label, "累计 step": None, "样本": result["samples"],
                     f"{result['metrics']['metric']} (%)": 100 * result["metrics"]["score"]}
                    for label, result in baselines.items())
    for summary in summaries:
        metrics = summary["metrics"]
        row = {"方法": Path(summary["training_run"]["directory"]).parent.name,
               "累计 step": summary["checkpoint_step"], "样本": metrics["samples"]}
        if summary["task"] == "test":
            row.update({name.upper(): metrics[name] for name in ("nll", "em", "f1")})
        else:
            row[f"{metrics['metric']} (%)"] = 100 * metrics["score"]
            if summary["task"] == "ruler":
                row["评估长度"] = ", ".join(str(value) for value in summary["protocol"]["lengths"])
                row["覆盖状态"] = "完整"
        rows.append(row)
    return rows


def _resource_rows(summaries, baselines):
    results = [(label, None, result) for label, result in baselines.items()]
    results.extend((Path(value["training_run"]["directory"]).parent.name,
                    value["checkpoint_step"], value) for value in summaries)
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


def _details(records, key):
    if not st.toggle("查看逐样本记录", key=f"{key}:details"):
        return
    checkpoints = {summary["checkpoint_step"]: (path, summary) for path, summary in records}
    step = st.selectbox("记录 checkpoint", list(checkpoints), index=len(checkpoints) - 1,
                        format_func=lambda value: f"global_step_{value}", key=f"{key}:details-step")
    directory, summary = checkpoints[step]
    count, page_size = summary["metrics"]["samples"], 20
    pages = max(1, math.ceil(count / page_size))
    page = st.number_input("页码", min_value=1, max_value=pages, value=1, step=1,
                           key=f"{key}:{step}:details-page")
    with (directory / "details.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in islice(stream, (page - 1) * page_size, page * page_size)]
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
            st.write("参考答案：" + " / ".join(row["references"]))
            flags = " · ".join(OUTPUT_LABELS[name] for name, value in output_flags(row).items() if value)
            if flags:
                st.warning(flags)
            st.code(visible_text(row["prediction"]), language=None, wrap_lines=True)
            with st.expander("原始记录与 token IDs"):
                st.json(row, expanded=1)


def _render_task(directory, task, baseline_results):
    records = load_completed_evaluations(directory / task)
    if not records:
        st.info(f"暂无已完成的 {TASK_LABELS[task]} 评估。")
        return
    key = f"{directory}:{task}"
    if st.toggle("全部 checkpoint", value=True, key=f"{key}:all-checkpoints"):
        selected = records
    else:
        steps = [summary["checkpoint_step"] for _, summary in records]
        chosen = st.multiselect("Checkpoint", steps, default=steps,
                                format_func=lambda value: f"global_step_{value}", key=f"{key}:steps")
        selected = [(path, summary) for path, summary in records if summary["checkpoint_step"] in chosen]
    if not selected:
        st.info("请选择至少一个 checkpoint。")
        return
    summaries = [summary for _, summary in selected]
    views = [_benchmark_view(summary) for summary in summaries] if task != "test" else []
    baselines = _baseline_selection(baseline_results, task, views, key) if views else {}
    st.subheader("质量指标")
    st.dataframe(_summary_rows(summaries, baselines), hide_index=True)
    if task == "ruler" and baselines:
        st.caption("基线 Overall 按对应 checkpoint 的长度范围重算；缺少任一长度时不显示总体分数。")
    ruler_length = None
    if task == "ruler" and len(summaries) > 1:
        lengths = sorted({row["length"] for summary in summaries for row in summary["metrics"]["by_length"]})
        ruler_length = st.selectbox("训练进度长度", [None, *lengths],
                                    format_func=lambda value: "Overall" if value is None else f"{value:,} tokens",
                                    key=f"{key}:progress-length")
    panels = (_test_charts(summaries) if task == "test"
              else evaluation_charts(views, baselines, ruler_length=ruler_length))
    for name, chart in panels.items():
        st.subheader(CHART_LABELS[name])
        st.echarts_chart(chart, height=460, theme=None, key=f"{key}:{name}")
    selected_step = st.selectbox(
        "明细 checkpoint", [summary["checkpoint_step"] for summary in summaries], index=len(summaries) - 1,
        format_func=lambda value: f"global_step_{value}", key=f"{key}:summary-step",
    )
    summary = next(value for value in summaries if value["checkpoint_step"] == selected_step)
    if task == "ruler":
        view = _benchmark_view(summary)
        if len(summaries) > 1:
            st.subheader("RULER 长度总览")
            st.echarts_chart(evaluation_charts([view], baselines)["evaluation/ruler_lengths"],
                             height=460, theme=None, key=f"{key}:{selected_step}:lengths")
        length = st.selectbox("任务明细长度", summary["protocol"]["lengths"],
                              format_func=lambda value: f"{value:,} tokens",
                              key=f"{key}:{selected_step}:task-length")
        st.subheader(f"RULER 任务明细 · {length:,} tokens")
        st.echarts_chart(ruler_task_chart(view, length, baselines), height=460, theme=None,
                         key=f"{key}:{selected_step}:tasks:{length}")
    st.subheader("资源测量与执行设置")
    st.dataframe(_resource_rows(summaries, baselines), hide_index=True)
    st.caption(
        "资源保留各运行的原始测量范围，未按质量共同范围裁剪。生成路径含 prefill、方法自身的压缩／写入和 decode；"
        "累计耗时与多卡墙钟不同。本次调度墙钟包含模型加载，缓存重试只统计本次执行，"
        "不能作为完整运行速度。硬件与批量记录不完整时，仅展示测量值，不计算加速比。"
    )
    st.subheader("输出诊断")
    st.dataframe([{"累计 step": value["checkpoint_step"],
                   **{OUTPUT_LABELS[name]: count for name, count in value["outputs"].items()}}
                  for value in summaries], hide_index=True)
    st.caption("字符标记仅用于展示，不改变保存的原文或评分；连续空白按至少 12 个字符统计。")
    with st.expander("评估配置与统计"):
        st.json(summary, expanded=1)
    _details(selected, key)


def main():
    args = parse_args()
    st.set_page_config(page_title="v4.1 评估", layout="wide")
    st.title("v4.1 评估结果")
    st.sidebar.caption(str(args.dashboard_run_dir))
    st.sidebar.button("刷新结果")
    evaluations = discover_evaluations(args.dashboard_run_dir)
    if not evaluations:
        st.info(f"未找到已完成评估：{args.dashboard_run_dir}/eval/<eval-id>/<task>/global_step_*/summary.json")
        return
    eval_ids = list(evaluations)
    eval_id = st.sidebar.selectbox("Eval ID", eval_ids,
                                   index=eval_ids.index("auto") if "auto" in evaluations else 0)
    directory = evaluations[eval_id]
    st.caption(f"{args.dashboard_run_dir.name} / {eval_id} · {directory}")
    tabs = st.tabs([TASK_LABELS[task] for task in TASKS], key=f"{directory}:task", on_change="rerun")
    for task, tab in zip(TASKS, tabs):
        if not tab.open:
            continue
        with tab:
            try:
                _render_task(directory, task, args.dashboard_baselines)
            except (OSError, ValueError, KeyError, TypeError) as error:
                st.error(f"{TASK_LABELS[task]} 结果无法展示：{error}")
                st.exception(error)


if __name__ == "__main__":
    main()
