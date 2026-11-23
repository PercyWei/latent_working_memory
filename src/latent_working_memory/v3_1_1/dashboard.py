"""v3.1.1 独立评估看板：官方指标、压缩分布与逐文本分析。"""

import argparse
import json
from pathlib import Path

from pyecharts import charts
import streamlit as st

from latent_working_memory.baselines.results import load_baseline_results
from latent_working_memory.benchmarks.data import RULER_TASKS
from latent_working_memory.benchmarks.protocol import protocol
from latent_working_memory.v4.reporting import evaluation_charts, ruler_overall, ruler_task_chart
from latent_working_memory.v3_1_1 import dashboard_compression as compression
from latent_working_memory.v3_1_1.evaluation.diagnostics import output_flags, visible_text
from latent_working_memory.v3_1_1.evaluation.records import completed_evaluation


TASKS = ("test", "ruler", "locomo")
TASK_LABELS = {"test": "Test", "ruler": "RULER", "locomo": "LoCoMo"}
CHART_LABELS = {
    "evaluation/nll": "NLL",
    "evaluation/em_f1": "EM / F1",
    "evaluation/ruler": "RULER checkpoint 对比",
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
    parser.add_argument("--dashboard-eval-id", help="启动时选中的评估标识，页面仍可切换")
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
        "model": summary["model"],
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
    if len(summaries) == 1:
        return {}
    steps = [summary["checkpoint_step"] for summary in summaries]
    panels = {}
    for name, metrics in (
        ("nll", ("qa_nll", "token_nll")), ("em_f1", ("em", "f1")),
    ):
        chart = charts.Line().add_xaxis(steps)
        for metric in metrics:
            chart.add_yaxis(metric, [summary["metrics"][metric] for summary in summaries],
                            is_symbol_show=True, symbol_size=8, label_opts={"show": False})
        chart.set_global_opts(
            tooltip_opts={"trigger": "axis"},
            xaxis_opts={"name": "optimizer step", "type": "value", "minInterval": 1},
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
               "累计 step": summary["checkpoint_step"]}
        if summary["task"] == "test":
            row.update({"文档": metrics["documents"], "问题": metrics["questions"]})
            row.update({name: metrics[name] for name in (
                "qa_nll", "token_nll", "em", "f1",
            )})
        else:
            row["样本"] = metrics["samples"]
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
        batch_sizes = resources["batch_sizes"] if step is not None else resources.get("batch_sizes")
        rows.append({
            "方法": label, "累计 step": step,
            "测量样本": resources.get("samples", (result["metrics"]["documents"]
                                                if result.get("task") == "test" else result["metrics"]["samples"])),
            "测量长度": ", ".join(map(str, resources.get("lengths", result["protocol"].get("lengths", [])))),
            "精度": (f"{model['backbone_dtype']} + BF16 autocast"
                     if model.get("mixed_precision", False) else model["backbone_dtype"]),
            "设备记录": resources.get("device"),
            "实际评估批量": ", ".join(map(str, batch_sizes)) if batch_sizes is not None else None,
            "批量单位": ({"test": "文档", "locomo": "对话", "ruler": "题目"}[result["task"]]
                         if step is not None else "题目"),
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


def _show_answer(answer, task, metric):
    if task == "test":
        quality = f"NLL {answer['nll']:.4f} · EM {answer['em']:.3f} · F1 {answer['f1']:.3f}"
        references = [answer["reference"]]
    else:
        quality = f"{metric} {answer['score']:.3f}"
        references = answer["references"]
    st.caption(f"{quality} · {answer['generated_tokens']}/{answer['max_new_tokens']} generated tokens · "
               f"{answer['stop_reason']}")
    if task == "test" and "segment_id" in answer:
        start, end = answer["segment_char_span"]
        st.caption(f"来源段 {answer['segment_id']} · 完整正文字符区间 [{start}, {end})")
    if answer["question"]:
        st.write(answer["question"])
    st.write("参考答案：" + " / ".join(references))
    flags = " · ".join(OUTPUT_LABELS[name] for name, value in output_flags(answer).items() if value)
    if flags:
        st.warning(flags)
    st.code(visible_text(answer["prediction"]), language=None, wrap_lines=True)


def _document_answers(document, directory, summary, key):
    if not st.toggle("查看问题与回答", key=f"{key}:answers"):
        return
    task = summary["task"]
    questions = document["questions"]
    if task == "locomo":
        categories = {row["category"]: row["category_name"] for row in questions}
        category = st.selectbox(
            "问题分类", [None, *sorted(categories)],
            format_func=lambda value: "全部分类" if value is None else categories[value],
            key=f"{key}:category",
        )
        questions = [row for row in questions if category is None or row["category"] == category]
        st.caption("分类筛选仅作用于下方问题；压缩统计按完整对话计算。")
        chosen = st.selectbox(
            "问题", range(len(questions)),
            format_func=lambda index: f"{questions[index]['id']} · {questions[index]['category_name']}",
            key=f"{key}:{category}:question",
        )
    else:
        chosen = 0
    row = compression.read_record(directory / "details.jsonl", questions[chosen]["offset"])
    if task == "test":
        answer_index = st.selectbox(
            "问题", range(len(row["qas"])),
            format_func=lambda index: f"{index + 1}. {row['qas'][index]['question']}",
            key=f"{key}:question",
        )
        answer = row["qas"][answer_index]
    else:
        answer = row
    _show_answer(answer, task, summary["metrics"].get("metric"))
    if task == "ruler":
        with st.expander("完整输入题面", key=f"{key}:prompt", on_change="rerun") as panel:
            if panel.open:
                data_dir = st.text_input("RULER 数据目录", value=summary["data"]["directory"],
                                         key=f"{key}:ruler-data-directory")
                try:
                    prompt = _ruler_prompt(data_dir, row)
                except (OSError, ValueError) as error:
                    st.warning(f"题面读取失败：{error}")
                else:
                    st.code(prompt, language=None, wrap_lines=True, height=350)
    with st.expander("原始记录与 token IDs", key=f"{key}:raw", on_change="rerun") as panel:
        if panel.open:
            st.json(row, expanded=1)


def _document_details(documents, directory, summary, key):
    st.subheader("逐文本分析")
    order = st.selectbox("文本排序", ["压缩倍率：低到高", "压缩倍率：高到低", "输入长度：长到短", "记忆块数：多到少"],
                         key=f"{key}:order")
    field = {"压缩倍率：低到高": "compression_rate", "压缩倍率：高到低": "compression_rate",
             "输入长度：长到短": "context_tokens", "记忆块数：多到少": "blocks"}[order]
    documents = sorted(documents, key=lambda row: row[field], reverse=order != "压缩倍率：低到高")
    rows = [{"文本": row["id"], "输入 tokens": row["context_tokens"], "记忆块数": row["blocks"],
             "记忆 slots": row["memory_slots"], "pending tokens": row["pending_tokens"],
             "recent tokens": row["recent_tokens"],
             "压缩倍率": row["compression_rate"],
             "问题数": row["question_count"]} for row in documents]
    selection = st.dataframe(
        rows, hide_index=True, height=280, on_select="rerun", selection_mode="single-row-required",
        column_config={"压缩倍率": st.column_config.NumberColumn(format="%.2f×")},
        key=f"{key}:{order}:documents",
    )
    document = documents[selection.selection.rows[0]]
    document_key = f"{key}:{document['id']}"
    st.markdown(f"**{document['id']}** · {document['compression_rate']:.2f}× · {document['blocks']} 个记忆块")
    st.caption("分段宽度对应输入 token 数，蓝色越深表示该文本内的块压缩倍率越高，黄色表示未压缩 pending，灰色表示固定 recent。拖动条内手柄调整显示范围，放大后拖动色块平移，悬停查看块信息。")
    st.echarts_chart(compression.segment_chart(document), height=120, theme=None,
                     key=f"{document_key}:segments")
    with st.expander("记忆块明细"):
        st.dataframe(compression.block_rows(document), hide_index=True,
                     column_config={"压缩倍率": st.column_config.NumberColumn(format="%.2f×")})
    _document_answers(document, directory, summary, document_key)


def _compression(directory, summary, key):
    path = directory / "details.jsonl"
    state = path.stat()
    documents = compression.load_index(str(path), summary["task"], state.st_mtime_ns, state.st_size)
    st.subheader("压缩概览")
    scope = "all"
    if summary["task"] == "ruler":
        left, right = st.columns(2)
        with left:
            length = st.selectbox(
                "压缩分析长度", [None, *sorted({row["length"] for row in documents})],
                format_func=lambda value: "全部长度" if value is None else f"{value:,} tokens",
                key=f"{key}:length",
            )
        documents = [row for row in documents if length is None or row["length"] == length]
        tasks = {row["task"] for row in documents}
        with right:
            subtask = st.selectbox(
                "压缩分析子任务", [None, *(task for task in RULER_TASKS if task in tasks)],
                format_func=lambda value: "全部子任务" if value is None else value,
                key=f"{key}:{length}:task",
            )
        documents = [row for row in documents if subtask is None or row["task"] == subtask]
        scope = f"{length}:{subtask}"
    if not documents:
        st.info("当前范围没有文本。")
        return
    values = compression.statistics(documents)
    st.caption(f"{len(documents):,} 篇文本 · {len(values['blocks']):,} 个记忆块 · "
               f"{sum(row['question_count'] for row in documents):,} 个问题。"
               + ("LoCoMo 同一对话计为一篇文本。" if summary["task"] == "locomo" else ""))
    for label, name in (("按篇", "documents"), ("按记忆块", "blocks")):
        rates = values[name]
        if not rates:
            continue
        for column, statistic, value in zip(st.columns(3), ("平均", "最低", "最高"),
                                             (sum(rates) / len(rates), min(rates), max(rates)), strict=True):
            column.metric(f"{label} · {statistic}压缩倍率", f"{value:.2f}×")
    left, right = st.columns(2)
    left.metric("平均记忆块数 / 篇", f"{values['mean_blocks']:.2f}")
    right.metric("生成单个记忆块的文本比例", f"{values['single_block_fraction']:.1%}")
    st.caption("按篇倍率 = 输入 tokens /（记忆 slots + pending tokens + recent tokens），各篇等权；按块倍率 = 该块输入 tokens / K，各块等权。"
               "输入长度取自本次评估记录的 Writer 输入 tokens。块数包含末尾强制提交块。")
    st.subheader("压缩分布")
    left, right = st.columns(2)
    with left:
        st.markdown("**整篇压缩倍率**")
        st.echarts_chart(compression.histogram(values["documents"], "文本数"), height=330,
                         theme=None, key=f"{key}:document-histogram")
    with right:
        st.markdown("**单块压缩倍率**")
        if values["blocks"]:
            st.echarts_chart(compression.histogram(values["blocks"], "记忆块数"), height=330,
                             theme=None, key=f"{key}:block-histogram")
        else:
            st.info("当前范围的正文均以 pending 和 recent 原文保留，未生成记忆块。")
    st.markdown("**输入长度—记忆块数**")
    st.echarts_chart(compression.length_scatter(documents), height=350,
                     theme=None, key=f"{key}:length-scatter")
    _document_details(documents, directory, summary, f"{key}:{scope}")


def _render_task(directory, task, baseline_results):
    records = load_completed_evaluations(directory / task)
    if not records:
        st.info(f"暂无已完成的 {TASK_LABELS[task]} 评估。")
        return
    key = f"{directory}:{task}"
    steps = [summary["checkpoint_step"] for _, summary in records]
    summaries = [summary for _, summary in records]
    views = [_benchmark_view(summary) for summary in summaries] if task != "test" else []
    st.header("多 checkpoint 比较" if len(records) > 1 else "评估概览")
    baselines = _baseline_selection(baseline_results, task, views, key) if views else {}
    st.subheader("质量指标")
    st.dataframe(_summary_rows(summaries, baselines), hide_index=True)
    if task == "ruler" and baselines:
        st.caption("基线 Overall 按对应 checkpoint 的长度范围计算。")
    ruler_length = None
    if task == "ruler" and len(summaries) > 1:
        lengths = sorted({row["length"] for summary in summaries for row in summary["metrics"]["by_length"]})
        ruler_length = st.selectbox("对比长度", [None, *lengths],
                                    format_func=lambda value: "Overall" if value is None else f"{value:,} tokens",
                                    key=f"{key}:compare-length")
    panels = (_test_charts(summaries) if task == "test"
              else evaluation_charts(views, baselines, ruler_length=ruler_length))
    if panels:
        containers = st.columns(len(panels)) if len(panels) > 1 else [st.container()]
        for container, (name, chart) in zip(containers, panels.items(), strict=True):
            with container:
                st.subheader(CHART_LABELS[name])
                st.echarts_chart(chart, height=400, theme=None, key=f"{key}:{name}")
    with st.expander("资源测量与执行设置"):
        st.dataframe(_resource_rows(summaries, baselines), hide_index=True)
        st.caption("生成路径包含正文压缩与回答，耗时按批内参与项分摊；多卡累计耗时与调度墙钟分别展示。"
                   "峰值显存为 worker 的批次观测。本次调度墙钟包含模型加载，续跑时仅覆盖本次执行。")
    with st.expander("输出诊断"):
        st.dataframe([{"累计 step": value["checkpoint_step"],
                       **{OUTPUT_LABELS[name]: count for name, count in value["outputs"].items()}}
                      for value in summaries], hide_index=True)
    st.divider()
    st.header("单 checkpoint 详情")
    if len(records) > 1:
        step = st.selectbox("明细 checkpoint", steps, index=len(steps) - 1,
                            format_func=lambda value: f"global_step_{value}", key=f"{key}:step")
    else:
        step = steps[0]
        st.caption(f"global_step_{step}")
    current_directory, summary = next(pair for pair in records if pair[1]["checkpoint_step"] == step)
    with st.expander("评估配置与统计"):
        st.json(summary, expanded=1)
    if task == "ruler":
        view = _benchmark_view(summary)
        if len(summaries) > 1:
            st.subheader(f"RULER 长度总览 · global_step_{step}")
            st.echarts_chart(evaluation_charts([view], baselines)["evaluation/ruler_lengths"],
                             height=400, theme=None, key=f"{key}:{step}:lengths")
        length = st.selectbox("任务明细长度", summary["protocol"]["lengths"],
                              format_func=lambda value: f"{value:,} tokens", key=f"{key}:{step}:task-length")
        st.subheader(f"RULER 任务明细 · {length:,} tokens")
        st.echarts_chart(ruler_task_chart(view, length, baselines), height=400, theme=None,
                         key=f"{key}:{step}:tasks:{length}")
    st.caption(f"以下压缩分析与逐文本记录：global_step_{step}")
    _compression(current_directory, summary, f"{key}:{step}")


def _evaluations(args):
    evaluations = discover_evaluations(args.dashboard_run_dir)
    if args.dashboard_eval_id is not None and args.dashboard_eval_id not in evaluations:
        st.error(f"指定评估没有已完成结果：{args.dashboard_run_dir}/eval/{args.dashboard_eval_id}")
        return
    if not evaluations:
        st.info("尚未找到已提交的独立评估。")
        return
    eval_ids = list(evaluations)
    eval_id = st.selectbox(
        "Eval ID", eval_ids,
        index=eval_ids.index(args.dashboard_eval_id) if args.dashboard_eval_id is not None else len(eval_ids) - 1,
        key=f"{args.dashboard_run_dir}:{args.dashboard_eval_id}:eval-id",
    )
    directory = evaluations[eval_id]
    st.caption(f"{args.dashboard_run_dir.name} / {eval_id}")
    tasks = [task for task in TASKS if next((directory / task).glob("global_step_*/summary.json"), None)]
    tabs = st.tabs([TASK_LABELS[task] for task in tasks], key=f"{directory}:task", on_change="rerun")
    for task, tab in zip(tasks, tabs):
        if not tab.open:
            continue
        with tab:
            try:
                _render_task(directory, task, args.dashboard_baselines)
            except (OSError, ValueError, KeyError, TypeError) as error:
                st.error(f"{TASK_LABELS[task]} 结果无法展示：{error}")
                st.exception(error)


def main():
    args = parse_args()
    st.set_page_config(page_title="v3.1.1 独立评估", layout="wide")
    st.title("v3.1.1 独立评估")
    st.sidebar.caption(str(args.dashboard_run_dir))
    st.sidebar.button("刷新结果")
    _evaluations(args)


if __name__ == "__main__":
    main()
