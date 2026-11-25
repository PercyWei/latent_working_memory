"""方法无关的 LoCoMo 看板组件：指标汇总、逐题查看与页面入口。"""

import argparse
import json
from pathlib import Path

import streamlit as st


@st.cache_data(show_spinner=False)
def load_details(path, modified_ns):
    """结果文件更新后刷新缓存。"""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def show_scores(metrics, extra_metrics=()):
    """显示官方 F1；方法可追加已格式化的（标签，数值）指标卡。"""
    cards = [("总体 F1", f"{metrics['score']:.2%}"), ("题数", f"{metrics['samples']:,}"), *extra_metrics]
    for column, (label, value) in zip(st.columns(len(cards)), cards, strict=True):
        column.metric(label, value)
    st.caption("总体及各题型 F1 均按题平均。")

    st.subheader("题型表现")
    st.dataframe([
        {"题型": group["category_name"], "题数": group["samples"], "F1 (%)": group["score"] * 100}
        for group in metrics["groups"]
    ], hide_index=True, column_config={
        "F1 (%)": st.column_config.ProgressColumn(format="%.2f%%", min_value=0, max_value=100),
    })


def show_answers(directory, metrics):
    st.subheader("逐题结果")
    path = directory / "details.jsonl"
    records = load_details(path, path.stat().st_mtime_ns)
    categories = {group["category"]: group["category_name"] for group in metrics["groups"]}
    conversations = list(dict.fromkeys(row["metadata"]["conversation_id"] for row in records))
    left, right = st.columns(2)
    category = left.selectbox(
        "题型", [None, *categories],
        format_func=lambda value: "全部" if value is None else categories[value], key="category",
    )
    conversation = right.selectbox(
        "对话", [None, *conversations],
        format_func=lambda value: "全部" if value is None else value, key="conversation",
    )
    records = [
        row for row in records
        if (category is None or row["metadata"]["category"] == category)
        and (conversation is None or row["metadata"]["conversation_id"] == conversation)
    ]
    if not records:
        st.info("当前筛选条件下没有题目。")
        return
    st.caption(f"符合筛选条件：{len(records):,} 道题")
    selected = st.selectbox(
        "题目", range(len(records)),
        format_func=lambda index: f"{records[index]['id']} · {records[index]['metadata']['question']}",
        key=f"question:{category}:{conversation}",
    )
    row = records[selected]
    st.text(row["metadata"]["question"])
    st.caption(f"F1 {row['score']:.2%}")
    prediction, references = st.columns(2)
    prediction.markdown("**模型回答**")
    prediction.code(row["prediction"], language=None, wrap_lines=True)
    references.markdown("**参考答案**")
    references.code("\n".join(row["references"]), language=None, wrap_lines=True)


def run_dashboard(method, title, render_results, argv=None):
    """方法维护入口与渲染回调；公共层读取单次运行的 LoCoMo 结果。"""
    parser = argparse.ArgumentParser(description=f"独立查看 {title} 的 benchmark 结果，当前支持 LoCoMo。")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="单次评估运行目录，其下包含 locomo/summary.json 和 locomo/details.jsonl")
    directory = parser.parse_args(argv).output_dir.expanduser().resolve()
    if not directory.is_dir():
        parser.error(f"--output-dir 必须指向已有的评估运行目录：{directory}")
    st.set_page_config(page_title=f"{title} · LoCoMo", layout="wide")
    st.title(f"{title} · {directory.name}")
    task_dir = directory / "locomo"
    summary_path = task_dir / "summary.json"
    if not summary_path.is_file():
        st.info(f"在 {directory} 下未找到 locomo/summary.json，请在评估完成后刷新看板。")
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["method"] != method or summary["benchmark"] != "locomo":
        st.error(f"该目录不是 {title} 的 LoCoMo 评估结果。")
        return
    render_results(task_dir, summary)
