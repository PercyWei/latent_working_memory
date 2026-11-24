"""从本地自主决策记录绘制跨 step 的压缩概率分布。"""

from collections import defaultdict
import json
from pathlib import Path

import pyarrow as pa
import streamlit as st

from benchmarks.data import RULER_TASKS


def _probability_bin(probability):
    # 最后一档 [0.9, 1.0] 包含右端点。
    return min(9, int(probability * 10))


@st.cache_data(max_entries=12, show_spinner="正在读取边界决策…")
def load_points(files, length, subtask):
    """按文件身份和评估范围缓存，保留重复轨迹中的每个真实决策。"""
    documents = set()
    trajectories = 0
    counts = [0] * 10
    decisions = []
    sums = defaultdict(float)
    sizes = defaultdict(int)
    for path, _mtime_ns, _size in files:
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if length is not None and row["length"] != length:
                    continue
                if subtask is not None and row["task"] != subtask:
                    continue
                documents.add(row["id"])
                trajectories += 1
                pairs = zip(row["pending_tokens_before"], row["compress_probs"], strict=True)
                for pending, probability in pairs:
                    counts[_probability_bin(probability)] += 1
                    decisions.append([pending, probability])
                    sums[pending] += probability
                    sizes[pending] += 1
    means = [[pending, sums[pending] / sizes[pending]] for pending in sorted(sums)]
    return len(documents), trajectories, counts, decisions, means


def _step_color(step):
    return f"hsl({(210 + step * 137.508) % 360:.1f}, 62%, 43%)"


def _histogram(rows):
    labels = [f"[{i / 10:.1f}, {(i + 1) / 10:.1f}{']' if i == 9 else ')'}" for i in range(10)]
    bars = []
    for step, name, counts in rows:
        total = sum(counts)
        bars.append({
            "name": name, "type": "bar", "itemStyle": {"color": _step_color(step)},
            "data": [100 * count / total if total else None for count in counts],
        })
    return {
        "animation": False, "legend": {"type": "scroll", "data": [name for _, name, _ in rows], "top": 0},
        "grid": {"left": 64, "right": 25, "top": 70, "bottom": 69},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "xAxis": {"type": "category", "data": labels, "axisLabel": {"rotate": 25, "interval": 0},
                  "name": "压缩概率", "nameLocation": "middle", "nameGap": 48},
        "yAxis": {"type": "value", "name": "自主决策位置占比", "min": 0, "max": 100,
                  "axisLabel": {"formatter": "{value}%"}},
        "series": bars,
    }


def conditional_histogram(snapshots, pending):
    rows = []
    for step, _, (_, _, _, decisions, _) in snapshots:
        counts = [0] * 10
        for length, probability in decisions:
            if length == pending:
                counts[_probability_bin(probability)] += 1
        rows.append((step, f"step {step} · {sum(counts):,} 个自主决策", counts))
    return _histogram(rows)


def _scatter(snapshots, selected_pending):
    tables = []
    steps = [step for step, _, _ in snapshots]
    for step, _, (_, _, _, decisions, means) in snapshots:
        for is_mean, pairs in ((False, decisions), (True, means)):
            tables.append(pa.table({
                "pending": pa.array((x for x, _ in pairs), type=pa.int32()),
                "probability": pa.array((p for _, p in pairs), type=pa.float64()),
                "step": pa.repeat(step, len(pairs)), "mean": pa.repeat(is_mean, len(pairs)),
            }))

    position = {
        "x": {"field": "pending", "type": "quantitative", "title": "决策前 pending 长度（tokens）",
              "scale": {"zero": True}},
        "y": {"field": "probability", "type": "quantitative", "title": "压缩概率",
              "scale": {"domain": [0, 1]}},
        "color": {"field": "step", "type": "nominal", "scale": {
            "domain": steps, "range": [_step_color(step) for step in steps],
        }, "legend": {"orient": "top", "title": "step"}},
    }
    tooltip = [
        {"field": "step", "type": "nominal", "title": "step"},
        {"field": "pending", "type": "quantitative", "title": "pending"},
        {"field": "probability", "type": "quantitative", "title": "压缩概率", "format": ".4f"},
    ]
    visible = {"filter": {"not": {"param": "hidden_steps", "empty": False}}}
    mean_filter = [visible, {"filter": "datum.mean"}]
    rule = {"type": "rule", "strokeDash": [5, 4], "color": "#777"}
    layers = [
        {"data": {"values": [{}]}, "mark": rule, "encoding": {"y": {"datum": 0.5}}},
        {"data": {"values": [{}]}, "mark": rule, "encoding": {"x": {"datum": selected_pending}}},
        {
            "name": "decisions", "transform": [visible, {"filter": "!datum.mean"}],
            "mark": {"type": "circle", "size": 23, "opacity": 0.35, "cursor": "pointer"},
            "encoding": {**position, "tooltip": tooltip},
            "params": [{"name": "pick", "select": {
                "type": "point", "fields": ["pending"], "toggle": False, "clear": False,
                "on": "@decisions_marks:click, @means_marks:click",
            }}, {"name": "hidden_steps", "select": {"type": "point", "fields": ["step"], "toggle": "true"},
                 "bind": "legend"}],
        },
        {
            "transform": mean_filter, "mark": {"type": "line", "strokeWidth": 2},
            "encoding": {**position, "order": {"field": "pending", "type": "quantitative"}},
        },
        {
            "name": "means", "transform": mean_filter,
            "mark": {"type": "point", "shape": "square", "filled": True, "size": 35, "cursor": "pointer"},
            "encoding": {**position, "tooltip": [*tooltip[:-1], {**tooltip[-1], "title": "平均压缩概率"}]},
        },
    ]
    scatter = {"layer": layers, "usermeta": {"embedOptions": {"renderer": "canvas"}}}
    return pa.concat_tables(tables), scatter


def render(sources, key, length=None, subtask=None):
    steps = sorted(sources)
    if not steps:
        st.info("当前范围暂无已完成的边界诊断记录。")
        return
    defaults = sorted({steps[0], steps[len(steps) // 2], steps[-1]})
    chosen = st.multiselect("对比 step", steps, default=defaults, key=f"{key}:steps")
    if not chosen:
        return
    snapshots = []
    for step in sorted(chosen):
        paths, temperature = sources[step]
        files = []
        for path in paths:
            state = path.stat()
            files.append((str(path), state.st_mtime_ns, state.st_size))
        snapshots.append((step, temperature, load_points(tuple(files), length, subtask)))
    rows = []
    for step, temperature, (documents, trajectories, counts, _, _) in snapshots:
        name = f"step {step} · {documents:,} 篇 · {trajectories:,} 条轨迹 · {sum(counts):,} 个决策 · T={temperature:g}"
        rows.append((step, name, counts))
    st.markdown("**压缩概率分布**")
    st.echarts_chart(_histogram(rows), height=355, theme=None, key=f"{key}:probability-histogram")
    lengths = sorted({pending for _, _, (_, _, _, _, means) in snapshots for pending, _ in means})
    if not lengths:
        st.info("所选 step 没有自主决策位置。")
        return
    pending_key = f"{key}:pending"
    if st.session_state.get(pending_key) not in lengths:
        st.session_state[pending_key] = lengths[0]
    pending = st.session_state[pending_key]
    # 换长度后重置点选事件，允许下拉选择后再次点击原来的点。
    chart_key = f"{key}:probability-scatter:{pending}"

    def select_point():
        picked = st.session_state[chart_key].selection["pick"]
        if picked:
            st.session_state[pending_key] = picked[0]["pending"]

    data, scatter = _scatter(snapshots, pending)
    st.markdown("**pending 长度与压缩概率**")
    st.caption("圆点表示自主决策，折线连接各长度的均值。点击圆点或均值标记，可查看该长度的概率分布。")
    st.vega_lite_chart(data, scatter, width="stretch", height=375, theme=None,
                       key=chart_key, on_select=select_point, selection_mode="pick")
    st.selectbox("pending 长度（tokens）", lengths, key=pending_key)
    st.markdown(f"**pending = {pending:,} 时的压缩概率分布**")
    st.echarts_chart(conditional_histogram(snapshots, pending), height=355, theme=None,
                     key=f"{key}:pending-histogram")


def training(run_dir):
    root = run_dir / "train" / "boundary"
    split = st.radio("数据集", ["dev", "train"], horizontal=True, key=f"{root}:split")
    if split == "dev":
        st.caption("固定 dev 文档上的 greedy 轨迹，概率按温度 1 计算。")
    sources = {}
    for path in root.glob(f"global_step_*/{split}/summary.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        paths = [path.parent / f"rank_{rank}.jsonl" for rank in range(summary["world_size"])]
        sources[summary["step"]] = paths, summary["temperature"]
    render(sources, f"{root}:{split}")


def evaluation(records, task, key):
    st.subheader("边界决策")
    sources = {}
    length = subtask = None
    if task == "ruler":
        lengths = sorted({length for _, summary in records for length in summary["protocol"]["lengths"]})
        left, right = st.columns(2)
        length = left.selectbox("边界分析长度", lengths,
                                format_func=lambda value: f"{value:,} tokens", key=f"{key}:boundary-length")
        subtask = right.selectbox("边界分析子任务", [None, *RULER_TASKS],
                                  format_func=lambda value: value or "全部子任务", key=f"{key}:boundary-task")
        records = [(directory / "lengths" / str(length), summary) for directory, summary in records
                   if length in summary["protocol"]["lengths"]]
    for directory, summary in records:
        path = directory / "boundary.jsonl"
        if path.is_file():
            step = summary["checkpoint_step"]
            sources[step] = [path], 1.0
    missing = [str(summary["checkpoint_step"]) for _, summary in records
               if summary["checkpoint_step"] not in sources]
    if missing and sources:
        st.caption("缺少边界诊断的 step：" + "、".join(missing))
    render(sources, f"{key}:boundary:{length}:{subtask}", length, subtask)
