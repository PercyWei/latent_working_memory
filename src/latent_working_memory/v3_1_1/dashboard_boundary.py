"""从本地自主决策记录绘制跨 step 的压缩概率分布。"""

from collections import defaultdict
import json
from pathlib import Path

import streamlit as st

from latent_working_memory.benchmarks.data import RULER_TASKS


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
                    counts[min(9, int(probability * 10))] += 1
                    decisions.append([pending, probability])
                    sums[pending] += probability
                    sizes[pending] += 1
    means = [[pending, sums[pending] / sizes[pending]] for pending in sorted(sums)]
    return len(documents), trajectories, counts, decisions, means


def charts(snapshots):
    """两张图共享 step 颜色、图例与自主决策口径。"""
    labels = [f"[{i / 10:.1f}, {(i + 1) / 10:.1f}{']' if i == 9 else ')'}" for i in range(10)]
    legend = []
    bars, points = [], []
    for step, temperature, (documents, trajectories, counts, decisions, means) in snapshots:
        total = sum(counts)
        name = f"step {step} · {documents:,} 篇 · {trajectories:,} 条轨迹 · {total:,} 个决策 · T={temperature:g}"
        legend.append(name)
        # 按真实 step 定色，在筛选 step 或切换数据来源后保持一致。
        color = f"hsl({(210 + step * 137.508) % 360:.1f}, 62%, 43%)"
        bars.append({
            "name": name, "type": "bar", "itemStyle": {"color": color},
            "data": [100 * count / total if total else None for count in counts],
        })
        for label, data, symbol, size, opacity, z in (
            ("压缩概率", decisions, "circle", 5, 0.35, 2),
            ("平均压缩概率", means, "rect", [13, 3], 1, 4),
        ):
            points.append({
                "name": name, "type": "scatter", "data": data, "symbol": symbol,
                "symbolSize": size, "itemStyle": {"color": color, "opacity": opacity},
                "tooltip": {"formatter": f"step {step}<br/>pending, {label}：{{c}}"},
                "progressive": 5000, "progressiveThreshold": 10000,
                "z": z,
            })
    common = {
        "animation": False, "legend": {"type": "scroll", "data": legend, "top": 0},
        "grid": {"left": 64, "right": 25, "top": 70, "bottom": 69},
    }
    histogram = {
        **common, "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "xAxis": {"type": "category", "data": labels, "axisLabel": {"rotate": 25, "interval": 0},
                  "name": "压缩概率", "nameLocation": "middle", "nameGap": 48},
        "yAxis": {"type": "value", "name": "自主决策位置占比", "min": 0, "max": 100,
                  "axisLabel": {"formatter": "{value}%"}},
        "series": bars,
    }
    scatter = {
        **common, "tooltip": {"trigger": "item"},
        "xAxis": {"type": "value", "min": 0, "minInterval": 1, "name": "决策前 pending 长度（tokens）",
                  "nameLocation": "middle", "nameGap": 36},
        "yAxis": {"type": "value", "min": 0, "max": 1, "name": "压缩概率"},
        "series": points,
    }
    points[0]["markLine"] = {
        "silent": True, "symbol": "none", "label": {"show": False},
        "lineStyle": {"color": "#777", "type": "dashed", "width": 1},
        "data": [{"yAxis": 0.5}],
    }
    return histogram, scatter


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
    histogram, scatter = charts(snapshots)
    st.markdown("**压缩概率分布**")
    st.echarts_chart(histogram, height=355, theme=None, key=f"{key}:probability-histogram")
    st.markdown("**pending 长度与压缩概率**")
    st.caption("每个圆点表示一次自主决策时的压缩概率，短横线表示同一 step、同一 pending 长度的均值。")
    st.echarts_chart(scatter, height=375, theme=None, key=f"{key}:probability-scatter")


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
