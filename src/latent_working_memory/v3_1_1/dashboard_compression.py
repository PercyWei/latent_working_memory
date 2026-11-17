"""独立评估的轻量文本索引与压缩图，不保存图表副本。"""

from html import escape
import json
import math
from pathlib import Path

import streamlit as st


@st.cache_data(max_entries=8, show_spinner="正在整理压缩记录…")
def load_index(path, task, modified_ns, size):
    """按文件身份缓存压缩字段与行偏移；答案和 token IDs 按需读取。"""
    documents = {}
    with Path(path).open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            row = json.loads(line)
            metadata = row.get("metadata", {})
            identity = metadata["conversation_id"] if task == "locomo" else row["id"]
            if identity not in documents:
                documents[identity] = {
                    "id": identity, "context_tokens": row["context_tokens"],
                    "blocks": row["blocks"], "memory_slots": row["memory_slots"],
                    "block_lengths": row["block_lengths"],
                    "compression_rate": row["context_tokens"] / row["memory_slots"],
                    "slots_per_block": row["memory_slots"] // row["blocks"],
                    "length": metadata.get("length"), "task": metadata.get("task"),
                    "questions": [], "question_count": 0,
                }
            document = documents[identity]
            document["questions"].append({
                "offset": offset, "id": row["id"],
                "category": metadata.get("category"),
                "category_name": metadata.get("category_name"),
            })
            document["question_count"] += len(row["qas"]) if task == "test" else 1
    return list(documents.values())


def read_record(path, offset):
    with Path(path).open("rb") as stream:
        stream.seek(offset)
        return json.loads(stream.readline())


def statistics(documents):
    document_rates = [row["compression_rate"] for row in documents]
    block_rates = [length / row["slots_per_block"]
                   for row in documents for length in row["block_lengths"]]
    return {
        "documents": document_rates, "blocks": block_rates,
        "mean_blocks": sum(row["blocks"] for row in documents) / len(documents),
        "single_block_fraction": sum(row["blocks"] == 1 for row in documents) / len(documents),
    }


def histogram(values, unit):
    lower, upper = min(values), max(values)
    count = min(40, max(1, math.ceil(math.sqrt(len(values))))) if lower != upper else 1
    width = (upper - lower) / count if lower != upper else 1
    frequencies = [0] * count
    for value in values:
        frequencies[min(count - 1, int((value - lower) / width))] += 1
    precision = max(2, min(6, 1 - math.floor(math.log10(width))))
    labels = ([f"{lower:.2f}×"] if lower == upper else [
        f"{lower + index * width:.{precision}f}–{lower + (index + 1) * width:.{precision}f}×"
        for index in range(count)
    ])
    return {
        "animation": False,
        "grid": {"left": 55, "right": 20, "bottom": 70, "top": 35},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "xAxis": {"type": "category", "data": labels, "name": "压缩倍率",
                  "nameLocation": "middle", "nameGap": 50, "axisLabel": {"rotate": 25}},
        "yAxis": {"type": "value", "name": unit, "minInterval": 1},
        "series": [{"type": "bar", "name": unit, "data": frequencies,
                    "barCategoryGap": "5%", "itemStyle": {"color": "#31688e"}}],
    }


def length_scatter(documents):
    data = [{
        "value": [row["context_tokens"], row["blocks"]],
        "tooltip": {"formatter": (
            f"{escape(str(row['id']))}<br>输入：{row['context_tokens']:,} tokens"
            f"<br>记忆块：{row['blocks']}<br>压缩倍率：{row['compression_rate']:.2f}×"
        )},
    } for row in documents]
    return {
        "animation": False,
        "grid": {"left": 55, "right": 30, "bottom": 60, "top": 35},
        "tooltip": {"trigger": "item"},
        "xAxis": {"type": "value", "name": "输入长度（tokens）", "nameLocation": "middle", "nameGap": 35},
        "yAxis": {"type": "value", "name": "记忆块数", "minInterval": 1},
        "series": [{"type": "scatter", "data": data, "symbolSize": 9,
                    "itemStyle": {"color": "#31688e", "opacity": 0.7}}],
    }


def block_rows(document):
    rows, start = [], 0
    slots = document["slots_per_block"]
    for index, length in enumerate(document["block_lengths"]):
        rows.append({
            "块": index + 1, "起始 token": start, "结束 token": start + length,
            "输入 tokens": length, "记忆 slots": slots, "压缩倍率": length / slots,
            "提交方式": "末尾强制提交" if index + 1 == document["blocks"] else "策略提交",
        })
        start += length
    return rows


def segment_chart(document):
    rows = block_rows(document)
    rates = [row["压缩倍率"] for row in rows]
    lower, upper = min(rates), max(rates)
    track = {"left": 24, "right": 40, "top": 16, "height": 42}
    series = []
    for row in rows:
        fraction = (row["压缩倍率"] - lower) / (upper - lower) if upper != lower else 0.5
        color = "#" + "".join(f"{round(light + (dark - light) * fraction):02x}"
                              for light, dark in zip((186, 214, 232), (32, 93, 137), strict=True))
        tooltip = (f"记忆块 {row['块']} · {row['提交方式']}"
                   f"<br>token 区间：[{row['起始 token']:,}, {row['结束 token']:,})"
                   f"<br>输入：{row['输入 tokens']:,} tokens → {row['记忆 slots']} slots"
                   f"<br>压缩倍率：{row['压缩倍率']:.2f}×")
        series.append({
            "name": f"块 {row['块']}", "type": "bar", "stack": "document", "barWidth": track["height"],
            "z": 2, "emphasis": {"disabled": True},
            "data": [row["输入 tokens"]], "tooltip": {"formatter": tooltip},
            "itemStyle": {"color": color, "borderColor": "#ffffff", "borderWidth": 1},
        })
    return {
        "animation": False, "grid": track,
        "tooltip": {"trigger": "item"},
        "xAxis": {"type": "value", "min": 0, "max": document["context_tokens"],
                  "name": "输入 token 位置", "nameLocation": "middle", "nameGap": 28},
        "yAxis": {"type": "category", "data": ["正文"], "show": False},
        # 同层时色块高于透明滑轨、低于手柄；关闭色块高亮，避免悬停时盖住手柄。
        "dataZoom": [{"type": "slider", "xAxisIndex": 0, "filterMode": "none", **track,
                      "z": 2, "backgroundColor": "transparent", "fillerColor": "transparent",
                      "borderColor": "transparent", "showDataShadow": False, "showDetail": False,
                      "brushSelect": False, "handleStyle": {"color": "#ffffff", "borderColor": "#31688e"}},
                     {"type": "inside", "xAxisIndex": 0, "filterMode": "none"}],
        "series": series,
    }
