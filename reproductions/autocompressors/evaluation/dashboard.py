"""AutoCompressors 的 benchmark 结果看板。"""

from pathlib import Path

import streamlit as st

from benchmarks.checkpoint_dashboard import run_dashboard


def show_memory(runs):
    overview = []
    conversations = []
    for run in runs:
        label = run["label"]
        summary = run["summary"]
        settings = summary["settings"]
        memory = summary["method_metrics"]
        overview.append({
            "压缩设置": label,
            "模型": Path(settings["checkpoint"]).name,
            "分段长度": settings["segment_length"],
            "平均摘要 token 数": memory["mean_memory_tokens"],
            "平均保留比例 (%)": 100 * memory["mean_retention_ratio"],
        })
        for conversation_id, stats in summary["conversations"].items():
            conversations.append({
                "压缩设置": label,
                "对话": conversation_id,
                "正文 token 数": stats["context_tokens"],
                "摘要 token 数": stats["memory_tokens"],
                "保留比例 (%)": 100 * stats["memory_tokens"] / stats["context_tokens"],
            })

    st.subheader("LoCoMo 摘要用量")
    st.caption("保留比例为每段对话的摘要 token 数除以正文 token 数，表中均值按对话等权计算。")
    st.dataframe(overview, hide_index=True, column_config={
        "平均摘要 token 数": st.column_config.NumberColumn(format="%.1f"),
        "平均保留比例 (%)": st.column_config.NumberColumn(format="%.2f%%"),
    })
    with st.expander("各段对话的摘要用量"):
        st.dataframe(conversations, hide_index=True, column_config={
            "保留比例 (%)": st.column_config.NumberColumn(format="%.2f%%"),
        })


if __name__ == "__main__":
    run_dashboard("autocompressors", "AutoCompressors", show_memory)
