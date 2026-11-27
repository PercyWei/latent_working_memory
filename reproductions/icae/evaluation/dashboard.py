"""ICAE 的 benchmark 结果看板。"""

from pathlib import Path

import streamlit as st

from benchmarks.checkpoint_dashboard import run_dashboard


STAGE_LABELS = {"pretrained": "预训练", "finetuned": "指令微调"}


def show_memory(runs):
    overview = []
    conversations = []
    for run in runs:
        label = run["label"]
        summary = run["summary"]
        settings = summary["settings"]
        memory = summary["method_metrics"]
        overview.append({
            "运行": label,
            "阶段": STAGE_LABELS[settings["stage"]],
            "基座": Path(settings["model_path"]).name,
            "每块记忆槽数": settings["fixed_mem_size"],
            "目标压缩率": settings["mean_compression_rate"],
            "平均记忆 token 数": memory["mean_memory_tokens"],
            "平均保留比例 (%)": 100 * memory["mean_retention_ratio"],
        })
        for conversation_id, stats in summary["conversations"].items():
            conversations.append({
                "运行": label,
                "对话": conversation_id,
                "正文 token 数": stats["context_tokens"],
                "记忆 token 数": stats["memory_tokens"],
                "保留比例 (%)": 100 * stats["memory_tokens"] / stats["context_tokens"],
            })

    st.subheader("LoCoMo 记忆用量")
    st.caption("保留比例为每段对话的记忆 token 数除以正文 token 数，表中均值按对话等权计算。")
    st.dataframe(overview, hide_index=True, column_config={
        "平均记忆 token 数": st.column_config.NumberColumn(format="%.1f"),
        "平均保留比例 (%)": st.column_config.NumberColumn(format="%.2f%%"),
    })
    with st.expander("各段对话的记忆用量"):
        st.dataframe(conversations, hide_index=True, column_config={
            "保留比例 (%)": st.column_config.NumberColumn(format="%.2f%%"),
        })


if __name__ == "__main__":
    run_dashboard("icae", "ICAE", show_memory)
