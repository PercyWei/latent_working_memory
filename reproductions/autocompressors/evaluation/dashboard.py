"""AutoCompressors 的 benchmark 结果看板。"""

from pathlib import Path

import streamlit as st

from benchmarks.locomo.dashboard import run_dashboard, show_answers, show_scores


def show_results(directory, summary):
    settings = summary["settings"]
    memory = summary["method_metrics"]
    conversations = summary["conversations"]
    st.caption(f"LoCoMo · {Path(settings['checkpoint']).name} · 分段长度 {settings['segment_length']:,}")
    show_scores(summary["metrics"], [
        ("平均记忆 token 数", f"{memory['mean_memory_tokens']:,.1f}"),
        ("平均保留比例", f"{memory['mean_retention_ratio']:.2%}"),
    ])
    st.caption("记忆用量按对话平均；保留比例为摘要 token 数除以正文 token 数。")
    with st.expander("各段对话的记忆用量"):
        st.dataframe([
            {"对话": cid, "正文 token 数": stats["context_tokens"],
             "记忆 token 数": stats["memory_tokens"],
             "保留比例 (%)": stats["memory_tokens"] / stats["context_tokens"] * 100}
            for cid, stats in conversations.items()
        ], hide_index=True, column_config={
            "保留比例 (%)": st.column_config.NumberColumn(format="%.2f%%"),
        })
    show_answers(directory, summary["metrics"])


if __name__ == "__main__":
    run_dashboard("autocompressors", "AutoCompressors", show_results)
