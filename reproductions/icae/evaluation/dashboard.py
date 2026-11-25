"""ICAE 的 benchmark 结果看板。"""

from pathlib import Path

import streamlit as st

from benchmarks.locomo.dashboard import run_dashboard, show_answers, show_scores


STAGE_LABELS = {"pretrained": "预训练", "finetuned": "指令微调"}


def show_results(directory, summary):
    settings = summary["settings"]
    memory = summary["method_metrics"]
    st.caption(f"LoCoMo · {STAGE_LABELS[settings['stage']]} · {Path(settings['model_path']).name}"
               f" · 每块 {settings['fixed_mem_size']} 槽 · 目标压缩率 {settings['mean_compression_rate']}×")
    show_scores(summary["metrics"], [
        ("平均记忆 token 数", f"{memory['mean_memory_tokens']:,.1f}"),
        ("平均保留比例", f"{memory['mean_retention_ratio']:.2%}"),
    ])
    st.caption("记忆用量按对话平均；保留比例为记忆 token 数除以正文 token 数。")
    with st.expander("各段对话的记忆用量"):
        st.dataframe([
            {"对话": cid, "正文 token 数": stats["context_tokens"],
             "记忆 token 数": stats["memory_tokens"],
             "保留比例 (%)": stats["memory_tokens"] / stats["context_tokens"] * 100}
            for cid, stats in summary["conversations"].items()
        ], hide_index=True, column_config={
            "保留比例 (%)": st.column_config.NumberColumn(format="%.2f%%"),
        })
    show_answers(directory, summary["metrics"])


if __name__ == "__main__":
    run_dashboard("icae", "ICAE", show_results)
