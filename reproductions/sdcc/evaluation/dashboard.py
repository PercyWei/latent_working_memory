"""SDCC 的 LoCoMo 结果与各对话的压缩档位。"""

from pathlib import Path

import streamlit as st

from benchmarks.locomo.dashboard import run_dashboard, show_answers, show_scores


def show_results(directory, summary):
    settings = summary["settings"]
    memory = summary["method_metrics"]
    st.caption(f"LoCoMo · {Path(settings['model_path']).name} · 压缩偏移 {settings['compress_ratio_scale']:g}")
    show_scores(summary["metrics"], [
        ("平均记忆 token 数", f"{memory['mean_memory_tokens']:,.1f}"),
        ("平均保留比例", f"{memory['mean_retention_ratio']:.2%}"),
    ])
    st.caption("记忆用量按对话平均。选中档位是池化窗口的大小，实际保留比例按记忆 token 数除以正文 token 数计算。")
    with st.expander("各段对话的压缩档位与记忆用量"):
        st.dataframe([
            {
                "对话": cid,
                "选中档位": f"{stats['compression_factor']}×",
                "正文 token 数": stats["context_tokens"],
                "记忆 token 数": stats["memory_tokens"],
                "保留比例 (%)": 100 * stats["memory_tokens"] / stats["context_tokens"],
            }
            for cid, stats in summary["conversations"].items()
        ], hide_index=True, column_config={
            "保留比例 (%)": st.column_config.NumberColumn(format="%.2f%%"),
        })
    show_answers(directory, summary["metrics"])


if __name__ == "__main__":
    run_dashboard("sdcc", "SDCC", show_results, qasper=True)
