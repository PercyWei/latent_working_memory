"""E²-TTT 的 benchmark 结果看板。"""

from pathlib import Path

import streamlit as st

from benchmarks.locomo.dashboard import run_dashboard, show_answers, show_scores


def show_results(directory, summary):
    st.caption(f"LoCoMo · {Path(summary['settings']['checkpoint']).name}")
    show_scores(summary["metrics"])
    show_answers(directory, summary["metrics"])


if __name__ == "__main__":
    run_dashboard("e2-ttt", "E²-TTT", show_results)
