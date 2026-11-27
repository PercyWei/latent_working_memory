"""按论文展示同一 QASPER 划分中各运行的 Answer F1 和逐题回答。"""

import argparse
import json
from pathlib import Path

import streamlit as st

from benchmarks.dashboard_data import load_details
from benchmarks.qasper.data import SPLIT_FILES


TASK_LABELS = {f"qasper/{split}": f"QASPER · {split}" for split in SPLIT_FILES}


def load_runs(runs):
    """读取一组（展示标签，已评分的划分目录），按同一划分的 question_id 对齐。"""
    loaded = []
    identity = None
    for label, directory in runs:
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if summary["benchmark"] != "qasper" or summary["metrics"]["metric"] != "answer_f1":
            raise ValueError(f"{label} 需要 QASPER Answer F1 结果。")
        source = (summary["split"], summary["data"]["fingerprint"])
        if identity is not None and source != identity:
            raise ValueError("参与比较的运行需要使用相同的 QASPER 划分和数据。")
        identity = source
        path = directory / "details.jsonl"
        records = {r["question_id"]: r for r in load_details(path, path.stat().st_mtime_ns)}
        loaded.append({"label": label, "directory": directory, "summary": summary, "records": records})
    return loaded


def show_scores(runs):
    totals, groups = [], []
    for run in runs:
        label, summary = run["label"], run["summary"]
        metrics = summary["metrics"]
        totals.append({"运行": label, "划分": summary["split"], "题数": metrics["samples"],
                       "Answer F1": metrics["score"]})
        groups.extend({"运行": label, "答案类型": item["answer_type"], "题数": item["samples"],
                       "Answer F1": item["score"]} for item in metrics["groups"])
    f1_format = {"Answer F1": st.column_config.NumberColumn(format="percent")}
    for title, rows in (("总体表现", totals), ("分答案类型表现", groups)):
        st.subheader(title)
        st.dataframe(rows, hide_index=True, column_config=f1_format)


def show_answers(runs):
    st.subheader("逐题回答")
    questions = runs[0]["records"]
    papers = list(dict.fromkeys(row["paper_id"] for row in questions.values()))
    paper_id = st.selectbox("论文", papers)
    question_ids = [qid for qid, row in questions.items() if row["paper_id"] == paper_id]
    qid = st.selectbox("问题", question_ids, format_func=lambda q: questions[q]["question"])
    row = questions[qid]
    st.caption(f"题号 {qid} · 人工参考答案")
    st.code("\n".join(ref["answer"] for ref in row["references"]), language=None, wrap_lines=True)
    columns = st.columns(min(3, len(runs)))
    for i, run in enumerate(runs):
        answer = run["records"][qid]
        with columns[i % len(columns)]:
            st.markdown(f"**{run['label']}**")
            st.caption(f"{answer['answer_type']} · Answer F1 {answer['score']:.2%}")
            st.code(answer["predicted_answer"], language=None, wrap_lines=True)


def show_runs(runs):
    try:
        loaded = load_runs(runs)
    except (OSError, KeyError, ValueError) as error:
        st.error(str(error))
        return
    show_scores(loaded)
    show_answers(loaded)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs="+", action="extend", required=True,
                        metavar="LABEL DIRECTORY",
                        help="连续传入一组或多组标签和目录；目录直接包含 summary.json、details.jsonl")
    args = parser.parse_args(argv)
    if len(args.run) % 2:
        parser.error("--run 必须按“标签 目录”成对传入。")
    paths = {}
    for label, directory in zip(args.run[::2], args.run[1::2], strict=True):
        if not label.strip() or label in paths:
            parser.error("运行标签必须非空且不重复。")
        paths[label] = Path(directory).expanduser().resolve()
    st.set_page_config(page_title="QASPER · 运行比较", layout="wide")
    st.title("QASPER · 运行比较")
    show_runs(paths.items())


if __name__ == "__main__":
    main()
