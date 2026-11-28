"""按推理跳数展示 MuSiQue 各运行的 Answer F1、EM、Judge Accuracy 和逐题回答。"""

import json

import streamlit as st

from benchmarks.comparison_dashboard import run_comparison, show_judge_accuracy_chart
from benchmarks.dashboard_data import load_details, load_judge_results, show_judge_answer, show_judge_statistics
from benchmarks.musique.data import SPLIT_FILES


TASK_LABELS = {f"musique/{split}": f"MuSiQue · {split}" for split in SPLIT_FILES}


def comparison_panels(metrics):
    """分别比较 F1 与 EM，各图均按总体、2–4 跳排列。"""
    groups = {item["hops"]: item for item in metrics["groups"]}
    return {
        title: {f"总体 {title}": metrics[field], **{
            f"{hops}-hop": groups[hops][field] if groups[hops]["samples"] else None
            for hops in (2, 3, 4)
        }}
        for title, field in (("Answer F1", "score"), ("Answer EM", "exact_match"))
    }


def load_runs(runs):
    """读取一组（展示标签，已评分的划分目录），按 question_id 对齐。"""
    loaded = []
    identity = None
    for label, directory in runs:
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if summary["benchmark"] != "musique" or summary["metrics"]["metric"] != "answer_f1":
            raise ValueError(f"{label} 需要 MuSiQue Answer F1 结果。")
        data = summary["data"]
        if summary["split"] not in SPLIT_FILES or data["split"] != summary["split"]:
            raise ValueError(f"{label} 需要 MuSiQue dev 的评分结果。")
        source = (summary["split"], data["filename"], data["source_release"], data["fingerprint"],
                  summary["scoring"]["source_commit"])
        if identity is not None and source != identity:
            raise ValueError("参与比较的运行需要使用相同的 MuSiQue 划分、数据和评分器。")
        identity = source
        path = directory / "details.jsonl"
        rows = load_details(path, path.stat().st_mtime_ns)
        records = {row["question_id"]: row for row in rows}
        if not records or len(records) != len(rows) or any(not qid for qid in records):
            raise ValueError(f"{label} 的逐题结果包含空题号、重复题号或没有题目。")
        if len(records) != summary["metrics"]["samples"] or len(records) != data["samples"]:
            raise ValueError(f"{label} 的逐题结果没有覆盖完整的已评分题目集合。")
        if loaded:
            first = loaded[0]["records"]
            if records.keys() != first.keys() or any(
                any(row[field] != first[qid][field] for field in ("question", "hops", "references"))
                for qid, row in records.items()
            ):
                raise ValueError(f"{label} 的题目、推理跳数或参考答案与其他运行不一致。")
        judge = load_judge_results(directory, "musique", records.keys())
        loaded.append({"label": label, "directory": directory, "summary": summary, "records": records,
                       "judge": judge})
    return loaded


def show_scores(runs):
    totals, groups = [], []
    for run in runs:
        label, summary = run["label"], run["summary"]
        metrics = summary["metrics"]
        totals.append({"运行": label, "划分": summary["split"], "题数": metrics["samples"],
                       "Answer F1": metrics["score"], "Answer EM": metrics["exact_match"]})
        groups.extend({"运行": label, "推理跳数": item["hops"], "题数": item["samples"],
                       "Answer F1": item["score"], "Answer EM": item["exact_match"]}
                      for item in metrics["groups"])
    score_format = {name: st.column_config.NumberColumn(format="percent")
                    for name in ("Answer F1", "Answer EM")}
    for title, rows in (("总体表现", totals), ("分推理跳数表现", groups)):
        st.subheader(title)
        st.dataframe(rows, hide_index=True, column_config=score_format)
    show_judge_statistics(runs)


def show_answers(runs):
    st.subheader("逐题回答")
    questions = runs[0]["records"]
    hops = st.selectbox("推理跳数", [None, *sorted({row["hops"] for row in questions.values()})],
                        format_func=lambda value: "全部" if value is None else f"{value}-hop")
    question_ids = [qid for qid, row in questions.items() if hops is None or row["hops"] == hops]
    qid = st.selectbox("问题", question_ids, format_func=lambda q: questions[q]["question"])
    row = questions[qid]
    st.text(row["question"])
    st.caption(f"题号 {qid} · {row['hops']}-hop · 参考答案")
    st.code("\n".join(row["references"]), language=None, wrap_lines=True)
    for start in range(0, len(runs), 3):
        row_runs = runs[start:start + 3]
        for column, run in zip(st.columns(len(row_runs)), row_runs):
            answer = run["records"][qid]
            with column.container(border=True):
                st.markdown(f"**{run['label']}**")
                st.caption(f"Answer F1 {answer['score']:.2%} · Answer EM {answer['exact_match']:.2%}")
                st.code(answer["predicted_answer"], language=None, wrap_lines=True)
                show_judge_answer(run.get("judge"), qid)


def show_runs(runs):
    try:
        loaded = load_runs(runs)
    except (OSError, KeyError, ValueError) as error:
        st.error(str(error))
        return
    show_judge_accuracy_chart(loaded)
    show_scores(loaded)
    show_answers(loaded)


def main(argv=None):
    run_comparison("MuSiQue", load_runs, {"musique": "MuSiQue · dev"},
                   comparison_panels, show_scores, show_answers, argv)


if __name__ == "__main__":
    main()
