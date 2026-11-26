"""在一张 GPU 上复用论文状态，按批回答尚未评分的问题。"""

import argparse
from importlib import import_module
import json
from pathlib import Path

from benchmarks.documents import QuestionInput, evaluate_documents
from benchmarks.io import write_json
from benchmarks.progress import QuestionTimer
from benchmarks.vendor.qasper import score_answer
from .data import QUESTION_FORMAT, load_qasper, qasper_data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    args = parser.parse_args()
    plan = json.loads((args.directory / "evaluation_config.json").read_text(encoding="utf-8"))
    split, data_dir = plan["split"], plan["data"]["directory"]
    papers = load_qasper(split, data_dir)
    if qasper_data_identity(papers, data_dir, split) != plan["data"]:
        raise ValueError("QASPER data changed after the run was planned")
    pending = {}
    by_history = {}
    for i in args.indices:
        paper = papers[i]
        for j, question in enumerate(paper.questions):
            path = args.directory / "progress" / f"{i}-{j}.json"
            if not path.is_file():
                pending[question.id] = (paper.id, question, path)
                by_history.setdefault(paper.history, []).append(question)

    budget = plan["settings"]["max_new_tokens"]
    documents = (
        (history, [QuestionInput(q.id, history + QUESTION_FORMAT.format(question=q.question), budget)
                   for q in questions])
        for history, questions in by_history.items()
    )
    total = len(pending)
    timer = QuestionTimer()
    factory = plan["adapter"]
    adapter = getattr(import_module(factory["module"]), factory["factory"])(plan["settings"])

    def commit(sample, answer, statistics):
        prediction = answer["prediction"]
        if not isinstance(prediction, str):
            raise ValueError(f"{sample.id}: adapter must return a string prediction")
        paper_id, question, path = pending.pop(sample.id)
        score, answer_type = score_answer(prediction, question.references)
        write_json(path, {
            "paper_id": paper_id, "question_id": sample.id, "question": question.question,
            "predicted_answer": prediction, "references": question.references,
            "answer_type": answer_type, "score": score,
        })
        completed = total - len(pending)
        print(f"QASPER {split} | {completed}/{total} | {paper_id} | {sample.id} | {timer.format_rate(completed)}",
              flush=True)

    evaluate_documents(documents, adapter, args.batch_size, commit)


if __name__ == "__main__":
    main()
