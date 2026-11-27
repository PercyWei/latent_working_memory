"""在一张 GPU 上按批回答 MuSiQue 问题，并保存逐题 EM/F1。"""

import argparse
from importlib import import_module
import json
from pathlib import Path

from benchmarks.documents import QuestionInput, evaluate_documents
from benchmarks.io import write_json
from benchmarks.progress import QuestionTimer
from benchmarks.vendor.musique import score_answer
from .data import QUESTION_FORMAT, load_musique, musique_data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    args = parser.parse_args()
    plan = json.loads((args.directory / "evaluation_config.json").read_text(encoding="utf-8"))
    split, data_dir = plan["split"], plan["data"]["directory"]
    samples = load_musique(split, data_dir)
    if musique_data_identity(samples, data_dir, split) != plan["data"]:
        raise ValueError("MuSiQue data changed after the run was planned")
    pending = {}
    questions = []
    for i in args.indices:
        sample = samples[i]
        path = args.directory / "progress" / f"{i}.json"
        if not path.is_file():
            pending[sample.id] = (sample, path)
            questions.append(sample)

    budget = plan["settings"]["max_new_tokens"]
    documents = (
        (q.history, [QuestionInput(q.id, q.history + QUESTION_FORMAT.format(question=q.question), budget)])
        for q in questions
    )
    total = len(pending)
    timer = QuestionTimer()
    factory = plan["adapter"]
    adapter = getattr(import_module(factory["module"]), factory["factory"])(plan["settings"])

    def commit(sample, answer, statistics):
        prediction = answer["prediction"]
        if not isinstance(prediction, str):
            raise ValueError(f"{sample.id}: adapter must return a string prediction")
        question, path = pending.pop(sample.id)
        f1, em = score_answer(prediction, question.references)
        write_json(path, {
            "question_id": sample.id, "question": question.question, "hops": question.hops,
            "predicted_answer": prediction, "references": question.references,
            "score": f1, "exact_match": em,
        })
        completed = total - len(pending)
        print(f"MuSiQue {split} | {completed}/{total} | {sample.id} | {timer.format_rate(completed)}",
              flush=True)

    evaluate_documents(documents, adapter, args.batch_size, commit)


if __name__ == "__main__":
    main()
