"""每张可见 GPU 运行一个独立进程，按原始题序合并已落盘的回答。"""

import json

from benchmarks.documents import QuestionInput, evaluate_documents, group_by_document, shard_groups
from benchmarks.io import write_json, write_jsonl
from benchmarks.parallel import run_workers
from benchmarks.progress import QuestionTimer
from .data import format_history, format_question


def history_groups(samples, indices):
    # 正文键排除问题和提问日期。相同历史始终分给同一个 worker。
    return group_by_document(indices, lambda i: format_history(samples[i].input))


def generate_assigned(directory, samples, groups, batch_size, load_adapter, settings):
    def documents():
        for indices in groups:
            history = format_history(samples[indices[0]].input)
            questions = [
                QuestionInput(samples[i].id, history + format_question(samples[i].input), settings["max_new_tokens"])
                for i in indices
            ]
            yield history, questions

    remaining = {samples[i].id: i for group in groups for i in group}
    total = len(remaining)
    timer = QuestionTimer()
    adapter = load_adapter(settings)

    def commit(question, answer, statistics):
        prediction = answer["prediction"]
        if not isinstance(prediction, str):
            raise ValueError(f"{question.id}: expected a string prediction")
        i = remaining.pop(question.id)
        write_json(directory / "progress" / f"{i}.json", {"question_id": question.id, "hypothesis": prediction})
        completed = total - len(remaining)
        print(f"{directory.name} | {completed}/{total} | {question.id} | {timer.format_rate(completed)}", flush=True)

    evaluate_documents(documents(), adapter, batch_size, commit)


def generate_predictions(directory, samples, rows, devices, batch_size, lock, load_adapter, settings):
    progress = directory / "progress"
    progress.mkdir(exist_ok=True)
    predictions = directory / "predictions.jsonl"
    prefix_size = len(rows)
    unfinished = range(prefix_size, len(samples))
    pending = [i for i in unfinished if not (progress / f"{i}.json").is_file()]
    if pending and devices is not None and not devices:
        raise ValueError("LongMemEval generation requires a visible GPU")
    already_completed = len(samples) - len(pending)
    timer = QuestionTimer(already_completed)
    print(f"{directory.name} | {already_completed}/{len(samples)} | {directory}", flush=True)

    def report_progress(completed=None):
        if completed is None:
            completed = already_completed + sum((progress / f"{i}.json").is_file() for i in pending)
        print(f"{directory.name} | {completed}/{len(samples)} | {timer.format_rate(completed)}",
              flush=True)

    groups = history_groups(samples, pending)
    if pending and devices is None:
        generate_assigned(directory, samples, groups, batch_size, load_adapter, settings)
    elif pending:
        shards = shard_groups(groups, len(devices))
        run_workers("benchmarks.longmemeval.worker", directory, devices, shards, batch_size, lock, report_progress)
    for i in unfinished:
        path = progress / f"{i}.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["question_id"] != samples[i].id or not isinstance(row["hypothesis"], str):
            raise ValueError(f"{path}: invalid prediction for {samples[i].id}")
        rows.append(row)
    if unfinished:
        write_jsonl(predictions, rows)
    if pending:
        report_progress(len(rows))
    # 如果上次在合并后中断，完整预测仍是有效的，续评只需清理分片。
    for path in progress.glob("[0-9]*.json"):
        path.unlink()
