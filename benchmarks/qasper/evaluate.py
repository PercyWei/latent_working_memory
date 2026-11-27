"""按论文调度 QASPER 问答，并汇总逐题计算的官方 Answer F1。"""

import json
from pathlib import Path
import time

from benchmarks.documents import group_by_document, shard_groups
from benchmarks.io import run_lock, write_json, write_jsonl
from benchmarks.output import add_output_arguments, evaluation_directory
from benchmarks.parallel import resolve_generation_devices, run_workers
from benchmarks.progress import QuestionTimer
from benchmarks.vendor.qasper import ANSWER_TYPES, EVALUATOR_REVISION
from .data import QUESTION_FORMAT, SPLIT_FILES, SYSTEM_PROMPT, load_qasper, qasper_data_identity


def add_evaluation_arguments(parser, output_root, checkpoint_layout=False):
    parser.allow_abbrev = False
    parser.add_argument("--split", choices=SPLIT_FILES, default="dev", help="QASPER 官方划分，默认 dev")
    parser.add_argument("--benchmark-data", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data/benchmarks/qasper")
    add_output_arguments(parser, output_root, checkpoint_layout)
    parser.add_argument("--batch-size-per-gpu", type=int, default=1, help="每卡批处理数，默认 1")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="每题生成预算，默认 128")
    parser.add_argument("--resume", action="store_true", help="沿用原配置和已评分的逐题结果，补齐剩余题目")


def evaluate_papers(directory, papers, devices, batch_size, lock):
    progress = directory / "progress"
    progress.mkdir(exist_ok=True)
    paths = [[progress / f"{i}-{j}.json" for j in range(len(paper.questions))]
             for i, paper in enumerate(papers)]
    total = sum(map(len, paths))

    pending_papers, pending_paths = [], []
    for i, paper_paths in enumerate(paths):
        missing = [path for path in paper_paths if not path.is_file()]
        if missing:
            pending_papers.append(i)
            pending_paths.extend(missing)
    if pending_papers and not devices:
        raise ValueError("QASPER generation requires a visible GPU")
    already_completed = total - len(pending_paths)
    timer = QuestionTimer(already_completed)
    print(f"QASPER {directory.name} | {already_completed}/{total} | {directory}", flush=True)

    def report_progress(completed=None):
        if completed is None:
            completed = already_completed + sum(path.is_file() for path in pending_paths)
        elapsed = time.monotonic() - timer.started
        print(f"QASPER {directory.name} | {completed}/{total} | {elapsed:.0f}s | {timer.format_rate(completed)}",
              flush=True)

    shards = shard_groups(group_by_document(pending_papers, lambda i: papers[i].history), len(devices))
    run_workers("benchmarks.qasper.worker", directory, devices, shards, batch_size, lock, report_progress)
    rows = []
    for paper, paper_paths in zip(papers, paths):
        for question, path in zip(paper.questions, paper_paths):
            row = json.loads(path.read_text(encoding="utf-8"))
            if row["question_id"] != question.id or row["paper_id"] != paper.id:
                raise ValueError(f"{path}: expected a result for {question.id} in {paper.id}")
            rows.append(row)
    if pending_paths:
        report_progress(len(rows))
    return rows


def summarize_scores(rows):
    groups = []
    for kind in ANSWER_TYPES:
        scores = [row["score"] for row in rows if row["answer_type"] == kind]
        groups.append({"answer_type": kind, "samples": len(scores),
                       "score": sum(scores) / len(scores) if scores else 0.0})
    return {
        "metric": "answer_f1", "score": sum(row["score"] for row in rows) / len(rows),
        "samples": len(rows), "missing_predictions": 0, "groups": groups,
    }


def run_evaluation(args, method, settings, load_adapter, output_name, checkpoint_name=None):
    directory = evaluation_directory(args, f"qasper/{args.split}", output_name, checkpoint_name)
    devices = resolve_generation_devices(args)
    papers = load_qasper(args.split, args.benchmark_data)
    config = {
        "benchmark": "qasper", "split": args.split, "method": method,
        "settings": {**settings, "system_prompt": SYSTEM_PROMPT, "question_format": QUESTION_FORMAT,
                     "decoding": "greedy", "max_new_tokens": args.max_new_tokens},
        "adapter": {"module": load_adapter.__module__, "factory": load_adapter.__name__},
        "data": qasper_data_identity(papers, args.benchmark_data, args.split),
        "scoring": {"metric": "answer_f1", "source_commit": EVALUATOR_REVISION},
    }
    if not args.resume:
        directory.mkdir(parents=True, exist_ok=False)
    config_path = directory / "evaluation_config.json"
    with run_lock(directory) as lock:
        if args.resume:
            if json.loads(config_path.read_text(encoding="utf-8")) != config:
                raise ValueError("saved QASPER settings or data differ")
        else:
            write_json(config_path, config)
        summary_path = directory / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            rows = evaluate_papers(directory, papers, devices, args.batch_size_per_gpu, lock)
            summary = {**config, "metrics": summarize_scores(rows)}
            write_jsonl(directory / "details.jsonl", rows)
            write_jsonl(directory / "predictions.jsonl", (
                {"question_id": row["question_id"], "predicted_answer": row["predicted_answer"]} for row in rows
            ))
            write_json(summary_path, summary)
        for path in (directory / "progress").glob("[0-9]*-[0-9]*.json"):
            path.unlink()

    metrics = summary["metrics"]
    print(f"QASPER {args.split} | Answer F1 {metrics['score']:.2%} ({metrics['samples']} questions)", flush=True)
    return summary
