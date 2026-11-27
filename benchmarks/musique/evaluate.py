"""调度 MuSiQue-Ans dev 问答，汇总官方 Answer F1 与 EM。"""

import json
from pathlib import Path
import time

from benchmarks.documents import shard_groups
from benchmarks.io import run_lock, write_json, write_jsonl
from benchmarks.output import add_output_arguments, evaluation_directory
from benchmarks.parallel import resolve_generation_devices, run_workers
from benchmarks.progress import QuestionTimer
from benchmarks.vendor.musique import EVALUATOR_REVISION
from .data import QUESTION_FORMAT, SPLIT_FILES, SYSTEM_PROMPT, load_musique, musique_data_identity


def add_evaluation_arguments(parser, output_root, checkpoint_layout=False):
    parser.allow_abbrev = False
    parser.add_argument("--split", choices=SPLIT_FILES, default="dev", help="MuSiQue-Ans 划分，默认 dev")
    parser.add_argument("--benchmark-data", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data/benchmarks/musique")
    add_output_arguments(parser, output_root, checkpoint_layout)
    parser.add_argument("--batch-size-per-gpu", type=int, default=1, help="每卡批处理数，默认 1")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="每题生成预算，默认 128")
    parser.add_argument("--resume", action="store_true", help="沿用原配置和已评分的逐题结果，补齐剩余题目")


def evaluate_samples(directory, samples, devices, batch_size, lock):
    progress = directory / "progress"
    progress.mkdir(exist_ok=True)
    paths = [progress / f"{i}.json" for i in range(len(samples))]
    pending = [i for i, path in enumerate(paths) if not path.is_file()]
    if pending and not devices:
        raise ValueError("MuSiQue generation requires a visible GPU")
    already_completed = len(samples) - len(pending)
    timer = QuestionTimer(already_completed)
    print(f"MuSiQue {directory.name} | {already_completed}/{len(samples)} | {directory}", flush=True)

    def report_progress(completed=None):
        if completed is None:
            completed = already_completed + sum(paths[i].is_file() for i in pending)
        elapsed = time.monotonic() - timer.started
        print(f"MuSiQue {directory.name} | {completed}/{len(samples)} | {elapsed:.0f}s | "
              f"{timer.format_rate(completed)}", flush=True)

    shards = shard_groups([[i] for i in pending], len(devices))
    run_workers("benchmarks.musique.worker", directory, devices, shards, batch_size, lock, report_progress)
    rows = []
    for sample, path in zip(samples, paths):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["question_id"] != sample.id:
            raise ValueError(f"{path}: expected a result for {sample.id}")
        rows.append(row)
    if pending:
        report_progress(len(rows))
    return rows


def summarize_scores(rows):
    groups = []
    for hops in (2, 3, 4):
        group = [row for row in rows if row["hops"] == hops]
        groups.append({
            "hops": hops, "samples": len(group),
            "score": sum(row["score"] for row in group) / len(group) if group else 0.0,
            "exact_match": sum(row["exact_match"] for row in group) / len(group) if group else 0.0,
        })
    return {
        "metric": "answer_f1", "score": sum(row["score"] for row in rows) / len(rows),
        "exact_match": sum(row["exact_match"] for row in rows) / len(rows),
        "samples": len(rows), "missing_predictions": 0, "groups": groups,
    }


def run_evaluation(args, method, settings, load_adapter, output_name, checkpoint_name=None):
    directory = evaluation_directory(args, f"musique/{args.split}", output_name, checkpoint_name)
    devices = resolve_generation_devices(args)
    samples = load_musique(args.split, args.benchmark_data)
    config = {
        "benchmark": "musique", "split": args.split, "method": method,
        "settings": {**settings, "system_prompt": SYSTEM_PROMPT, "question_format": QUESTION_FORMAT,
                     "decoding": "greedy", "max_new_tokens": args.max_new_tokens},
        "adapter": {"module": load_adapter.__module__, "factory": load_adapter.__name__},
        "data": musique_data_identity(samples, args.benchmark_data, args.split),
        "scoring": {"metric": "answer_f1", "source_commit": EVALUATOR_REVISION},
    }
    if not args.resume:
        directory.mkdir(parents=True, exist_ok=False)
    config_path = directory / "evaluation_config.json"
    with run_lock(directory) as lock:
        if args.resume:
            if json.loads(config_path.read_text(encoding="utf-8")) != config:
                raise ValueError("saved MuSiQue settings or data differ")
        else:
            write_json(config_path, config)
        summary_path = directory / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            rows = evaluate_samples(directory, samples, devices, args.batch_size_per_gpu, lock)
            summary = {**config, "metrics": summarize_scores(rows)}
            write_jsonl(directory / "details.jsonl", rows)
            write_jsonl(directory / "predictions.jsonl", (
                {"question_id": row["question_id"], "predicted_answer": row["predicted_answer"]} for row in rows
            ))
            write_json(summary_path, summary)
        for path in (directory / "progress").glob("[0-9]*.json"):
            path.unlink()

    metrics = summary["metrics"]
    print(f"MuSiQue {args.split} | Answer F1 {metrics['score']:.2%} | "
          f"EM {metrics['exact_match']:.2%} ({metrics['samples']} questions)", flush=True)
    return summary
