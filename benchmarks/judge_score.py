"""为已完成的 LoCoMo、QASPER 或 MuSiQue-Ans 运行补评 Judge Accuracy。"""

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.answer_judge import JUDGE_MODEL, JUDGE_OPTIONS, judge_answer, judge_system_prompt
from benchmarks.io import run_lock, write_json
from benchmarks.judge_pool import DEFAULT_CONCURRENCY, load_judgments, score_pending


GROUP_FIELDS = {"locomo": "category", "qasper": "answer_type", "musique": "hops"}


def score_run(run_dir, resume=False, concurrency=DEFAULT_CONCURRENCY):
    """使用现有逐题产物评分，保留原有 F1 和 EM 文件。"""
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    directory = Path(run_dir).expanduser().resolve()
    with run_lock(directory):
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        benchmark = summary["benchmark"]
        if benchmark not in GROUP_FIELDS:
            raise ValueError(f"Judge Accuracy is not supported for {benchmark}")
        metrics = summary["metrics"]
        if metrics["metric"] != ("f1" if benchmark == "locomo" else "answer_f1"):
            raise ValueError(f"{directory}: expected completed lexical scores")

        content = (directory / "details.jsonl").read_bytes()
        details = [json.loads(line) for line in content.splitlines()]
        id_field = "id" if benchmark == "locomo" else "question_id"
        question_ids = [row[id_field] for row in details]
        total = len(details)
        if not total or total != metrics["samples"] or total != summary["data"]["samples"]:
            raise ValueError(f"{directory}: details do not cover the evaluated question set")
        if any(not isinstance(qid, str) or not qid.strip() for qid in question_ids) or len(set(question_ids)) != total:
            raise ValueError(f"{directory}: details contain invalid or duplicate question IDs")

        group_field = GROUP_FIELDS[benchmark]
        groups = {group[group_field]: [] for group in metrics["groups"]}
        group_values = [(row["metadata"] if benchmark == "locomo" else row)[group_field] for row in details]
        if any(group not in groups for group in group_values):
            raise ValueError(f"{directory}: details contain a group missing from summary.json")

        system_prompt = judge_system_prompt(benchmark)
        config = {
            "benchmark": benchmark, "model": JUDGE_MODEL, **JUDGE_OPTIONS, "system_prompt": system_prompt,
            "data_fingerprint": summary["data"]["fingerprint"],
            "details_sha256": hashlib.sha256(content).hexdigest(),
        }
        config_path = directory / "judge_config.json"
        details_path = directory / "judge_details.jsonl"
        summary_path = directory / "judge_summary.json"
        rows = {}
        if config_path.exists():
            if not resume:
                raise FileExistsError(f"judge scoring already started in {directory}; use --resume")
            if json.loads(config_path.read_text(encoding="utf-8")) != config:
                raise ValueError("saved judge settings, evaluated data or details differ")
            rows = load_judgments(details_path, question_ids)
        else:
            if details_path.exists() or summary_path.exists():
                raise FileNotFoundError(f"existing judge results require {config_path}")
            write_json(config_path, config)

        def judge_row(index):
            judgment = judge_answer(benchmark, details[index], system_prompt)
            return {"question_id": question_ids[index], **judgment}

        score_pending(total, rows, details_path, benchmark, concurrency, judge_row)
        for index, group in enumerate(group_values):
            groups[group].append(rows[index]["correct"])
        scores = {
            "metric": "judge_accuracy", "score": sum(row["correct"] for row in rows.values()) / total,
            "samples": total,
            "groups": [{group_field: group, "samples": len(values), "score": sum(values) / len(values) if values else None}
                       for group, values in groups.items()],
        }
        result = {**config, "metrics": scores}
        write_json(summary_path, result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="包含 summary.json 和 details.jsonl 的结果目录")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help="并发裁判请求数，默认 %(default)s")
    parser.add_argument("--resume", action="store_true", help="沿用本次裁判配置，只补评缺少判定的题目")
    args = parser.parse_args(argv)
    scores = score_run(args.run_dir, args.resume, args.concurrency)["metrics"]
    print(f"Judge Accuracy {scores['score']:.2%} ({scores['samples']} questions)", flush=True)


if __name__ == "__main__":
    main()
