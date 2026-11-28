"""使用 LongMemEval 官方提示词评判已生成的回答，按题保存评分。"""

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.data import load_samples
from benchmarks.io import run_lock, write_json
from benchmarks.judge_pool import DEFAULT_CONCURRENCY, load_judgments, score_pending
from benchmarks.protocol import data_identity
from .data import LONGMEMEVAL_FILES
from .generate import read_predictions
from .judge import JUDGE_MODEL, JUDGE_OPTIONS, JUDGE_REVISION, judge_answer, summarize_scores


def score_run(run_dir, resume=False, data_dir=None, concurrency=DEFAULT_CONCURRENCY):
    """评分一份已完成生成的 S 或 Oracle 运行；续评只提交尚未得到判定的回答。"""
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    directory = Path(run_dir).expanduser().resolve()
    with run_lock(directory):
        config = json.loads((directory / "evaluation_config.json").read_text(encoding="utf-8"))
        benchmark = config["benchmark"]
        if benchmark not in LONGMEMEVAL_FILES:
            raise ValueError(f"unsupported LongMemEval setting: {benchmark}")
        if data_dir is None:
            data_dir = config["data"]["directory"]
        samples = load_samples(benchmark, data_dir)
        actual_data = data_identity(samples, data_dir, benchmark)
        if actual_data != {**config["data"], "directory": actual_data["directory"]}:
            raise ValueError("data differ from the saved generation run")

        path = directory / "predictions.jsonl"
        predictions = read_predictions(path, samples)
        if len(predictions) != len(samples):
            raise ValueError(f"generation incomplete: {len(predictions)} of {len(samples)} predictions")
        with path.open("rb") as stream:
            prediction_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        scoring = {
            "model": JUDGE_MODEL, **JUDGE_OPTIONS, "source_commit": JUDGE_REVISION,
            "benchmark": benchmark, "data_fingerprint": config["data"]["fingerprint"],
            "predictions_sha256": prediction_digest,
        }
        scoring_path = directory / "scoring_config.json"
        details_path = directory / "details.jsonl"
        rows = {}
        if scoring_path.exists():
            if not resume:
                raise FileExistsError(f"scoring already started in {directory}; use --resume")
            if json.loads(scoring_path.read_text(encoding="utf-8")) != scoring:
                raise ValueError("saved judge settings, predictions or data differ")
            rows = load_judgments(details_path, (prediction["question_id"] for prediction in predictions))
            for index, row in rows.items():
                if any(row[key] != value for key, value in predictions[index].items()):
                    raise ValueError(f"{details_path}: saved judgment differs for {row['question_id']}")
        else:
            write_json(scoring_path, scoring)

        def judge_row(index):
            sample, prediction = samples[index], predictions[index]
            return {
                **prediction,
                "metadata": {"question_type": sample.question_type, "question": sample.input.question,
                             "question_date": sample.input.question_date},
                "reference": sample.answer,
                "autoeval_label": judge_answer(sample, prediction["hypothesis"]),
            }

        score_pending(len(samples), rows, details_path, benchmark, concurrency, judge_row)
        summary = {**config, "judge": scoring, "metrics": summarize_scores(rows.values())}
        write_json(directory / "summary.json", summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="包含 evaluation_config.json 和 predictions.jsonl 的设置目录")
    parser.add_argument("--benchmark-data", type=Path,
                        help="在另一台机器评分时指定同一份数据的本地目录")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help="并发裁判请求数，默认 %(default)s")
    parser.add_argument("--resume", action="store_true", help="沿用本次裁判配置和已保存的逐题判定")
    args = parser.parse_args()
    metrics = score_run(args.run_dir, args.resume, args.benchmark_data, args.concurrency)["metrics"]
    print(f"Overall Accuracy {metrics['score']:.2%} ({metrics['samples']} questions)")


if __name__ == "__main__":
    main()
