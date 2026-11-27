"""使用 LongMemEval 官方提示词评判已生成的回答，按题保存评分。"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
from itertools import islice
import json
from pathlib import Path

from benchmarks.data import load_samples
from benchmarks.io import run_lock, write_json, write_jsonl
from benchmarks.progress import QuestionTimer
from benchmarks.protocol import data_identity
from .data import LONGMEMEVAL_FILES
from .generate import read_predictions
from .judge import JUDGE_MODEL, JUDGE_OPTIONS, JUDGE_REVISION, judge_answer, summarize_scores


DEFAULT_CONCURRENCY = 8


def _score_pending(samples, predictions, rows, details_path, benchmark, concurrency):
    timer = QuestionTimer(len(rows))
    pending = (index for index in range(len(samples)) if index not in rows)
    failure = None
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        in_flight = {}
        while True:
            if failure is None:
                for index in islice(pending, concurrency - len(in_flight)):
                    future = executor.submit(judge_answer, samples[index], predictions[index]["hypothesis"])
                    in_flight[future] = index
            if not in_flight:
                break
            completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in completed:
                index = in_flight.pop(future)
                try:
                    judgment = future.result()
                except Exception as error:
                    # 停止派发新题，继续保存已经发出的请求中成功的判定。
                    if failure is None:
                        failure = error
                    continue
                sample = samples[index]
                rows[index] = {
                    **predictions[index],
                    "metadata": {"question_type": sample.question_type, "question": sample.input.question,
                                 "question_date": sample.input.question_date},
                    "reference": sample.answer,
                    "autoeval_label": judgment,
                }
                write_jsonl(details_path, (rows[index] for index in sorted(rows)))
                print(f"{benchmark} | judged {len(rows)}/{len(samples)} | {timer.format_rate(len(rows))}",
                      flush=True)
    if failure is not None:
        raise failure


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
            if details_path.exists():
                positions = {prediction["question_id"]: index for index, prediction in enumerate(predictions)}
                previous_index = -1
                with details_path.open(encoding="utf-8") as stream:
                    for line_number, line in enumerate(stream, 1):
                        row = json.loads(line)
                        index = positions.get(row["question_id"], -1)
                        if index <= previous_index or any(
                            row[key] != value for key, value in predictions[index].items()
                        ):
                            raise ValueError(f"{details_path}:{line_number}: saved judgment differs or is out of order")
                        rows[index] = row
                        previous_index = index
        else:
            write_json(scoring_path, scoring)

        _score_pending(samples, predictions, rows, details_path, benchmark, concurrency)
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
