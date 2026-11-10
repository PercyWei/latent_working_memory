"""Run original OpenCompass generation and scoring over preconstructed RULER rows."""

import argparse
from datetime import datetime
import logging
import json
import os
from pathlib import Path
import signal
from zoneinfo import ZoneInfo
import time

import torch
from opencompass.datasets.ruler.ruler_cwe import RulerCweEvaluator
from opencompass.datasets.ruler.ruler_fwe import RulerFweEvaluator
from opencompass.datasets.ruler.ruler_niah import RulerNiahEvaluator
from opencompass.datasets.ruler.ruler_qa import RulerQaEvaluator
from opencompass.datasets.ruler.ruler_vt import RulerVtEvaluator
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

from provenance import UPSTREAM_REVISION, capture_source
from runtime import (
    GENERATION_KWARGS,
    MODEL_CLASSES,
    MODEL_KWARGS,
    TOKENIZER_KWARGS,
    generate_one,
    load_model,
    runtime_info,
)


TASKS = {
    "niah_single_1": RulerNiahEvaluator,
    "niah_single_2": RulerNiahEvaluator,
    "niah_single_3": RulerNiahEvaluator,
    "niah_multikey_1": RulerNiahEvaluator,
    "niah_multikey_2": RulerNiahEvaluator,
    "niah_multikey_3": RulerNiahEvaluator,
    "niah_multivalue": RulerNiahEvaluator,
    "niah_multiquery": RulerNiahEvaluator,
    "vt": RulerVtEvaluator,
    "cwe": RulerCweEvaluator,
    "fwe": RulerFweEvaluator,
    "qa_1": RulerQaEvaluator,
    "qa_2": RulerQaEvaluator,
}


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d %H:%M:%S Asia/Shanghai")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_rows(path, count):
    with path.open() as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != count:
        raise ValueError(f"{path}: expected {count} rows, got {len(rows)}")
    for index, row in enumerate(rows):
        if not isinstance(row["input"], str) or not isinstance(row["answer_prefix"], str):
            raise ValueError(f"{path}:{index + 1}: input and answer_prefix must be strings")
        if not isinstance(row["outputs"], list) or not row["outputs"]:
            raise ValueError(f"{path}:{index + 1}: outputs must be a nonempty list")
        if any(not isinstance(value, str) or not value for value in row["outputs"]):
            raise ValueError(f"{path}:{index + 1}: outputs must contain nonempty strings")
    return rows


def load_expected(plan):
    expected = {}
    for length in plan["lengths"]:
        for task in plan["tasks"]:
            source = Path(plan["ruler_data"]) / str(length) / task / "test.jsonl"
            rows = read_rows(source, plan["samples_per_task"])
            for index, row in enumerate(rows):
                if plan["shard_index"] is None or index % plan["num_shards"] == plan["shard_index"]:
                    expected[length, task, index] = (str(source), row)
    return expected


def validate_prediction(record, expected, shard_index):
    key = record["length"], record["task"], record["row"]
    if key not in expected or record["shard_index"] != shard_index:
        raise ValueError(f"Prediction is outside the planned shard: {key}")
    source, row = expected[key]
    if (
        record["source"] != source
        or record["source_index"] != row["index"]
        or record["gold"] != row["outputs"]
    ):
        raise ValueError(f"Saved prediction does not match current data: {key}")
    if record["generated_tokens"] != len(record["token_ids"]):
        raise ValueError(f"Saved prediction token count differs from token IDs: {key}")
    prediction = extract_non_reasoning_content(record["raw_prediction"])
    if prediction != record["prediction"]:
        raise ValueError(f"Saved prediction postprocessing does not match: {key}")
    return key


def read_predictions(path, expected, shard_index, execution_attempt):
    if not path.exists():
        return {}
    completed = {}
    with path.open("r+b") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if line.endswith(b"\n") or stream.read(1):
                    raise
                stream.truncate(offset)
                logging.warning("Removed incomplete final prediction at byte %s", offset)
                break
            key = validate_prediction(record, expected, shard_index)
            if key in completed:
                raise ValueError(f"Duplicate prediction key: {key}")
            if not 0 <= record["execution_attempt"] < execution_attempt:
                raise ValueError(f"Saved prediction has an invalid execution attempt: {key}")
            completed[key] = record
            if not line.endswith(b"\n"):
                stream.write(b"\n")
    return completed


def build_summary(plan, expected, completed, output):
    if completed.keys() != expected.keys():
        raise ValueError("Predictions do not exactly cover the planned samples")
    summary = {
        "checkpoint": plan["checkpoint"],
        "inference_mode": plan["inference_mode"],
        "mlp_chunk_size": plan["mlp_chunk_size"],
        "num_shards": plan["num_shards"],
        "shard_index": plan["shard_index"],
        "completed": True,
        "lengths": {},
        "samples": len(completed),
    }
    for length in plan["lengths"]:
        length_results = {}
        for task, evaluator_class in TASKS.items():
            keys = [key for key in expected if key[0] == length and key[1] == task]
            native_score = evaluator_class().score(
                predictions=[
                    extract_non_reasoning_content(completed[key]["raw_prediction"])
                    for key in keys
                ],
                gold=[expected[key][1]["outputs"] for key in keys],
            )
            length_results[task] = native_score
            result_dir = output / "results" / str(length)
            result_dir.mkdir(parents=True, exist_ok=True)
            write_json(result_dir / f"{task}.json", native_score)
        summary["lengths"][str(length)] = {
            "tasks": length_results,
            "score": sum(result["score"] for result in length_results.values()) / len(TASKS),
        }
    return summary


def evaluate(plan, wrapper, resume, completed, execution_attempt, expected):
    output = Path(plan["output_dir"])
    budget = plan["tokens_to_generate"]
    current_group = None
    with (output / "predictions.jsonl").open("a" if resume else "w") as predictions_file:
        for key, (source, row) in expected.items():
            if key in completed:
                continue
            length, task, index = key
            if current_group != (length, task):
                evaluator = TASKS[task]()
                current_group = length, task
            prompt = row["input"] + row["answer_prefix"]
            start = time.perf_counter()
            # Every generate call creates a new upstream cache, including TTT state.
            generated = generate_one(wrapper, prompt, budget, length)
            prediction = extract_non_reasoning_content(generated["raw_prediction"])
            record = {
                "execution_attempt": execution_attempt,
                "shard_index": plan["shard_index"],
                "length": length,
                "task": task,
                "row": index,
                "source": source,
                "source_index": row["index"],
                **generated,
                "prediction": prediction,
                "gold": row["outputs"],
                "native_score": evaluator.score([prediction], [row["outputs"]]),
                "seconds": time.perf_counter() - start,
            }
            predictions_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            predictions_file.flush()
            completed[key] = record
            logging.info("%s/%s: row=%s shard=%s/%s", length, task, index, plan["shard_index"], plan["num_shards"])
    write_json(output / "summary.json", build_summary(plan, expected, completed, output))


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inference-mode", choices=("original", "optimized"), default="original")
    parser.add_argument("--mlp-chunk-size", type=int, default=16384)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--ruler-data",
        type=Path,
        default=root / "data/benchmarks/ruler-v1/reproductions/in-place-ttt-qwen3-4b",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[4096, 8192, 16384, 32768, 65536, 131072, 262144],
    )
    args = parser.parse_args()
    if not 1 <= args.num_shards <= 100 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Use 1 <= num-shards <= 100 and 0 <= shard-index < num-shards.")
    physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    if physical_gpu not in {"0", "1"}:
        parser.error("CUDA_VISIBLE_DEVICES must select a single physical GPU: 0 or 1.")
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_gpu
    if any(length <= 64 for length in args.lengths) or len(set(args.lengths)) != len(args.lengths):
        parser.error("--lengths must contain distinct integers greater than 64.")
    if args.mlp_chunk_size < 0:
        parser.error("--mlp-chunk-size must be nonnegative; 0 disables MLP chunking.")
    checkpoint = args.checkpoint.expanduser().resolve()
    data = args.ruler_data.expanduser().resolve()
    if not checkpoint.is_dir():
        parser.error(f"Checkpoint directory does not exist: {checkpoint}")
    for length in args.lengths:
        for task in TASKS:
            path = data / str(length) / task / "test.jsonl"
            if not path.is_file():
                parser.error(f"RULER data file does not exist: {path}")
    output = args.output_dir.expanduser().resolve()
    run_source = capture_source(__file__, {args.inference_mode: MODEL_CLASSES[args.inference_mode]})
    plan = {
        "created_at": timestamp(),
        "source": run_source,
        "upstream_revision": UPSTREAM_REVISION,
        "checkpoint": str(checkpoint),
        "checkpoint_config": json.loads((checkpoint / "config.json").read_text()),
        "inference_mode": args.inference_mode,
        "mlp_chunk_size": args.mlp_chunk_size,
        "compact_tails": args.inference_mode == "optimized",
        "ruler_data": str(data),
        "lengths": args.lengths,
        "samples_per_task": 100,
        "tasks": list(TASKS),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "assigned_samples": len(args.lengths) * len(TASKS) * len(range(args.shard_index, 100, args.num_shards)),
        "tokens_to_generate": 64,
        "prompt": "input + answer_prefix; base completion; no chat template",
        "model_wrapper": "runtime.EvaluationModel",
        "generation_path": (
            "OpenCompass HuggingFaceBaseModel.generate"
            if args.inference_mode == "original"
            else "runtime.generate_one; single tokenization and native generation parameters"
        ),
        "model_kwargs": MODEL_KWARGS.copy(),
        "tokenizer_kwargs": TOKENIZER_KWARGS.copy(),
        "generation_kwargs": GENERATION_KWARGS.copy(),
        "postprocessor": "opencompass.utils.text_postprocessors.extract_non_reasoning_content",
        "max_seq_len": max(args.lengths),
        "cuda_visible_devices": physical_gpu,
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "device": "cuda:0",
        "output_dir": str(output),
        "sample_timing": "input encoding, generation, decoding, postprocessing and native sample scoring",
    }
    plan_path = output / "plan.json"
    if args.resume:
        if not output.is_dir():
            parser.error("--resume requires an existing output directory")
    else:
        output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(output / "evaluation.log"), logging.StreamHandler()],
    )
    execution_path = output / "execution.json"
    execution = (
        json.loads(execution_path.read_text())
        if args.resume
        else {"attempts": [], "elapsed_seconds": 0.0, "peak_memory_allocated_bytes": 0}
    )
    execution_attempt = len(execution["attempts"])
    attempt = {
        "execution_attempt": execution_attempt,
        "started_at": timestamp(),
        "resume": args.resume,
        "status": "running",
        "source": run_source,
        "inference_mode": args.inference_mode,
        "mlp_chunk_size": args.mlp_chunk_size,
    }
    execution["attempts"].append(attempt)
    write_json(execution_path, execution)
    started = time.perf_counter()

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        if args.resume:
            previous = json.loads(plan_path.read_text())
            previous_settings = {
                key: value for key, value in previous.items() if key not in {"created_at", "source"}
            }
            current_settings = {
                key: value for key, value in plan.items() if key not in {"created_at", "source"}
            }
            if previous_settings != current_settings:
                raise ValueError("Resume settings do not match the initial plan.json")
            plan = previous
        else:
            write_json(plan_path, plan)
        expected = load_expected(plan)
        completed = (
            read_predictions(output / "predictions.jsonl", expected, plan["shard_index"], execution_attempt)
            if args.resume
            else {}
        )
        attempt["resumed_samples"] = len(completed)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(0)
        logging.info(
            "Starting evaluation; resume=%s, cached_samples=%s", args.resume, len(completed)
        )
        wrapper = load_model(
            plan["checkpoint"],
            plan["inference_mode"],
            plan["mlp_chunk_size"],
            plan["max_seq_len"],
        )
        runtime = runtime_info(wrapper)
        attempt["runtime"] = runtime
        write_json(execution_path, execution)
        runtime_path = output / "runtime.json"
        if args.resume and runtime_path.exists():
            if json.loads(runtime_path.read_text()) != runtime:
                raise ValueError("Resume model, optimization, backend or package settings differ")
        elif completed:
            raise FileNotFoundError("Existing predictions require their initial runtime.json")
        else:
            write_json(runtime_path, runtime)
        logging.info("Runtime: %s", json.dumps(runtime, ensure_ascii=False))
        evaluate(plan, wrapper, args.resume, completed, execution_attempt, expected)
        attempt["status"] = "completed"
        attempt["exit_code"] = 0
    except BaseException as error:
        attempt.update(
            status="failed",
            exit_code=130 if isinstance(error, KeyboardInterrupt) else 1,
            error=f"{type(error).__name__}: {error}",
        )
        logging.exception("Evaluation failed")
        raise
    finally:
        attempt["finished_at"] = timestamp()
        attempt["elapsed_seconds"] = time.perf_counter() - started
        attempt["peak_memory_allocated_bytes"] = (
            torch.cuda.max_memory_allocated(0) if torch.cuda.is_initialized() else 0
        )
        execution["elapsed_seconds"] += attempt["elapsed_seconds"]
        execution["peak_memory_allocated_bytes"] = max(
            execution["peak_memory_allocated_bytes"], attempt["peak_memory_allocated_bytes"]
        )
        write_json(execution_path, execution)


if __name__ == "__main__":
    main()
