"""按已分配的样本 ID 组批生成，逐条保存 In-Place TTT 原生评分。"""

import argparse
import json
import logging
import os
from pathlib import Path
import signal
import tempfile
import time

from in_place_ttt.evaluation.provenance import capture_source
from in_place_ttt.evaluation.batching import execution_parameters, prepare_batches
from in_place_ttt.evaluation.records import (
    TASKS as TASKS,
    load_samples,
    timestamp,
    validate_sample_ids,
    write_json,
)
from in_place_ttt.evaluation.runtime import MODEL_CLASSES, generate_batch, generate_one, load_model, runtime_info
from opencompass.utils.text_postprocessors import extract_non_reasoning_content
import torch


def check_runtime(path, runtime):
    """多个 worker 首次启动时，只发布一份完整的共享 runtime。"""
    if path.is_file():
        if json.loads(path.read_text()) != runtime:
            raise ValueError("Resume model, optimization, backend or package settings differ")
        return
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(runtime, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        except BaseException:
            temporary.unlink()
            raise
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if json.loads(path.read_text()) != runtime:
                raise ValueError("Concurrent evaluation workers have different runtime settings")
    finally:
        temporary.unlink()


def evaluate(
    protocol, wrapper, assigned, output, execution, execution_attempt, shard_index,
    worker_started, batch_parameters,
):
    current_group = None
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as predictions:
        batches = prepare_batches(
            assigned, wrapper.tokenizer, batch_parameters,
            protocol["inference_mode"], protocol["tokens_to_generate"],
        )
        for batch_index, (length, batch) in enumerate(batches):
            prompts = [sample[4]["input"] + sample[4]["answer_prefix"] for _, sample, _ in batch]
            logging.info(
                "Evaluating length=%s batch=%s size=%s input_tokens=%s",
                length, batch_index, len(batch),
                [len(token_ids) for _, _, token_ids in batch] if protocol["inference_mode"] == "optimized" else None,
            )
            started = time.perf_counter()
            # 每次生成创建新 cache；批内各条样本的 KV 和 TTT 状态独立。
            if protocol["inference_mode"] == "original":
                outputs = [generate_one(wrapper, prompts[0], protocol["tokens_to_generate"], length)]
            else:
                outputs = generate_batch(
                    wrapper, prompts, protocol["tokens_to_generate"], length,
                    token_ids=[token_ids for _, _, token_ids in batch],
                )
            batch_generation_seconds = time.perf_counter() - started
            for (sample_id, sample, _), generated in zip(batch, outputs, strict=True):
                _, task, index, source, row = sample
                score_started = time.perf_counter()
                if current_group != (length, task):
                    evaluator = TASKS[task]()
                    current_group = length, task
                prediction = extract_non_reasoning_content(generated["raw_prediction"])
                native_score = evaluator.score([prediction], [row["outputs"]])
                record = {
                    "id": sample_id,
                    "execution_attempt": execution_attempt,
                    "shard_index": shard_index,
                    "length": length,
                    "task": task,
                    "row": index,
                    "source": str(source),
                    "source_index": row["index"],
                    **generated,
                    "prediction": prediction,
                    "gold": row["outputs"],
                    "native_score": native_score,
                    "batch_index": batch_index,
                    "batch_size": len(batch),
                    "batch_generation_seconds": batch_generation_seconds,
                    "seconds": batch_generation_seconds / len(batch) + time.perf_counter() - score_started,
                }
                predictions.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                predictions.flush()
                execution["completed_samples"] += 1
                execution.update(
                    seconds=time.perf_counter() - worker_started,
                    peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(0),
                )
                write_json(output / "execution.json", execution)
                logging.info(
                    "%s: tokens=%s score=%s seconds=%.3f batch=%s size=%s shard=%s",
                    sample_id, record["generated_tokens"], record["native_score"],
                    record["seconds"], batch_index, len(batch), shard_index,
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-file", type=Path, required=True)
    parser.add_argument("--sample-ids-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-file", type=Path, required=True)
    parser.add_argument("--execution-attempt", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-size-by-length", nargs="+", default=[], metavar="LENGTH=BATCH")
    args = parser.parse_args()
    if args.execution_attempt < 0 or args.shard_index < 0:
        parser.error("Execution attempt and shard index must be nonnegative")
    physical_gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    if physical_gpu not in {"0", "1"}:
        parser.error("CUDA_VISIBLE_DEVICES must select a single physical GPU: 0 or 1")

    protocol = json.loads(args.protocol_file.read_text())["protocol"]
    batch_parameters = execution_parameters(
        args.batch_size, args.batch_size_by_length, protocol["lengths"], protocol["inference_mode"]
    )
    expected = load_samples(protocol)
    sample_ids = json.loads(args.sample_ids_file.read_text())
    validate_sample_ids(sample_ids, expected)
    assigned = {sample_id: expected[sample_id] for sample_id in sample_ids}
    if not assigned:
        parser.error("An evaluation worker must have assigned samples")
    source = capture_source(__file__, {protocol["inference_mode"]: MODEL_CLASSES[protocol["inference_mode"]]})
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(output / "evaluation.log"), logging.StreamHandler()],
    )
    write_json(output / "plan.json", {
        "created_at": timestamp(), "protocol": protocol, "sample_ids": sample_ids,
        "assigned_samples": len(sample_ids),
        "source": source, "execution_attempt": args.execution_attempt,
        "physical_gpu": physical_gpu, "shard_index": args.shard_index,
        "batch_parameters": batch_parameters,
        "sample_timing": (
            "seconds = batch generation wall / actual batch_size + per-row postprocessing and native scoring; "
            "optimized grouping tokenization is excluded; original tokenization remains inside generation"
        ),
    })
    execution = {
        "execution_attempt": args.execution_attempt,
        "started_at": timestamp(), "status": "running", "source": source,
        "assigned_samples": len(assigned), "completed_samples": 0,
        "seconds": 0.0, "peak_memory_allocated_bytes": 0,
    }
    execution_path = output / "execution.json"
    write_json(execution_path, execution)
    started = time.perf_counter()

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(0)
        wrapper = load_model(
            protocol["checkpoint"], protocol["inference_mode"],
            protocol["mlp_chunk_size"], max(protocol["lengths"]),
        )
        runtime = runtime_info(wrapper)
        execution["runtime"] = runtime
        write_json(output / "runtime.json", runtime)
        check_runtime(args.runtime_file.expanduser().resolve(), runtime)
        write_json(execution_path, execution)
        logging.info("Runtime: %s", json.dumps(runtime, ensure_ascii=False))
        logging.info("Starting evaluation; assigned_samples=%s batch_parameters=%s", len(assigned), batch_parameters)
        evaluate(
            protocol, wrapper, assigned, output, execution,
            args.execution_attempt, args.shard_index, started, batch_parameters,
        )
        execution.update(status="completed", exit_code=0)
    except BaseException as error:
        execution.update(
            status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
            exit_code=(
                130 if isinstance(error, KeyboardInterrupt)
                else error.code if isinstance(error, SystemExit) else 1
            ),
            error=f"{type(error).__name__}: {error}",
        )
        logging.exception("Evaluation worker did not complete")
        raise
    finally:
        execution.update(
            finished_at=timestamp(), seconds=time.perf_counter() - started,
            peak_memory_allocated_bytes=(
                torch.cuda.max_memory_allocated(0) if torch.cuda.is_initialized() else 0
            ),
        )
        write_json(execution_path, execution)


if __name__ == "__main__":
    main()
