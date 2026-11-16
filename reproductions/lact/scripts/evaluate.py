"""使用 LaCT 的原始或增量推理路径评估本地 RULER 数据。"""

import argparse
import json
import logging
import os
from pathlib import Path
import signal
import time

from batching import parse_batch_size_by_length
from lact_model import LaCTForCausalLM, LaCTSWIGLUConfig
from incremental import IncrementalLaCTForCausalLM
from provenance import capture_source
from records import (
    BATCHING, GENERATION_BUDGET, LENGTHS, SAMPLES_PER_TASK_LENGTH, TASKS as TASKS,
    build_protocol, build_summary, load_samples, sample_index,
    read_samples as read_samples, timestamp, validate_prediction, write_json,
)
import torch
from transformers import AutoTokenizer

from latent_working_memory.benchmarks.vendor.ruler import (
    postprocess_pred,
    string_match_all,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def read_predictions(path, expected, logger, shard_index):
    records = {}
    if not path.exists():
        return records
    with path.open("r+b") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if line.endswith(b"\n"):
                    raise
                logger.warning("Removing incomplete final prediction at byte %s", offset)
                stream.seek(offset)
                stream.truncate()
                break
            sample_id = record["id"]
            if sample_id in records:
                raise ValueError(f"Duplicate saved prediction: {sample_id}")
            validate_prediction(record, expected, shard_index)
            records[sample_id] = record
            if not line.endswith(b"\n"):
                stream.write(b"\n")
    return records


def load_checkpoint(checkpoint, inference_mode="original"):
    config = LaCTSWIGLUConfig.from_pretrained(checkpoint, local_files_only=True)
    model_class = IncrementalLaCTForCausalLM if inference_mode == "incremental" else LaCTForCausalLM
    model, loading = model_class.from_pretrained(
        checkpoint,
        config=config,
        torch_dtype=torch.float32,
        local_files_only=True,
        output_loading_info=True,
    )
    failures = {
        key: loading.get(key, [])
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if loading.get(key)
    }
    if failures:
        raise ValueError(f"Checkpoint did not load strictly: {failures}")
    # 所有慢参数只读。原始 TTT 更新已写成显式矩阵公式，不需要外层 autograd。
    model.requires_grad_(False)
    model.eval()
    model.to("cuda:0")
    return model, config


def prepare_batches(expected, records, tokenizer, batch_size, batch_size_by_length=None):
    """各评估长度内按实际 token 数稳定降序，相邻样本组成批次。"""
    for length in dict.fromkeys(sample[0] for sample in expected.values()):
        limit = (batch_size_by_length or {}).get(str(length), batch_size)
        pending = []
        for sample_id, (sample_length, task, position, row) in expected.items():
            if sample_length != length or sample_id in records:
                continue
            token_ids = tokenizer.encode(
                row["input"] + row["answer_prefix"], truncation=False
            )
            input_tokens = len(token_ids)
            if input_tokens + GENERATION_BUDGET > length:
                raise ValueError(
                    f"{sample_id}: prompt ({input_tokens}) + "
                    f"generation budget ({GENERATION_BUDGET}) exceeds {length}"
                )
            pending.append(
                (sample_id, task, position, row, token_ids)
            )
        pending.sort(key=lambda sample: -len(sample[4]))
        for start in range(0, len(pending), limit):
            batch = pending[start : start + limit]
            yield length, len(batch[0][4]), batch


def run(args, output, logger, run_source, execution_attempt):
    checkpoint = args.checkpoint.expanduser().resolve()
    data = args.ruler_data.expanduser().resolve()
    if not checkpoint.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint}")
    if not args.lengths or len(set(args.lengths)) != len(args.lengths):
        raise ValueError("lengths must be nonempty and unique")
    samples = load_samples({"ruler_data": str(data), "lengths": args.lengths})
    all_samples = sample_index(samples)
    if args.sample_ids_file is not None:
        assigned = json.loads(args.sample_ids_file.read_text(encoding="utf-8"))
        if not isinstance(assigned, list) or any(not isinstance(key, str) for key in assigned):
            raise ValueError("Worker sample IDs must be a JSON list of strings")
        if len(set(assigned)) != len(assigned) or any(key not in all_samples for key in assigned):
            raise ValueError("Worker sample IDs must be unique and belong to this evaluation")
        expected = {key: all_samples[key] for key in assigned}
    else:
        expected = {
            key: sample for key, sample in all_samples.items()
            if sample[2] % args.num_shards == args.shard_index
        }
    plan = {
        **build_protocol(checkpoint, data, args.lengths, args.inference_mode, samples),
        "created_at": timestamp(),
        "source": run_source,
        "batch_size": args.batch_size,
        "batch_size_by_length": args.batch_size_by_length,
        "batching": BATCHING,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "assigned_samples": len(expected),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    }
    if args.sample_ids_file is not None:
        plan["assigned_sample_ids"] = list(expected)
    if args.resume:
        saved_plan = json.loads((output / "plan.json").read_text())
        for key, value in plan.items():
            if (
                key not in {"created_at", "cuda_visible_devices", "source", "batch_size", "batch_size_by_length"}
                and saved_plan[key] != value
            ):
                raise ValueError(f"Resume setting differs from saved plan: {key}")
        plan = saved_plan
        plan["cuda_visible_devices"] = os.environ["CUDA_VISIBLE_DEVICES"]
        records = read_predictions(output / "predictions.jsonl", expected, logger, args.shard_index)
        logger.info("Resuming with %s/%s completed samples", len(records), len(expected))
    else:
        records = {}
        write_json(output / "plan.json", plan)
    if not torch.cuda.is_available():
        raise RuntimeError("LaCT evaluation requires the native CUDA method environment")

    logger.info("Loading checkpoint strictly: %s", checkpoint)
    model, config = load_checkpoint(checkpoint, args.inference_mode)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    first_layer = model.model.layers[0].attn
    loaded_model = {
        "num_lact_heads": config.num_lact_heads,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "lact_chunk_size": config.lact_chunk_size,
        "max_position_embeddings": config.max_position_embeddings,
        "use_muon": config.use_muon,
        "ttt_prenorm": config.ttt_prenorm,
        "ttt_nope": config.ttt_nope,
        "use_fused_kernel": config.use_fused_kernel,
        "fp32_states": config.fp32_states,
        "first_layer_w1_shape": list(first_layer.w1.shape),
    }
    if args.resume and "loaded_model" in plan and plan["loaded_model"] != loaded_model:
        raise ValueError("Loaded model differs from saved plan")
    plan["loaded_model"] = loaded_model
    plan["runtime"] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    write_json(output / "plan.json", plan)
    logger.info(
        "Actual fast-weight heads: %s; first layer w1: %s",
        config.num_lact_heads,
        first_layer.w1.shape,
    )
    logger.info("Inference mode: %s", args.inference_mode)
    with (output / "predictions.jsonl").open(
        "a" if args.resume else "x", encoding="utf-8"
    ) as predictions:
        eos_ids = model.generation_config.eos_token_id
        eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
        batches = prepare_batches(
            expected, records, tokenizer, args.batch_size, args.batch_size_by_length
        )
        for batch_index, (length, batch_width, batch) in enumerate(batches):
            encoded = tokenizer.pad(
                {"input_ids": [sample[4] for sample in batch]},
                padding=True, return_tensors="pt",
            ).to("cuda:0")
            input_ids = encoded.input_ids
            logger.info(
                "Evaluating length=%s batch_width=%s input_tokens=%s batch=%s size=%s shard=%s/%s",
                length, batch_width, [len(sample[4]) for sample in batch],
                batch_index, len(batch), args.shard_index, args.num_shards,
            )
            start = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=encoded.attention_mask,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=GENERATION_BUDGET,
                    use_cache=args.inference_mode == "incremental",
                    logits_to_keep=1,
                    return_dict_in_generate=False,
                    output_scores=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            continuations = generated[:, batch_width:].cpu().tolist()
            torch.cuda.synchronize()
            batch_seconds = time.perf_counter() - start
            for sample, continuation in zip(batch, continuations, strict=True):
                sample_id, task, position, row, token_ids = sample
                # HF 在已结束行填充 pad；只保留该行截至首个 EOS 的真实生成。
                end = next(
                    (index + 1 for index, token in enumerate(continuation) if token in eos_ids),
                    len(continuation),
                )
                continuation = continuation[:end]
                prediction = tokenizer.decode(continuation, skip_special_tokens=True)
                cleaned = postprocess_pred(prediction)
                score = string_match_all([cleaned], [row["outputs"]])
                record = {
                    "id": sample_id,
                    "execution_attempt": execution_attempt,
                    "shard_index": args.shard_index,
                    "length": length,
                    "task": task,
                    "sample_position": position,
                    "source_index": row["index"],
                    "prediction": prediction,
                    "prediction_for_scoring": cleaned,
                    "references": row["outputs"],
                    "score": score,
                    "input_tokens": len(token_ids),
                    "batch_input_width": batch_width,
                    "generated_tokens": len(continuation),
                    "token_ids": continuation,
                    "batch_index": batch_index,
                    "batch_size": len(batch),
                    "batch_seconds": batch_seconds,
                    "seconds": batch_seconds / len(batch),
                }
                predictions.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                predictions.flush()
                records[sample_id] = record
                logger.info(
                    "%s: score=%.4f, tokens=%s, seconds=%.3f",
                    record["id"],
                    score,
                    len(continuation),
                    record["seconds"],
                )

    summary = (
        {"completed_at": timestamp(), "samples": len(records)}
        if args.sample_ids_file is not None else build_summary(plan, samples, records, output)
    )
    write_json(output / "summary.json", summary)
    logger.info("Completed %s samples; summary: %s", len(records), output / "summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inference-mode", choices=("original", "incremental"), default="original")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1, help="Maximum samples per length-sorted batch")
    parser.add_argument("--batch-size-by-length", nargs="+", default=[], metavar="LENGTH=BATCH")
    parser.add_argument("--sample-ids-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--num-shards", type=int, default=1, help="Independent row shards per task and length")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based row shard index")
    parser.add_argument(
        "--ruler-data",
        type=Path,
        default=PROJECT_ROOT / "data/benchmarks/ruler-v1/reproductions/lact-760m",
    )
    parser.add_argument("--lengths", type=int, nargs="+", choices=LENGTHS, default=list(LENGTHS))
    parser.add_argument(
        "--resume", action="store_true", help="Continue an existing output directory"
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    try:
        args.batch_size_by_length = parse_batch_size_by_length(args.batch_size_by_length, args.lengths)
    except ValueError as error:
        parser.error(str(error))
    effective_sizes = [args.batch_size_by_length.get(str(length), args.batch_size) for length in args.lengths]
    if max(effective_sizes) > 1 and args.inference_mode != "incremental":
        parser.error("--batch-size > 1 requires --inference-mode incremental")
    if not 1 <= args.num_shards <= SAMPLES_PER_TASK_LENGTH:
        parser.error(f"--num-shards must be between 1 and {SAMPLES_PER_TASK_LENGTH}")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, num_shards)")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "1"}:
        parser.error("CUDA_VISIBLE_DEVICES must be 0 or 1; select one physical GPU")
    model_class = (
        IncrementalLaCTForCausalLM if args.inference_mode == "incremental" else LaCTForCausalLM
    )
    run_source = capture_source(__file__, {args.inference_mode: model_class})
    output = args.output_dir.expanduser().resolve()
    if args.resume:
        if not output.is_dir():
            parser.error("--resume requires an existing output directory")
    else:
        output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "evaluation.log")],
    )
    logger = logging.getLogger("lact-evaluation")
    execution_path = output / "execution.json"
    execution = (
        json.loads(execution_path.read_text())
        if args.resume and execution_path.exists()
        else {
            "attempts": [],
            "seconds": 0.0,
            "peak_cuda_memory_allocated_bytes": 0,
        }
    )
    execution_attempt = len(execution["attempts"])
    attempt = {
        "execution_attempt": execution_attempt,
        "started_at": timestamp(),
        "resume": args.resume,
        "status": "running",
        "source": run_source,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "batch_size": args.batch_size,
        "batch_size_by_length": args.batch_size_by_length,
        "batching": BATCHING,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }
    execution["attempts"].append(attempt)
    write_json(execution_path, execution)
    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        run(args, output, logger, run_source, execution_attempt)
        attempt["status"] = "completed"
    except BaseException as error:
        attempt["status"] = "failed"
        attempt["error"] = f"{type(error).__name__}: {error}"
        logger.exception("Evaluation failed; partial predictions are not a completed result")
        raise
    finally:
        attempt["finished_at"] = timestamp()
        attempt["seconds"] = time.perf_counter() - start
        attempt["peak_cuda_memory_allocated_bytes"] = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )
        execution["seconds"] += attempt["seconds"]
        execution["peak_cuda_memory_allocated_bytes"] = max(
            execution["peak_cuda_memory_allocated_bytes"],
            attempt["peak_cuda_memory_allocated_bytes"],
        )
        write_json(execution_path, execution)
        if attempt["status"] == "completed":
            summary_path = output / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["seconds"] = execution["seconds"]
            summary["peak_cuda_memory_allocated_bytes"] = execution[
                "peak_cuda_memory_allocated_bytes"
            ]
            write_json(summary_path, summary)


if __name__ == "__main__":
    main()
