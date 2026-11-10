"""使用 LaCT 的原始或增量推理路径评估本地 RULER 数据。"""

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import signal
import time
from zoneinfo import ZoneInfo

from lact_model import LaCTForCausalLM, LaCTSWIGLUConfig
from incremental import IncrementalLaCTForCausalLM
from provenance import UPSTREAM_REVISION, capture_source
import torch
from transformers import AutoTokenizer

from latent_working_memory.benchmarks.vendor.ruler import (
    postprocess_pred,
    string_match_all,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multiquery",
    "niah_multivalue",
)
LENGTHS = (4096, 8192, 16384, 32768)
GENERATION_BUDGET = 128
SAMPLES_PER_TASK_LENGTH = 500


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d %H:%M:%S Asia/Shanghai")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_samples(path):
    samples = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            prompt = row["input"]
            prefix = row["answer_prefix"]
            references = row["outputs"]
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"{path}:{line_number}: input must be nonempty text")
            if not isinstance(prefix, str):
                raise ValueError(f"{path}:{line_number}: answer_prefix must be text")
            if not isinstance(references, list) or not references:
                raise ValueError(f"{path}:{line_number}: outputs must be a nonempty list")
            if any(not isinstance(value, str) or not value for value in references):
                raise ValueError(f"{path}:{line_number}: outputs must contain nonempty strings")
            samples.append(row)
    if len(samples) != SAMPLES_PER_TASK_LENGTH:
        raise ValueError(f"{path}: expected {SAMPLES_PER_TASK_LENGTH} samples, got {len(samples)}")
    return samples


def validate_prediction(record, expected, shard_index):
    sample_id = record["id"]
    if sample_id not in expected:
        raise ValueError(f"Prediction is outside the planned shard: {sample_id}")
    length, task, position, row = expected[sample_id]
    if (
        record["length"] != length
        or record["task"] != task
        or record["sample_position"] != position
        or record["source_index"] != row["index"]
        or record["references"] != row["outputs"]
        or record["shard_index"] != shard_index
    ):
        raise ValueError(f"Saved prediction does not match the current sample: {sample_id}")
    # 重新计分时直接使用原始 prediction；已保存的 score 不是汇总依据。
    postprocess_pred(record["prediction"])


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


def build_summary(plan, samples, records, output):
    shard_index, num_shards = plan["shard_index"], plan["num_shards"]
    positions = {
        key: [
            position
            for position in range(len(rows))
            if shard_index is None or position % num_shards == shard_index
        ]
        for key, rows in samples.items()
    }
    expected_ids = {
        f"ruler/{length}/{task}/{position}"
        for (length, task), selected in positions.items()
        for position in selected
    }
    if records.keys() != expected_ids:
        raise ValueError("Predictions do not exactly cover the planned samples")
    group_scores = {}
    for (length, task), rows in samples.items():
        scores = [
            string_match_all(
                [postprocess_pred(records[f"ruler/{length}/{task}/{position}"]["prediction"])],
                [rows[position]["outputs"]],
            )
            for position in positions[(length, task)]
        ]
        group_scores[(length, task)] = sum(scores) / len(scores)
    by_length = []
    for length in plan["lengths"]:
        means = [group_scores[(length, task)] for task in TASKS]
        by_length.append(
            {
                "length": length,
                "tasks": {
                    task: {
                        "samples": len(positions[(length, task)]),
                        "accuracy": round(group_scores[(length, task)], 2),
                    }
                    for task in TASKS
                },
                "s_niah_average": round(sum(means[:3]) / 3, 2),
                "multi_niah_average": round(sum(means[3:]) / 3, 2),
                "six_task_average": round(sum(means) / len(means), 2),
            }
        )
    return {
        "completed_at": timestamp(),
        "benchmark": "ruler",
        "plan_path": str(output / "plan.json"),
        "execution_path": str(output / "execution.json"),
        "checkpoint": plan["checkpoint"],
        "num_shards": num_shards,
        "shard_index": shard_index,
        "samples": len(records),
        "score_unit": "percent",
        "task_length_macro_average": round(sum(group_scores.values()) / len(group_scores), 2),
        "by_length": by_length,
        "generation": plan["generation"],
    }


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


def run(args, output, logger, run_source, execution_attempt):
    checkpoint = args.checkpoint.expanduser().resolve()
    data = args.ruler_data.expanduser().resolve()
    if not checkpoint.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint}")
    if not args.lengths or len(set(args.lengths)) != len(args.lengths):
        raise ValueError("lengths must be nonempty and unique")
    samples = {
        (length, task): read_samples(data / str(length) / task / "test.jsonl")
        for length in args.lengths
        for task in TASKS
    }
    expected = {
        f"ruler/{length}/{task}/{position}": (length, task, position, row)
        for (length, task), rows in samples.items()
        for position, row in enumerate(rows)
        if position % args.num_shards == args.shard_index
    }
    plan = {
        "created_at": timestamp(),
        "method": "lact",
        "source": run_source,
        "inference_mode": args.inference_mode,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "assigned_samples": len(expected),
        "checkpoint": str(checkpoint),
        "checkpoint_config": json.loads((checkpoint / "config.json").read_text()),
        "ruler_data": str(data),
        "lengths": args.lengths,
        "tasks": list(TASKS),
        "samples_per_task_length": [
            {"length": length, "task": task, "samples": len(rows)}
            for (length, task), rows in samples.items()
        ],
        "prompt": "base; input + answer_prefix; tokenizer default special tokens; no truncation",
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": GENERATION_BUDGET,
            "use_cache": args.inference_mode == "incremental",
            "logits_to_keep": 1,
            "recompute_complete_prefix_per_token": args.inference_mode == "original",
        },
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "parameter_dtype": "float32",
        "compute_autocast_dtype": "bfloat16",
        "upstream_revision": UPSTREAM_REVISION,
        "scoring": "NVIDIA RULER string_match_all; case-insensitive reference recall, percent",
    }
    if args.resume:
        saved_plan = json.loads((output / "plan.json").read_text())
        for key, value in plan.items():
            if (
                key not in {"created_at", "cuda_visible_devices", "source"}
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
        for (length, task), rows in samples.items():
            assigned = sum(
                position % args.num_shards == args.shard_index for position in range(len(rows))
            )
            logger.info(
                "Evaluating length=%s task=%s shard=%s/%s assigned=%s source_samples=%s",
                length, task, args.shard_index, args.num_shards, assigned, len(rows),
            )
            for position, row in enumerate(rows):
                if position % args.num_shards != args.shard_index:
                    continue
                sample_id = f"ruler/{length}/{task}/{position}"
                if sample_id in records:
                    continue
                prompt = row["input"] + row["answer_prefix"]
                encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
                input_tokens = encoded["input_ids"].shape[-1]
                if input_tokens + GENERATION_BUDGET > length:
                    raise ValueError(
                        f"{length}/{task}/{position}: prompt ({input_tokens}) + "
                        f"generation budget ({GENERATION_BUDGET}) exceeds {length}"
                    )
                inputs = {name: value.to("cuda:0") for name, value in encoded.items()}
                start = time.perf_counter()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    generated = model.generate(
                        **inputs,
                        do_sample=False,
                        num_beams=1,
                        max_new_tokens=GENERATION_BUDGET,
                        use_cache=args.inference_mode == "incremental",
                        logits_to_keep=1,
                        return_dict_in_generate=False,
                        output_scores=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                continuation = generated[0, input_tokens:]
                prediction = tokenizer.decode(continuation, skip_special_tokens=True)
                cleaned = postprocess_pred(prediction)
                score = string_match_all([cleaned], [row["outputs"]])
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
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
                    "input_tokens": input_tokens,
                    "generated_tokens": int(continuation.numel()),
                    "seconds": seconds,
                }
                predictions.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                predictions.flush()
                records[sample_id] = record
                logger.info(
                    "%s: score=%.4f, tokens=%s, seconds=%.3f",
                    record["id"],
                    score,
                    continuation.numel(),
                    seconds,
                )

    summary = build_summary(plan, samples, records, output)
    write_json(output / "summary.json", summary)
    logger.info("Completed %s samples; summary: %s", len(records), output / "summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inference-mode", choices=("original", "incremental"), default="original")
    parser.add_argument("--output-dir", type=Path, required=True)
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
