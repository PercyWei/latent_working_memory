"""比较 LaCT 原始前缀重算与增量缓存；结果仅用于验证，不计入正式 RULER。"""

import argparse
import gc
import json
import logging
import os
from pathlib import Path
import statistics
import time

import torch
from transformers import AutoTokenizer

from evaluate import (
    GENERATION_BUDGET,
    LENGTHS,
    PROJECT_ROOT,
    SAMPLES_PER_TASK_LENGTH,
    TASKS,
    load_checkpoint,
    postprocess_pred,
    read_samples,
    string_match_all,
    timestamp,
    write_json,
)


VALIDATION_TASKS = ("niah_single_1", "niah_multiquery")
TEACHER_STEPS = 4
SYNTHETIC_LENGTHS = (2046, 2048, 4094)
RELATIVE_DENOMINATOR_FLOOR = 1e-6


def append_record(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def compare_logits(reference, incremental):
    reference = reference.detach().float().cpu()
    incremental = incremental.detach().float().cpu()
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(incremental).all())
    if not finite:
        return {
            "finite": False,
            "bitwise_equal": False,
            "argmax_equal": False,
            "max_abs": None,
            "max_rel": None,
        }
    difference = (incremental - reference).abs()
    return {
        "finite": True,
        "bitwise_equal": bool(torch.equal(reference, incremental)),
        "argmax_equal": bool(torch.equal(reference.argmax(-1), incremental.argmax(-1))),
        "max_abs": difference.max().item(),
        "max_rel": (difference / reference.abs().clamp_min(RELATIVE_DENOMINATOR_FLOOR))
        .max()
        .item(),
    }


def compare_tokens(reference, incremental):
    common = min(len(reference), len(incremental))
    first_difference = next(
        (index for index in range(common) if reference[index] != incremental[index]),
        None,
    )
    if first_difference is None and len(reference) != len(incremental):
        first_difference = common
    return {
        "token_ids_equal": reference == incremental,
        "first_different_generated_position": first_difference,
        "reference_generated_tokens": len(reference),
        "incremental_generated_tokens": len(incremental),
    }


def teacher_forced(model, prefix, continuation, sample_id, output, logger):
    prefix = prefix.to("cuda:0")
    continuation = continuation.to("cuda:0")
    cache = None
    records = []
    for step in range(TEACHER_STEPS + 1):
        complete = torch.cat((prefix, continuation[:, :step]), dim=-1)
        mask = torch.ones_like(complete)
        new_tokens = prefix if step == 0 else continuation[:, step - 1 : step]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            reference = model(
                input_ids=complete,
                attention_mask=mask,
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
            incremental = model(
                input_ids=new_tokens,
                attention_mask=mask,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
        cache = incremental.past_key_values
        if cache is None:
            raise RuntimeError("Incremental forward did not return past_key_values")
        comparison = compare_logits(reference.logits[:, -1], incremental.logits[:, -1])
        record = {
            "id": sample_id,
            "teacher_forced_step": step,
            "consumed_tokens": complete.shape[-1],
            **comparison,
        }
        append_record(output / "teacher_forced.jsonl", record)
        records.append(record)
        logger.info(
            "%s teacher step=%s tokens=%s argmax_equal=%s max_abs=%s max_rel=%s",
            sample_id,
            step,
            complete.shape[-1],
            comparison["argmax_equal"],
            comparison["max_abs"],
            comparison["max_rel"],
        )
        del reference, incremental
    # Cache ownership is local to this sample; the next sample starts with None.
    del cache
    return records


def measure_generation(model, tokenizer, input_ids, use_cache):
    gc.collect()
    torch.cuda.empty_cache()
    input_ids = input_ids.to("cuda:0")
    attention_mask = torch.ones_like(input_ids)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    start = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            do_sample=False,
            num_beams=1,
            max_new_tokens=GENERATION_BUDGET,
            use_cache=use_cache,
            logits_to_keep=1,
            return_dict_in_generate=False,
            output_scores=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    continuation = generated[0, input_ids.shape[-1] :].cpu().tolist()
    return {
        "use_cache": use_cache,
        "seconds": seconds,
        "allocated_before_bytes": allocated_before,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "generated_tokens": len(continuation),
        "generated_full_budget": len(continuation) == GENERATION_BUDGET,
        "token_ids": continuation,
        "prediction": tokenizer.decode(continuation, skip_special_tokens=True),
    }


def estimate_time(generations, lengths):
    by_length = []
    selected_totals = {"original": 0.0, "incremental": 0.0}
    for length in lengths:
        matching = [record for record in generations if record["length"] == length]
        entry = {"length": length, "validation_samples": len(matching)}
        for mode in selected_totals:
            seconds = [record[mode]["seconds"] for record in matching]
            mean = statistics.mean(seconds)
            estimated = len(TASKS) * SAMPLES_PER_TASK_LENGTH * mean
            selected_totals[mode] += estimated
            entry[mode] = {
                "mean_seconds": mean,
                "min_seconds": min(seconds),
                "max_seconds": max(seconds),
                "formal_samples": len(TASKS) * SAMPLES_PER_TASK_LENGTH,
                "estimated_generation_seconds": estimated,
            }
        entry["speedup"] = entry["original"]["mean_seconds"] / entry["incremental"]["mean_seconds"]
        by_length.append(entry)
    result = {
        "by_length": by_length,
        "selected_length_generation_seconds": selected_totals,
        "selected_formal_samples": len(lengths) * len(TASKS) * SAMPLES_PER_TASK_LENGTH,
        "note": (
            "按每长度六任务各500条外推，计时均值来自两种任务的验证样本。"
            "不含模型加载、独立warmup、分词和结果写入；小样本外推不是置信区间。"
        ),
    }
    if set(lengths) == set(LENGTHS):
        result["estimated_generation_seconds_12000"] = selected_totals
        result["estimated_generation_hours_12000"] = {
            mode: seconds / 3600 for mode, seconds in selected_totals.items()
        }
    return result


def run(args, output, logger):
    checkpoint = args.checkpoint.expanduser().resolve()
    data = args.ruler_data.expanduser().resolve()
    if not checkpoint.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint}")
    if len(set(args.lengths)) != len(args.lengths):
        raise ValueError("lengths must be unique")
    samples = {
        (length, task): read_samples(data / str(length) / task / "test.jsonl")[0]
        for length in args.lengths
        for task in VALIDATION_TASKS
    }
    plan = {
        "created_at": timestamp(),
        "purpose": "incremental correctness and timing validation; not formal benchmark",
        "checkpoint": str(checkpoint),
        "checkpoint_config": json.loads((checkpoint / "config.json").read_text()),
        "ruler_data": str(data),
        "lengths": args.lengths,
        "tasks": list(VALIDATION_TASKS),
        "sample_position_per_task": 0,
        "real_samples": len(samples),
        "synthetic_prefix_lengths": list(SYNTHETIC_LENGTHS),
        "teacher_forced_continuation_tokens": TEACHER_STEPS,
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": GENERATION_BUDGET,
            "respect_original_eos": True,
            "logits_to_keep": 1,
            "fresh_past_key_values_each_call": True,
        },
        "prompt": "input + answer_prefix; tokenizer default special tokens; no truncation",
        "warmup": "每长度每路径各完整生成一次，单独记录，不计入稳定耗时",
        "logit_relative_denominator_floor": RELATIVE_DENOMINATOR_FLOOR,
        "acceptance": (
            "所有teacher-forced logits有限且argmax一致；"
            "独立greedy生成token IDs严格相等；跨样本重放结果严格相等。"
            "另报告logits最大绝对和相对差，不把浮点逐位相同作为要求。"
        ),
        "parameter_dtype": "float32",
        "compute_autocast_dtype": "bfloat16",
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    }
    write_json(output / "plan.json", plan)
    if not torch.cuda.is_available():
        raise RuntimeError("Incremental validation requires the native CUDA LaCT environment")
    start = time.perf_counter()
    model, config = load_checkpoint(checkpoint, "incremental")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    plan["runtime"] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "model_class": type(model).__name__,
        "actual_parameter_dtype": str(next(model.parameters()).dtype),
        "num_lact_heads": config.num_lact_heads,
        "lact_chunk_size": config.lact_chunk_size,
        "use_muon": config.use_muon,
        "ttt_prenorm": config.ttt_prenorm,
        "ttt_nope": config.ttt_nope,
        "fp32_states": config.fp32_states,
    }
    write_json(output / "plan.json", plan)
    if config.lact_chunk_size != 2048:
        raise ValueError("This validation uses boundaries for lact_chunk_size=2048")
    continuation_ids = tokenizer.encode(
        " The answer is remembered in the library beside the river.",
        add_special_tokens=False,
    )[:TEACHER_STEPS]
    if len(continuation_ids) != TEACHER_STEPS:
        raise ValueError("Teacher continuation does not provide four tokens")
    continuation = torch.tensor([continuation_ids], dtype=torch.long)
    plan["teacher_forced_token_ids"] = continuation_ids
    write_json(output / "plan.json", plan)
    synthetic_ids = tokenizer.encode(
        "Alice stored the number 1729 in a blue notebook. Bob kept 2468 in a red box. "
        "The river passes the library, and the museum stands beside a quiet garden. ",
        add_special_tokens=False,
    )
    teacher_records = []
    for length in SYNTHETIC_LENGTHS:
        repeated = (synthetic_ids * (length // len(synthetic_ids) + 1))[:length]
        teacher_records.extend(
            teacher_forced(
                model,
                torch.tensor([repeated], dtype=torch.long),
                continuation,
                f"synthetic/{length}",
                output,
                logger,
            )
        )
    generations = []
    warmups = []
    first_input = None
    for length in args.lengths:
        for task_index, task in enumerate(VALIDATION_TASKS):
            row = samples[length, task]
            sample_id = f"ruler/{length}/{task}/0"
            encoded = tokenizer(row["input"] + row["answer_prefix"], return_tensors="pt")
            input_ids = encoded["input_ids"]
            input_tokens = input_ids.shape[-1]
            if input_tokens + GENERATION_BUDGET > length:
                raise ValueError(f"{sample_id}: prompt plus generation budget exceeds {length}")
            if first_input is None:
                first_input = input_ids.clone()
            if task_index == 0:
                for mode, use_cache in (("original", False), ("incremental", True)):
                    logger.info("Warmup length=%s mode=%s", length, mode)
                    measured = measure_generation(model, tokenizer, input_ids, use_cache)
                    warmup = {"id": sample_id, "length": length, "mode": mode, **measured}
                    append_record(output / "warmup.jsonl", warmup)
                    warmups.append(warmup)
            teacher_records.extend(
                teacher_forced(model, input_ids, continuation, sample_id, output, logger)
            )
            record = {
                "id": sample_id,
                "length": length,
                "task": task,
                "sample_position": 0,
                "source_index": row["index"],
                "input_tokens": input_tokens,
                "references": row["outputs"],
            }
            modes = (("original", False), ("incremental", True))
            if task_index % 2:
                modes = tuple(reversed(modes))
            for mode, use_cache in modes:
                logger.info("Generating %s mode=%s", sample_id, mode)
                measured = measure_generation(model, tokenizer, input_ids, use_cache)
                measured["score"] = string_match_all(
                    [postprocess_pred(measured["prediction"])], [row["outputs"]]
                )
                record[mode] = measured
            record["comparison"] = compare_tokens(
                record["original"]["token_ids"], record["incremental"]["token_ids"]
            )
            record["speedup"] = record["original"]["seconds"] / record["incremental"]["seconds"]
            append_record(output / "generation.jsonl", record)
            generations.append(record)
            logger.info(
                "%s token_ids_equal=%s original=%.3fs incremental=%.3fs speedup=%.2f",
                sample_id,
                record["comparison"]["token_ids_equal"],
                record["original"]["seconds"],
                record["incremental"]["seconds"],
                record["speedup"],
            )
    logger.info("Checking sample isolation by replaying the first input after all other inputs")
    replay = measure_generation(model, tokenizer, first_input, True)
    isolation = {
        "replayed_id": generations[0]["id"],
        "preceding_id": generations[-1]["id"],
        "original_incremental_token_ids": generations[0]["incremental"]["token_ids"],
        "replay": replay,
        "comparison": compare_tokens(
            generations[0]["incremental"]["token_ids"], replay["token_ids"]
        ),
    }
    write_json(output / "cache_isolation.json", isolation)
    all_greedy_equal = all(record["comparison"]["token_ids_equal"] for record in generations)
    all_teacher_equal = all(record["argmax_equal"] for record in teacher_records)
    all_finite = all(record["finite"] for record in teacher_records)
    isolation_equal = isolation["comparison"]["token_ids_equal"]
    passed = all_greedy_equal and all_teacher_equal and all_finite and isolation_equal
    measured_runs = warmups + [replay]
    measured_runs.extend(
        record[mode] for record in generations for mode in ("original", "incremental")
    )
    summary = {
        "finished_at": timestamp(),
        "passed": passed,
        "real_samples": len(generations),
        "teacher_forced_comparisons": len(teacher_records),
        "all_greedy_token_ids_equal": all_greedy_equal,
        "all_teacher_forced_argmax_equal": all_teacher_equal,
        "all_teacher_forced_logits_finite": all_finite,
        "cache_isolation_equal": isolation_equal,
        "all_teacher_forced_logits_bitwise_equal": all(
            record["bitwise_equal"] for record in teacher_records
        ),
        "max_logit_abs_difference": max(
            (record["max_abs"] for record in teacher_records if record["finite"]), default=None
        ),
        "max_logit_relative_difference": max(
            (record["max_rel"] for record in teacher_records if record["finite"]), default=None
        ),
        "load_seconds": load_seconds,
        "warmup_seconds": sum(record["seconds"] for record in warmups),
        "peak_generation_allocated_bytes": max(
            record["peak_allocated_bytes"] for record in measured_runs
        ),
        "peak_generation_reserved_bytes": max(
            record["peak_reserved_bytes"] for record in measured_runs
        ),
        "timing": estimate_time(generations, args.lengths),
        "formal_benchmark_completed": False,
    }
    write_json(output / "summary.json", summary)
    if not passed:
        raise RuntimeError(
            "Incremental equivalence validation failed; see summary.json and per-sample records"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ruler-data",
        type=Path,
        default=PROJECT_ROOT / "data/benchmarks/ruler-v1/reproductions/lact-760m",
    )
    parser.add_argument("--lengths", type=int, nargs="+", choices=LENGTHS, default=list(LENGTHS))
    args = parser.parse_args()
    visible = os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if visible not in {"0", "1"}:
        raise ValueError("CUDA_VISIBLE_DEVICES must select physical GPU 0 or 1")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "validation.log")],
    )
    logger = logging.getLogger(__name__)
    execution = {"started_at": timestamp(), "status": "running"}
    write_json(output / "execution.json", execution)
    start = time.perf_counter()
    try:
        run(args, output, logger)
        execution["status"] = "complete"
    except BaseException as error:
        execution["status"] = "failed"
        execution["error"] = f"{type(error).__name__}: {error}"
        logger.exception("Validation failed")
        raise
    finally:
        execution["finished_at"] = timestamp()
        execution["wall_seconds"] = time.perf_counter() - start
        write_json(output / "execution.json", execution)


if __name__ == "__main__":
    main()
