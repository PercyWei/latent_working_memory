"""从 v4 checkpoint 对 RULER v1／LoCoMo 执行独立生成和官方评分。"""

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import torch
from transformers import set_seed

from latent_working_memory.runtime import initialize_device
from latent_working_memory.v4.benchmarks.data import (
    LOCOMO_COMMIT,
    RULER_COMMIT,
    load_samples,
)
from latent_working_memory.v4.benchmarks.scoring import score_sample, summarize_scores
from latent_working_memory.v4.checkpoint import read_checkpoint
from latent_working_memory.v4.evaluation_records import (
    EvaluationHistory,
    evaluation_config_path,
    load_evaluation_settings,
)
from latent_working_memory.v4.evaluation_runtime import (
    load_evaluation_inputs,
    load_evaluation_model,
)
from latent_working_memory.v4.tracking import publish_evaluations, validate_evaluation_publication


DEFAULT_RULER_LENGTHS = (8192, 16384, 32768)


def evaluate(
    benchmark, checkpoint_path, output_dir, data_dir=None, lengths=None, device="cuda", tracking=None,
    defer_publication=False,
):
    if benchmark not in {"ruler", "locomo"}:
        raise ValueError("benchmark must be ruler or locomo")
    if benchmark == "locomo" and lengths is not None:
        raise ValueError("lengths applies only to RULER")
    if tracking not in {None, "online", "disabled"}:
        raise ValueError("tracking must be online or disabled")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("benchmark evaluation uses one process")
    output_dir = Path(output_dir).expanduser().resolve()
    evaluation_path = evaluation_config_path(output_dir)
    evaluation, seed = load_evaluation_settings(output_dir)
    if data_dir is None:
        if evaluation is None:
            raise ValueError(f"{evaluation_path} is missing; specify --benchmark-data")
        data_dir = evaluation["data"]["directory"]
    if benchmark == "ruler" and lengths is None and evaluation is not None:
        lengths = evaluation["protocol"]["lengths"]
    set_seed(seed)
    data_dir = Path(data_dir).expanduser().resolve()
    selected_lengths = DEFAULT_RULER_LENGTHS if lengths is None else tuple(lengths)
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = read_checkpoint(checkpoint_path)
    run, step = checkpoint["run"], checkpoint["cursor"]["step"]
    if tracking is None:
        tracking = "online" if run["training"]["swanlab_project"] is not None else "disabled"
    if tracking == "online":
        validate_evaluation_publication(run, checkpoint_path, step)

    samples = load_samples(benchmark, data_dir, selected_lengths)
    fingerprint = hashlib.blake2b(digest_size=32)
    for sample in samples:
        fingerprint.update(
            json.dumps(asdict(sample), ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        fingerprint.update(b"\n")

    config, tokenizer, backbone_config = load_evaluation_inputs(checkpoint)
    model_window = backbone_config.max_position_embeddings
    evaluation_config = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "split": "test",
        "benchmark": benchmark,
        "checkpoint": {"path": str(checkpoint_path), "step": step},
        "training_run": run["training"]["output_dir"],
        "seed": seed,
        "model": asdict(config),
        "data": {
            "directory": str(data_dir),
            "source_commit": RULER_COMMIT if benchmark == "ruler" else LOCOMO_COMMIT,
            "samples": len(samples),
            "fingerprint": fingerprint.hexdigest(),
        },
        "device": str(device),
        "tracking": tracking,
        "model_window": model_window,
        "protocol": {
            "decoding": "greedy; base model tokenizer chat template; enable_thinking=False",
            "memory": "each question starts from initial memory; local TTT during streaming",
            "input": "complete benchmark prompt; no truncation or training length filtering",
            "generation_budget": "RULER task budget; LoCoMo 50 tokens",
            "reference": "scoring only; never passed to memory writes or generation",
            **(
                {"lengths": list(selected_lengths), "answer_prefix": "omitted in chat protocol"}
                if benchmark == "ruler"
                else {"categories": [1, 2, 3, 4], "history_order": "chronological"}
            ),
        },
    }
    history = EvaluationHistory(output_dir, evaluation_config)
    with history.lock():
        summary = history.cached_summary()
        if summary is not None:
            print(f"{benchmark} reusing cached summary: {output_dir / 'summary.json'}", flush=True)
            if tracking == "online" and not defer_publication:
                publish_evaluations(run, [(checkpoint_path, summary, output_dir)])
            return summary

        device = initialize_device(device)
        evaluation_config["device"] = str(device)
        model = load_evaluation_model(checkpoint, device, config)
        del checkpoint
        model.evaluation_phase = "generation"
        slots = [layer.initial_slots for layer in model.memory_layers]
        evaluation_config["memory"] = {
            "layers": len(slots),
            "slots_per_layer": config.num_slots,
            "dimension": config.memory_dim,
            "total_slots": len(slots) * config.num_slots,
            "elements": sum(value.numel() for value in slots),
            "bytes": sum(value.numel() * value.element_size() for value in slots),
            "dtype": str(slots[0].dtype),
        }
        records = []
        input_tokens, generated_tokens, write_events = 0, 0, 0
        generation_seconds, write_seconds = 0.0, 0.0
        peak_allocated, peak_reserved = 0, 0
        started = time.perf_counter()
        with (output_dir / "details.jsonl").open("w", encoding="utf-8") as stream:
            for index, sample in enumerate(samples, 1):
                ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": sample.prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                total_budget = len(ids) + sample.max_new_tokens
                limit = sample.metadata["length"] if benchmark == "ruler" else model_window
                if total_budget > min(limit, model_window):
                    raise ValueError(
                        f"{sample.id}: prompt ({len(ids)}) + generation budget "
                        f"({sample.max_new_tokens}) exceeds limit {min(limit, model_window)}"
                    )
                model.write_measurements.clear()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                generated_at = time.perf_counter()
                with torch.no_grad():
                    output = model.generate(
                        torch.tensor(ids, dtype=torch.long, device=device),
                        sample.max_new_tokens,
                        tokenizer.eos_token_id,
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
                    peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))
                elapsed = time.perf_counter() - generated_at
                generated = output[len(ids) :].tolist()
                prediction = tokenizer.decode(generated, skip_special_tokens=True)
                timing = model.write_timing(device)["generation"]
                record = {
                    "id": sample.id,
                    "metadata": sample.metadata,
                    "references": list(sample.references),
                    "prediction": prediction,
                    "score": score_sample(benchmark, sample, prediction),
                    "prompt_tokens": len(ids),
                    "generated_tokens": len(generated),
                    "generated_token_ids": generated,
                    "generation_budget": sample.max_new_tokens,
                    "finish_reason": (
                        "eos" if generated[-1] == tokenizer.eos_token_id else "max_new_tokens"
                    ),
                    "generation_seconds": elapsed,
                    **timing,
                }
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                records.append(record)
                input_tokens += len(ids)
                generated_tokens += len(generated)
                generation_seconds += elapsed
                write_seconds += timing["write_seconds"]
                write_events += timing["write_events"]
                print(
                    f"{benchmark} {index}/{len(samples)} | score {record['score']:.4f} | "
                    f"elapsed {time.perf_counter() - started:.1f}s",
                    flush=True,
                )
        summary = {
            **evaluation_config,
            "samples": len(records),
            "metrics": summarize_scores(benchmark, records),
            "input_tokens": input_tokens,
            "generated_tokens": generated_tokens,
            "write_events": write_events,
            "resources": {
                "evaluation_seconds": time.perf_counter() - started,
                "generation_seconds": generation_seconds,
                "write_seconds": write_seconds,
                "scope": "evaluation excludes model/data loading; generation includes prefill and decode",
                "write_measurement": "CUDA events" if device.type == "cuda" else "CPU wall time",
                **(
                    {
                        "peak_memory_allocated_bytes": peak_allocated,
                        "peak_memory_reserved_bytes": peak_reserved,
                    }
                    if device.type == "cuda"
                    else {}
                ),
            },
        }
        history.save_summary(summary)
        if tracking == "online" and not defer_publication:
            publish_evaluations(run, [(checkpoint_path, summary, output_dir)])
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("ruler", "locomo"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--benchmark-data",
        type=Path,
        help="默认沿用已保存的评估数据目录；无评估配置时必填",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--lengths", type=int, nargs="+",
        help="RULER 长度，沿用评估配置；无配置时默认 8192 16384 32768",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tracking", choices=("online", "disabled"))
    parser.add_argument("--defer-publication", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    evaluate(
        args.benchmark,
        args.checkpoint,
        args.output_dir,
        args.benchmark_data,
        args.lengths,
        args.device,
        args.tracking,
        args.defer_publication,
    )


if __name__ == "__main__":
    main()
