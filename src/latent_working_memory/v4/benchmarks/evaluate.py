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

from latent_working_memory.runtime import initialize_device
from latent_working_memory.v4.benchmarks.data import (
    LOCOMO_COMMIT,
    RULER_COMMIT,
    load_samples,
)
from latent_working_memory.v4.benchmarks.reporting import benchmark_media
from latent_working_memory.v4.benchmarks.scoring import score_sample, summarize_scores
from latent_working_memory.v4.checkpoint import read_checkpoint
from latent_working_memory.v4.evaluation_records import EvaluationHistory
from latent_working_memory.v4.evaluation_runtime import (
    load_evaluation_inputs,
    load_evaluation_model,
)
from latent_working_memory.v4.io import write_json
from latent_working_memory.v4.tracking import append_evaluation, validate_evaluation_publication


DEFAULT_DATA_DIRS = {
    "ruler": Path("data/benchmarks/ruler-v1"),
    "locomo": Path("data/benchmarks/locomo"),
}
DEFAULT_RULER_LENGTHS = (8192, 16384, 32768)


def evaluate(
    benchmark, checkpoint_path, output_dir, data_dir=None, lengths=None, device="cuda", tracking=None,
    checkpoint_series=False,
):
    if benchmark not in DEFAULT_DATA_DIRS:
        raise ValueError("benchmark must be ruler or locomo")
    if benchmark == "locomo" and lengths is not None:
        raise ValueError("lengths applies only to RULER")
    if tracking not in {None, "online", "disabled"}:
        raise ValueError("tracking must be online or disabled")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("benchmark evaluation uses one process")
    output_dir = Path(output_dir).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = read_checkpoint(checkpoint_path)
    run, step = checkpoint["run"], checkpoint["cursor"]["step"]
    if tracking is None:
        tracking = "online" if run["training"]["swanlab_project"] is not None else "disabled"
    if tracking == "online":
        validate_evaluation_publication(run, checkpoint_path, step)

    if data_dir is None:
        data_dir = DEFAULT_DATA_DIRS[benchmark]
        if benchmark == "ruler":
            data_dir = data_dir / Path(run["model"]["model_name_or_path"]).name
    data_dir = Path(data_dir).expanduser().resolve()
    selected_lengths = DEFAULT_RULER_LENGTHS if lengths is None else tuple(lengths)
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
        "model": asdict(config),
        "tokenizer": str(Path(run["training"]["output_dir"]) / "tokenizer"),
        "data": {
            "directory": str(data_dir),
            "source_commit": RULER_COMMIT if benchmark == "ruler" else LOCOMO_COMMIT,
            "samples": len(samples),
            "fingerprint": fingerprint.hexdigest(),
        },
        "device": str(device),
        "tracking": tracking,
        "checkpoint_series": checkpoint_series,
        "model_window": model_window,
        "protocol": {
            "decoding": "greedy; checkpoint tokenizer chat template; enable_thinking=False",
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
            if tracking == "online":
                append_evaluation(
                    run, checkpoint_path, summary, output_dir,
                    media=None if checkpoint_series else benchmark_media(summary),
                    checkpoint_series=checkpoint_series,
                )
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
        with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
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
        write_json(output_dir / "summary.json", summary)
        if tracking == "online":
            append_evaluation(
                run, checkpoint_path, summary, output_dir,
                media=None if checkpoint_series else benchmark_media(summary),
                checkpoint_series=checkpoint_series,
            )
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("ruler", "locomo"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--benchmark-data",
        type=Path,
        help="默认 RULER 为 data/benchmarks/ruler-v1/<model-name>，LoCoMo 为 data/benchmarks/locomo",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", help="RULER 长度，默认 8192 16384 32768")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tracking", choices=("online", "disabled"))
    parser.add_argument("--checkpoint-series", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    evaluate(
        args.benchmark,
        args.checkpoint,
        args.output_dir,
        args.benchmark_data,
        args.lengths,
        args.device,
        args.tracking,
        args.checkpoint_series,
    )


if __name__ == "__main__":
    main()
