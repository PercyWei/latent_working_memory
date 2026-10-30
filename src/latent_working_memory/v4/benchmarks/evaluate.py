"""从 v4 checkpoint 对 RULER v1／LoCoMo 执行独立生成和官方评分。"""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import torch
from transformers import set_seed

from latent_working_memory.runtime import initialize_device
from latent_working_memory.benchmarks.comparison import comparison_data
from latent_working_memory.benchmarks.data import load_samples
from latent_working_memory.benchmarks.protocol import (
    DEFAULT_RULER_LENGTHS,
    data_identity,
    encode_prompt,
    protocol,
)
from latent_working_memory.benchmarks.scoring import score_sample, summarize_scores
from latent_working_memory.v4.benchmarks.ruler_results import (
    length_config,
    load_length_summaries,
    merge_length_summaries,
    ruler_data_identity,
)
from latent_working_memory.v4.checkpoint import read_checkpoint
from latent_working_memory.v4.evaluation_records import (
    EvaluationHistory,
    evaluation_config_path,
    load_evaluation_settings,
    normalize_ruler_lengths,
    print_evaluation_summary,
)
from latent_working_memory.v4.evaluation_runtime import (
    load_evaluation_inputs,
    load_evaluation_model,
)
from latent_working_memory.v4.io import write_json


def _evaluate_samples(benchmark, samples, output_dir, evaluation_config, model, tokenizer, device):
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    input_tokens, generated_tokens, write_events = 0, 0, 0
    generation_seconds, write_seconds = 0.0, 0.0
    peak_allocated, peak_reserved = 0, 0
    started = time.perf_counter()
    temporary = output_dir / "details.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        for index, sample in enumerate(samples, 1):
            ids = encode_prompt(sample, tokenizer, evaluation_config["model_window"], benchmark)
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
            length = f" length={sample.metadata['length']}" if benchmark == "ruler" else ""
            print(
                f"{benchmark}{length} {index}/{len(samples)} | score {record['score']:.4f} | "
                f"elapsed {time.perf_counter() - started:.1f}s",
                flush=True,
            )
    summary = {
        **evaluation_config,
        "comparison_data": comparison_data(samples, benchmark),
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
    temporary.replace(output_dir / "details.jsonl")
    return summary


def evaluate(
    benchmark, checkpoint_path, output_dir, data_dir=None, lengths=None, device="cuda",
):
    if benchmark not in {"ruler", "locomo"}:
        raise ValueError("benchmark must be ruler or locomo")
    if benchmark == "locomo" and lengths is not None:
        raise ValueError("lengths applies only to RULER")
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
    if benchmark == "ruler":
        selected_lengths = tuple(normalize_ruler_lengths(selected_lengths))
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = read_checkpoint(checkpoint_path)
    run, step = checkpoint["run"], checkpoint["cursor"]["step"]

    samples = load_samples(benchmark, data_dir, selected_lengths)
    config, tokenizer, backbone_config = load_evaluation_inputs(checkpoint)
    evaluation_config = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "eval_id": output_dir.parents[1].name,
        "split": "test",
        "benchmark": benchmark,
        "checkpoint": {"path": str(checkpoint_path), "step": step},
        "training_run": run["training"]["output_dir"],
        "seed": seed,
        "model": asdict(config),
        "data": (
            ruler_data_identity(samples, data_dir)
            if benchmark == "ruler" else data_identity(samples, data_dir, benchmark)
        ),
        "device": str(device),
        "model_window": backbone_config.max_position_embeddings,
        "protocol": protocol(
            benchmark, selected_lengths,
            "each question starts from initial memory; local TTT during streaming",
        ),
    }
    history = EvaluationHistory(output_dir, evaluation_config)
    with history.lock():
        summary = history.cached_summary()
        completed = load_length_summaries(output_dir, evaluation_config) if benchmark == "ruler" else {}
        missing_lengths = [length for length in selected_lengths if length not in completed]
        if summary is not None and (
            benchmark == "locomo"
            or not missing_lengths and summary["protocol"]["lengths"] == list(selected_lengths)
        ):
            print(f"{benchmark} reusing cached summary: {output_dir / 'summary.json'}", flush=True)
            print_evaluation_summary(summary)
            return summary
        if benchmark == "ruler" and not missing_lengths:
            summary = merge_length_summaries(output_dir, evaluation_config, completed, history, samples)
            print_evaluation_summary(summary)
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
        if benchmark == "locomo":
            summary = _evaluate_samples(
                benchmark, samples, output_dir, evaluation_config, model, tokenizer, device,
            )
            history.save_summary(summary)
        else:
            for length in missing_lengths:
                set_seed(seed)
                length_dir = output_dir / "lengths" / str(length)
                print(f"ruler evaluating missing length {length}", flush=True)
                completed[length] = _evaluate_samples(
                    benchmark,
                    [sample for sample in samples if sample.metadata["length"] == length],
                    length_dir,
                    length_config(evaluation_config, [length], samples),
                    model,
                    tokenizer,
                    device,
                )
                write_json(length_dir / "summary.json", completed[length])
                summary = merge_length_summaries(output_dir, evaluation_config, completed, history, samples)
        print_evaluation_summary(summary)
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
    args = parser.parse_args()
    evaluate(
        args.benchmark,
        args.checkpoint,
        args.output_dir,
        args.benchmark_data,
        args.lengths,
        args.device,
    )


if __name__ == "__main__":
    main()
