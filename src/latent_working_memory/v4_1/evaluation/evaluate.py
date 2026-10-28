"""读取 v4.1 checkpoint，执行独立 QA 或公共 benchmark 评估。"""

import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import time

import torch
from transformers import AutoConfig, AutoTokenizer, set_seed

from latent_working_memory.benchmarks.data import DEFAULT_RULER_LENGTHS, load_samples
from latent_working_memory.benchmarks.protocol import data_identity, encode_prompt, protocol
from latent_working_memory.benchmarks.scoring import score_sample, summarize_scores
from latent_working_memory.data_preparation.personamem.audit import scores
from latent_working_memory.runtime import initialize_device
from latent_working_memory.v4_1.config import config_from_dict
from latent_working_memory.v4_1.data import dataset_identity, load_episodes
from latent_working_memory.v4_1.evaluation.report import build_report, summarize_outputs
from latent_working_memory.v4_1.modeling.model import OnlineMemoryLM
from latent_working_memory.v4_1.training.checkpoint import checkpoint_file, read_checkpoint
from latent_working_memory.v4_1.training.train import tracking_run


def _benchmark_selection(samples, task, max_samples):
    if max_samples is None:
        return samples
    if task != "ruler":
        return samples[:max_samples]
    # 保留所有 task/length，才能使用官方的先任务后长度的汇总规则。
    counts, selected = Counter(), []
    for sample in samples:
        group = (sample.metadata["length"], sample.metadata["task"])
        if counts[group] < max_samples:
            selected.append(sample)
            counts[group] += 1
    return tuple(selected)


def evaluate(checkpoint_path, data, output_dir, task="test", device="cuda",
             max_new_tokens=None, max_samples=None, lengths=DEFAULT_RULER_LENGTHS):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("v4.1 evaluation supports one process; do not use torchrun")
    for name, value in (("max_new_tokens", max_new_tokens), ("max_samples", max_samples)):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be a positive integer")
    checkpoint_path = checkpoint_file(checkpoint_path)
    state = read_checkpoint(checkpoint_path)
    config = config_from_dict(state["config"])
    training_directory = Path(config.training.output_dir).expanduser().resolve()
    training_run = {"directory": str(training_directory), "swanlab": None}
    if config.training.tracking == "swanlab":
        identity = json.loads((training_directory / "swanlab.json").read_text(encoding="utf-8"))
        training_run["swanlab"] = {
            "project": config.training.swanlab_project, "id": identity["id"], "url": identity["url"],
        }
    device = initialize_device(device)
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)
    model_window = AutoConfig.from_pretrained(config.model.model_name_or_path).max_position_embeddings
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"evaluation output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if task == "test":
        samples, references, statistics = load_episodes(
            data, tokenizer, split="test", model_window=model_window,
            max_samples=max_samples, seed=config.seed,
        )
        test_documents = set(statistics["raw_document_ids"])
        if any(test_documents & set(state["binding"][f"{split}_documents"]) for split in ("train", "dev")):
            raise ValueError("test documents overlap the checkpoint's train/dev documents")
        identity = {**dataset_identity(samples), "statistics": statistics}
        evaluation_protocol = {
            "input": "explicit document/question boundary; no writes during QA",
            "generation": "greedy; independent memory per question and per NLL/generation path",
            "generation_budget": max_new_tokens or 128,
            "nll": "answer-token weighted; includes assistant ending tokens",
            "scoring": "single-reference QA EM/token-overlap F1; normalized case/articles/punctuation",
            "teacher": "no offline teacher cache; online local inner AM only",
            "write_diagnostics": "generation path only; inner losses averaged over write events and layers",
        }
    else:
        samples = _benchmark_selection(load_samples(task, data, tuple(lengths)), task, max_samples)
        if max_new_tokens is not None:
            samples = tuple(replace(sample, max_new_tokens=max_new_tokens) for sample in samples)
        identity = data_identity(samples, data, task)
        evaluation_protocol = protocol(
            task, lengths,
            "per-layer latent memory KV plus retained raw KV; compression during complete prompt prefill",
        )
        evaluation_protocol.update({
            "write_boundary": "complete public prompt (including question) is context; freeze memory during generation",
            "training_boundary_difference": "training/test QA freeze memory before separately encoded question",
            "teacher": "no offline teacher cache; online local inner AM only",
            "write_diagnostics": "inner losses averaged over write events and layers",
            "selection": "first max_samples per task/length" if task == "ruler" else "first max_samples overall",
            "max_samples": max_samples,
        })
        if max_new_tokens is not None:
            evaluation_protocol["generation_budget"] = f"explicit override: {max_new_tokens} tokens"
    model = OnlineMemoryLM(config.model).to(device)
    model.load_trainable_state_dict(state["model"])
    model.eval()
    step = state["cursor"]["step"]
    del state
    records = []
    nll_sum = target_tokens = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with (output_dir / "details.jsonl").open("w", encoding="utf-8") as stream:
        for index, sample in enumerate(samples, 1):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            sample_started = time.perf_counter()
            if task == "test":
                context_ids, question_ids = sample.context_ids, sample.question_ids
                requested_budget = max_new_tokens or 128
                budget = min(requested_budget, model_window - len(context_ids) - len(question_ids))
                if budget < 1:
                    raise ValueError(f"{sample.id}: no model-window capacity remains for generation")
                nll_started = time.perf_counter()
                with torch.no_grad():
                    output = model(sample, teacher=None, differentiable=False)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                nll_seconds = time.perf_counter() - nll_started
                nll = float(output["qa_loss"])
                nll_sum += nll * output["target_tokens"]
                target_tokens += output["target_tokens"]
                reference = references[sample.id]
                row = {
                    "id": sample.id, "document_id": sample.document_id,
                    "question": reference["question"], "references": [reference["answer"]],
                    "nll": nll, "target_tokens": output["target_tokens"],
                    "nll_seconds": nll_seconds,
                }
            else:
                context_ids = tuple(encode_prompt(sample, tokenizer, model_window, task))
                question_ids = ()
                requested_budget = budget = sample.max_new_tokens
                row = {
                    "id": sample.id, "question": sample.metadata.get("question", ""),
                    "references": list(sample.references), "metadata": sample.metadata,
                }
            generation_started = time.perf_counter()
            with torch.no_grad():
                result = model.generate(context_ids, question_ids, budget, tokenizer.eos_token_id)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            generation_seconds = time.perf_counter() - generation_started
            generated = result["token_ids"]
            prediction = tokenizer.decode(generated, skip_special_tokens=True)
            stop_reason = result["stop_reason"]
            if stop_reason == "max_new_tokens" and budget < requested_budget:
                stop_reason = "model_window"
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            row.update({
                "prediction": prediction,
                "raw_prediction": tokenizer.decode(generated, skip_special_tokens=False),
                "generated_token_ids": generated,
                "prompt_tokens": len(context_ids) + len(question_ids),
                "context_tokens": len(context_ids), "question_tokens": len(question_ids),
                "generated_tokens": len(generated), "max_new_tokens": budget,
                "requested_max_new_tokens": requested_budget,
                "stop_reason": stop_reason, "write_events": result["write_events"],
                "inner_before": result["inner_before"], "inner_after": result["inner_after"],
                "generation_seconds": generation_seconds,
                "seconds": time.perf_counter() - sample_started,
            })
            if task == "test":
                row.update(scores(prediction, reference["answer"]))
            else:
                row["score"] = score_sample(task, sample, prediction)
            records.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            quality = row["f1"] if task == "test" else row["score"]
            metric_name = "accuracy" if task == "ruler" else "f1"
            print(
                f"{task} {index}/{len(samples)} {sample.id} {metric_name}={quality:.4f} "
                f"tokens={len(generated)}/{budget} stop={stop_reason} "
                f"generation_seconds={generation_seconds:.1f} total_seconds={row['seconds']:.1f}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    if task == "test":
        nll = nll_sum / target_tokens
        metrics = {
            "samples": len(records), "nll": nll, "perplexity": math.exp(nll),
            "em": sum(row["em"] for row in records) / len(records),
            "f1": sum(row["f1"] for row in records) / len(records),
            "target_tokens": target_tokens,
        }
    else:
        metrics = summarize_scores(task, records)
    summary = {
        "task": task, "checkpoint": str(checkpoint_path), "checkpoint_step": step,
        "training_run": training_run,
        "model": asdict(config.model), "model_window": model_window, "seed": config.seed,
        "data": identity, "protocol": evaluation_protocol, "metrics": metrics,
        "outputs": summarize_outputs(records), "stop_reasons": dict(Counter(row["stop_reason"] for row in records)),
        "resources": {
            "device": str(device), "seconds": elapsed,
            "generated_tokens": sum(row["generated_tokens"] for row in records),
            "generation_seconds": sum(row["generation_seconds"] for row in records),
            "generation_timing_scope": "context prefill, inner writes, question prefill and autoregressive decode",
            "nll_seconds": sum(row.get("nll_seconds", 0.0) for row in records),
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = build_report(output_dir)
    if config.training.tracking == "swanlab":
        with tracking_run(config, training_directory, True, device) as tracking:
            values = {f"evaluation/{task}/{key}": metrics[key] for key in ("nll", "em", "f1", "score") if key in metrics}
            for name in values:
                tracking.define_metric(name, x_axis="_step", section_name="evaluation")
            tracking.log(values, step=step)
    print(json.dumps({"metrics": metrics, "report": str(report)}, ensure_ascii=False), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="QA 数据目录或 JSONL；公共 benchmark 原生目录")
    parser.add_argument("--task", choices=("test", "locomo", "ruler"), default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-samples", type=int, help="最多 QA 数；RULER 为每个 task/length 的上限")
    parser.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_RULER_LENGTHS)
    args = parser.parse_args(argv)
    evaluate(args.checkpoint, args.data, args.output_dir, args.task, args.device,
             args.max_new_tokens, args.max_samples, args.lengths)


if __name__ == "__main__":
    main()
