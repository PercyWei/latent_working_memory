"""读取 v3.1.1 checkpoint，按完整文档或公共 benchmark 样本独立评估。"""

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, set_seed

from latent_working_memory.benchmarks.data import load_samples
from latent_working_memory.benchmarks.protocol import data_identity, protocol
from latent_working_memory.runtime import initialize_device
from ..cli import DEFAULT_EVAL_BATCH_SIZE, positive_int
from ..config import config_from_dict
from ..data import load_evaluation_documents, tokenization_identity
from ..modeling.compressor import MemoryModel
from ..training.checkpoint import checkpoint_file, read_checkpoint
from .batching import compute_batch, sample_batches
from .records import (
    check_evaluation, completed_evaluation, evaluation_lock, normalize_lengths,
    prepare_evaluation, read_details, ruler_data_identity,
)
from .shards import commit_results, load_progress, write_work


def _benchmark_selection(samples, task, max_samples):
    if max_samples is None:
        return samples
    if task != "ruler":
        return samples[:max_samples]
    counts, selected = Counter(), []
    for sample in samples:
        group = (sample.metadata["length"], sample.metadata["task"])
        if counts[group] < max_samples:
            selected.append(sample)
            counts[group] += 1
    return tuple(selected)


def _evaluation_inputs(checkpoint_path, data, task, max_new_tokens, max_samples, lengths,
                       eval_id):
    for name, value in (("max_new_tokens", max_new_tokens), ("max_samples", max_samples)):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be a positive integer")
    if task not in {"test", "locomo", "ruler"}:
        raise ValueError("task must be test, locomo or ruler")
    if task != "test" and max_new_tokens is not None:
        raise ValueError("max_new_tokens applies only to test evaluation")
    if task != "ruler" and lengths is not None:
        raise ValueError("lengths applies only to RULER")
    lengths = normalize_lengths(lengths) if task == "ruler" else ()
    checkpoint_path = checkpoint_file(checkpoint_path)
    state = read_checkpoint(checkpoint_path)
    config = config_from_dict(state["config"])
    if config.stage != "post":
        raise ValueError("independent QA and benchmark evaluation requires a post checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    model_window = AutoConfig.from_pretrained(
        config.model_name_or_path,
    ).max_position_embeddings
    if task == "test":
        documents, statistics = load_evaluation_documents(
            data, tokenizer, max_samples=max_samples, seed=config.seed,
        )
        samples = tuple(documents.values())
        identity = statistics
        evaluation_protocol = {
            "input": "complete document; separately encoded question; no writes during QA",
            "generation": "greedy; fresh generation config; all QA share one document memory",
            "generation_budget": max_new_tokens or 128,
            "max_samples": max_samples,
            "selection": "stable document hash sampling; no training length filtering",
            "nll": "QA-token mean, then QA mean per document, then document mean; assistant ending included",
            "scoring": "single-reference QA EM/token-overlap F1; normalized case/articles/punctuation",
        }
    else:
        samples = _benchmark_selection(load_samples(task, data, tuple(lengths)), task, max_samples)
        identity = (ruler_data_identity(samples, data, lengths) if task == "ruler"
                    else data_identity(samples, data, task))
        evaluation_protocol = protocol(
            task, lengths, "immutable compressed embedding blocks; each block has fixed K slots",
        )
        evaluation_protocol.update({
            "write_boundary": "complete public prompt including question is compressed; no writes during generation",
            "selection": "first max_samples per task/length" if task == "ruler" else "first max_samples overall",
            "max_samples": max_samples,
        })
    evaluation_protocol.update({
        "boundary": "commit at probability >= 0.5; forced final pending flush",
        "unit_tokens": config.method.rollout.unit_tokens,
        "tokenization": tokenization_identity(tokenizer),
    })
    request = json.loads(json.dumps({
        "eval_id": eval_id, "task": task, "checkpoint": str(checkpoint_path),
        "checkpoint_step": state["cursor"]["step"],
        "training_run": {"directory": config.training.run.output_dir},
        "model": {
            "model_name_or_path": config.model_name_or_path,
            **asdict(config.model.structure), **asdict(config.model.lora),
            **asdict(config.compute), "backbone_dtype": config.compute.dtype,
        },
        "model_window": model_window, "seed": config.seed,
        "data": identity, "protocol": evaluation_protocol,
    }))
    return state, config, tokenizer, request, samples


def _check_complete_details(output_dir, summary, samples):
    if summary is not None:
        records = read_details(output_dir / "details.jsonl")
        if [row["id"] for row in records] != [sample.id for sample in samples]:
            raise ValueError("completed evaluation details differ from the selected sample order")
    return summary


def load_cached_evaluation(checkpoint_path, data, output_dir, task="test", max_new_tokens=None,
                           max_samples=None, lengths=None, eval_id=None):
    output_dir = Path(output_dir).expanduser().resolve()
    if not output_dir.exists():
        return None
    state, config, tokenizer, request, samples = _evaluation_inputs(
        checkpoint_path, data, task, max_new_tokens, max_samples, lengths, eval_id,
    )
    del state, config, tokenizer
    with evaluation_lock(output_dir):
        check_evaluation(output_dir, request)
        return _check_complete_details(output_dir, completed_evaluation(output_dir, request), samples)


def prepare_sample_work(checkpoint_path, data, output_dir, task="test", max_new_tokens=None,
                        max_samples=None, lengths=None, eval_id=None, num_workers=1):
    state, config, tokenizer, request, samples = _evaluation_inputs(
        checkpoint_path, data, task, max_new_tokens, max_samples, lengths, eval_id,
    )
    del state, config, tokenizer
    output_dir = Path(output_dir).expanduser().resolve()
    with evaluation_lock(output_dir):
        prepare_evaluation(output_dir, request)
        summary = _check_complete_details(output_dir, completed_evaluation(output_dir, request), samples)
        if summary is not None:
            return {"total_samples": len(samples), "completed_samples": len(samples),
                    "pending_samples": 0, "workers": []}
        entries = load_progress(output_dir, request, samples, repair=True)
        return write_work(output_dir, samples, entries, num_workers)


def evaluate_shard(checkpoint_path, data, output_dir, worker_rank, task="test",
                   max_new_tokens=None, max_samples=None, lengths=None, eval_id=None,
                   eval_batch_size_per_gpu=DEFAULT_EVAL_BATCH_SIZE):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("evaluation workers do not use torchrun")
    if type(worker_rank) is not int or worker_rank < 0:
        raise ValueError("worker_rank must be a non-negative integer")
    if type(eval_batch_size_per_gpu) is not int or eval_batch_size_per_gpu < 1:
        raise ValueError("eval_batch_size_per_gpu must be a positive integer")
    output_dir = Path(output_dir).expanduser().resolve()
    state, config, tokenizer, request, samples = _evaluation_inputs(
        checkpoint_path, data, task, max_new_tokens, max_samples, lengths, eval_id,
    )
    check_evaluation(output_dir, request)
    work = json.loads((output_dir / "work.json").read_text(encoding="utf-8"))
    selected_ids = work["workers"][worker_rank]
    by_id = {sample.id: sample for sample in samples}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run v3.1.1 evaluation")
    device = initialize_device("cuda")
    set_seed(config.seed)
    model = MemoryModel(config.model_name_or_path, asdict(config.model), asdict(config.compute), device)
    model.load_trainable_state_dict(state["model"])
    model.eval()
    del state, tokenizer
    progress = output_dir / "progress"
    progress.mkdir(parents=True, exist_ok=True)
    with (progress / f"worker_{worker_rank}.jsonl").open("a", encoding="utf-8") as stream:
        completed = 0

        def commit(entry):
            nonlocal completed
            stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            completed += 1
            record = entry["record"]
            print(f"{task} worker={worker_rank} {completed}/{len(selected_ids)} {record['id']} "
                  f"blocks={record['blocks']} batch={record['eval_batch_size']} "
                  f"seconds={record['seconds']:.1f}", flush=True)

        for batch in sample_batches([by_id[sample_id] for sample_id in selected_ids],
                                    task, eval_batch_size_per_gpu):
            compute_batch(batch, request, model, config, max_new_tokens, commit)
    del model


def finalize_sample_work(checkpoint_path, data, output_dir, task="test", max_new_tokens=None,
                         max_samples=None, lengths=None, eval_id=None):
    state, config, tokenizer, request, samples = _evaluation_inputs(
        checkpoint_path, data, task, max_new_tokens, max_samples, lengths, eval_id,
    )
    del state, config, tokenizer
    output_dir = Path(output_dir).expanduser().resolve()
    with evaluation_lock(output_dir):
        check_evaluation(output_dir, request)
        summary = _check_complete_details(output_dir, completed_evaluation(output_dir, request), samples)
        if summary is not None:
            return summary
        return commit_results(output_dir, request, samples, load_progress(output_dir, request, samples))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--task", choices=("test", "locomo", "ruler"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=positive_int)
    parser.add_argument("--max-samples", type=positive_int)
    parser.add_argument("--lengths", nargs="+", type=positive_int)
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--worker-rank", type=int, required=True)
    parser.add_argument("--eval-batch-size-per-gpu", type=positive_int,
                        default=DEFAULT_EVAL_BATCH_SIZE)
    args = parser.parse_args(argv)
    evaluate_shard(
        args.checkpoint, args.data, args.output_dir, args.worker_rank, task=args.task,
        max_new_tokens=args.max_new_tokens, max_samples=args.max_samples,
        lengths=args.lengths, eval_id=args.eval_id,
        eval_batch_size_per_gpu=args.eval_batch_size_per_gpu,
    )


if __name__ == "__main__":
    main()
