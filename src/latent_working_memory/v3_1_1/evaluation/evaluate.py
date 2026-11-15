"""共用 checkpoint、模型、样本进度与评估结果执行机制。"""

from dataclasses import asdict
import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, set_seed

from latent_working_memory.runtime import initialize_device
from ..config import config_from_dict
from ..data import tokenization_identity
from ..modeling.compressor import MemoryModel
from ..training.checkpoint import checkpoint_file, read_checkpoint
from .records import (
    check_evaluation, completed_evaluation, evaluation_lock,
    prepare_evaluation, read_details,
)
from .shards import commit_results, load_progress, write_work


def _evaluation_inputs(checkpoint_path, data, task_module, max_new_tokens, max_samples, lengths,
                       eval_id):
    for name, value in (("max_new_tokens", max_new_tokens), ("max_samples", max_samples)):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be a positive integer")
    checkpoint_path = checkpoint_file(checkpoint_path)
    state = read_checkpoint(checkpoint_path)
    config = config_from_dict(state["config"])
    if config.stage != "post":
        raise ValueError("independent QA and benchmark evaluation requires a post checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    model_window = AutoConfig.from_pretrained(
        config.model_name_or_path,
    ).max_position_embeddings
    samples, identity, evaluation_protocol = task_module.load_data(
        config, tokenizer, model_window, data, max_samples, max_new_tokens, lengths,
    )
    evaluation_protocol.update({
        "boundary": "commit at probability >= 0.5; forced final pending flush",
        "unit_tokens": config.model.structure.unit_tokens,
        "tokenization": tokenization_identity(tokenizer),
    })
    request = json.loads(json.dumps({
        "eval_id": eval_id, "task": task_module.NAME, "checkpoint": str(checkpoint_path),
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


def load_cached_evaluation(checkpoint_path, data, output_dir, task_module, max_new_tokens=None,
                           max_samples=None, lengths=None, eval_id=None):
    output_dir = Path(output_dir).expanduser().resolve()
    if not output_dir.exists():
        return None
    state, config, tokenizer, request, samples = _evaluation_inputs(
        checkpoint_path, data, task_module, max_new_tokens, max_samples, lengths, eval_id,
    )
    del state, config, tokenizer
    with evaluation_lock(output_dir):
        check_evaluation(output_dir, request)
        return _check_complete_details(output_dir, completed_evaluation(output_dir, request), samples)


def prepare_sample_work(checkpoint_path, data, output_dir, task_module, max_new_tokens=None,
                        max_samples=None, lengths=None, eval_id=None, num_workers=1):
    state, config, tokenizer, request, samples = _evaluation_inputs(
        checkpoint_path, data, task_module, max_new_tokens, max_samples, lengths, eval_id,
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


def evaluate_shard(checkpoint_path, data, output_dir, worker_rank, task_module, execution,
                   max_new_tokens=None, max_samples=None, lengths=None, eval_id=None):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("evaluation workers do not use torchrun")
    if type(worker_rank) is not int or worker_rank < 0:
        raise ValueError("worker_rank must be a non-negative integer")
    output_dir = Path(output_dir).expanduser().resolve()
    state, config, tokenizer, request, samples = _evaluation_inputs(
        checkpoint_path, data, task_module, max_new_tokens, max_samples, lengths, eval_id,
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
            print(f"{task_module.NAME} worker={worker_rank} {completed}/{len(selected_ids)} {record['id']} "
                  f"blocks={record['blocks']} batch={record['eval_batch_size']} "
                  f"seconds={record['seconds']:.1f}", flush=True)

        for batch in task_module.sample_batches(
            [by_id[sample_id] for sample_id in selected_ids], execution,
        ):
            with torch.no_grad(), model.forward_context():
                task_module.compute_batch(batch, request, model, config, max_new_tokens, commit)
    del model


def finalize_sample_work(checkpoint_path, data, output_dir, task_module, max_new_tokens=None,
                         max_samples=None, lengths=None, eval_id=None):
    state, config, tokenizer, request, samples = _evaluation_inputs(
        checkpoint_path, data, task_module, max_new_tokens, max_samples, lengths, eval_id,
    )
    del state, config, tokenizer
    output_dir = Path(output_dir).expanduser().resolve()
    with evaluation_lock(output_dir):
        check_evaluation(output_dir, request)
        summary = _check_complete_details(output_dir, completed_evaluation(output_dir, request), samples)
        if summary is not None:
            return summary
        return commit_results(
            output_dir, request, samples, load_progress(output_dir, request, samples), task_module,
        )
