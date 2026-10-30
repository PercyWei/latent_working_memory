"""读取 v4.1 checkpoint，执行独立 QA 或公共 benchmark 评估。"""

import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoConfig, AutoTokenizer, set_seed

from latent_working_memory.benchmarks.data import load_samples
from latent_working_memory.benchmarks.protocol import data_identity, encode_prompt, protocol
from latent_working_memory.benchmarks.scoring import score_sample
from latent_working_memory.data_preparation.personamem.audit import scores
from latent_working_memory.runtime import initialize_device
from latent_working_memory.v4_1.config import config_from_dict
from latent_working_memory.v4_1.data import dataset_identity, load_episodes
from latent_working_memory.v4_1.evaluation.records import (
    check_evaluation, completed_evaluation, evaluation_lock, normalize_lengths,
    prepare_evaluation, read_details, ruler_data_identity,
)
from latent_working_memory.v4_1.evaluation.shards import (
    IncompleteEvaluationError, commit_results, load_progress, write_work,
)
from latent_working_memory.v4_1.modeling.model import OnlineMemoryLM
from latent_working_memory.v4_1.training.checkpoint import checkpoint_file, read_checkpoint


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


def _evaluation_inputs(checkpoint_path, data, task, max_new_tokens, max_samples, lengths, eval_id):
    for name, value in (("max_new_tokens", max_new_tokens), ("max_samples", max_samples)):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be a positive integer")
    if task not in {"test", "locomo", "ruler"}:
        raise ValueError("task must be test, locomo or ruler")
    if task != "ruler" and lengths is not None:
        raise ValueError("lengths applies only to RULER")
    lengths = normalize_lengths(lengths) if task == "ruler" else ()
    checkpoint_path = checkpoint_file(checkpoint_path)
    state = read_checkpoint(checkpoint_path)
    config = config_from_dict(state["config"])
    training_directory = Path(config.training.output_dir).expanduser().resolve()
    training_run = {"directory": str(training_directory)}
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)
    model_window = AutoConfig.from_pretrained(config.model.model_name_or_path).max_position_embeddings
    references = None
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
            "max_samples": max_samples,
            "selection": "QA hash sampling after length filtering",
            "nll": "answer-token weighted; includes assistant ending tokens",
            "scoring": "single-reference QA EM/token-overlap F1; normalized case/articles/punctuation",
            "teacher": "no offline teacher cache; online local inner AM only",
            "write_diagnostics": "generation path only; inner losses averaged over write events and layers",
        }
    else:
        samples = _benchmark_selection(load_samples(task, data, tuple(lengths)), task, max_samples)
        if max_new_tokens is not None:
            samples = tuple(replace(sample, max_new_tokens=max_new_tokens) for sample in samples)
        identity = (ruler_data_identity(samples, data, lengths) if task == "ruler"
                    else data_identity(samples, data, task))
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
    request = json.loads(json.dumps({
        "eval_id": eval_id, "task": task, "checkpoint": str(checkpoint_path),
        "checkpoint_step": state["cursor"]["step"], "training_run": training_run,
        "model": asdict(config.model),
        "model_window": model_window, "seed": config.seed,
        "data": identity, "protocol": evaluation_protocol,
    }))
    return state, config, tokenizer, request, samples, references



def _compute_sample(sample, references, request, model, tokenizer, device, max_new_tokens):
    task, model_window = request["task"], request["model_window"]
    # 各样本的随机状态独立于分片数量、调度顺序及此前失败位置。
    set_seed(request["seed"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
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
        reference = references[sample.id]
        row = {
            "id": sample.id, "document_id": sample.document_id,
            "question": reference["question"], "references": [reference["answer"]],
            "nll": float(output["qa_loss"]), "target_tokens": output["target_tokens"],
            "nll_seconds": nll_seconds,
        }
    else:
        context_ids = tuple(encode_prompt(sample, tokenizer, model_window, task))
        question_ids = ()
        requested_budget = budget = sample.max_new_tokens
        row = {"id": sample.id, "question": sample.metadata.get("question", ""),
               "references": list(sample.references), "metadata": sample.metadata}
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
    row.update({
        "prediction": prediction, "raw_prediction": tokenizer.decode(generated, skip_special_tokens=False),
        "generated_token_ids": generated, "prompt_tokens": len(context_ids) + len(question_ids),
        "context_tokens": len(context_ids), "question_tokens": len(question_ids),
        "generated_tokens": len(generated), "max_new_tokens": budget,
        "requested_max_new_tokens": requested_budget, "stop_reason": stop_reason,
        "write_events": result["write_events"], "inner_before": result["inner_before"],
        "inner_after": result["inner_after"], "generation_seconds": generation_seconds,
        "seconds": time.perf_counter() - started,
    })
    if task == "test":
        row.update(scores(prediction, reference["answer"]))
    else:
        row["score"] = score_sample(task, sample, prediction)
    device_label = str(device)
    if device.type == "cuda":
        device_label += f" (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')})"
    return {"record": row, "device": device_label, "peak_memory_bytes": (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    )}


def _check_complete_details(output_dir, summary, samples):
    if summary is not None:
        records = read_details(output_dir / "details.jsonl")
        if [row["id"] for row in records] != [sample.id for sample in samples]:
            raise ValueError("completed evaluation details differ from the selected sample order")
    return summary


def load_cached_evaluation(checkpoint_path, data, output_dir, task="test", max_new_tokens=None,
                           max_samples=None, lengths=None, eval_id=None):
    """只校验已有完整结果，不加载 LM 或初始化 CUDA。"""
    output_dir = Path(output_dir).expanduser().resolve()
    if not output_dir.exists():
        return None
    state, config, tokenizer, request, samples, references = _evaluation_inputs(
        checkpoint_path, data, task, max_new_tokens, max_samples, lengths, eval_id,
    )
    del state, config, tokenizer, references
    with evaluation_lock(output_dir):
        check_evaluation(output_dir, request)
        return _check_complete_details(output_dir, completed_evaluation(output_dir, request), samples)


def prepare_sample_work(checkpoint_path, data, output_dir, task="test", max_new_tokens=None,
                        max_samples=None, lengths=None, eval_id=None, num_workers=1):
    """在无 worker 运行时固定本轮分工，只重新分配未完成样本。"""
    state, config, tokenizer, request, samples, references = _evaluation_inputs(
        checkpoint_path, data, task, max_new_tokens, max_samples, lengths, eval_id,
    )
    del state, config, tokenizer, references
    output_dir = Path(output_dir).expanduser().resolve()
    with evaluation_lock(output_dir):
        prepare_evaluation(output_dir, request)
        summary = _check_complete_details(output_dir, completed_evaluation(output_dir, request), samples)
        if summary is not None:
            work = {"total_samples": len(samples), "completed_samples": len(samples),
                    "pending_samples": 0, "workers": []}
        else:
            entries = load_progress(output_dir, request, samples, repair=True)
            work = write_work(output_dir, samples, entries, num_workers)
    return work


def evaluate_shard(checkpoint_path, data, output_dir, worker_rank, task="test", device="cuda",
                   max_new_tokens=None, max_samples=None, lengths=None, eval_id=None):
    """每个进程加载一个模型，逐样本追加自己的已完成记录；不生成局部评分汇总。"""
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("evaluation workers do not use torchrun")
    if type(worker_rank) is not int or worker_rank < 0:
        raise ValueError("worker_rank must be a non-negative integer")
    output_dir = Path(output_dir).expanduser().resolve()
    state, config, tokenizer, request, samples, references = _evaluation_inputs(
        checkpoint_path, data, task, max_new_tokens, max_samples, lengths, eval_id,
    )
    check_evaluation(output_dir, request)
    work = json.loads((output_dir / "work.json").read_text(encoding="utf-8"))
    selected_ids = work["workers"][worker_rank]
    by_id = {sample.id: sample for sample in samples}
    device = initialize_device(device)
    set_seed(config.seed)
    model = OnlineMemoryLM(config.model).to(device)
    model.load_trainable_state_dict(state["model"])
    model.eval()
    del state
    progress = output_dir / "progress"
    progress.mkdir(parents=True, exist_ok=True)
    with (progress / f"worker_{worker_rank}.jsonl").open("a", encoding="utf-8") as stream:
        for index, sample_id in enumerate(selected_ids, 1):
            entry = _compute_sample(by_id[sample_id], references, request, model, tokenizer, device, max_new_tokens)
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            stream.flush()
            row = entry["record"]
            quality = row["f1"] if task == "test" else row["score"]
            metric = "accuracy" if task == "ruler" else "f1"
            print(f"{task} worker={worker_rank} {index}/{len(selected_ids)} {sample_id} "
                  f"{metric}={quality:.4f} tokens={row['generated_tokens']}/{row['max_new_tokens']} "
                  f"stop={row['stop_reason']} seconds={row['seconds']:.1f}", flush=True)
    del model


def finalize_sample_work(checkpoint_path, data, output_dir, task="test", max_new_tokens=None,
                         max_samples=None, lengths=None, eval_id=None):
    state, config, tokenizer, request, samples, references = _evaluation_inputs(
        checkpoint_path, data, task, max_new_tokens, max_samples, lengths, eval_id,
    )
    del state, config, tokenizer, references
    output_dir = Path(output_dir).expanduser().resolve()
    with evaluation_lock(output_dir):
        check_evaluation(output_dir, request)
        summary = _check_complete_details(output_dir, completed_evaluation(output_dir, request), samples)
        if summary is not None:
            return summary
        return commit_results(output_dir, request, samples, load_progress(output_dir, request, samples))


def evaluate(checkpoint_path, data, output_dir, task="test", device="cuda",
             max_new_tokens=None, max_samples=None, lengths=None, eval_id=None):
    options = dict(task=task, max_new_tokens=max_new_tokens, max_samples=max_samples,
                   lengths=lengths, eval_id=eval_id)
    work = prepare_sample_work(checkpoint_path, data, output_dir, **options)
    try:
        if work["pending_samples"]:
            evaluate_shard(checkpoint_path, data, output_dir, worker_rank=0, device=device, **options)
    except BaseException:
        # 成功样本已逐条提交；尽可能保存齐全的长度，不用汇总异常掩盖计算错误。
        try:
            finalize_sample_work(checkpoint_path, data, output_dir, **options)
        except IncompleteEvaluationError:
            pass
        raise
    summary = finalize_sample_work(checkpoint_path, data, output_dir, **options)
    print(json.dumps({"metrics": summary["metrics"], "output_dir": str(output_dir)}, ensure_ascii=False), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--task", choices=("test", "locomo", "ruler"), default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-id")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--lengths", type=int, nargs="+")
    parser.add_argument("--worker-rank", type=int, help="调度器分配的样本 worker 编号")
    args = parser.parse_args(argv)
    options = dict(task=args.task, max_new_tokens=args.max_new_tokens, max_samples=args.max_samples,
                   lengths=args.lengths, eval_id=args.eval_id)
    if args.worker_rank is None:
        evaluate(args.checkpoint, args.data, args.output_dir, **options)
    else:
        evaluate_shard(args.checkpoint, args.data, args.output_dir, worker_rank=args.worker_rank, **options)


if __name__ == "__main__":
    main()
