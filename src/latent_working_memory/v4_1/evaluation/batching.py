"""长度分组、批量计算与逐题提交；执行批量不参与评估身份。"""

from itertools import groupby
import os
import time

import torch
from transformers import set_seed

from latent_working_memory.benchmarks.protocol import encode_prompt
from latent_working_memory.benchmarks.scoring import score_sample
from latent_working_memory.data_preparation.personamem.audit import scores


def sample_batches(samples, task, batch_size):
    """仅排序样本引用；当前批次才分词，不缓存整个 benchmark 的 token IDs。"""
    if task == "ruler":
        ordered = sorted(samples, key=lambda sample: (sample.metadata["length"], len(sample.prompt)))
        groups = (group for _, group in groupby(ordered, key=lambda sample: sample.metadata["length"]))
    else:
        ordered = sorted(samples, key=(
            (lambda sample: len(sample.context_ids) + len(sample.question_ids))
            if task == "test" else (lambda sample: len(sample.prompt))
        ))
        groups = (ordered,)
    for group in groups:
        batch = []
        for sample in group:
            batch.append(sample)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def compute_batch(samples, references, request, model, tokenizer, device, max_new_tokens, commit):
    """完成一题立即评分提交；后续样本失败时保留已提交记录。"""
    task, model_window = request["task"], request["model_window"]
    count = len(samples)
    set_seed(request["seed"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    contexts, questions, requested_budgets, budgets, rows = [], [], [], [], []
    for sample in samples:
        if task == "test":
            context, question = sample.context_ids, sample.question_ids
            requested = max_new_tokens or 128
            budget = min(requested, model_window - len(context) - len(question))
            reference = references[sample.id]
            row = {"id": sample.id, "document_id": sample.document_id,
                   "question": reference["question"], "references": [reference["answer"]]}
        else:
            context, question = tuple(encode_prompt(sample, tokenizer, model_window, task)), ()
            requested = budget = sample.max_new_tokens
            row = {"id": sample.id, "question": sample.metadata.get("question", ""),
                   "references": list(sample.references), "metadata": sample.metadata}
        if not context or budget < 1:
            raise ValueError(
                f"{sample.id}: generation requires nonempty context and remaining model-window capacity"
            )
        contexts.append(context)
        questions.append(question)
        requested_budgets.append(requested)
        budgets.append(budget)
        rows.append(row)
    preparation_seconds = (time.perf_counter() - started) / count
    nll_seconds = 0.0
    if task == "test":
        nll_started = time.perf_counter()
        with torch.no_grad():
            outputs = model.forward_batch(samples, differentiable=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        nll_seconds = (time.perf_counter() - nll_started) / count
        for row, output in zip(rows, outputs):
            row.update(nll=float(output["qa_loss"]), target_tokens=output["target_tokens"],
                       nll_seconds=nll_seconds)
        del outputs

    generation_seconds = [0.0] * count
    active = set(range(count))
    device_label = str(device)
    if device.type == "cuda":
        device_label += f" (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')})"
    last_finished = time.perf_counter()

    def completed(index, result):
        nonlocal last_finished
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        finished = time.perf_counter()
        # 一段批量计算时间只计一次，在当时仍活跃的样本之间均摊。
        share = (finished - last_finished) / len(active)
        for pending in active:
            generation_seconds[pending] += share
        active.remove(index)
        scored_at = time.perf_counter()
        generated = result["token_ids"]
        prediction = tokenizer.decode(generated, skip_special_tokens=True)
        stop_reason = result["stop_reason"]
        if stop_reason == "max_new_tokens" and budgets[index] < requested_budgets[index]:
            stop_reason = "model_window"
        row = rows[index]
        row.update({
            "prediction": prediction,
            "raw_prediction": tokenizer.decode(generated, skip_special_tokens=False),
            "generated_token_ids": generated,
            "prompt_tokens": len(contexts[index]) + len(questions[index]),
            "context_tokens": len(contexts[index]), "question_tokens": len(questions[index]),
            "generated_tokens": len(generated), "max_new_tokens": budgets[index],
            "requested_max_new_tokens": requested_budgets[index], "stop_reason": stop_reason,
            "write_events": result["write_events"], "inner_before": result["inner_before"],
            "inner_after": result["inner_after"], "generation_seconds": generation_seconds[index],
            "eval_batch_size": count,
        })
        if task == "test":
            row.update(scores(prediction, references[samples[index].id]["answer"]))
        else:
            row["score"] = score_sample(task, samples[index], prediction)
        row["seconds"] = (preparation_seconds + nll_seconds + generation_seconds[index]
                          + (time.perf_counter() - scored_at))
        entry = {"record": row, "device": device_label, "peak_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        )}
        commit(entry)
        # 逐题评分与文件写入不重复计入其余样本的 GPU 生成时间。
        last_finished = time.perf_counter()

    with torch.no_grad():
        model.generate_batch(contexts, questions, budgets, tokenizer.eos_token_id, on_complete=completed)
