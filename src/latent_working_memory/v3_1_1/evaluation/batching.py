"""长度分组、批量压缩与读取；一篇文档的全部 QA 共用同一份记忆。"""

from itertools import groupby
import os
import time

import torch

from latent_working_memory.benchmarks.protocol import encode_prompt
from latent_working_memory.benchmarks.scoring import score_sample
from latent_working_memory.data_preparation.personamem.audit import scores
from ..rollout import execute_batch


def sample_batches(samples, task, batch_size):
    """仅排序引用；RULER 同一批次保持相同长度，题面在计算时编码。"""
    if task == "ruler":
        ordered = sorted(samples, key=lambda sample: (sample.metadata["length"], len(sample.prompt)))
        groups = (group for _, group in groupby(ordered, key=lambda sample: sample.metadata["length"]))
    else:
        ordered = sorted(samples, key=(
            (lambda sample: len(sample.input_ids)) if task == "test"
            else (lambda sample: len(sample.prompt))
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


@torch.no_grad()
def compute_batch(samples, request, model, config, max_new_tokens, commit):
    """真实批量前向；完整文档／题目完成即提交，共享计算时间仅分摊一次。"""
    task = request["task"]
    count = len(samples)
    device = model.memory_embeddings.device
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    contexts = [sample.input_ids if task == "test" else tuple(encode_prompt(
        sample, model.tokenizer, model.max_positions, task,
    )) for sample in samples]
    preparation = time.perf_counter() - started
    started = time.perf_counter()
    paths = execute_batch(model, contexts, config.method.rollout.unit_tokens)
    torch.cuda.synchronize(device)
    compression = time.perf_counter() - started
    memories = [path[0] for path in paths]
    generation_seconds = [compression / count] * count
    nll_seconds = [0.0] * count
    seconds = [(preparation + compression) / count] * count
    rows = []
    started = time.perf_counter()
    for index, (sample, path) in enumerate(zip(samples, paths, strict=True)):
        blocks, actions, log_probs, entropy, lengths = path
        common = {
            "id": sample.id, "context_tokens": len(contexts[index]),
            "actions": list(actions), "block_lengths": list(lengths),
            "blocks": len(blocks), "memory_slots": len(blocks) * model.memory_slots,
            "policy_log_probs": log_probs.tolist(), "policy_entropies": entropy.tolist(),
            "eval_batch_size": count,
        }
        if task == "test":
            common.update(document_id=sample.document_id, qas=[])
        else:
            common.update(question=sample.metadata.get("question", ""),
                          references=list(sample.references), metadata=sample.metadata)
        rows.append(common)
    metadata_seconds = (time.perf_counter() - started) / count
    seconds = [value + metadata_seconds for value in seconds]
    device_label = f"{device} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')})"

    def submit(index):
        row = rows[index]
        row.update(seconds=seconds[index], generation_seconds=generation_seconds[index],
                   nll_seconds=nll_seconds[index])
        if task == "test":
            row["generated_tokens"] = sum(answer["generated_tokens"] for answer in row["qas"])
        commit({"record": row, "device": device_label,
                "peak_memory_bytes": torch.cuda.max_memory_allocated(device)})

    rounds = max(len(sample.qas) for sample in samples) if task == "test" else 1
    for qa_index in range(rounds):
        indices = ([index for index, sample in enumerate(samples) if qa_index < len(sample.qas)]
                   if task == "test" else list(range(count)))
        prompts = ([samples[index].qas[qa_index].prompt_ids for index in indices]
                   if task == "test" else [()] * count)
        requested = ([max_new_tokens or 128] * len(indices) if task == "test"
                     else [samples[index].max_new_tokens for index in indices])
        budgets = [min(budget, model.max_positions - rows[index]["memory_slots"] - len(prompt))
                   if task == "test" else budget
                   for index, prompt, budget in zip(indices, prompts, requested, strict=True)]
        if any(budget < 1 for budget in budgets):
            raise ValueError("no model-window capacity remains for generation")
        nll_values = None
        if task == "test":
            started = time.perf_counter()
            values = model.answer_nll_batch(
                [memories[index] for index in indices], prompts,
                [samples[index].qas[qa_index].target_ids for index in indices],
            )
            torch.cuda.synchronize(device)
            nll_values = values.tolist()
            share = (time.perf_counter() - started) / len(indices)
            for index in indices:
                seconds[index] += share
                nll_seconds[index] += share

        active = set(range(len(indices)))
        last_finished = time.perf_counter()

        def completed(position, result):
            nonlocal last_finished
            torch.cuda.synchronize(device)
            finished = time.perf_counter()
            share = (finished - last_finished) / len(active)
            for pending in active:
                index = indices[pending]
                seconds[index] += share
                generation_seconds[index] += share
            active.remove(position)
            scored_at = time.perf_counter()
            index = indices[position]
            sample, row = samples[index], rows[index]
            tokens = result["token_ids"]
            prediction = model.tokenizer.decode(tokens, skip_special_tokens=True).strip()
            reason = result["stop_reason"]
            if reason == "max_new_tokens" and budgets[position] < requested[position]:
                reason = "model_window"
            answer = {
                "prediction": prediction,
                "raw_prediction": model.tokenizer.decode(tokens, skip_special_tokens=False),
                "generated_token_ids": list(tokens), "generated_tokens": len(tokens),
                "max_new_tokens": budgets[position],
                "requested_max_new_tokens": requested[position], "stop_reason": reason,
            }
            if task == "test":
                qa = sample.qas[qa_index]
                answer.update(qa_index=qa_index, question=qa.question, reference=qa.answer,
                              nll=nll_values[position], target_tokens=len(qa.target_ids),
                              **scores(prediction, qa.answer))
                if qa.segment_id is not None:
                    answer.update(segment_id=qa.segment_id,
                                  segment_char_span=list(qa.segment_char_span))
                row["qas"].append(answer)
            else:
                row.update(answer, score=score_sample(task, sample, prediction))
            seconds[index] += time.perf_counter() - scored_at
            if task != "test" or qa_index + 1 == len(sample.qas):
                submit(index)
            # 已完成样本的评分与写盘时间不重复计入仍活跃的 GPU 任务。
            last_finished = time.perf_counter()

        model.generate_batch(
            [memories[index] for index in indices], prompts, budgets, on_complete=completed,
        )
