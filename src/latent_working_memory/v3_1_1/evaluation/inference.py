"""共用正文压缩、逐题回答、可选 NLL 与资源计时。"""

from dataclasses import dataclass
import os
import time

import torch

from ..rollout import execute_batch


@dataclass(frozen=True)
class EvaluationQuestion:
    prompt_ids: tuple[int, ...]
    max_new_tokens: int
    target_ids: tuple[int, ...] | None = None
    truncate_to_window: bool = False


@dataclass(frozen=True)
class EvaluationInput:
    context_ids: tuple[int, ...]
    questions: tuple[EvaluationQuestion, ...]


def encode_context_question(context, question, encoder, max_new_tokens):
    """分别编码原始题面的两个连续片段，问题保留自带的分隔符。"""
    prompt_ids = tuple(encoder.tokenizer.encode(
        question + encoder.suffix, add_special_tokens=False,
    ))
    return EvaluationInput(
        encoder.context(context), (EvaluationQuestion(prompt_ids, max_new_tokens),),
    )


@torch.no_grad()
def compute_batch(samples, model, config, commit, encode, record, reader_prefix_ids,
                  commit_per_question=False):
    """正文只压缩一次；LoCoMo 逐题提交，其他任务在整篇回答完成后提交。"""
    count = len(samples)
    device = model.memory_embeddings.device
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    inputs = [encode(sample) for sample in samples]
    contexts = [value.context_ids for value in inputs]
    preparation = time.perf_counter() - started
    started = time.perf_counter()
    paths = execute_batch(
        model, contexts, config.model.structure.pending_stride,
        pending_size=config.model.structure.pending_size,
        recent_size=config.recent_size(),
    )
    torch.cuda.synchronize(device)
    compression = time.perf_counter() - started
    memories = [path.blocks for path in paths]
    generation_seconds = [compression / count] * count
    nll_seconds = [0.0] * count
    seconds = [(preparation + compression) / count] * count
    rows, answers = [], [[] for _ in samples]
    started = time.perf_counter()
    for sample, context, path in zip(samples, contexts, paths, strict=True):
        slots = len(path.blocks) * model.memory_slots
        retained = slots + len(path.retained_ids)
        prefix_length = len(context) - len(path.recent_ids)
        prefix_retention = (slots + len(path.pending_ids)) / prefix_length if prefix_length else 0.0
        rows.append({
            "id": sample.id, "context_tokens": len(context),
            "actions": list(path.actions), "block_lengths": list(path.block_lengths),
            "blocks": len(path.blocks), "memory_slots": slots,
            "slots_per_block": model.memory_slots, "recent_tokens": len(path.recent_ids),
            "pending_tokens": len(path.pending_ids),
            "retained_tokens": retained, "retention_ratio": retained / len(context),
            "prefix_retention_ratio": prefix_retention,
            "memory_cost": prefix_retention ** config.training.objective.memory_cost_power,
            "policy_log_probs": path.log_probs.tolist(), "policy_entropies": path.entropy.tolist(),
            "policy_decision_mask": list(path.decision_mask),
            "eval_batch_size": count,
        })
    metadata_seconds = (time.perf_counter() - started) / count
    seconds = [value + metadata_seconds for value in seconds]
    device_label = f"{device} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')})"
    window_after_prefix = model.max_positions - len(reader_prefix_ids)

    for question_index in range(max(len(value.questions) for value in inputs)):
        indices = [index for index, value in enumerate(inputs) if question_index < len(value.questions)]
        questions = [inputs[index].questions[question_index] for index in indices]
        prompts = [paths[index].retained_ids + question.prompt_ids
                   for index, question in zip(indices, questions, strict=True)]
        requested = [question.max_new_tokens for question in questions]
        capacities = [window_after_prefix - rows[index]["memory_slots"] - len(prompt)
                      for index, prompt in zip(indices, prompts, strict=True)]
        budgets = [min(budget, capacity) if question.truncate_to_window else budget
                   for question, budget, capacity in zip(questions, requested, capacities, strict=True)]
        if any(budget < 1 or budget > capacity
               for budget, capacity in zip(budgets, capacities, strict=True)):
            raise ValueError("memory, reader prompt and generation budget exceed the model window")

        nll_values = {}
        nll_indices = [index for index in indices
                       if inputs[index].questions[question_index].target_ids is not None]
        if nll_indices:
            started = time.perf_counter()
            values = model.answer_nll_batch(
                [memories[index] for index in nll_indices],
                [paths[index].retained_ids + inputs[index].questions[question_index].prompt_ids
                 for index in nll_indices],
                [inputs[index].questions[question_index].target_ids for index in nll_indices],
                reader_prefix_ids=reader_prefix_ids,
            )
            torch.cuda.synchronize(device)
            nll_values = dict(zip(nll_indices, values.tolist(), strict=True))
            share = (time.perf_counter() - started) / len(nll_indices)
            for index in nll_indices:
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
            tokens = result["token_ids"]
            reason = result["stop_reason"]
            if reason == "max_new_tokens" and budgets[position] < requested[position]:
                reason = "model_window"
            answer = {
                "prediction": model.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                "raw_prediction": model.tokenizer.decode(tokens, skip_special_tokens=False),
                "generated_token_ids": list(tokens), "generated_tokens": len(tokens),
                "max_new_tokens": budgets[position],
                "requested_max_new_tokens": requested[position], "stop_reason": reason,
            }
            if index in nll_values:
                answer.update(nll=nll_values[index], target_tokens=len(questions[position].target_ids))
            answers[index].append(answer)
            complete = commit_per_question or question_index + 1 == len(inputs[index].questions)
            if complete:
                row = {**rows[index], **record(samples[index], answers[index])}
                if all("nll" in value for value in answers[index]):
                    nll = sum(value["nll"] for value in answers[index]) / len(answers[index])
                    row["objective"] = nll + config.training.objective.memory_cost_weight * row["memory_cost"]
                row["generated_tokens"] = (answer["generated_tokens"] if commit_per_question else
                                           sum(value["generated_tokens"] for value in answers[index]))
            seconds[index] += time.perf_counter() - scored_at
            if complete:
                row.update(seconds=seconds[index], generation_seconds=generation_seconds[index],
                           nll_seconds=nll_seconds[index])
                commit({"record": row, "device": device_label,
                        "peak_memory_bytes": torch.cuda.max_memory_allocated(device)})
                seconds[index] = generation_seconds[index] = nll_seconds[index] = 0.0
            # 评分和写盘时间不重复计入仍活跃的 GPU 任务。
            last_finished = time.perf_counter()

        model.generate_batch(
            [memories[index] for index in indices], prompts, budgets, on_complete=completed,
            reader_prefix_ids=reader_prefix_ids,
        )
