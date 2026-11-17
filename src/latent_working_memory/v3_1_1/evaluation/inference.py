"""共用正文压缩、逐题回答、可选 NLL 与资源计时。"""

from dataclasses import dataclass
import os
import time

import torch

from latent_working_memory.v4_1.data import EpisodeEncoder
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


def encode_context_question(context, question, tokenizer, max_new_tokens):
    """正文沿用训练编码；任务指令和问题作为未压缩的 Reader 提示。"""
    encoder = EpisodeEncoder(tokenizer)
    prompt_ids = tuple(tokenizer.encode(
        f"\n\n{question}{encoder.suffix}", add_special_tokens=False,
    ))
    return EvaluationInput(
        encoder.context(context), (EvaluationQuestion(prompt_ids, max_new_tokens),),
    )


@torch.no_grad()
def compute_batch(samples, model, config, commit, encode, record, commit_per_question=False):
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
    paths = execute_batch(model, contexts, config.model.structure.unit_tokens)
    torch.cuda.synchronize(device)
    compression = time.perf_counter() - started
    memories = [path[0] for path in paths]
    generation_seconds = [compression / count] * count
    nll_seconds = [0.0] * count
    seconds = [(preparation + compression) / count] * count
    rows, answers = [], [[] for _ in samples]
    started = time.perf_counter()
    for sample, context, path in zip(samples, contexts, paths, strict=True):
        blocks, actions, log_probs, entropy, lengths = path
        rows.append({
            "id": sample.id, "context_tokens": len(context),
            "actions": list(actions), "block_lengths": list(lengths),
            "blocks": len(blocks), "memory_slots": len(blocks) * model.memory_slots,
            "policy_log_probs": log_probs.tolist(), "policy_entropies": entropy.tolist(),
            "eval_batch_size": count,
        })
    metadata_seconds = (time.perf_counter() - started) / count
    seconds = [value + metadata_seconds for value in seconds]
    device_label = f"{device} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')})"

    for question_index in range(max(len(value.questions) for value in inputs)):
        indices = [index for index, value in enumerate(inputs) if question_index < len(value.questions)]
        questions = [inputs[index].questions[question_index] for index in indices]
        prompts = [question.prompt_ids for question in questions]
        requested = [question.max_new_tokens for question in questions]
        capacities = [model.max_positions - rows[index]["memory_slots"] - len(prompt)
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
                [inputs[index].questions[question_index].prompt_ids for index in nll_indices],
                [inputs[index].questions[question_index].target_ids for index in nll_indices],
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
        )
