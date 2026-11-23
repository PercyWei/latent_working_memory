"""为同一段 LoCoMo 对话预填充一次 KV，再从该前缀独立回答各题。"""

from collections import defaultdict
from copy import copy
import time

import torch
from transformers import DynamicCache

from latent_working_memory.benchmarks.protocol import encode_prompt

from .generation import generated_ids, left_pad


def _encode_conversation(questions, tokenizer, model_window):
    """在完整 chat token 序列上找公共前缀，只保存一次正文及每题的剩余 token。"""
    first = encode_prompt(questions[0], tokenizer, model_window, "locomo", use_qa_system=True)
    # 相同问题或单题会话也留一个 token，通过 generate 取得首个答案 token。
    prefix, suffixes = first[:-1], [first[-1:]]
    for question in questions[1:]:
        tokens = encode_prompt(question, tokenizer, model_window, "locomo", use_qa_system=True)
        limit = min(len(prefix), len(tokens) - 1)
        shared = next((i for i in range(limit) if prefix[i] != tokens[i]), limit)
        if shared < len(prefix):
            tail = prefix[shared:]
            suffixes = [tail + suffix for suffix in suffixes]
            del prefix[shared:]
        suffixes.append(tokens[shared:])
    if not prefix:
        raise ValueError(f"LoCoMo conversation {questions[0].id} has no shared prompt tokens")
    return prefix, list(zip(questions, suffixes, strict=True))


def _question_cache(prefix_cache, num_questions):
    """独立的动态层状态共享只读原文 KV。动态层追加时以 cat 生成新张量。"""
    branch = copy(prefix_cache)
    branch.layers = [copy(layer) for layer in prefix_cache.layers]
    for layer in branch.layers:
        layer.keys = layer.keys.expand(num_questions, -1, -1, -1)
        layer.values = layer.values.expand(num_questions, -1, -1, -1)
    return branch


def _question_batches(turns, sliding, batch_size):
    # 滑动窗口按物理列计算距离。把等长后缀放在一起，正文与问题之间就没有 padding。
    groups = defaultdict(list)
    for sample, suffix in turns:
        key = (sample.max_new_tokens, len(suffix) if sliding else None)
        groups[key].append((sample, suffix))
    for group in groups.values():
        for start in range(0, len(group), batch_size):
            yield group[start:start + batch_size]


def evaluate_locomo(samples, tokenizer, model, generation, model_window, device, batch_size, commit):
    conversations = defaultdict(list)
    for sample in samples:
        conversations[sample.metadata["conversation_id"]].append(sample)
    for questions in sorted(conversations.values(), key=lambda group: len(group[0].prompt)):
        prefix, turns = _encode_conversation(questions, tokenizer, model_window)
        prefix_ids = torch.tensor([prefix], dtype=torch.long, device=device)
        prefix_mask = torch.ones_like(prefix_ids)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        cache = model.base_model(
            input_ids=prefix_ids, attention_mask=prefix_mask,
            past_key_values=DynamicCache(config=model.config), use_cache=True, return_dict=True,
        ).past_key_values
        torch.cuda.synchronize(device)
        prefill_per_question = (time.perf_counter() - started) / len(turns)

        for batch in _question_batches(turns, any(cache.is_sliding), batch_size):
            batch_questions, suffixes = zip(*batch)
            num_questions = len(batch)
            suffix_ids, suffix_mask = left_pad(suffixes, generation.pad_token_id, device)
            # generate 按缓存宽度切去已处理的输入，完整 mask 给每个真实 token 原来的位置。
            ids = torch.cat((prefix_ids.expand(num_questions, -1), suffix_ids), dim=1)
            mask = torch.cat((prefix_mask.expand(num_questions, -1), suffix_mask), dim=1)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            tokens = model.generate(
                input_ids=ids, attention_mask=mask, past_key_values=_question_cache(cache, num_questions),
                generation_config=generation, max_new_tokens=batch_questions[0].max_new_tokens,
                use_model_defaults=False,
            )
            torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - started) / num_questions
            answers = tokens[:, ids.shape[1]:].tolist()
            for (sample, suffix), answer in zip(batch, answers, strict=True):
                commit(
                    sample, len(prefix) + len(suffix), generated_ids(answer, generation.eos_token_id),
                    prefill_per_question + elapsed, 1, num_questions,
                )
        del cache
