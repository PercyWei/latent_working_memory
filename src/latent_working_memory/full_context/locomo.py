"""为同一段 LoCoMo 对话预填充一次 KV，再从该前缀独立回答各题。"""

from collections import defaultdict, deque
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


def _conversation_batches(samples, tokenizer, model_window, batch_size):
    conversations = defaultdict(list)
    for sample in samples:
        conversations[sample.metadata["conversation_id"]].append(sample)
    ordered = sorted(conversations.values(), key=lambda questions: len(questions[0].prompt))

    batch = []
    prefix_width = answer_width = 0
    for questions in ordered:
        prefix, turns = _encode_conversation(questions, tokenizer, model_window)
        width = max(len(suffix) + sample.max_new_tokens for sample, suffix in turns)
        padded_length = max(prefix_width, len(prefix)) + max(answer_width, width)
        if batch and (len(batch) == batch_size or padded_length > model_window):
            yield batch
            batch = []
            prefix_width = answer_width = 0
        batch.append((prefix, turns))
        prefix_width = max(prefix_width, len(prefix))
        answer_width = max(answer_width, width)
    if batch:
        yield batch


def _question_cache(prefix_cache, rows):
    """独立的动态层状态共享只读原文 KV。动态层追加时以 cat 生成新张量。"""
    if all(row == rows[0] for row in rows):
        selection = slice(rows[0], rows[0] + 1)
    elif rows == list(range(rows[0], rows[0] + len(rows))):
        selection = slice(rows[0], rows[-1] + 1)
    else:
        selection = rows
    branch = copy(prefix_cache)
    branch.layers = [copy(layer) for layer in prefix_cache.layers]
    for layer in branch.layers:
        # 同一段正文的多个题尾共用张量视图，第一次追加时各行才获得私有 KV。
        layer.keys = layer.keys[selection].expand(len(rows), -1, -1, -1)
        layer.values = layer.values[selection].expand(len(rows), -1, -1, -1)
    return branch


def _question_batches(conversations, sliding, batch_size):
    # 滑动窗口按物理列计算距离。把等长后缀放在一起，正文与问题之间就没有 padding。
    groups = defaultdict(lambda: defaultdict(deque))
    for row, (_prefix, questions) in enumerate(conversations):
        for sample, suffix in questions:
            key = (sample.max_new_tokens, len(suffix) if sliding else None)
            groups[key][row].append((sample, suffix))
    for by_conversation in groups.values():
        pending = deque(by_conversation.items())
        batch = []
        while pending:
            row, questions = pending.popleft()
            batch.append((row, *questions.popleft()))
            if questions:
                pending.append((row, questions))
            if len(batch) == batch_size or not pending:
                yield batch
                batch = []


def evaluate_locomo(samples, tokenizer, model, generation, model_window, device, batch_size, commit):
    for conversations in _conversation_batches(samples, tokenizer, model_window, batch_size):
        prefixes = [prefix for prefix, _questions in conversations]
        prefix_ids, prefix_mask = left_pad(prefixes, generation.pad_token_id, device)
        positions = (prefix_mask.cumsum(-1) - 1).masked_fill(prefix_mask == 0, 1)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        cache = model.base_model(
            input_ids=prefix_ids, attention_mask=prefix_mask, position_ids=positions,
            past_key_values=DynamicCache(config=model.config), use_cache=True, return_dict=True,
        ).past_key_values
        torch.cuda.synchronize(device)
        prefill_per_conversation = (time.perf_counter() - started) / len(conversations)
        prefill_per_question = [prefill_per_conversation / len(turns) for _, turns in conversations]

        for batch in _question_batches(conversations, any(cache.is_sliding), batch_size):
            rows = [row for row, _sample, _suffix in batch]
            suffix_ids, suffix_mask = left_pad(
                [suffix for _row, _sample, suffix in batch], generation.pad_token_id, device,
            )
            # generate 按缓存宽度切去已处理的输入，完整 mask 给每个真实 token 原来的位置。
            ids = torch.cat((prefix_ids[rows], suffix_ids), dim=1)
            mask = torch.cat((prefix_mask[rows], suffix_mask), dim=1)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            tokens = model.generate(
                input_ids=ids, attention_mask=mask, past_key_values=_question_cache(cache, rows),
                generation_config=generation, max_new_tokens=batch[0][1].max_new_tokens,
                use_model_defaults=False,
            )
            torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - started) / len(batch)
            answers = tokens[:, ids.shape[1]:].tolist()
            for (row, sample, suffix), answer in zip(batch, answers, strict=True):
                commit(
                    sample, len(prefixes[row]) + len(suffix),
                    generated_ids(answer, generation.eos_token_id),
                    prefill_per_question[row] + elapsed, len(conversations), len(batch),
                )
        del cache
