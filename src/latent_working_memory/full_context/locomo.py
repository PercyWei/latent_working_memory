"""为同一段 LoCoMo 对话预填充一次 KV，再从该前缀独立回答各题。"""

import time

import torch
from transformers import DynamicCache

from benchmarks.protocol import encode_prompt
from benchmarks.locomo.engine import (
    PreparedConversation, PreparedQuestion, evaluate_conversations,
)
from benchmarks.locomo.reader import fork_dynamic_cache

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



class FullContextLoCoMo:
    def __init__(self, tokenizer, model, generation, model_window, device, mixed_precision=False):
        self.tokenizer = tokenizer
        self.model = model
        self.generation = generation
        self.model_window = model_window
        self.device = device
        self.mixed_precision = mixed_precision

    @torch.no_grad()
    def prepare_context(self, history, questions):
        prefix, turns = _encode_conversation(questions, self.tokenizer, self.model_window)
        ids = torch.tensor([prefix], dtype=torch.long, device=self.device)
        mask = torch.ones_like(ids)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.mixed_precision):
            cache = self.model.base_model(
                input_ids=ids, attention_mask=mask,
                past_key_values=DynamicCache(config=self.model.config),
                use_cache=True, return_dict=True,
            ).past_key_values
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        prefill_seconds = (time.perf_counter() - started) / len(questions)
        sliding = any(cache.is_sliding)
        prepared = [
            PreparedQuestion(sample, suffix, (sample.max_new_tokens, len(suffix) if sliding else None))
            for sample, suffix in turns
        ]
        return PreparedConversation((cache, ids, mask, prefill_seconds), prepared)

    @torch.no_grad()
    def answer_batch(self, state, questions, complete):
        cache, prefix_ids, prefix_mask, prefill_seconds = state
        suffixes = [question.inputs for question in questions]
        size = len(questions)
        suffix_ids, suffix_mask = left_pad(suffixes, self.generation.pad_token_id, self.device)
        ids = torch.cat((prefix_ids.expand(size, -1), suffix_ids), dim=1)
        mask = torch.cat((prefix_mask.expand(size, -1), suffix_mask), dim=1)
        started = time.perf_counter()
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.mixed_precision):
            tokens = self.model.generate(
                input_ids=ids, attention_mask=mask, past_key_values=fork_dynamic_cache(cache, size),
                generation_config=self.generation, max_new_tokens=questions[0].sample.max_new_tokens,
                use_model_defaults=False,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = (time.perf_counter() - started) / size
        for question, row in zip(questions, tokens[:, ids.shape[1]:].tolist(), strict=True):
            answer = generated_ids(row, self.generation.eos_token_id)
            complete(question, {
                "prediction": self.tokenizer.decode(answer, skip_special_tokens=True),
                "generated_token_ids": answer, "generated_tokens": len(answer),
                "prompt_tokens": prefix_ids.shape[1] + len(question.inputs),
                "stop_reason": "eos" if answer[-1] == self.generation.eos_token_id else "max_new_tokens",
                "decode_batch_size": size, "generation_seconds": prefill_seconds + elapsed,
            })


def evaluate_locomo(samples, tokenizer, model, generation, model_window, device, batch_size, commit):
    adapter = FullContextLoCoMo(tokenizer, model, generation, model_window, device,
                               mixed_precision=torch.is_autocast_enabled(device.type))
    def completed(sample, answer, statistics):
        commit(sample, answer["prompt_tokens"], answer["generated_token_ids"],
               answer["generation_seconds"], 1, answer["decode_batch_size"])
    evaluate_conversations(samples, adapter, batch_size, completed)
