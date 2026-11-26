"""复用完整前缀状态，以模型原生缓存续写规则独立回答各题。"""

import torch

from benchmarks.causal import CachedCausalReader
from benchmarks.data import locomo_question_prompt
from .engine import PreparedConversation, PreparedQuestion


class CausalLoCoMo(CachedCausalReader):
    def __init__(self, model, tokenizer, fork_cache, input_format,
                 max_positions=None, mixed_precision=False):
        super().__init__(model, tokenizer, input_format, max_positions, mixed_precision)
        self.fork_cache = fork_cache

    @torch.no_grad()
    def prepare_context(self, history, questions):
        formatting = self.input_format
        prefix_text = self.input_builder.parts().prefix + history
        prefix = self.tokenizer.encode(
            prefix_text,
            add_special_tokens=formatting["add_special_tokens"] and not formatting["use_chat_template"],
        )
        # 固定探针确定正文末尾的 BPE 边界；prefix 不随待答题集合或续评变化。
        probe = self.input_builder.encode(
            history + locomo_question_prompt("", 4),
            add_special_tokens=formatting["add_special_tokens"],
        )
        shared = next((i for i, (a, b) in enumerate(zip(prefix, probe)) if a != b),
                      min(len(prefix), len(probe)))
        prefix = prefix[:shared]
        if not prefix:
            raise ValueError("LoCoMo input has no reusable context prefix")
        prepared = []
        for sample in questions:
            tokens = self.input_builder.encode(
                sample.prompt,
                add_special_tokens=formatting["add_special_tokens"],
            )
            if tokens[:shared] != prefix or len(tokens) == shared:
                raise ValueError(f"question does not extend the context prefix: {sample.id}")
            if self.max_positions is not None and len(tokens) + sample.max_new_tokens > self.max_positions:
                raise ValueError(f"input and generation budget exceed model window: {sample.id}")
            suffix = tokens[shared:]
            prepared.append(PreparedQuestion(sample, suffix, (len(suffix), sample.max_new_tokens)))
        ids = torch.tensor([prefix], device=self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.mixed_precision):
            cache = self.forward(ids, None, torch.ones_like(ids)).past_key_values
        return PreparedConversation((cache, shared), prepared)

    @torch.no_grad()
    def answer_batch(self, state, questions, complete):
        prefix, prefix_length = state
        size = len(questions)
        cache = self.fork_cache(prefix, size)
        ids = torch.tensor([question.inputs for question in questions], device=self.device)
        mask = torch.ones((size, prefix_length), dtype=torch.long, device=self.device)

        def completed(index, tokens, reason):
            complete(questions[index], {
                "prediction": self.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                "generated_token_ids": tokens, "stop_reason": reason,
            })

        self.read_cached(cache, mask, ids, torch.ones_like(ids),
                         [q.sample.max_new_tokens for q in questions], completed)
