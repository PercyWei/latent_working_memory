"""复用完整正文的原生 TTT 状态，按题分支并批量回答。"""

import torch

from benchmarks.causal import CachedCausalReader
from benchmarks.documents import PreparedConversation, PreparedQuestion
from latent_working_memory.full_context.generation import left_pad


class CausalLoCoMo(CachedCausalReader):
    def __init__(self, model, tokenizer, fork_cache, input_format,
                 max_positions=None, mixed_precision=False):
        super().__init__(model, tokenizer, input_format, max_positions, mixed_precision)
        self.fork_cache = fork_cache

    @torch.no_grad()
    def prepare_contexts(self, documents):
        add_special = self.input_format["add_special_tokens"]
        system = self.input_builder.parts().prefix
        prefixes, groups = [], []
        for history, questions in documents:
            prefix = self.tokenizer.encode(
                system + history, add_special_tokens=add_special and not self.input_format["use_chat_template"],
            )
            prepared = []
            for question in questions:
                tokens = self.input_builder.encode(question.prompt, add_special_tokens=add_special)
                if not prepared:
                    # 用该 benchmark 的真实问题分隔符确定正文末尾的 BPE 边界。
                    boundary = next((i for i, (a, b) in enumerate(zip(prefix, tokens)) if a != b),
                                    min(len(prefix), len(tokens)))
                    prefix = prefix[:boundary]
                if not prefix or len(tokens) <= len(prefix) or tokens[:len(prefix)] != prefix:
                    raise ValueError(f"question does not extend the shared document prefix: {question.id}")
                if self.max_positions is not None and len(tokens) + question.max_new_tokens > self.max_positions:
                    raise ValueError(f"input and generation budget exceed model window: {question.id}")
                prepared.append(PreparedQuestion(question, tokens[len(prefix):]))
            prefixes.append(prefix)
            groups.append(prepared)

        ids, mask = left_pad(prefixes, self.tokenizer.eos_token_id, self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.mixed_precision):
            batch_cache = self.forward(ids, None, mask).past_key_values
        # 单独保存各正文，后续批次仍可复用其中一篇并释放其余正文的状态。
        caches = ([batch_cache] if len(prefixes) == 1 else
                  [self.fork_cache([(batch_cache, row)]) for row in range(len(prefixes))])
        return [PreparedConversation((cache, len(prefix)), questions)
                for cache, prefix, questions in zip(caches, prefixes, groups, strict=True)]

    @torch.no_grad()
    def answer_batch(self, states, questions, complete):
        cache = self.fork_cache([(prefix, 0) for prefix, _ in states])
        lengths = torch.tensor([length for _, length in states], device=self.device)
        width = max(length for _, length in states)
        mask = (torch.arange(width, device=self.device) >= (width - lengths[:, None])).long()
        ids, suffix_mask = left_pad([q.inputs for q in questions], self.tokenizer.eos_token_id, self.device)

        def completed(index, tokens, reason):
            complete(questions[index], {
                "prediction": self.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                "generated_token_ids": tokens, "stop_reason": reason,
            })

        self.read_cached(cache, mask, ids, suffix_mask, [q.sample.max_new_tokens for q in questions], completed)
