"""LaCT、E²-TTT 和 In-Place TTT 的 LongMemEval 独立历史批量推理。"""

import torch

from benchmarks.causal import CachedCausalReader
from benchmarks.longmemeval.data import format_history, format_question
from latent_working_memory.full_context.generation import left_pad


class TTTLongMemEval(CachedCausalReader):
    def __init__(self, model, tokenizer, settings, max_positions=None):
        super().__init__(model, tokenizer, settings["input_format"], max_positions)
        self.max_new_tokens = settings["max_new_tokens"]

    @torch.no_grad()
    def answer_batch(self, inputs):
        prefixes, suffixes = [], []
        system = self.input_builder.parts().prefix
        add_special = self.input_format["add_special_tokens"]
        for item in inputs:
            history = format_history(item)
            prefix = self.tokenizer.encode(system + history, add_special_tokens=add_special)
            tokens = self.input_builder.encode(history + format_question(item), add_special_tokens=add_special)
            if self.max_positions is not None and len(tokens) + self.max_new_tokens > self.max_positions:
                raise ValueError(f"input ({len(tokens)}) + generation budget ({self.max_new_tokens}) "
                                 f"exceeds the model window ({self.max_positions})")
            # 预填截止于正文末尾的稳定 BPE 边界，保证两段合起来等于完整提示的分词。
            boundary = next((i for i, (a, b) in enumerate(zip(prefix, tokens)) if a != b),
                            min(len(prefix), len(tokens)))
            prefixes.append(tokens[:boundary])
            suffixes.append(tokens[boundary:])

        padding_id = self.tokenizer.eos_token_id
        ids, mask = left_pad(prefixes, padding_id, self.device)
        cache = self.forward(ids, None, mask).past_key_values
        question_ids, question_mask = left_pad(suffixes, padding_id, self.device)
        answers = [""] * len(inputs)

        def completed(index, tokens, reason):
            answers[index] = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()

        self.read_cached(cache, mask, question_ids, question_mask,
                         [self.max_new_tokens] * len(inputs), completed)
        return answers
