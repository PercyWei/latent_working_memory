"""复用完整前缀状态，以模型原生缓存续写规则独立回答各题。"""

import torch

from benchmarks.data import locomo_question_prompt
from latent_working_memory.prompts import QAInputBuilder
from .engine import PreparedConversation, PreparedQuestion


class CausalLoCoMo:
    def __init__(self, model, tokenizer, fork_cache, input_format,
                 max_positions=None, mixed_precision=False):
        self.model = model
        self.tokenizer = tokenizer
        self.fork_cache = fork_cache
        self.input_format = dict(input_format)
        self.input_builder = QAInputBuilder(tokenizer, self.input_format)
        self.max_positions = max_positions
        self.mixed_precision = mixed_precision
        self.device = next(model.parameters()).device
        eos = model.generation_config.eos_token_id
        if eos is None:
            eos = tokenizer.eos_token_id
        self.eos_ids = {eos} if isinstance(eos, int) else set(eos)

    def forward(self, ids, cache, mask):
        return self.model(input_ids=ids, past_key_values=cache, attention_mask=mask,
                          use_cache=True, return_dict=True, logits_to_keep=1)

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
        generated = [[] for _ in questions]
        active = [True] * size
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.mixed_precision):
            # 部分原生递归状态只支持单 token 续写；保留其更新时机和位置规则。
            for column in range(ids.shape[1]):
                mask = torch.cat((mask, mask.new_ones(size, 1)), dim=1)
                output = self.forward(ids[:, column:column + 1], cache, mask)
                cache = output.past_key_values
            for _ in range(max(question.sample.max_new_tokens for question in questions)):
                predicted = output.logits[:, -1].argmax(dim=-1)
                for index, token in enumerate(predicted.tolist()):
                    if not active[index]:
                        continue
                    generated[index].append(token)
                    question = questions[index]
                    reason = ("eos" if token in self.eos_ids else "max_new_tokens"
                              if len(generated[index]) == question.sample.max_new_tokens else None)
                    if reason is not None:
                        active[index] = False
                        complete(question, {
                            "prediction": self.tokenizer.decode(generated[index], skip_special_tokens=True).strip(),
                            "generated_token_ids": generated[index], "stop_reason": reason,
                        })
                if not any(active):
                    break
                # 已结束行不再写入 KV 或递归状态，其余行继续原生自回归更新。
                mask = torch.cat((mask, torch.tensor(active, device=self.device)[:, None]), dim=1)
                output = self.forward(predicted[:, None], cache, mask)
                cache = output.past_key_values
