"""按 token 续写推理，支持各行的独立缓存、问题长度和停止位置。"""

import torch

from latent_working_memory.prompts import QAInputBuilder


class CachedCausalReader:
    def __init__(self, model, tokenizer, input_format, max_positions=None, mixed_precision=False):
        self.model = model
        self.tokenizer = tokenizer
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
    def read_cached(self, cache, mask, ids, suffix_mask, budgets, complete):
        """从完整正文状态读入左侧补齐的问题，完成一行即提交生成 token。"""
        generated = [[] for _ in budgets]
        active = [True] * len(budgets)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.mixed_precision):
            # 部分原生递归状态只支持单 token 续写。padding 不推进各行真实状态。
            for column in range(ids.shape[1]):
                mask = torch.cat((mask, suffix_mask[:, column:column + 1]), dim=1)
                output = self.forward(ids[:, column:column + 1], cache, mask)
                cache = output.past_key_values
            for _ in range(max(budgets)):
                predicted = output.logits[:, -1].argmax(dim=-1)
                for index, token in enumerate(predicted.tolist()):
                    if not active[index]:
                        continue
                    generated[index].append(token)
                    reason = ("eos" if token in self.eos_ids else "max_new_tokens"
                              if len(generated[index]) == budgets[index] else None)
                    if reason is not None:
                        active[index] = False
                        complete(index, generated[index], reason)
                if not any(active):
                    break
                mask = torch.cat((mask, torch.tensor(active, device=self.device)[:, None]), dim=1)
                output = self.forward(predicted[:, None], cache, mask)
                cache = output.past_key_values
