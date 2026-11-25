"""AutoCompressors 的正文压缩、只读前缀与批量回答。"""

import torch

from benchmarks.data import locomo_question_prompt
from benchmarks.locomo.engine import PreparedConversation, PreparedQuestion
from benchmarks.locomo.reader import greedy_read
from latent_working_memory.prompts import QAInputBuilder, qa_input_format


INPUT_FORMAT = {
    **qa_input_format(use_chat_template=False),
    "system_add_special_tokens": True, "question_add_special_tokens": False,
}


class AutoCompressorLoCoMo:
    def __init__(self, model, tokenizer, segment_length, input_format):
        self.model = model
        self.tokenizer = tokenizer
        self.segment_length = segment_length
        self.input_format = dict(input_format)
        self.prompt_parts = QAInputBuilder(tokenizer, self.input_format).parts()
        self.window = model.config.max_position_embeddings

    @torch.inference_mode()
    def encode_history(self, text):
        input_ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        summary = None
        slots = self.model.config.summary_length
        for index, segment in enumerate(input_ids.split(self.segment_length, dim=1)):
            write_length = index * slots + segment.size(1) + slots
            if write_length > self.window:
                raise ValueError(f"AutoCompressor write needs {write_length} positions (window {self.window})")
            summary = self.model(
                segment.to(self.model.device), softprompt=summary, output_softprompt=True, use_cache=False,
            ).softprompt

        # 写入仅接触正文，QA 系统提示位于摘要前。该前缀在所有问题间只读。
        system_ids = self.tokenizer(
            self.prompt_parts.prefix,
            add_special_tokens=self.input_format["system_add_special_tokens"], return_tensors="pt",
        ).input_ids.to(self.model.device)
        prefix = torch.cat((self.model.get_input_embeddings()(system_ids), summary), dim=1)
        if prefix.size(1) > self.window:
            raise ValueError(f"AutoCompressor reader prefix needs {prefix.size(1)} positions (window {self.window})")
        cache = self.model.model(inputs_embeds=prefix, use_cache=True).past_key_values
        return cache, {"context_tokens": input_ids.size(1), "memory_tokens": summary.size(1)}

    def prepare_context(self, history, questions):
        state, stats = self.encode_history(history)
        prefix_length = state[0][1]
        prepared = []
        for sample in questions:
            prompt = locomo_question_prompt(sample.metadata["question"], sample.metadata["category"])
            ids = self.tokenizer.encode(
                prompt + self.prompt_parts.suffix,
                add_special_tokens=self.input_format["question_add_special_tokens"],
            )
            read_length = prefix_length + len(ids) + sample.max_new_tokens
            if read_length > self.window:
                raise ValueError(f"AutoCompressor reader needs {read_length} positions "
                                 f"(window {self.window}): {sample.id}")
            prepared.append(PreparedQuestion(sample, ids, (len(ids), sample.max_new_tokens)))
        return PreparedConversation(state, prepared, stats)

    @torch.inference_mode()
    def answer_batch(self, state, questions, complete):
        size = len(questions)
        ids = torch.tensor([q.inputs for q in questions], device=self.model.device)
        budgets = [q.sample.max_new_tokens for q in questions]
        capacity = state[0][1] + ids.size(1) + max(budgets)
        # 官方缓存为 (KV 存储, 有效长度)，追加时原地写入。
        # 为每行复制前缀并预留问题和答案空间，不能仅 expand 共享存储。
        cache = []
        for kv, length in state:
            branch = kv.new_empty((size, capacity, *kv.shape[2:]))
            branch[:, :length].copy_(kv[:, :length])
            cache.append((branch, length))

        def completed(index, result):
            tokens = result["generated_token_ids"]
            result.update(prediction=self.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                          generated_tokens=len(tokens), decode_batch_size=size)
            complete(questions[index], result)

        greedy_read(self.model, tuple(cache), self.model.get_input_embeddings()(ids),
                    budgets, self.tokenizer.eos_token_id, completed)


def summarize_metrics(records, conversations):
    return {
        "mean_memory_tokens": sum(s["memory_tokens"] for s in conversations.values()) / len(conversations),
        "mean_retention_ratio": sum(s["memory_tokens"] / s["context_tokens"]
                                    for s in conversations.values()) / len(conversations),
    }
