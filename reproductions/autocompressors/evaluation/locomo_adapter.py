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
    def __init__(self, model, tokenizer, segment_length, input_format, fit_segments=False):
        self.model = model
        self.tokenizer = tokenizer
        self.segment_length = segment_length
        self.fit_segments = fit_segments
        self.input_format = dict(input_format)
        self.prompt_parts = QAInputBuilder(tokenizer, self.input_format).parts()
        self.window = model.config.max_position_embeddings

    @torch.inference_mode()
    def encode_history(self, text):
        input_ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        summary = None
        slots = self.model.config.summary_length
        lengths = []
        remaining = input_ids.size(1)
        while remaining:
            summary_positions = (len(lengths) + 1) * slots
            length = min(self.segment_length, remaining)
            if self.fit_segments:
                length = min(length, self.window - summary_positions)
            if length <= 0 or length + summary_positions > self.window:
                raise ValueError(f"AutoCompressor cannot write the full {input_ids.size(1)}-token history: "
                                 f"{summary_positions} summary positions in a {self.window}-position window")
            lengths.append(length)
            remaining -= length
        for segment in input_ids.split(lengths, dim=1):
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

    def question_ids(self, state, prompt, max_new_tokens):
        ids = self.tokenizer.encode(
            prompt + self.prompt_parts.suffix,
            add_special_tokens=self.input_format["question_add_special_tokens"],
        )
        read_length = state[0][1] + len(ids) + max_new_tokens
        if read_length > self.window:
            raise ValueError(f"AutoCompressor reader needs {read_length} positions (window {self.window})")
        return ids

    def prepare_context(self, history, questions):
        state, stats = self.encode_history(history)
        prepared = []
        for sample in questions:
            prompt = locomo_question_prompt(sample.metadata["question"], sample.metadata["category"])
            ids = self.question_ids(state, prompt, sample.max_new_tokens)
            prepared.append(PreparedQuestion(sample, ids, (len(ids), sample.max_new_tokens)))
        return PreparedConversation(state, prepared, stats)

    @torch.inference_mode()
    def _decode(self, state, question_ids, budgets, complete):
        size = len(question_ids)
        ids = torch.tensor(question_ids, device=self.model.device)
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
            complete(index, result)

        greedy_read(self.model, tuple(cache), self.model.get_input_embeddings()(ids),
                    budgets, self.tokenizer.eos_token_id, completed)

    def answer_batch(self, state, questions, complete):
        self._decode(state, [q.inputs for q in questions], [q.sample.max_new_tokens for q in questions],
                     lambda index, result: complete(questions[index], result))

    def answer(self, state, prompt, max_new_tokens):
        ids = self.question_ids(state, prompt, max_new_tokens)
        answers = []
        self._decode(state, [ids], [max_new_tokens], lambda _, result: answers.append(result["prediction"]))
        return answers[0]


def summarize_metrics(records, conversations):
    return {
        "mean_memory_tokens": sum(s["memory_tokens"] for s in conversations.values()) / len(conversations),
        "mean_retention_ratio": sum(s["memory_tokens"] / s["context_tokens"]
                                    for s in conversations.values()) / len(conversations),
    }
