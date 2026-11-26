"""AutoCompressors 的正文压缩、只读前缀与批量回答。"""

import torch

from benchmarks.documents import PreparedConversation, PreparedQuestion
from latent_working_memory.prompts import QA_SYSTEM_PROMPT, QAInputBuilder, qa_input_format


INPUT_FORMAT = {
    **qa_input_format(use_chat_template=False),
    "system_add_special_tokens": True, "question_add_special_tokens": False,
}


class AutoCompressorLoCoMo:
    def __init__(self, model, tokenizer, segment_length, input_format, fit_segments=False,
                 system_prompt=QA_SYSTEM_PROMPT):
        self.model = model
        self.tokenizer = tokenizer
        self.segment_length = segment_length
        self.fit_segments = fit_segments
        self.input_format = dict(input_format)
        self.prompt_parts = QAInputBuilder(tokenizer, self.input_format, system_prompt).parts()
        self.window = model.config.max_position_embeddings

    def segment_lengths(self, token_count):
        lengths = []
        remaining = token_count
        while remaining:
            slots = (len(lengths) + 1) * self.model.config.summary_length
            length = min(self.segment_length, remaining)
            if self.fit_segments:
                length = min(length, self.window - slots)
            if length <= 0 or length + slots > self.window:
                raise ValueError(f"AutoCompressor cannot write the full {token_count}-token history: "
                                 f"{slots} summary positions in a {self.window}-position window")
            lengths.append(length)
            remaining -= length
        return lengths

    @torch.inference_mode()
    def prepare_contexts(self, documents):
        # 第 t 轮每个活跃正文均有 t 组摘要；正文尾段的 padding 由官方 varlen writer 去除。
        texts = [self.tokenizer.encode(history, add_special_tokens=False) for history, _ in documents]
        segments = [self.segment_lengths(len(ids)) for ids in texts]
        summaries = [None] * len(documents)
        offsets = [0] * len(documents)
        device = self.model.device
        for step in range(max(map(len, segments))):
            active = [i for i, lengths in enumerate(segments) if step < len(lengths)]
            width = max(segments[i][step] for i in active)
            ids = torch.zeros((len(active), width), dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            for row, i in enumerate(active):
                count = segments[i][step]
                ids[row, :count] = torch.tensor(texts[i][offsets[i]:offsets[i] + count], device=device)
                mask[row, :count] = 1
                offsets[i] += count
            prior = torch.cat([summaries[i] for i in active]) if step else None
            summary = self.model(ids, attention_mask=mask, softprompt=prior,
                                 output_softprompt=True, use_cache=False).softprompt
            for row, i in enumerate(active):
                summaries[i] = summary[row:row + 1]

        system_ids = self.tokenizer(self.prompt_parts.prefix,
                                    add_special_tokens=self.input_format["system_add_special_tokens"],
                                    return_tensors="pt").input_ids.to(device)
        system = self.model.get_input_embeddings()(system_ids)[0]
        prefixes = [torch.cat((system, summary[0])) for summary in summaries]
        width = max(map(len, prefixes))
        if width > self.window:
            raise ValueError(f"AutoCompressor reader prefix needs {width} positions (window {self.window})")
        # 官方缓存模式使用稠密因果 attention。右侧补齐不会改变任何真实前缀 token。
        padded = torch.nn.utils.rnn.pad_sequence(prefixes, batch_first=True)
        caches = self.model.model(inputs_embeds=padded, use_cache=True).past_key_values
        prepared = []
        for i, (history, questions) in enumerate(documents):
            length = len(prefixes[i])
            cache = tuple((kv[i:i + 1, :length].clone() if len(documents) > 1 else kv, length)
                          for kv, _ in caches)
            turns = []
            for sample in questions:
                ids = self.tokenizer.encode(sample.prompt.removeprefix(history) + self.prompt_parts.suffix,
                                            add_special_tokens=self.input_format["question_add_special_tokens"])
                if length + len(ids) + sample.max_new_tokens > self.window:
                    raise ValueError(f"{sample.id}: AutoCompressor answer exceeds reader window {self.window}")
                turns.append(PreparedQuestion(sample, ids))
            stats = {"context_tokens": len(texts[i]), "memory_tokens": summaries[i].size(1)}
            prepared.append(PreparedConversation(cache, turns, stats))
        return prepared

    @torch.inference_mode()
    def answer_batch(self, states, questions, complete):
        def completed(index, result):
            tokens = result["generated_token_ids"]
            result.update(prediction=self.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                          generated_tokens=len(tokens), decode_batch_size=len(questions))
            complete(questions[index], result)

        self.model.read_questions(states, [q.inputs for q in questions], [q.sample.max_new_tokens for q in questions],
                                  self.tokenizer.eos_token_id, completed)


def summarize_metrics(records, conversations):
    return {
        "mean_memory_tokens": sum(s["memory_tokens"] for s in conversations.values()) / len(conversations),
        "mean_retention_ratio": sum(s["memory_tokens"] / s["context_tokens"]
                                    for s in conversations.values()) / len(conversations),
    }
