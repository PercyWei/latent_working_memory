"""SDCC 对完整正文预测一次压缩率，各题复用回答端前缀。"""

import torch
from torch.nn.utils.rnn import pad_sequence

from benchmarks.documents import PreparedConversation, PreparedQuestion
from benchmarks.locomo.reader import prefill_prefixes, read_prefixes
from latent_working_memory.prompts import QAInputBuilder, qa_input_format


def sdcc_input_format(tokenizer, placeholder_id):
    """聊天包装取自基座 tokenizer，记忆占位符取自 SDCC checkpoint。"""
    settings = qa_input_format(tokenizer)
    placeholder = tokenizer.convert_ids_to_tokens(placeholder_id)
    if not placeholder or placeholder == tokenizer.unk_token or tokenizer.encode(
        placeholder, add_special_tokens=False
    ) != [placeholder_id]:
        raise ValueError(f"tokenizer must encode SDCC placeholder {placeholder_id} as one token")
    if tokenizer.eos_token_id is None:
        raise ValueError("SDCC writer requires a tokenizer EOS token")
    return {**settings, "writer_eos_id": tokenizer.eos_token_id, "context_marker": "Context: " + placeholder}


class SDCCLoCoMo:
    def __init__(self, bridge, tokenizer, scale, input_format):
        self.bridge = bridge
        self.tokenizer = tokenizer
        self.scale = scale
        self.decoder = bridge.decoder.get_base_model()
        self.prompt = QAInputBuilder(tokenizer, input_format)
        # 保留官方 user 消息中的上下文标记。用实际 placeholder 确定 BPE 边界。
        self.context_marker = input_format["context_marker"]
        self.writer_eos_id = input_format["writer_eos_id"]
        frame = self.prompt.encode(self.context_marker)
        if frame.count(bridge.placeholder_token_id) != 1:
            raise ValueError("SDCC chat template must preserve one official placeholder")
        self.prefix_ids = frame[:frame.index(bridge.placeholder_token_id) + 1]

    @torch.inference_mode()
    def encode_histories(self, histories):
        encoded = self.tokenizer(histories, add_special_tokens=True, return_special_tokens_mask=True)
        context_lengths = [mask.count(0) for mask in encoded["special_tokens_mask"]]
        docs = []
        for tokens in encoded["input_ids"]:
            # 每篇正文末尾均有 EOS，供该篇的倍率选择器读取。
            if not tokens or tokens[-1] != self.writer_eos_id:
                tokens.append(self.writer_eos_id)
            if len(tokens) > self.bridge.encoder.config.max_position_embeddings:
                raise ValueError(f"SDCC encoder needs {len(tokens)} positions "
                                 f"(window {self.bridge.encoder.config.max_position_embeddings})")
            docs.append(torch.tensor(tokens, device=self.decoder.device))
        # 官方变倍率池化从最右端对齐有效正文；每行的末位均是自己的 EOS。
        ids = pad_sequence(docs, batch_first=True, padding_value=self.writer_eos_id, padding_side="left")
        lengths = torch.tensor([len(doc) for doc in docs], device=ids.device)
        mask = torch.arange(ids.size(1), device=ids.device) >= ids.size(1) - lengths[:, None]
        frame = torch.tensor(self.prefix_ids, device=ids.device).expand(len(docs), -1)
        embeds, _, _, counts, prefix_lengths, prediction, _ = self.bridge.compute_inputs_embeds(
            ids, frame, context_attention_mask=mask.long(),
            attention_mask=torch.ones_like(frame), compress_ratio_scale=self.scale,
        )
        retentions = self.bridge._discretize_ratio_log(
            prediction, self.bridge.comp_ratio_or_len, self.bridge.discretize_ratio_mode,
            self.bridge.discretize_compare_in_log, values_in_log_space=True,
        ).flatten().tolist()
        if max(prefix_lengths) >= self.decoder.config.max_position_embeddings:
            raise ValueError(f"SDCC memory prefix fills decoder window: {max(prefix_lengths)} positions")
        # frame 本身等长且无 padding，官方展开 placeholder 时将填充放在右侧。
        caches = prefill_prefixes(self.decoder, [row[:n] for row, n in zip(embeds, prefix_lengths, strict=True)])
        return [((cache, length), {
            "context_tokens": context, "encoder_tokens": len(doc),
            "memory_tokens": count, "compression_factor": round(1 / retention),
        }) for cache, length, context, doc, count, retention in zip(
            caches, prefix_lengths, context_lengths, docs, counts.tolist(), retentions, strict=True
        )]

    def question_ids(self, state, prompt, max_new_tokens):
        ids = self.prompt.encode(self.context_marker + prompt)
        if ids[:len(self.prefix_ids)] != self.prefix_ids:
            raise ValueError("question changes SDCC memory prefix")
        suffix = ids[len(self.prefix_ids):]
        length = state[1] + len(suffix) + max_new_tokens
        window = self.decoder.config.max_position_embeddings
        if length > window:
            raise ValueError(f"SDCC reader needs {length} positions (window {window})")
        return suffix

    def prepare_contexts(self, documents):
        encoded = self.encode_histories([history for history, _ in documents])
        contexts = []
        for (history, questions), (state, stats) in zip(documents, encoded, strict=True):
            prepared = []
            for question in questions:
                task = question.prompt.removeprefix(history)
                ids = self.question_ids(state, task, question.max_new_tokens)
                prepared.append(PreparedQuestion(question, ids))
            contexts.append(PreparedConversation(state, prepared, stats))
        return contexts

    @torch.inference_mode()
    def answer_batch(self, states, questions, complete):
        embed = self.decoder.get_input_embeddings()
        embeddings = [embed(torch.tensor(q.inputs, device=self.decoder.device)) for q in questions]

        def completed(index, result):
            tokens = result["generated_token_ids"]
            result.update(prediction=self.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                          generated_tokens=len(tokens), decode_batch_size=len(questions))
            complete(questions[index], result)

        caches, lengths = zip(*states)
        read_prefixes(self.decoder, caches, lengths, embeddings,
                      [q.sample.max_new_tokens for q in questions], self.tokenizer.eos_token_id, completed)


def summarize_metrics(records, conversations):
    stats = conversations.values()
    return {
        "mean_memory_tokens": sum(s["memory_tokens"] for s in stats) / len(conversations),
        "mean_retention_ratio": sum(s["memory_tokens"] / s["context_tokens"] for s in stats)
        / len(conversations),
    }
