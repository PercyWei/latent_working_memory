"""SDCC 对完整正文预测一次压缩率，各题复用回答端前缀。"""

import torch

from benchmarks.data import locomo_question_prompt
from benchmarks.locomo.engine import PreparedConversation, PreparedQuestion
from benchmarks.locomo.reader import fork_dynamic_cache, greedy_read
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
    def encode_history(self, history):
        encoded = self.tokenizer(history, add_special_tokens=True, return_special_tokens_mask=True)
        tokens = encoded["input_ids"]
        context_tokens = encoded["special_tokens_mask"].count(0)
        # 与作者评估相同，在正文末尾补一个 EOS 供比例选择器读取。
        if not tokens or tokens[-1] != self.writer_eos_id:
            tokens.append(self.writer_eos_id)
        writer_window = self.bridge.encoder.config.max_position_embeddings
        if len(tokens) > writer_window:
            raise ValueError(f"SDCC encoder needs {len(tokens)} positions (window {writer_window})")
        context_ids = torch.tensor([tokens], device=self.decoder.device)
        frame_ids = torch.tensor([self.prefix_ids], device=self.decoder.device)
        embeds, mask, _, counts, _, prediction, _ = self.bridge.compute_inputs_embeds(
            context_ids, frame_ids, context_attention_mask=torch.ones_like(context_ids),
            attention_mask=torch.ones_like(frame_ids), compress_ratio_scale=self.scale,
        )
        retention = self.bridge._discretize_ratio_log(
            prediction, self.bridge.comp_ratio_or_len, self.bridge.discretize_ratio_mode,
            self.bridge.discretize_compare_in_log, values_in_log_space=True,
        ).item()
        if embeds.size(1) >= self.decoder.config.max_position_embeddings:
            raise ValueError(f"SDCC memory prefix fills decoder window: {embeds.size(1)} positions")
        cache = self.decoder.model(inputs_embeds=embeds, attention_mask=mask, use_cache=True).past_key_values
        return (cache, embeds.size(1)), {
            "context_tokens": context_tokens, "encoder_tokens": len(tokens),
            "memory_tokens": counts.item(), "compression_factor": round(1 / retention),
        }

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

    def prepare_context(self, history, questions):
        state, stats = self.encode_history(history)
        prepared = []
        for question in questions:
            task = locomo_question_prompt(question.metadata["question"], question.metadata["category"])
            suffix = self.question_ids(state, task, question.max_new_tokens)
            prepared.append(PreparedQuestion(question, suffix, (len(suffix), question.max_new_tokens)))
        return PreparedConversation(state, prepared, stats)

    @torch.inference_mode()
    def _decode(self, state, question_ids, budgets, complete):
        size = len(question_ids)
        ids = torch.tensor(question_ids, device=self.decoder.device)
        cache = fork_dynamic_cache(state[0], size)

        def completed(index, result):
            tokens = result["generated_token_ids"]
            result.update(prediction=self.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                          generated_tokens=len(tokens), decode_batch_size=size)
            complete(index, result)

        greedy_read(self.decoder, cache, self.decoder.get_input_embeddings()(ids),
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
    stats = conversations.values()
    return {
        "mean_memory_tokens": sum(s["memory_tokens"] for s in stats) / len(conversations),
        "mean_retention_ratio": sum(s["memory_tokens"] / s["context_tokens"] for s in stats)
        / len(conversations),
    }
