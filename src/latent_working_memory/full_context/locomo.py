"""为同一段对话共享前缀 KV，再从该前缀独立回答各题。"""

import time

import torch

from benchmarks.documents import PreparedConversation, PreparedQuestion
from benchmarks.locomo.reader import prefill_prefixes, read_prefixes
from benchmarks.protocol import encode_document_prompt
from latent_working_memory.prompts import QAInputBuilder, qa_input_format

from .model import load_model


def _encode_document(history, questions, builder, model_window):
    """在完整聊天分词的边界切分正文与问题，保留可能跨过边界的 BPE token。"""
    prefix = builder.tokenizer.encode(
        builder.parts().prefix + history, add_special_tokens=not builder.input_format["use_chat_template"],
    )
    prompts = [encode_document_prompt(q, builder, model_window) for q in questions]
    length = len(prefix)
    for tokens in prompts:
        limit = min(length, len(tokens) - 1)
        length = next((i for i in range(limit) if tokens[i] != prefix[i]), limit)
    if not length:
        raise ValueError(f"{questions[0].id}: missing shared document prefix")
    return prefix[:length], [PreparedQuestion(q, ids[length:]) for q, ids in zip(questions, prompts, strict=True)]


class FullContextLoCoMo:
    def __init__(self, tokenizer, model, generation, model_window, device, mixed_precision=False,
                 prefill_chunk_size=None):
        self.tokenizer = tokenizer
        self.prompt = QAInputBuilder(tokenizer, qa_input_format(tokenizer))
        self.model = model
        self.generation = generation
        self.model_window = model_window
        self.device = device
        self.mixed_precision = mixed_precision
        self.prefill_chunk_size = prefill_chunk_size

    @torch.no_grad()
    def prepare_contexts(self, documents):
        encoded = [_encode_document(history, questions, self.prompt, self.model_window)
                   for history, questions in documents]
        ids = [torch.tensor(prefix, device=self.device) for prefix, _ in encoded]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.mixed_precision):
            caches = prefill_prefixes(self.model, ids, self.prefill_chunk_size)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        prefill_seconds = (time.perf_counter() - started) / sum(len(turns) for _, turns in encoded)
        return [PreparedConversation((cache, len(prefix), prefill_seconds), turns)
                for cache, (prefix, turns) in zip(caches, encoded, strict=True)]

    @torch.no_grad()
    def answer_batch(self, states, questions, complete):
        caches, lengths, times = zip(*states, strict=True)
        size = len(questions)
        started = time.perf_counter()

        def completed(index, result):
            question = questions[index]
            answer = result["generated_token_ids"]
            result.update(prediction=self.tokenizer.decode(answer, skip_special_tokens=True),
                          generated_tokens=len(answer), prompt_tokens=lengths[index] + len(question.inputs),
                          decode_batch_size=size,
                          generation_seconds=times[index] + (time.perf_counter() - started) / size)
            complete(question, result)

        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.mixed_precision):
            suffixes = [self.model.get_input_embeddings()(torch.tensor(q.inputs, device=self.device))
                        for q in questions]
            read_prefixes(self.model, caches, lengths, suffixes, [q.sample.max_new_tokens for q in questions],
                          self.generation.eos_token_id, completed, native_logits=True)


def load_adapter(settings, prefill_chunk_size=None):
    device = torch.device("cuda", 0)
    tokenizer, model, generation = load_model(settings["model"], settings["seed"], device)
    if qa_input_format(tokenizer) != settings["input_format"]:
        raise ValueError("loaded chat input format differs from the evaluation plan")
    return FullContextLoCoMo(tokenizer, model, generation, settings["model_window"], device,
                            mixed_precision=settings["model"]["mixed_precision"],
                            prefill_chunk_size=prefill_chunk_size)
