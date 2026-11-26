"""v3.1.1 的会话压缩和冻结 Reader KV 复用。"""

import time
import torch

from benchmarks.data import locomo_question_prompt
from benchmarks.documents import PreparedConversation, PreparedQuestion
from benchmarks.locomo.reader import prefill_prefixes, read_prefixes
from ..data import QAEncoder
from ..rollout import execute_batch


class MemoryAdapter:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.encoder = QAEncoder(model.tokenizer)

    @torch.no_grad()
    def prepare_contexts(self, documents):
        model, config = self.model, self.config
        contexts = [self.encoder.context(history) for history, _ in documents]
        started = time.perf_counter()
        with model.forward_context():
            paths = execute_batch(
                model, contexts, config.model.structure.pending_stride,
                pending_size=config.model.structure.pending_size,
                recent_size=config.recent_size(), collect_boundary=True,
            )
            prefixes = [model._inputs(model.decoder, path.blocks, path.retained_ids,
                                      self.encoder.reader_prefix_ids) for path in paths]
            caches = prefill_prefixes(model.decoder, prefixes)
        seconds = (time.perf_counter() - started) / len(documents)
        return [
            self._prepare_questions(questions, context, path, cache, len(prefix), seconds)
            for (_, questions), context, path, cache, prefix in zip(
                documents, contexts, paths, caches, prefixes, strict=True
            )
        ]

    def _prepare_questions(self, questions, context, path, cache, reader_length, seconds):
        model, config = self.model, self.config
        slots = len(path.blocks) * model.memory_slots
        prefix_length = len(context) - len(path.recent_ids)
        retention = (slots + len(path.pending_ids)) / prefix_length if prefix_length else 0.0
        statistics = {
            "context_tokens": len(context), "memory_tokens": slots,
            "raw_history_tokens": len(path.retained_ids),
            "actions": list(path.actions), "block_lengths": list(path.block_lengths),
            "blocks": len(path.blocks), "memory_slots": slots,
            "slots_per_block": model.memory_slots, "recent_tokens": len(path.recent_ids),
            "pending_tokens": len(path.pending_ids), "retained_tokens": slots + len(path.retained_ids),
            "retention_ratio": (slots + len(path.retained_ids)) / len(context),
            "prefix_retention_ratio": retention,
            "memory_cost": retention ** config.training.objective.memory_cost_power,
            "policy_log_probs": path.log_probs.tolist(), "policy_entropies": path.entropy.tolist(),
            "policy_decision_mask": list(path.decision_mask),
        }
        prepared = []
        for sample in questions:
            text = ("\n\n" + locomo_question_prompt(sample.metadata["question"], sample.metadata["category"])
                    + self.encoder.suffix)
            ids = model.tokenizer.encode(text, add_special_tokens=False)
            if reader_length + len(ids) + sample.max_new_tokens > model.max_positions:
                raise ValueError(f"reader exceeds position window: {sample.id}")
            prepared.append(PreparedQuestion(sample, ids))
        first_per_conversation = {}
        for question in questions:
            first_per_conversation.setdefault(question.metadata["conversation_id"], question.id)
        state = {"cache": cache, "prefix_length": reader_length, "boundary": path.boundary_trace,
                 "trace_questions": set(first_per_conversation.values()),
                 "first": questions[0].id, "preparation_seconds": seconds}
        return PreparedConversation(state, prepared, statistics)

    @torch.no_grad()
    def answer_batch(self, states, questions, complete):
        model = self.model
        size = len(questions)
        num_contexts = len({id(state) for state in states})
        embed = model.decoder.get_input_embeddings()
        device = model.memory_embeddings.device
        suffixes = [embed(torch.tensor(q.inputs, device=device)) for q in questions]
        started = time.perf_counter()
        def completed(index, result):
            tokens = result["generated_token_ids"]
            question, state = questions[index], states[index]
            preparation = state["preparation_seconds"] if question.sample.id == state["first"] else 0.0
            result.update(
                prediction=model.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                raw_prediction=model.tokenizer.decode(tokens, skip_special_tokens=False),
                generated_tokens=len(tokens), max_new_tokens=question.sample.max_new_tokens,
                requested_max_new_tokens=question.sample.max_new_tokens,
                eval_batch_size=num_contexts, decode_batch_size=size, nll_seconds=0.0,
                generation_seconds=preparation + (time.perf_counter() - started) / size,
                seconds=preparation + (time.perf_counter() - started) / size,
            )
            if question.sample.id in state["trace_questions"]:
                result["boundary"] = state["boundary"]
            complete(question, result)
        with model.forward_context():
            read_prefixes(model.decoder, [s["cache"] for s in states], [s["prefix_length"] for s in states],
                          suffixes, [q.sample.max_new_tokens for q in questions],
                          model.tokenizer.eos_token_id, completed)
