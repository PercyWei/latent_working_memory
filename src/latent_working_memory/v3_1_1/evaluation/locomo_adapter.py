"""v3.1.1 的会话压缩和冻结 Reader KV 复用。"""

import time
import torch

from benchmarks.data import locomo_question_prompt
from benchmarks.locomo.engine import PreparedConversation, PreparedQuestion
from benchmarks.locomo.reader import fork_dynamic_cache, greedy_read
from ..data import QAEncoder
from ..rollout import execute_batch


class MemoryAdapter:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.encoder = QAEncoder(model.tokenizer)

    @torch.no_grad()
    def prepare_context(self, history, questions):
        model, config = self.model, self.config
        context = self.encoder.context(history)
        started = time.perf_counter()
        with model.forward_context():
            path = execute_batch(
                model, [context], config.model.structure.pending_stride,
                pending_size=config.model.structure.pending_size,
                recent_size=config.recent_size(), collect_boundary=True,
            )[0]
            inputs = model._inputs(model.decoder, path.blocks, path.retained_ids,
                                   self.encoder.reader_prefix_ids)
            cache = model.decoder.model(inputs_embeds=inputs.unsqueeze(0),
                                        use_cache=True, return_dict=True).past_key_values
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
            if len(inputs) + len(ids) + sample.max_new_tokens > model.max_positions:
                raise ValueError(f"reader exceeds position window: {sample.id}")
            prepared.append(PreparedQuestion(sample, ids, (len(ids), sample.max_new_tokens)))
        state = {"cache": cache, "boundary": path.boundary_trace, "first": questions[0].id,
                 "preparation_seconds": time.perf_counter() - started}
        return PreparedConversation(state, prepared, statistics)

    @torch.no_grad()
    def answer_batch(self, state, questions, complete):
        model = self.model
        size = len(questions)
        ids = torch.tensor([q.inputs for q in questions], device=model.memory_embeddings.device)
        cache = fork_dynamic_cache(state["cache"], size)
        started = time.perf_counter()
        def completed(index, result):
            tokens = result["generated_token_ids"]
            question = questions[index]
            preparation = state["preparation_seconds"] if question.sample.id == state["first"] else 0.0
            result.update(
                prediction=model.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                raw_prediction=model.tokenizer.decode(tokens, skip_special_tokens=False),
                generated_tokens=len(tokens), max_new_tokens=question.sample.max_new_tokens,
                requested_max_new_tokens=question.sample.max_new_tokens,
                eval_batch_size=1, decode_batch_size=size, nll_seconds=0.0,
                generation_seconds=preparation + (time.perf_counter() - started) / size,
                seconds=preparation + (time.perf_counter() - started) / size,
            )
            if question.sample.id == state["first"]:
                result["boundary"] = state["boundary"]
            complete(question, result)
        with model.forward_context():
            greedy_read(model.decoder, cache, model.decoder.get_input_embeddings()(ids),
                        [q.sample.max_new_tokens for q in questions], model.tokenizer.eos_token_id, completed)
