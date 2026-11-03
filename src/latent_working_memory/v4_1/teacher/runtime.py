"""Student 建图前重建 outer 参照；AM 与 distillation 共用原文及离线响应。"""

from contextlib import contextmanager

import torch
from transformers.cache_utils import DynamicCache

from ..data import compression_boundaries
from .cache import content_identity
from .replay import TeacherReplay


class RuntimeTeacher:
    def __init__(self, model, config, cache):
        self.model = model
        self.config = config
        self.cache = cache
        self.prompts = tuple(tuple(ids) for ids in cache.identity["prompt_token_ids"])
        self.replay = TeacherReplay(
            model, config.model.prefill_chunk_size, config.teacher.batch_size
        )

    @contextmanager
    def episode(self, episode, generator=None):
        trajectory = TeacherTrajectory()
        prefix = DynamicCache(config=self.model.config)
        context_identity = content_identity(episode.context_ids)
        boundaries = compression_boundaries(
            len(episode.context_ids), self.config.model.pending_size, self.config.model.recent_size,
        )
        try:
            for boundary in boundaries:
                trajectory.references[boundary] = self._reference(
                    episode, prefix, context_identity, *boundary, generator,
                )
            # Teacher 临时 batch 与 student 元训练图分开执行。
            del prefix
            yield trajectory
        finally:
            trajectory.references.clear()

    @torch.no_grad()
    def _reference(self, episode, prefix, context_identity, write_index,
                   processed_end, compressed_end, generator=None):
        record, cached = self.cache.read(episode.document_id, write_index)
        if (record["processed_end"], record["compressed_end"]) != (processed_end, compressed_end):
            raise ValueError("teacher cache boundary does not match the online trajectory")
        if record["context_tokens"] != context_identity:
            raise ValueError("teacher cache tokenized context differs from the current episode")
        indices = torch.randperm(
            self.config.teacher.pool_size, generator=generator,
            device="cpu" if generator is None else generator.device,
        )[:self.config.teacher.am_sample_size].cpu()
        positions = cached["positions"].index_select(0, indices)
        tokens, offsets = cached["response_token_ids"].tolist(), cached["response_offsets"].tolist()
        sequences = [
            (*prompt, *tokens[offsets[index]:offsets[index + 1]])
            for index, prompt in enumerate(self.prompts)
        ]
        replay = self.replay
        replay.advance(episode.context_ids, prefix, processed_end)
        base = self.model.config
        shape = (base.num_hidden_layers, len(indices), base.num_attention_heads, base.head_dim)
        result = {
            "queries": torch.empty(shape, device=replay.device, dtype=torch.float32),
            "outputs": torch.empty(shape, device=replay.device, dtype=torch.float32),
            "log_mass": torch.empty(shape[:-1], device=replay.device, dtype=torch.float32),
            "scales_o": cached["scales_o"].to(replay.device),
            "scales_z": cached["scales_z"].to(replay.device),
            "position_indices": positions,
        }

        def collect(layer_index, pool_indices, queries, outputs, log_mass):
            for name, value in (("queries", queries), ("outputs", outputs), ("log_mass", log_mass)):
                result[name][layer_index].index_copy_(0, pool_indices, value)

        replay.replay(prefix, sequences, positions, compressed_end, collect)
        if self.config.training.outer_write_obj == "distill":
            prompt_index = int(torch.randint(
                len(self.prompts), (1,), generator=generator,
                device="cpu" if generator is None else generator.device,
            ).item())
            prompt = self.prompts[prompt_index]
            response = tokens[offsets[prompt_index]:offsets[prompt_index + 1]]
            branches = [((*prompt, *response[:-1]), len(prompt) - 1)]
            continuation = episode.context_ids[
                processed_end:processed_end + self.config.teacher.response_max_tokens
            ]
            if len(continuation) >= 2:
                branches.append((tuple(continuation[:-1]), 0))
            result["distill_branches"] = replay.distill_branches(prefix, branches)
        return result


class TeacherTrajectory:
    """固定参照按轮消费；本对象不保留 teacher 的完整原文 KV。"""

    def __init__(self):
        self.references = {}

    def reference(self, write_index, processed_end, compressed_end, generator=None):
        return self.references.pop((write_index, processed_end, compressed_end))
