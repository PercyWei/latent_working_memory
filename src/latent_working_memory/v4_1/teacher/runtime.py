"""训练时按采样位置重建 outer AM 参照；每个 QA episode 有独立原文 KV。"""

from contextlib import contextmanager

import torch
from transformers.cache_utils import DynamicCache

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
    def episode(self, episode):
        trajectory = TeacherTrajectory(self, episode)
        try:
            yield trajectory
        finally:
            # teacher 参照不依赖这些 KV 的计算图，student backward 前即可释放。
            trajectory.prefix = None


class TeacherTrajectory:
    def __init__(self, teacher, episode):
        self.teacher = teacher
        self.context_ids = episode.context_ids
        self.document_id = episode.document_id
        self.context_identity = content_identity(episode.context_ids)
        self.prefix = DynamicCache(config=teacher.model.config)

    @torch.no_grad()
    def reference(self, write_index, processed_end, compressed_end, generator=None):
        teacher = self.teacher
        record, cached = teacher.cache.read(self.document_id, write_index)
        if (record["processed_end"], record["compressed_end"]) != (processed_end, compressed_end):
            raise ValueError("teacher cache boundary does not match the online trajectory")
        if record["context_tokens"] != self.context_identity:
            raise ValueError("teacher cache tokenized context differs from the current episode")
        indices = torch.randperm(
            teacher.config.teacher.pool_size, generator=generator,
            device="cpu" if generator is None else generator.device,
        )[:teacher.config.teacher.am_sample_size].cpu()
        positions = cached["positions"].index_select(0, indices)
        tokens, offsets = cached["response_token_ids"].tolist(), cached["response_offsets"].tolist()
        sequences = [
            (*prompt, *tokens[offsets[index]:offsets[index + 1]])
            for index, prompt in enumerate(teacher.prompts)
        ]
        replay = teacher.replay
        replay.advance(self.context_ids, self.prefix, processed_end)
        base = teacher.model.config
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

        replay.replay(self.prefix, sequences, positions, compressed_end, collect)
        return result
