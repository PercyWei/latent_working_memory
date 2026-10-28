"""从完整、未压缩的已到达历史准备 outer AM 监督，不读取下游 QA。"""

import argparse
import hashlib
import random

import torch
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

from ..config import load_config
from ..data import compression_boundaries, load_episodes
from .cache import build_cache_identity, content_identity, initialize_cache
from .replay import TeacherReplay, load_teacher


def reservoir_positions(lengths, pool_size, rng):
    """对全部真实位置做等概率无放回抽样；不足固定容量时明确失败。"""
    reservoir = []
    count = 0
    for prompt_index, length in enumerate(lengths):
        for position in range(length):
            count += 1
            if len(reservoir) < pool_size:
                reservoir.append((prompt_index, position))
            else:
                replacement = rng.randrange(count)
                if replacement < pool_size:
                    reservoir[replacement] = (prompt_index, position)
    if count < pool_size:
        raise ValueError(
            f"teacher generated {count} candidate positions, below required pool_size={pool_size}; "
            "increase prompt/response coverage rather than duplicating positions"
        )
    return torch.tensor(reservoir, dtype=torch.int64), count


class PoolScales:
    """按层和 head 合并均值与 M2，仅保留完整位置池的 FP64 统计量。"""

    def __init__(self, layers, query_heads, head_dim, device=None):
        self.head_dim = head_dim
        self.counts = [0] * layers
        self.mean_o = torch.zeros(layers, query_heads, dtype=torch.float64, device=device)
        self.mean_z = torch.zeros_like(self.mean_o)
        self.m2_o = torch.zeros_like(self.mean_o)
        self.m2_z = torch.zeros_like(self.mean_o)

    @staticmethod
    def _combine(mean, m2, old_count, batch_count, batch_variance, batch_mean):
        total = old_count + batch_count
        delta = batch_mean - mean
        m2.add_(batch_variance * batch_count + delta.square() * (old_count * batch_count / total))
        mean.add_(delta * (batch_count / total))

    def update(self, layer_index, pool_indices, queries, outputs, log_mass):
        count = outputs.shape[0]
        variance_o, mean_o = torch.var_mean(outputs.double(), dim=(0, 2), correction=0)
        variance_z, mean_z = torch.var_mean(log_mass.double(), dim=0, correction=0)
        old_count = self.counts[layer_index]
        self._combine(
            self.mean_o[layer_index], self.m2_o[layer_index], old_count * self.head_dim,
            count * self.head_dim, variance_o, mean_o,
        )
        self._combine(
            self.mean_z[layer_index], self.m2_z[layer_index], old_count,
            count, variance_z, mean_z,
        )
        self.counts[layer_index] += count

    def finish(self, pool_size, floor):
        if any(count != pool_size for count in self.counts):
            raise ValueError("scale statistics must cover every cached position in every layer")
        return {
            "scales_o": (self.m2_o / (pool_size * self.head_dim)).sqrt().float().clamp_min(floor).cpu(),
            "scales_z": (self.m2_z / pool_size).sqrt().float().clamp_min(floor).cpu(),
        }


class TeacherPreparer:
    """逐提示保存响应；固定位置池仅用于流式尺度统计，不保存稠密监督张量。"""

    def __init__(self, model, config, cache):
        self.replay = TeacherReplay(
            model, config.model.prefill_chunk_size, config.teacher.branch_batch_size
        )
        self.model = model
        self.config = config
        self.cache = cache
        self.prompts = cache.identity["prompt_token_ids"]
        self.eos_ids = set(cache.identity["eos_token_ids"])

    def _generate(self, prefix):
        responses = []
        for prompt_index, prompt in enumerate(self.prompts):
            branch = DynamicCache(ddp_cache_data=prefix)
            result = self.replay.forward(prompt, branch, logits=True)
            response = []
            for _ in range(self.config.teacher.response_max_tokens):
                token = int(result.logits[0, -1].argmax())
                response.append(token)
                if token in self.eos_ids:
                    break
                if len(response) < self.config.teacher.response_max_tokens:
                    result = self.replay.forward([token], branch, logits=True)
            # 最后生成的 token 会在下一次回放中实际前向，保证其 query 也在候选池中。
            responses.append(tuple(response))
            del result, branch
            if (prompt_index + 1) % 48 == 0:
                print(f"teacher rollout prompts={prompt_index + 1}/{len(self.prompts)}", flush=True)
        return responses

    def _collect(self, prefix_cache, responses, positions, compressed_end):
        architecture = self.cache.identity["architecture"]
        scales = PoolScales(
            architecture["layers"], architecture["query_heads"], architecture["head_dim"],
            self.replay.device,
        )
        sequences = [(*prompt, *response) for prompt, response in zip(self.prompts, responses)]
        self.replay.replay(prefix_cache, sequences, positions, compressed_end, scales.update)
        lengths = torch.tensor([len(response) for response in responses], dtype=torch.int64)
        return {
            "response_token_ids": torch.tensor(
                [token for response in responses for token in response], dtype=torch.int64
            ),
            "response_offsets": torch.cat((lengths.new_zeros(1), lengths.cumsum(0))),
            "positions": positions,
            **scales.finish(self.config.teacher.pool_size, self.config.model.am_std_floor),
        }

    @torch.no_grad()
    def prepare_document(self, document_id, context_ids):
        boundaries = compression_boundaries(
            len(context_ids), self.config.model.pending_size, self.config.model.recent_size
        )
        context_identity = content_identity(context_ids)
        cached = set()
        for write_index, processed_end, compressed_end in boundaries:
            path = self.cache.path(document_id, write_index)
            if path.exists():
                record = self.cache.metadata(document_id, write_index)
                if (
                    record["processed_end"] != processed_end
                    or record["compressed_end"] != compressed_end
                    or record["context_tokens"] != context_identity
                ):
                    raise ValueError(f"{path}: boundary or tokenized context does not match")
                cached.add(write_index)
        missing = [index for index, _, _ in boundaries if index not in cached]
        if not missing:
            print(f"teacher {document_id}: all {len(boundaries)} boundaries cached", flush=True)
            return
        state = DynamicCache(config=self.model.config)
        for write_index, processed_end, compressed_end in boundaries[:max(missing)]:
            # 即使已有某边界缓存，后续边界也仍从完整原文继续 prefill。
            self.replay.advance(context_ids, state, processed_end)
            if write_index in cached:
                continue
            if processed_end + max(map(len, self.prompts)) + self.config.teacher.response_max_tokens > (
                self.model.config.max_position_embeddings
            ):
                raise ValueError(f"{document_id} write={write_index}: teacher branch exceeds model window")
            prefix = tuple((layer.keys, layer.values) for layer in state.layers)
            print(
                f"teacher {document_id} write={write_index}/{len(boundaries)}: generating prompts",
                flush=True,
            )
            responses = self._generate(prefix)
            seed = int.from_bytes(hashlib.blake2b(
                f"{self.config.seed}:{document_id}:{write_index}".encode(), digest_size=8
            ).digest(), "little")
            positions, candidate_count = reservoir_positions(
                [len(prompt) + len(response) for prompt, response in zip(self.prompts, responses)],
                self.config.teacher.pool_size,
                random.Random(seed),
            )
            tensors = self._collect(state, responses, positions, compressed_end)
            self.cache.write(
                document_id, write_index, tensors, processed_end, compressed_end,
                context_ids, candidate_count,
            )
            print(
                f"teacher {document_id} write={write_index} context={processed_end} "
                f"compressed={compressed_end} candidates={candidate_count} pool={len(positions)}",
                flush=True,
            )
            del prefix, responses, tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--splits", nargs="+", choices=("train", "dev"), default=["train", "dev"])
    args = parser.parse_args()
    config = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)
    model = load_teacher(config.model, args.device)
    cache = initialize_cache(args.output_dir, build_cache_identity(config, tokenizer))
    preparer = TeacherPreparer(model, config, cache)
    documents = {}
    split_documents = {}
    for split in args.splits:
        episodes, _, statistics = load_episodes(
            config.training.dataset_dir, tokenizer, split=split,
            min_text_tokens=config.model.pending_size + config.model.recent_size,
            max_text_tokens=config.training.max_train_data_tokens if split == "train" else None,
            model_window=model.config.max_position_embeddings,
            max_samples=(
                config.training.max_train_samples if split == "train"
                else config.training.max_dev_samples
            ),
            seed=config.seed,
        )
        raw_ids = set(statistics["raw_document_ids"])
        for other_split, other_ids in split_documents.items():
            if raw_ids & other_ids:
                raise ValueError(f"{split} and {other_split} contain overlapping documents")
        split_documents[split] = raw_ids
        for episode in episodes:
            documents[episode.document_id] = episode.context_ids
    for document_id, context_ids in documents.items():
        preparer.prepare_document(document_id, context_ids)


if __name__ == "__main__":
    main()
