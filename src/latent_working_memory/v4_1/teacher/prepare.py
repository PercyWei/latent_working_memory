"""从完整、未压缩的已到达历史准备 outer AM 监督，不读取下游 QA。"""

import argparse
from collections import defaultdict
import hashlib
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import rotate_half

from ..config import load_config
from ..data import EpisodeEncoder, compression_boundaries, load_episodes
from ..modeling.am import local_read, teacher_scales
from .cache import build_cache_identity, content_identity, initialize_cache
from .prompts import SOURCE_PROMPTS


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


class TeacherPreparer:
    """同一边界只保留一份 prefix KV；逐提示生成，再回放选中位置收集监督。"""

    def __init__(self, model, tokenizer, config, cache):
        if model.config.model_type != "qwen3":
            raise ValueError("v4.1 teacher currently supports native Qwen3 only")
        self.model = model.eval().requires_grad_(False)
        self.encoder = EpisodeEncoder(tokenizer)
        self.config = config
        self.cache = cache
        self.device = next(model.parameters()).device
        self.prompts = [
            self.encoder.question(prompt)
            for prompt in SOURCE_PROMPTS[:config.teacher.num_prompts]
        ]
        eos = model.generation_config.eos_token_id
        self.eos_ids = {eos} if isinstance(eos, int) else set(eos)

    def _forward(self, tokens, cache, logits=False):
        ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        if logits:
            return self.model(
                input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1
            )
        return self.model.model(input_ids=ids, past_key_values=cache, use_cache=True)

    def _generate(self, prefix):
        sequences = []
        for prompt_index, prompt in enumerate(self.prompts):
            branch = DynamicCache(ddp_cache_data=prefix)
            result = self._forward(prompt, branch, logits=True)
            response = []
            for _ in range(self.config.teacher.response_max_tokens):
                token = int(result.logits[0, -1].argmax())
                response.append(token)
                if token in self.eos_ids:
                    break
                if len(response) < self.config.teacher.response_max_tokens:
                    result = self._forward([token], branch, logits=True)
            # 最后生成的 token 会在下一次回放中实际前向，保证其 query 也在候选池中。
            sequences.append((*prompt, *response))
            del result, branch
            if (prompt_index + 1) % 48 == 0:
                print(f"teacher rollout prompts={prompt_index + 1}/{len(self.prompts)}", flush=True)
        return sequences

    def _collect(self, prefix, sequences, positions, compressed_end):
        architecture = self.cache.identity["architecture"]
        shape = (
            architecture["layers"], len(positions), architecture["query_heads"],
            architecture["head_dim"],
        )
        tensors = {
            "queries": torch.empty(shape, dtype=torch.float32),
            "outputs": torch.empty(shape, dtype=torch.float32),
            "log_mass": torch.empty(shape[:-1], dtype=torch.float32),
            "positions": positions,
        }
        selected = defaultdict(list)
        for pool_index, (prompt_index, position) in enumerate(positions.tolist()):
            selected[prompt_index].append((pool_index, position))
        rotary = {}
        current = {}
        bias = torch.zeros(
            architecture["kv_heads"], compressed_end, device=self.device, dtype=torch.float32
        )

        def capture_rotary(module, args, kwargs):
            rotary["cos"], rotary["sin"] = kwargs["position_embeddings"]

        def query_hook(layer_index):
            def capture(module, args, output):
                indices = current["token_indices"]
                queries = output[0, indices]
                cos = rotary["cos"][0, indices].unsqueeze(1)
                sin = rotary["sin"][0, indices].unsqueeze(1)
                queries = (queries * cos + rotate_half(queries) * sin).float()
                keys, values = prefix[layer_index]
                outputs, log_mass = local_read(
                    queries, keys[0, :, :compressed_end], values[0, :, :compressed_end], bias
                )
                slots = current["pool_indices"]
                tensors["queries"][layer_index, slots] = queries.cpu()
                tensors["outputs"][layer_index, slots] = outputs.cpu()
                tensors["log_mass"][layer_index, slots] = log_mass.cpu()
            return capture

        handles = []
        for layer_index, layer in enumerate(self.model.model.layers):
            handles.append(layer.self_attn.register_forward_pre_hook(capture_rotary, with_kwargs=True))
            handles.append(layer.self_attn.q_norm.register_forward_hook(query_hook(layer_index)))
        try:
            for prompt_index, choices in selected.items():
                current["pool_indices"] = [item[0] for item in choices]
                current["token_indices"] = [item[1] for item in choices]
                self._forward(sequences[prompt_index], DynamicCache(ddp_cache_data=prefix))
        finally:
            for handle in handles:
                handle.remove()
        scales = [
            teacher_scales(outputs, log_mass, self.config.model.am_std_floor)
            for outputs, log_mass in zip(tensors["outputs"], tensors["log_mass"])
        ]
        tensors["scales_o"] = torch.stack([item[0] for item in scales])
        tensors["scales_z"] = torch.stack([item[1] for item in scales])
        return tensors

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
        previous_end = 0
        for write_index, processed_end, compressed_end in boundaries[:max(missing)]:
            # 即使已有某边界缓存，后续边界也仍从完整原文继续 prefill。
            for start in range(previous_end, processed_end, self.config.model.prefill_chunk_size):
                end = min(start + self.config.model.prefill_chunk_size, processed_end)
                self._forward(context_ids[start:end], state)
            previous_end = processed_end
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
            sequences = self._generate(prefix)
            seed = int.from_bytes(hashlib.blake2b(
                f"{self.config.seed}:{document_id}:{write_index}".encode(), digest_size=8
            ).digest(), "little")
            positions, candidate_count = reservoir_positions(
                [len(sequence) for sequence in sequences], self.config.teacher.pool_size,
                random.Random(seed),
            )
            tensors = self._collect(prefix, sequences, positions, compressed_end)
            self.cache.write(
                document_id, write_index, tensors, processed_end, compressed_end,
                context_ids, candidate_count,
            )
            print(
                f"teacher {document_id} write={write_index} context={processed_end} "
                f"compressed={compressed_end} candidates={candidate_count} pool={len(positions)}",
                flush=True,
            )
            del prefix, sequences, tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--splits", nargs="+", choices=("train", "dev"), default=["train", "dev"])
    args = parser.parse_args()
    config = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name_or_path, dtype=getattr(torch, config.model.backbone_dtype),
        attn_implementation="sdpa",
    ).to(args.device)
    cache = initialize_cache(
        args.output_dir, build_cache_identity(config, tokenizer), config.teacher.am_sample_size
    )
    preparer = TeacherPreparer(model, tokenizer, config, cache)
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
