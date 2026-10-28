"""从完整、未压缩的已到达历史准备 outer AM 监督，不读取下游 QA。"""

import argparse
from dataclasses import dataclass, replace
import hashlib
import os
import random

import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

from ..config import load_config
from ..data import compression_boundaries, load_episodes
from .cache import TeacherCache, build_cache_identity, content_identity, initialize_cache
from .generation import generate_reference_pool
from .replay import TeacherReplay, load_teacher


@dataclass(frozen=True)
class PreparationJob:
    document_id: str
    context_ids: tuple[int, ...]
    boundaries: tuple[tuple[int, int, int], ...]


def plan_preparation(documents, config, cache, num_workers):
    """校验已存边界，按剩余边界的历史长度估算工作量，整篇文档分配给一张卡。"""
    jobs = []
    for document_id, context_ids in documents.items():
        context_identity = content_identity(context_ids)
        boundaries = compression_boundaries(
            len(context_ids), config.model.pending_size, config.model.recent_size
        )
        missing = []
        for write_index, processed_end, compressed_end in boundaries:
            if cache.path(document_id, write_index).exists():
                record = cache.metadata(document_id, write_index)
                if (
                    (record["processed_end"], record["compressed_end"])
                    != (processed_end, compressed_end)
                    or record["context_tokens"] != context_identity
                ):
                    raise ValueError(f"{document_id} write={write_index}: cached context or boundary differs")
            else:
                missing.append((write_index, processed_end, compressed_end))
        if missing:
            jobs.append(PreparationJob(document_id, context_ids, tuple(missing)))
        else:
            print(f"teacher {document_id}: all {len(boundaries)} boundaries cached", flush=True)
    shards = [[] for _ in range(min(num_workers, len(jobs)))]
    loads = [0] * len(shards)
    weighted_jobs = [(sum(end for _, end, _ in job.boundaries), job) for job in jobs]
    for cost, job in sorted(weighted_jobs, key=lambda item: (-item[0], item[1].document_id)):
        rank = min(range(len(shards)), key=loads.__getitem__)
        shards[rank].append(job)
        loads[rank] += cost
    return shards


class TeacherPreparer:
    """批量生成响应；固定位置池仅用于流式尺度统计，不保存稠密监督张量。"""

    def __init__(self, model, config, cache):
        self.replay = TeacherReplay(
            model, config.model.prefill_chunk_size, config.teacher.batch_size
        )
        self.model = model
        self.config = config
        self.cache = cache
        self.prompts = cache.identity["prompt_token_ids"]
        self.eos_ids = set(cache.identity["eos_token_ids"])

    @torch.no_grad()
    def prepare_document(self, job):
        document_id, context_ids = job.document_id, job.context_ids
        state = DynamicCache(config=self.model.config)
        for write_index, processed_end, compressed_end in job.boundaries:
            # 跳过已有边界的生成，仍将此前完整原文追加到 teacher prefix KV。
            self.replay.advance(context_ids, state, processed_end)
            if processed_end + max(map(len, self.prompts)) + self.config.teacher.response_max_tokens > (
                self.model.config.max_position_embeddings
            ):
                raise ValueError(f"{document_id} write={write_index}: teacher branch exceeds model window")
            prefix = tuple((layer.keys, layer.values) for layer in state.layers)
            print(
                f"teacher device={self.replay.device} {document_id} write={write_index}: generating references",
                flush=True,
            )
            seed = int.from_bytes(hashlib.blake2b(
                f"{self.config.seed}:{document_id}:{write_index}".encode(), digest_size=8
            ).digest(), "little")
            tensors, candidate_count = generate_reference_pool(
                self.model, prefix, self.prompts, self.eos_ids,
                self.config.teacher.response_max_tokens, self.config.teacher.batch_size,
                compressed_end, self.config.teacher.pool_size, random.Random(seed),
                self.config.model.am_std_floor,
            )
            self.cache.write(
                document_id, write_index, tensors, processed_end, compressed_end,
                context_ids, candidate_count,
            )
            print(
                f"teacher device={self.replay.device} {document_id} write={write_index} context={processed_end} "
                f"compressed={compressed_end} candidates={candidate_count} pool={len(tensors['positions'])}",
                flush=True,
            )
            del prefix, tensors


def prepare_worker(rank, shards, config, cache_directory):
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    cache = TeacherCache(cache_directory)
    preparer = TeacherPreparer(load_teacher(config.model, device), config, cache)
    for job in shards[rank]:
        preparer.prepare_document(job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "dev"), default=["train", "dev"])
    parser.add_argument("--batch-size", type=int, help="每卡 teacher 批处理大小；默认使用配置值")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.batch_size is not None:
        config = replace(config, teacher=replace(config.teacher, batch_size=args.batch_size))
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("set CUDA_VISIBLE_DEVICES to the GPUs used for teacher preparation")
    num_workers = torch.cuda.device_count()
    if num_workers == 0:
        raise RuntimeError("teacher preparation requires a visible CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)
    cache = initialize_cache(args.output_dir, build_cache_identity(config, tokenizer))
    documents = {}
    split_documents = {}
    for split in args.splits:
        episodes, _, statistics = load_episodes(
            config.training.dataset_dir, tokenizer, split=split,
            min_text_tokens=config.model.pending_size + config.model.recent_size,
            max_text_tokens=config.training.max_train_data_tokens if split == "train" else None,
            model_window=cache.identity["architecture"]["max_position_embeddings"],
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
    shards = plan_preparation(documents, config, cache, num_workers)
    for rank, jobs in enumerate(shards):
        print(
            f"teacher cuda:{rank}: documents={len(jobs)} "
            f"remaining_boundaries={sum(len(job.boundaries) for job in jobs)} "
            f"batch_size={config.teacher.batch_size}",
            flush=True,
        )
    if len(shards) == 1:
        prepare_worker(0, shards, config, cache.directory)
    elif shards:
        mp.spawn(prepare_worker, args=(shards, config, cache.directory), nprocs=len(shards), join=True)


if __name__ == "__main__":
    main()
