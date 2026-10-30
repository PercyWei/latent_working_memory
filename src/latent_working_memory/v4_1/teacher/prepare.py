"""从完整、未压缩的已到达历史准备 outer AM 监督，不读取下游 QA。"""

import argparse
from dataclasses import dataclass
import hashlib
import os
import random

import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.utils.logging import disable_progress_bar

from ..cli import DEFAULT_CONFIG, add_config_arguments, config_overrides, positive_int
from ..config import load_config
from ..data import compression_boundaries, load_episodes
from .cache import TeacherCache, build_cache_identity, content_identity, initialize_cache
from .generation import generate_reference_pool
from .progress import PreparationProgress, WorkerProgress
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
            model, config.model.prefill_chunk_size, config.teacher.prepare_batch_size
        )
        if model.model.has_sliding_layers:
            raise ValueError("v4.1 teacher preparation requires full-attention Qwen3")
        self.model = model
        self.config = config
        self.cache = cache
        self.prompts = cache.identity["prompt_token_ids"]
        self.eos_ids = set(cache.identity["eos_token_ids"])

    @torch.no_grad()
    def prepare_document(self, job, progress=None):
        document_id, context_ids = job.document_id, job.context_ids
        state = DynamicCache(config=self.model.config)
        for write_index, processed_end, compressed_end in job.boundaries:
            if progress is not None:
                progress.set_phase("prefill", write_index)
            # 跳过已有边界的生成，仍将此前完整原文追加到 teacher prefix KV。
            self.replay.advance(context_ids, state, processed_end)
            if processed_end + max(map(len, self.prompts)) + self.config.teacher.response_max_tokens > (
                self.model.config.max_position_embeddings
            ):
                raise ValueError(f"{document_id} write={write_index}: teacher branch exceeds model window")
            prefix = tuple((layer.keys, layer.values) for layer in state.layers)
            if progress is not None:
                progress.set_phase("generating", write_index)
            seed = int.from_bytes(hashlib.blake2b(
                f"{self.config.seed}:{document_id}:{write_index}".encode(), digest_size=8
            ).digest(), "little")
            tensors, candidate_count = generate_reference_pool(
                self.model, prefix, self.prompts, self.eos_ids,
                self.config.teacher.response_max_tokens, self.config.teacher.prepare_batch_size,
                compressed_end, self.config.teacher.pool_size, random.Random(seed),
                self.config.model.am_std_floor,
                on_batch=progress.set_prompts if progress is not None else None,
            )
            if progress is not None:
                progress.set_phase("saving", write_index)
            self.cache.write(
                document_id, write_index, tensors, processed_end, compressed_end,
                context_ids, candidate_count,
            )
            if progress is not None:
                progress.finish_boundary()
            del prefix, tensors
        if progress is not None:
            progress.finish_document()


def prepare_worker(rank, shards, config, cache_directory, workers):
    disable_progress_bar()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device)[0] < 8:
        raise RuntimeError("FlashInfer BF16 teacher preparation requires an SM80+ GPU")
    cache = TeacherCache(cache_directory)
    preparer = TeacherPreparer(load_teacher(config.model, device), config, cache)
    for job in shards[rank]:
        preparer.prepare_document(job, workers[rank])
    workers[rank].set_phase("done")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--prepare-data", default=argparse.SUPPRESS,
                        help="离线准备使用的 train/dev 数据目录，默认使用 training.dataset_dir")
    parser.add_argument("--prepare-batch-size-per-gpu", type=positive_int, default=argparse.SUPPRESS,
                        help="每卡离线准备批量；默认使用 teacher.prepare_batch_size（64）")
    args = parser.parse_args()
    config = load_config(getattr(args, "config", DEFAULT_CONFIG), config_overrides(args))
    if config.model.backbone_dtype != "bfloat16":
        raise ValueError("FlashInfer teacher preparation requires backbone_dtype=bfloat16")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("set CUDA_VISIBLE_DEVICES to the GPUs used for teacher preparation")
    num_workers = torch.cuda.device_count()
    if num_workers == 0:
        raise RuntimeError("teacher preparation requires a visible CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)
    cache = initialize_cache(config.training.teacher_cache_dir, build_cache_identity(config, tokenizer))
    print(f"teacher cache_dir={cache.directory}", flush=True)
    documents = {}
    split_documents = {}
    for split in ("train", "dev"):
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
    total_boundaries = sum(len(compression_boundaries(
        len(context_ids), config.model.pending_size, config.model.recent_size,
    )) for context_ids in documents.values())
    workers = [WorkerProgress(mp.get_context("spawn")) for _ in shards]
    progress = PreparationProgress(
        shards, workers, len(documents), total_boundaries, config.teacher.num_prompts,
    )
    if not shards:
        progress.show(force=True, status="complete")
        return
    progress.show(force=True)
    processes = None
    try:
        processes = mp.spawn(
            prepare_worker, args=(shards, config, cache.directory, workers),
            nprocs=len(shards), join=False,
        )
        while not processes.join(timeout=1):
            progress.show()
    except BaseException as error:
        progress.show(force=True, status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed")
        raise
    else:
        progress.show(force=True, status="complete")
    finally:
        if processes is not None:
            for process in processes.processes:
                if process.is_alive():
                    process.terminate()
            for process in processes.processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()


if __name__ == "__main__":
    main()
