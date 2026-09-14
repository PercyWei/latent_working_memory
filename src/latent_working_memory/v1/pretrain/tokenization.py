"""Bounded CPU tokenization of current text blocks and upcoming training batches."""

from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import multiprocessing
import os

from latent_working_memory.data_preparation.pretrain.text_samples import TextSample, input_text_key
from latent_working_memory.v1.data import Episode
from latent_working_memory.v1.pretrain.prepared_data import eligible_lengths


@dataclass(frozen=True, slots=True)
class TokenizedSample:
    episode: Episode
    prompt_ids: tuple[int, ...]
    target_ids: tuple[int, ...]


def encode_texts(tokenizer, texts):
    return tokenizer(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]


def inspect_block(tokenizer, config, block):
    samples = [TextSample(**json.loads(line)) for _, line in block]
    texts = [s.text for s in samples]
    continuation_positions = [i for i, s in enumerate(samples) if s.task == "continuation"]
    texts.extend(samples[i].continuation for i in continuation_positions)
    encoded = encode_texts(tokenizer, texts)
    targets = {
        i: len(ids) for i, ids in zip(continuation_positions, encoded[len(samples) :], strict=True)
    }
    prompts = encode_texts(tokenizer, [config.ae_prompt, config.lm_prompt])
    prompt_lengths = dict(zip(("ae", "continuation"), map(len, prompts), strict=True))
    result = []
    for i, ((offset, _), sample) in enumerate(zip(block, samples, strict=True)):
        size = eligible_lengths(
            len(encoded[i]), targets.get(i, len(encoded[i])), sample.task, config, prompt_lengths
        )
        target = sample.text if sample.task == "ae" else sample.continuation
        content = (
            hashlib.blake2b(
                json.dumps(
                    (sample.task, input_text_key(sample.text), " ".join(target.split()))
                ).encode()
            ).hexdigest()
            if size is not None
            else None
        )
        result.append(
            (
                offset,
                sample.sample_id,
                sample.document_id,
                sample.source_id,
                sample.dedup_cluster,
                sample.task,
                sample.boundary_method,
                size,
                content,
            )
        )
    return result


def tokenize_entries(tokenizer, config, entries):
    """Read only this batch; AE input and target share its one encoding of X."""
    samples = []
    with ExitStack() as stack:
        handles = {}
        for path, offset, variant in entries:
            if path not in handles:
                handles[path] = stack.enter_context(open(path, "rb"))
            handles[path].seek(offset)
            samples.append((TextSample(**json.loads(handles[path].readline())), variant))
    texts = [s.text for s, _ in samples]
    continuation_positions = [i for i, (s, _) in enumerate(samples) if s.task == "continuation"]
    texts.extend(samples[i][0].continuation for i in continuation_positions)
    # Prompts belong to this request as well; no persistent token cache.
    texts.extend([config.ae_prompt, config.lm_prompt])
    encoded = encode_texts(tokenizer, texts)
    targets = dict(zip(continuation_positions, encoded[len(samples) : -2], strict=True))
    result = []
    for i, (sample, variant) in enumerate(samples):
        ids = tuple(encoded[i])
        prompt = encoded[-2] if sample.task == "ae" else encoded[-1]
        target = ids if sample.task == "ae" else tuple(targets[i])
        result.append(
            TokenizedSample(
                sample.to_episode_tokens(ids, config, variant),
                tuple(prompt),
                (*target, tokenizer.eos_token_id),
            )
        )
    return result


def initialize_worker(tokenizer, config):
    global worker_tokenizer, worker_config
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    worker_tokenizer, worker_config = tokenizer, config


def process_block(block):
    return inspect_block(worker_tokenizer, worker_config, block)


def process_entries(entries):
    return tokenize_entries(worker_tokenizer, worker_config, entries)


class TokenizationPool:
    """One spawn pool per training process; zero workers is explicit serial execution."""

    def __init__(self, tokenizer, config, workers=0):
        if type(workers) is not int or workers < 0:
            raise ValueError("tokenizer_workers must be a non-negative integer")
        self.tokenizer, self.config, self.workers = tokenizer, config, workers
        self.executor = None

    def __enter__(self):
        if self.workers:
            self.executor = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_worker,
                initargs=(self.tokenizer, self.config),
            )
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None

    def inspect_blocks(self, blocks):
        if not self.workers:
            for block in blocks:
                yield inspect_block(self.tokenizer, self.config, block)
            return
        blocks = iter(blocks)
        pending = deque()
        try:
            for _ in range(2 * self.workers):
                block = next(blocks, None)
                if block is None:
                    break
                pending.append(self.executor.submit(process_block, block))
            while pending:
                yield pending.popleft().result()
                block = next(blocks, None)
                if block is not None:
                    pending.append(self.executor.submit(process_block, block))
        finally:
            for future in pending:
                future.cancel()

    def submit_batch(self, entries):
        if self.workers:
            return self.executor.submit(process_entries, entries)
        future = Future()
        future.set_result(tokenize_entries(self.tokenizer, self.config, entries))
        return future


def text_blocks(path, batch_size):
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("tokenization_batch_size must be a positive integer")
    with open(path, "rb") as handle:
        block = []
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            block.append((offset, line))
            if len(block) == batch_size:
                yield block
                block = []
        if block:
            yield block
