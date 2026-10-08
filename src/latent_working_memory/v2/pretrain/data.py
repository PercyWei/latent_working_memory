"""读取多段候选文本，分词筛选一次；单次与多次写入共用连续 token 序列。"""

from collections import Counter
from dataclasses import dataclass, replace
from itertools import islice
import json
from pathlib import Path
import random

import torch

from latent_working_memory.data_preparation.fineweb_multisegment.config import DataPreparationConfig
from latent_working_memory.data_preparation.fineweb_multisegment.records import MultisegmentSample


@dataclass(frozen=True)
class Trajectory:
    document_id: str
    source_char_start: int
    token_ids: torch.Tensor
    write_ends: tuple[int, ...]
    capacity: int
    sample_id: str = ""


def rejection_reason(ends, token_count, preparation, max_positions, prompts):
    capacity = preparation.capacity
    continuation_tokens = preparation.continuation_tokens
    if token_count < ends[-1]:
        return "content_length"
    if token_count - ends[-1] < continuation_tokens:
        return "continuation_length"
    writer_lengths = [ends[-1]] + [
        end - start + (capacity if i else 0)
        for i, (start, end) in enumerate(zip((0,) + ends[:-1], ends, strict=True))
    ]
    if (
        max(
            *writer_lengths,
            capacity + prompts["ae"] + ends[-1] + 1,
            capacity + prompts["lm"] + continuation_tokens + 1,
        )
        > max_positions
    ):
        return "model_window"
    return None


def load_datasets(config, tokenizer, max_positions, training):
    root = Path(config.dataset_dir)
    metadata = json.loads((root / "preparation.json").read_text(encoding="utf-8"))
    preparation = DataPreparationConfig(**metadata["config"])
    capacity, q = preparation.capacity, preparation.continuation_tokens
    prompts = {
        task: len(tokenizer.encode(getattr(training, task + "_prompt"), add_special_tokens=False))
        for task in ("ae", "lm")
    }
    stages = ("warmup", "multiround") if training.warmup_epochs else ("multiround",)
    datasets = {stage: {} for stage in stages}
    filtering = {stage: {} for stage in stages}
    for split in ("train", "dev", "test"):
        path = root / f"{split}.jsonl"
        rows, rejected, candidates = [], Counter(), 0
        with path.open(encoding="utf-8") as stream:
            while lines := list(islice(stream, 64)):
                batch = [MultisegmentSample(**json.loads(line)) for line in lines]
                for sample in batch:
                    sample.validate_plan(preparation)
                candidates += len(batch)
                encoded = tokenizer(
                    [sample.text for sample in batch],
                    add_special_tokens=False,
                    truncation=False,
                )
                for sample, ids in zip(batch, encoded["input_ids"], strict=True):
                    # Tokenize the complete candidate once, including all reserve text.
                    # AE, LM and both write modes reuse the same contiguous token sequence.
                    ends = tuple(sample.write_token_ends)
                    reason = rejection_reason(ends, len(ids), preparation, max_positions, prompts)
                    if reason:
                        rejected[reason] += 1
                        continue
                    rows.append(
                        Trajectory(
                            sample.document_id,
                            sample.source["char_span"][0],
                            torch.tensor(ids[: ends[-1] + q], dtype=torch.long),
                            ends,
                            capacity,
                            sample.sample_id,
                        )
                    )
        if candidates and not rows:
            raise ValueError(f"no valid trajectories in {path}: {dict(rejected)}")
        for stage in stages:
            if (
                stage == "multiround"
                and split == "train"
                and not (training.multiround_epochs or training.independent_prefix_epochs)
            ):
                datasets[stage][split] = ()
                filtering[stage][split] = {"candidates": 0, "retained": 0, "rejected": {}}
                continue
            datasets[stage][split] = tuple(
                replace(row, write_ends=(row.write_ends[-1],)) if stage == "warmup" else row
                for row in rows
            )
            filtering[stage][split] = {
                "candidates": candidates,
                "retained": len(rows),
                "rejected": dict(rejected),
            }
            print(
                f"{stage}/{split}: retained {len(rows)}/{candidates}, rejected {dict(rejected)}",
                flush=True,
            )
    return datasets, metadata, filtering


def dataset_statistics(datasets):
    result = {}
    for stage, splits in datasets.items():
        result[stage] = {}
        for split, rows in splits.items():
            parts = [
                end - start
                for row in rows
                for start, end in zip((0,) + row.write_ends[:-1], row.write_ends, strict=True)
            ]
            result[stage][split] = {
                "trajectories": len(rows),
                "documents": len({r.document_id for r in rows}),
                "rounds": dict(Counter(len(r.write_ends) for r in rows)),
                "ratio_bins": dict(Counter((r.write_ends[-1] - 1) // r.capacity + 1 for r in rows)),
                "segment_min": min(parts, default=0),
                "segment_max": max(parts, default=0),
                "source_tokens": sum(r.write_ends[-1] for r in rows),
            }
    return result


def epoch_batches(rows, batch_size, seed, stage, epoch):
    indices = list(range(len(rows)))
    random.Random(f"{seed}:order:{stage}:{epoch}").shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [rows[i] for i in indices[start : start + batch_size]]
