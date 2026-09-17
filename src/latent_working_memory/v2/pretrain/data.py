"""读取已保存的原文／字符索引；启动时分词、筛选一次，各 epoch 复用。"""

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import random

import torch

from latent_working_memory.v2.pretrain.prepare_data import STAGE_DIRECTORIES


@dataclass(frozen=True)
class Trajectory:
    document_id: str
    source_char_start: int
    token_ids: torch.Tensor
    write_ends: tuple[int, ...]
    capacity: int
    sample_id: str = ""


def rejection_reason(
    ends, token_count, capacity, continuation_tokens, stage, max_positions, prompts
):
    if stage == "warmup":
        if len(ends) != 1 or not 2 * capacity <= ends[-1] <= 8 * capacity:
            return "single_length"
    else:
        parts = [end - start for start, end in zip((0,) + ends[:-1], ends, strict=True)]
        if (
            not 3 <= len(parts) <= 5
            or any(not capacity <= n <= 3 * capacity for n in parts)
            or ends[-1] > 8 * capacity
        ):
            return "multi_lengths"
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
    if not tokenizer.is_fast:
        raise ValueError("character indices require a fast tokenizer with offset mappings")
    root = Path(config.dataset_dir)
    metadata = json.loads((root / "preparation.json").read_text())
    documents = {
        row["document_id"]: row
        for row in (
            json.loads(line) for line in (root / "documents.jsonl").read_text().splitlines()
        )
    }
    prompts = {
        task: len(tokenizer.encode(getattr(training, task + "_prompt"), add_special_tokens=False))
        for task in ("ae", "lm")
    }
    q = metadata["config"]["continuation_tokens"]
    stages = ("warmup", "multiround") if training.warmup_epochs else ("multiround",)
    datasets, filtering = {}, {}
    for stage in stages:
        datasets[stage], filtering[stage] = {}, {}
        for split in ("train", "dev", "test"):
            if stage == "multiround" and split == "train" and not training.multiround_epochs:
                datasets[stage][split] = ()
                filtering[stage][split] = {"candidates": 0, "retained": 0, "rejected": {}}
                continue
            path = root / STAGE_DIRECTORIES[stage] / f"{split}.jsonl"
            indices = [json.loads(line) for line in path.read_text().splitlines()]
            rows, rejected = [], Counter()
            for start in range(0, len(indices), 64):
                batch = indices[start : start + 64]
                texts = []
                for row in batch:
                    document = documents[row["document_id"]]
                    if document["split"] != split:
                        raise ValueError(
                            f"{row['sample_id']}: source split differs from sample split"
                        )
                    text = document["text"][row["char_start"] : row["char_end"]]
                    cuts = row["write_char_ends"]
                    if (
                        row["char_start"] < 0
                        or len(text) != row["char_end"] - row["char_start"]
                        or not cuts
                        or cuts != sorted(set(cuts))
                        or not 0 < cuts[0] <= cuts[-1] < len(text)
                    ):
                        raise ValueError(f"{row['sample_id']}: invalid character indices")
                    texts.append(text)
                encoded = tokenizer(texts, add_special_tokens=False, return_offsets_mapping=True)
                for row, ids, offsets in zip(
                    batch, encoded["input_ids"], encoded["offset_mapping"], strict=True
                ):
                    # A token crossing a character cut belongs to the next write. All paths
                    # use this same full-window tokenization, including the one-shot control.
                    token_ends = [end for _, end in offsets]
                    ends = tuple(bisect_right(token_ends, cut) for cut in row["write_char_ends"])
                    reason = rejection_reason(
                        ends, len(ids), row["capacity"], q, stage, max_positions, prompts
                    )
                    if reason:
                        rejected[reason] += 1
                        continue
                    rows.append(
                        Trajectory(
                            row["document_id"],
                            row["char_start"],
                            torch.tensor(ids[: ends[-1] + q], dtype=torch.long),
                            ends,
                            row["capacity"],
                            row["sample_id"],
                        )
                    )
            filtering[stage][split] = {
                "candidates": len(indices),
                "retained": len(rows),
                "rejected": dict(rejected),
            }
            if indices and not rows:
                raise ValueError(f"no valid trajectories in {path}: {dict(rejected)}")
            datasets[stage][split] = tuple(rows)
            print(
                f"{stage}/{split}: retained {len(rows)}/{len(indices)}, rejected {dict(rejected)}",
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
