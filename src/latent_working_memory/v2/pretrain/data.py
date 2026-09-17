"""按已保存的位置读取原始 Parquet／字符索引；启动时分词、筛选一次，各 epoch 复用。"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random

import pyarrow.parquet as pq
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


def load_referenced_documents(root, indices):
    """Read each referenced row group once, decode only selected articles, and reuse them."""
    requests = defaultdict(lambda: defaultdict(set))
    for row in indices:
        if Path(row["source_file"]).is_absolute() or min(row["row_group"], row["row_index"]) < 0:
            raise ValueError(
                f"{row['sample_id']}: expected relative source path and nonnegative row location"
            )
        requests[row["source_file"]][row["row_group"]].add(row["row_index"])
    documents = {}
    for source_file, groups in requests.items():
        with pq.ParquetFile(root / source_file) as source:
            for group, selected in groups.items():
                positions = sorted(selected)
                table = source.read_row_group(group, columns=["id", "text"])
                records = table.take(positions).to_pylist()
                for position, record in zip(positions, records, strict=True):
                    documents[source_file, group, position] = record
    return documents


def load_datasets(config, tokenizer, max_positions, training):
    root = Path(config.dataset_dir)
    metadata = json.loads((root / "preparation.json").read_text())
    prompts = {
        task: len(tokenizer.encode(getattr(training, task + "_prompt"), add_special_tokens=False))
        for task in ("ae", "lm")
    }
    q = metadata["config"]["continuation_tokens"]
    stages = ("warmup", "multiround") if training.warmup_epochs else ("multiround",)
    indices_by_split = {}
    for stage in stages:
        for split in ("train", "dev", "test"):
            if stage == "multiround" and split == "train" and not training.multiround_epochs:
                indices_by_split[stage, split] = []
            else:
                path = root / STAGE_DIRECTORIES[stage] / f"{split}.jsonl"
                indices_by_split[stage, split] = [
                    json.loads(line) for line in path.read_text().splitlines()
                ]
    documents = load_referenced_documents(
        root, (row for indices in indices_by_split.values() for row in indices)
    )
    datasets, filtering = {}, {}
    for stage in stages:
        datasets[stage], filtering[stage] = {}, {}
        for split in ("train", "dev", "test"):
            if stage == "multiround" and split == "train" and not training.multiround_epochs:
                datasets[stage][split] = ()
                filtering[stage][split] = {"candidates": 0, "retained": 0, "rejected": {}}
                continue
            path = root / STAGE_DIRECTORIES[stage] / f"{split}.jsonl"
            indices = indices_by_split[stage, split]
            rows, rejected = [], Counter()
            for start in range(0, len(indices), 64):
                batch = indices[start : start + 64]
                texts = []
                for row in batch:
                    document = documents[row["source_file"], row["row_group"], row["row_index"]]
                    if document["id"] != row["document_id"]:
                        raise ValueError(
                            f"{row['sample_id']}: Parquet location does not match document_id"
                        )
                    text = document["text"][row["char_start"] : row["char_end"]]
                    cuts = row["write_token_ends"]
                    if (
                        row["char_start"] < 0
                        or len(text) != row["char_end"] - row["char_start"]
                        or not cuts
                        or cuts != sorted(set(cuts))
                        or not 0 < cuts[0] <= cuts[-1]
                    ):
                        raise ValueError(
                            f"{row['sample_id']}: invalid source interval or target token cuts"
                        )
                    texts.append(text)
                encoded = tokenizer(texts, add_special_tokens=False)
                for row, ids in zip(batch, encoded["input_ids"], strict=True):
                    # Reserve characters only enlarge the candidate window. The written prefix
                    # and all AE/LM targets use the same contiguous, tokenized sequence.
                    ends = tuple(row["write_token_ends"])
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
