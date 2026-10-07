"""Tokenize canonical FineWeb AE/continuation samples for v3 pretraining."""

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import pyarrow.parquet as pq

from latent_working_memory.data_preparation.pretrain.text_samples import TextSample


@dataclass(frozen=True, slots=True)
class PretrainExample:
    sample_id: str
    document_id: str
    dedup_cluster: str
    task: str
    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]


def load_pretraining(
    dataset_dir: str | Path, tokenizer, min_input_tokens: int, max_input_tokens: int
) -> tuple[dict[str, tuple[PretrainExample, ...]], dict]:
    """Filter whole inputs by current-tokenizer length; never crop source or targets."""
    if (
        type(min_input_tokens) is not int
        or type(max_input_tokens) is not int
        or not 1 <= min_input_tokens <= max_input_tokens
    ):
        raise ValueError("input token interval requires positive integers with min <= max")
    root = Path(dataset_dir)
    seen_samples, document_sources, cluster_splits = set(), {}, {}
    splits, statistics = {}, {}
    for split in ("train", "dev", "test"):
        examples = []
        counts = {
            "read": 0,
            "kept": 0,
            "filtered_too_short": 0,
            "filtered_too_long": 0,
            "read_by_task": {"ae": 0, "continuation": 0},
            "kept_by_task": {"ae": 0, "continuation": 0},
        }
        path = root / f"{split}.jsonl"
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    sample = TextSample(**json.loads(line))
                    if sample.sample_id in seen_samples:
                        raise ValueError("duplicate pretraining sample_id")
                    identity = (split, sample.dedup_cluster)
                    if (
                        sample.document_id in document_sources
                        and document_sources[sample.document_id] != identity
                    ):
                        raise ValueError("source document changes split or dedup cluster")
                    if (
                        sample.dedup_cluster in cluster_splits
                        and cluster_splits[sample.dedup_cluster] != split
                    ):
                        raise ValueError("dedup cluster occurs in multiple pretraining splits")
                    seen_samples.add(sample.sample_id)
                    document_sources[sample.document_id] = identity
                    cluster_splits[sample.dedup_cluster] = split
                    counts["read"] += 1
                    counts["read_by_task"][sample.task] += 1
                    input_ids = tuple(
                        tokenizer.encode(sample.text, add_special_tokens=False, truncation=False)
                    )
                    if len(input_ids) < min_input_tokens:
                        counts["filtered_too_short"] += 1
                        continue
                    if len(input_ids) > max_input_tokens:
                        counts["filtered_too_long"] += 1
                        continue
                    target_ids = (
                        input_ids
                        if sample.task == "ae"
                        else tuple(
                            tokenizer.encode(
                                sample.continuation, add_special_tokens=False, truncation=False
                            )
                        )
                    )
                    if not target_ids:
                        raise ValueError("continuation tokenized to an empty target")
                    examples.append(
                        PretrainExample(
                            sample.sample_id,
                            sample.document_id,
                            sample.dedup_cluster,
                            sample.task,
                            input_ids,
                            target_ids,
                        )
                    )
                    counts["kept"] += 1
                    counts["kept_by_task"][sample.task] += 1
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
        for name, field in (("input_tokens", "input_ids"), ("target_tokens", "target_ids")):
            lengths = [len(getattr(example, field)) for example in examples]
            counts[name] = {
                "min": min(lengths) if lengths else None,
                "max": max(lengths) if lengths else None,
                "total": sum(lengths),
            }
        splits[split] = tuple(examples)
        statistics[split] = counts
    return splits, {
        "input_token_interval": [min_input_tokens, max_input_tokens],
        "splits": statistics,
    }


def load_reconstruction_pretraining(
    dataset_dir: str | Path,
    tokenizer,
    min_input_tokens: int,
    max_input_tokens: int,
    view: str,
    lm_only: bool = False,
) -> tuple[dict[str, tuple[PretrainExample, ...]], dict]:
    """Read existing reconstruction indices without materializing another text dataset.

    A single index supplies its final prefix; a multi index supplies its first write.
    AE and continuation share that prefix, and LM starts immediately after its cut.
    The old index's memory capacity does not constrain the current model's slots.
    """
    if (
        type(min_input_tokens) is not int
        or type(max_input_tokens) is not int
        or not 1 <= min_input_tokens <= max_input_tokens
    ):
        raise ValueError("input token interval requires positive integers with min <= max")
    directories = {
        "reconstruction_single": "single",
        "reconstruction_first_write": "multi",
    }
    if view not in directories:
        raise ValueError(f"unknown reconstruction data view: {view}")
    root = Path(dataset_dir)
    metadata = json.loads((root / "preparation.json").read_text(encoding="utf-8"))
    continuation_tokens = metadata["config"]["continuation_tokens"]
    if type(continuation_tokens) is not int or continuation_tokens < 1:
        raise ValueError("reconstruction continuation_tokens must be a positive integer")
    tasks = ("continuation",) if lm_only else ("ae", "continuation")
    groups = defaultdict(lambda: defaultdict(list))
    indices_by_split, statistics = {}, {}
    seen_samples, document_sources, cluster_splits = set(), {}, {}
    for split in ("train", "dev", "test"):
        path = root / directories[view] / f"{split}.jsonl"
        indices_by_split[split] = []
        statistics[split] = {
            "source_index_candidates": 0,
            "kept_source_indices": 0,
            "read": 0,
            "kept": 0,
            "filtered_too_short": 0,
            "filtered_too_long": 0,
            "filtered_content_length": 0,
            "filtered_continuation_length": 0,
            "read_by_task": {"ae": 0, "continuation": 0},
            "kept_by_task": {"ae": 0, "continuation": 0},
        }
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                    sample_id = row["sample_id"]
                    if sample_id in seen_samples:
                        raise ValueError("duplicate pretraining sample_id")
                    identity = (split, row["dedup_cluster"])
                    if (
                        row["document_id"] in document_sources
                        and document_sources[row["document_id"]] != identity
                    ):
                        raise ValueError("source document changes split or dedup cluster")
                    if (
                        row["dedup_cluster"] in cluster_splits
                        and cluster_splits[row["dedup_cluster"]] != split
                    ):
                        raise ValueError("dedup cluster occurs in multiple pretraining splits")
                    if Path(row["source_file"]).is_absolute() or any(
                        type(row[name]) is not int or row[name] < 0
                        for name in ("row_group", "row_index")
                    ):
                        raise ValueError(
                            "expected relative source path and nonnegative row location"
                        )
                    cuts = row["write_token_ends"]
                    if (
                        not cuts
                        or any(type(cut) is not int or cut < 1 for cut in cuts)
                        or cuts != sorted(set(cuts))
                    ):
                        raise ValueError("invalid target token cuts")
                    start, end = row["char_start"], row["char_end"]
                    if type(start) is not int or type(end) is not int or not 0 <= start < end:
                        raise ValueError("invalid source character interval")
                    seen_samples.add(sample_id)
                    document_sources[row["document_id"]] = identity
                    cluster_splits[row["dedup_cluster"]] = split
                    indices_by_split[split].append(sample_id)
                    counts = statistics[split]
                    counts["source_index_candidates"] += 1
                    counts["read"] += len(tasks)
                    for task in tasks:
                        counts["read_by_task"][task] += 1
                    cut = cuts[-1 if view == "reconstruction_single" else 0]
                    if cut < min_input_tokens:
                        counts["filtered_too_short"] += len(tasks)
                        continue
                    if cut > max_input_tokens:
                        counts["filtered_too_long"] += len(tasks)
                        continue
                    groups[row["source_file"]][row["row_group"]].append(
                        (split, path, line_number, row)
                    )
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error

    examples_by_id = {}
    # Reuse integer objects across the vocabulary, and share each input tuple between
    # its AE/LM examples. This keeps Python-token storage small on multi-process runs.
    token_pool = {}
    for source_file, row_groups in groups.items():
        with pq.ParquetFile(root / source_file) as source:
            for row_group, requests in row_groups.items():
                positions = sorted({row["row_index"] for _, _, _, row in requests})
                table = source.read_row_group(row_group, columns=["id", "text"])
                documents = dict(zip(positions, table.take(positions).to_pylist(), strict=True))
                del table
                for split, path, line_number, row in requests:
                    try:
                        document = documents[row["row_index"]]
                        if document["id"] != row["document_id"]:
                            raise ValueError("Parquet location does not match document_id")
                        start, end = row["char_start"], row["char_end"]
                        if end > len(document["text"]):
                            raise ValueError("invalid source character interval")
                        cut = row["write_token_ends"][-1 if view == "reconstruction_single" else 0]
                        counts = statistics[split]
                        ids = tokenizer.encode(
                            document["text"][start:end],
                            add_special_tokens=False,
                            truncation=False,
                        )
                        if len(ids) < cut:
                            counts["filtered_content_length"] += len(tasks)
                            continue
                        if len(ids) < cut + continuation_tokens:
                            counts["filtered_continuation_length"] += len(tasks)
                            continue
                        input_ids = tuple(
                            token_pool.setdefault(token, token) for token in ids[:cut]
                        )
                        continuation = tuple(
                            token_pool.setdefault(token, token)
                            for token in ids[cut : cut + continuation_tokens]
                        )
                        examples_by_id[row["sample_id"]] = tuple(
                            PretrainExample(
                                f"{row['sample_id']}:{task}",
                                row["document_id"],
                                row["dedup_cluster"],
                                task,
                                input_ids,
                                input_ids if task == "ae" else continuation,
                            )
                            for task in tasks
                        )
                        counts["kept_source_indices"] += 1
                        counts["kept"] += len(tasks)
                        for task in tasks:
                            counts["kept_by_task"][task] += 1
                    except (ValueError, KeyError, TypeError) as error:
                        raise ValueError(f"{path}:{line_number}: {error}") from error
                del documents

    splits = {}
    for split, sample_ids in indices_by_split.items():
        examples = tuple(
            example for sample_id in sample_ids for example in examples_by_id.get(sample_id, ())
        )
        splits[split] = examples
        for name, field in (("input_tokens", "input_ids"), ("target_tokens", "target_ids")):
            lengths = [len(getattr(example, field)) for example in examples]
            statistics[split][name] = {
                "min": min(lengths) if lengths else None,
                "max": max(lengths) if lengths else None,
                "total": sum(lengths),
            }
    return splits, {
        "kind": "reconstruction_indices",
        "view": view,
        "index_directory": directories[view],
        "continuation_tokens": continuation_tokens,
        "input_token_interval": [min_input_tokens, max_input_tokens],
        "splits": statistics,
    }
