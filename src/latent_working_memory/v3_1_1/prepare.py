"""按字符数估算长度，从既有正文窗口准备单次压缩预训练数据。"""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import random
import shutil
import tempfile
import time
from zoneinfo import ZoneInfo

from latent_working_memory.data_preparation.fineweb_multisegment.records import MultisegmentSample
from latent_working_memory.data_preparation.segmentation import TOKEN_ESTIMATION_RULE
from .cli import optional_positive_int, positive_int
from .records import PreparedPretrainSample


CHARACTERS_PER_TOKEN = 4


def _sample(line, path, split, line_number):
    try:
        sample = MultisegmentSample(**json.loads(line))
        if sample.split != split:
            raise ValueError(f"record split must be {split}")
        return sample
    except (ValueError, TypeError, KeyError) as error:
        error.add_note(f"{path}:{line_number}")
        raise


def _scan(path, split):
    rows, seen = [], set()
    started = last_report = time.monotonic()
    print(f"[prepare {split}] scanning source estimated token capacities: {path}", flush=True)
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            sample = _sample(line, path, split, line_number)
            if sample.trajectory_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate trajectory_id")
            seen.add(sample.trajectory_id)
            capacity = (len(sample.text) + len(sample.continuation)) // CHARACTERS_PER_TOKEN
            rows.append((sample.trajectory_id, capacity))
            now = time.monotonic()
            if line_number % 1000 == 0 or now - last_report >= 15:
                print(f"[prepare {split}] scanned={line_number} seconds={now - started:.1f}",
                      flush=True)
                last_report = now
    print(f"[prepare {split}] scanned={len(rows)} seconds={time.monotonic() - started:.1f}",
          flush=True)
    return rows


def _assign(rows, buckets, seed, split, distribution, target_samples=None):
    """长目标先匹配长来源；计划仅保留 ID、估算容量和目标，不缓存正文。"""
    if target_samples is not None and (type(target_samples) is not int or target_samples < 1):
        raise ValueError("target_samples must be null or a positive integer")
    sources = ([index for index, (_, capacity) in enumerate(rows) if capacity >= buckets[0][0]]
               if buckets else [])
    target_samples = len(sources) if target_samples is None else target_samples
    if target_samples > len(sources):
        raise ValueError(
            f"{split}: target_samples={target_samples} exceeds the available sample count ({len(sources)})"
        )
    if not buckets:
        return [None] * len(rows), [], 0
    rng = random.Random(f"{seed}:prepare:{split}:lengths")
    if target_samples < len(sources):
        source_rng = random.Random(f"{seed}:prepare:{split}:sources")
        sources = source_rng.sample(sources, target_samples)
    if distribution == "uniform":
        count, remainder = divmod(len(sources), len(buckets))
        extra = set(rng.sample(range(len(buckets)), remainder))
        quotas = [count + (index in extra) for index in range(len(buckets))]
        choices = [index for index, quota in enumerate(quotas) for _ in range(quota)]
    else:
        choices = [rng.randrange(len(buckets)) for _ in sources]
        counts = Counter(choices)
        quotas = [counts[index] for index in range(len(buckets))]
    targets = []
    for index in choices:
        lower, upper = buckets[index]
        targets.append((rng.randint(lower, upper), lower, upper))
    rng.shuffle(targets)
    targets.sort(key=lambda item: item[0], reverse=True)
    rng.shuffle(sources)
    sources.sort(key=lambda index: rows[index][1], reverse=True)
    assignments, limited = [None] * len(rows), 0
    for index, target in zip(sources, targets, strict=True):
        capacity = rows[index][1]
        limited += capacity < target[0]
        length = min(capacity, target[0])
        lower, upper = next(bucket for bucket in buckets if bucket[0] <= length <= bucket[1])
        assignments[index] = (length, lower, upper)
    if limited:
        actual = Counter((item[1], item[2]) for item in assignments if item is not None)
        actual_counts = [actual[bucket] for bucket in buckets]
        print(f"[prepare {split}] capacity_limited_targets={limited}; "
              f"target_bin_counts={quotas} actual_bin_counts={actual_counts}", flush=True)
    return assignments, quotas, limited


def _write_split(path, destination, rows, assignments, quotas, limited, buckets,
                 seed, distribution, split, target_samples):
    lengths, bucket_counts = Counter(), Counter()
    documents, trajectories = set(), set()
    started = last_report = time.monotonic()
    eligible = sum(capacity >= buckets[0][0] for _, capacity in rows) if buckets else 0
    selected = sum(assignment is not None for assignment in assignments)
    print(f"[prepare {split}] writing {selected} cropped windows from {eligible} eligible "
          f"and {len(rows)} total sources", flush=True)
    with path.open(encoding="utf-8") as source, destination.open("w", encoding="utf-8") as output:
        for line_number, (line, row, assignment) in enumerate(
            zip(source, rows, assignments, strict=True), 1,
        ):
            sample = _sample(line, path, split, line_number)
            full_text = sample.text + sample.continuation
            identifier, capacity = row
            if (sample.trajectory_id != identifier
                or len(full_text) // CHARACTERS_PER_TOKEN != capacity):
                raise ValueError(f"{path}:{line_number}: source changed between preparation passes")
            if assignment is None:
                continue
            target, lower, upper = assignment
            characters = target * CHARACTERS_PER_TOKEN
            rng = random.Random(f"{seed}:prepare:{split}:crop:{sample.trajectory_id}")
            start = rng.randrange(len(full_text) - characters + 1)
            end = start + characters
            text = full_text[start:end]
            absolute_start, absolute_end = (sample.window_char_span[0] + value
                                            for value in (start, end))
            prepared = PreparedPretrainSample(
                trajectory_id=f"{sample.document_id}:{absolute_start}:{absolute_end}",
                document_id=sample.document_id, dedup_cluster=sample.dedup_cluster,
                split=split, source=sample.source,
                window_char_span=[absolute_start, absolute_end], text=text,
                estimated_tokens=len(text) / CHARACTERS_PER_TOKEN,
            )
            if prepared.trajectory_id in trajectories:
                raise ValueError(f"{split}: cropped trajectory ID is duplicated: {prepared.trajectory_id}")
            trajectories.add(prepared.trajectory_id)
            documents.add(sample.document_id)
            output.write(json.dumps(asdict(prepared), ensure_ascii=False) + "\n")
            lengths[target] += 1
            bucket_counts[(lower, upper)] += 1
            now = time.monotonic()
            if line_number % 1000 == 0 or now - last_report >= 15:
                print(f"[prepare {split}] processed={line_number}/{len(rows)} written={sum(lengths.values())} "
                      f"seconds={now - started:.1f}", flush=True)
                last_report = now
    counts = [bucket_counts[bucket] for bucket in buckets]
    statistics = {
        "source_windows": len(rows), "samples": sum(lengths.values()),
        "eligible_source_windows": eligible,
        "requested_target_samples": target_samples, "resolved_target_samples": selected,
        "excluded_short_source_windows": len(rows) - eligible,
        "unselected_source_windows": eligible - selected,
        "source_estimated_token_capacity_min": min((capacity for _, capacity in rows), default=None),
        "source_estimated_token_capacity_max": max((capacity for _, capacity in rows), default=None),
        "supported_min_estimated_tokens": buckets[0][0] if buckets else None,
        "supported_max_estimated_tokens": buckets[-1][1] if buckets else None,
        "length_bins": len(buckets), "length_distribution": distribution,
        "documents": len(documents),
        "capacity_limited_targets": limited,
        "text_estimated_tokens_min": min(lengths) if lengths else None,
        "text_estimated_tokens_max": max(lengths) if lengths else None,
        "text_estimated_tokens_distribution": dict(sorted(lengths.items())),
        "buckets": [{"min_estimated_tokens": lower, "max_estimated_tokens": upper,
                     "target_samples": quota, "samples": bucket_counts[(lower, upper)]}
                    for (lower, upper), quota in zip(buckets, quotas, strict=True)],
    }
    print(f"[prepare {split}] complete samples={statistics['samples']} documents={len(documents)} "
          f"buckets={counts} seconds={time.monotonic() - started:.1f}", flush=True)
    return statistics


def prepare_data(source_file, min_tokens, max_tokens, output_file,
                 length_bins=10, length_distribution="uniform", target_samples=None):
    seed = 42
    path = Path(source_file).expanduser().resolve()
    if path.name not in ("train.jsonl", "dev.jsonl"):
        raise ValueError("prepare_data must point to train.jsonl or dev.jsonl")
    split = path.stem
    for name, value in (("min_tokens", min_tokens), ("max_tokens", max_tokens)):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be null or a positive integer")
    if min_tokens is not None and max_tokens is not None and min_tokens > max_tokens:
        raise ValueError("min_tokens must not exceed max_tokens")
    if type(length_bins) is not int or length_bins < 1:
        raise ValueError("length_bins must be a positive integer")
    if length_distribution not in ("uniform", "random"):
        raise ValueError("length_distribution must be uniform or random")
    source = path.parent
    data_path = Path(output_file).expanduser().resolve()
    if data_path.name != path.name:
        raise ValueError(f"output file must be named {path.name} to preserve the source split")
    output = data_path.parent
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("source and output must be separate, non-overlapping directories")
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"output must be a directory: {output}")
    metadata_path = output / f"preparation-{split}.json"
    if data_path.exists():
        raise FileExistsError(f"output file already exists: {data_path}")
    started = time.monotonic()
    rows = _scan(path, split)
    capacity_min = min((capacity for _, capacity in rows), default=None)
    capacity_max = max((capacity for _, capacity in rows), default=None)
    if not rows and split == "train":
        raise ValueError("training source must contain at least one window")
    lower = max(capacity_min, min_tokens if min_tokens is not None else 1) if rows else None
    upper = min(capacity_max, max_tokens) if rows and max_tokens is not None else capacity_max
    intersects = bool(rows) and lower <= upper
    eligible = sum(capacity >= lower for _, capacity in rows) if intersects else 0
    supported = f"[{lower}, {upper}]" if intersects else "none"
    requested = json.dumps([min_tokens, max_tokens])
    print(f"[prepare {split}] source_estimated_capacity=[{capacity_min}, {capacity_max}] "
          f"requested_estimated_tokens={requested} supported={supported} "
          f"eligible={eligible}/{len(rows)} excluded_below_min={len(rows) - eligible}", flush=True)
    if rows and not intersects:
        raise ValueError(
            f"{split}: requested estimated token range {requested} has no intersection with "
            f"source capacity range [{capacity_min}, {capacity_max}]"
        )
    if eligible < len(rows):
        print(f"[prepare {split}] excluding {len(rows) - eligible} source windows "
              f"shorter than {lower} estimated tokens", flush=True)
    if eligible and min_tokens is not None and lower > min_tokens:
        print(f"[prepare {split}] raising minimum from {min_tokens} to {lower} "
              "estimated tokens to match the source capacity range", flush=True)
    if eligible and max_tokens is not None and upper < max_tokens:
        print(f"[prepare {split}] reducing maximum from {max_tokens} to {upper} "
              "estimated tokens to match the available source", flush=True)
    if not eligible:
        buckets = []
    else:
        size = upper - lower + 1
        count = min(length_bins, size)
        buckets = [(lower + index * size // count, lower + (index + 1) * size // count - 1)
                   for index in range(count)]
    assignments, quotas, limited = _assign(
        rows, buckets, seed, split, length_distribution, target_samples,
    )
    resolved_target = sum(assignment is not None for assignment in assignments)
    print(f"[prepare {split}] requested_target_samples={json.dumps(target_samples)} "
          f"resolved_target_samples={resolved_target} length_distribution={length_distribution} "
          f"bins={len(buckets)}/{length_bins} bin_ranges={buckets} "
          f"target_bin_counts={quotas}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.preparing-", dir=output.parent))
    try:
        statistics = _write_split(
            path, staging / data_path.name, rows, assignments, quotas, limited, buckets,
            seed, length_distribution, split, target_samples,
        )
        metadata = {
            "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
            "dataset": output.name, "output_file": str(data_path),
            "split": split,
            "source_file": str(path),
            "estimated_tokens_rule": TOKEN_ESTIMATION_RULE,
            "characters_per_token": CHARACTERS_PER_TOKEN,
            "seed": seed, "length_bins": length_bins, "length_distribution": length_distribution,
            "min_estimated_tokens": min_tokens, "max_estimated_tokens": max_tokens,
            "target_samples": target_samples, "resolved_target_samples": resolved_target,
            "length_selection": f"seeded without-replacement source selection; {length_distribution} target bin selection; "
                                "equal-width integer bins and uniform random integer lengths within each bin; "
                                "descending targets matched to descending source estimated capacities; "
                                "capacity-limited targets move to their actual bucket",
            "text_construction": "one raw character crop per selected source window from text + continuation; "
                                 "four characters per estimated token; output text is a complete sample without reserved continuation",
            "statistics": statistics, "elapsed_seconds": time.monotonic() - started,
        }
        (staging / metadata_path.name).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        output.mkdir(parents=True, exist_ok=True)
        if data_path.exists():
            raise FileExistsError(f"output file appeared during preparation: {data_path}")
        (staging / data_path.name).replace(data_path)
        (staging / metadata_path.name).replace(metadata_path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"preparation complete: {data_path}; seconds={time.monotonic() - started:.1f}", flush=True)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-data", required=True, help="source train.jsonl or dev.jsonl file")
    parser.add_argument("--output-file", required=True, help="output JSONL file with the same name as the source")
    parser.add_argument("--min-tokens", type=optional_positive_int, required=True,
                        help="minimum estimated sample tokens, inclusive; null uses the source boundary")
    parser.add_argument("--max-tokens", type=optional_positive_int, required=True,
                        help="maximum estimated sample tokens, inclusive; null uses the source boundary")
    parser.add_argument("--length-bins", type=positive_int, default=10)
    parser.add_argument("--length-distribution", choices=("uniform", "random"), default="uniform")
    parser.add_argument("--target-samples", type=optional_positive_int, default=None,
                        help="number of samples to generate; null generates all samples meeting the length requirements")
    args = parser.parse_args(argv)
    return prepare_data(
        args.prepare_data, args.min_tokens, args.max_tokens, args.output_file,
        args.length_bins, args.length_distribution, args.target_samples,
    )


if __name__ == "__main__":
    main()
