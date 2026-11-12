"""按完整原文的实际 token 数，构造单次压缩预训练样本。"""

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import random
import shutil
import tempfile
import time
from zoneinfo import ZoneInfo

from transformers import AutoTokenizer

from .cli import nonnegative_int, optional_positive_int, positive_int


def _scan(path, tokenizer):
    capacities, split = [], None
    started = last_report = time.monotonic()
    print(f"[prepare-pre] scanning source token capacities: {path}", flush=True)
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            sample = json.loads(line)
            if not capacities:
                split = sample["split"]
            capacity = len(tokenizer.encode(
                sample["text"] + sample["continuation"], add_special_tokens=False,
            ))
            capacities.append(capacity)
            now = time.monotonic()
            if line_number % 1000 == 0 or now - last_report >= 15:
                print(f"[prepare-pre {split}] scanned={line_number} seconds={now - started:.1f}",
                      flush=True)
                last_report = now
    print(f"[prepare-pre {split}] scanned={len(capacities)} seconds={time.monotonic() - started:.1f}",
          flush=True)
    return capacities, split


def _assign(capacities, buckets, seed, split, distribution, target_samples=None):
    """长目标先匹配长来源；计划仅保留 token 容量和目标，不缓存正文。"""
    sources = ([index for index, capacity in enumerate(capacities) if capacity >= buckets[0][0]]
               if buckets else [])
    target_samples = len(sources) if target_samples is None else target_samples
    if target_samples > len(sources):
        raise ValueError(
            f"{split}: target_samples={target_samples} exceeds the available sample count ({len(sources)})"
        )
    if not buckets:
        return [None] * len(capacities), [], 0
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
    sources.sort(key=lambda index: capacities[index], reverse=True)
    assignments, limited = [None] * len(capacities), 0
    for index, target in zip(sources, targets, strict=True):
        capacity = capacities[index]
        limited += capacity < target[0]
        length = min(capacity, target[0])
        lower, upper = next(bucket for bucket in buckets if bucket[0] <= length <= bucket[1])
        assignments[index] = (length, lower, upper)
    if limited:
        actual = Counter((item[1], item[2]) for item in assignments if item is not None)
        actual_counts = [actual[bucket] for bucket in buckets]
        print(f"[prepare-pre {split}] capacity_limited_targets={limited}; "
              f"target_bin_counts={quotas} actual_bin_counts={actual_counts}", flush=True)
    return assignments, quotas, limited


def _crop(full_text, target, lower, upper, rng, tokenizer):
    """按 token 偏移定位原文子串，再以最终正文的实际长度修正边界。"""
    offsets = tokenizer(
        full_text, add_special_tokens=False, return_offsets_mapping=True,
    )["offset_mapping"]
    first = rng.randrange(len(offsets) - target + 1)
    start, end = offsets[first][0], offsets[first + target - 1][1]
    visited = set()
    while start < end <= len(full_text) and (start, end) not in visited:
        visited.add((start, end))
        text = full_text[start:end]
        length = len(tokenizer.encode(text, add_special_tokens=False))
        if lower <= length <= upper:
            return text, length
        if length > upper:
            end -= 1
        elif end < len(full_text):
            end += 1
        elif start:
            start -= 1
        else:
            break
    raise ValueError(f"cannot crop the selected text to [{lower}, {upper}] tokens; target={target}")


def _write_split(path, destination, capacities, assignments, quotas, limited, buckets,
                 seed, distribution, split, target_samples, tokenizer):
    lengths, bucket_counts = Counter(), Counter()
    started = last_report = time.monotonic()
    eligible = sum(capacity >= buckets[0][0] for capacity in capacities) if buckets else 0
    selected = sum(assignment is not None for assignment in assignments)
    print(f"[prepare-pre {split}] writing {selected} cropped windows from {eligible} eligible "
          f"and {len(capacities)} total sources", flush=True)
    with path.open(encoding="utf-8") as source, destination.open("w", encoding="utf-8") as output:
        for line_number, (line, assignment) in enumerate(zip(source, assignments), 1):
            if assignment is None:
                continue
            sample = json.loads(line)
            full_text = sample["text"] + sample["continuation"]
            target, lower, upper = assignment
            rng = random.Random(f"{seed}:prepare:{split}:crop:{sample['trajectory_id']}")
            try:
                text, length = _crop(full_text, target, lower, upper, rng, tokenizer)
            except ValueError as error:
                error.add_note(f"{path}:{line_number}")
                raise
            output.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            lengths[length] += 1
            bucket_counts[next(bucket for bucket in buckets if bucket[0] <= length <= bucket[1])] += 1
            now = time.monotonic()
            if line_number % 1000 == 0 or now - last_report >= 15:
                print(f"[prepare-pre {split}] processed={line_number}/{len(capacities)} written={sum(lengths.values())} "
                      f"seconds={now - started:.1f}", flush=True)
                last_report = now
    counts = [bucket_counts[bucket] for bucket in buckets]
    statistics = {
        "source_windows": len(capacities), "samples": sum(lengths.values()),
        "eligible_source_windows": eligible,
        "requested_target_samples": target_samples, "resolved_target_samples": selected,
        "excluded_short_source_windows": len(capacities) - eligible,
        "unselected_source_windows": eligible - selected,
        "source_token_capacity_min": min(capacities, default=None),
        "source_token_capacity_max": max(capacities, default=None),
        "supported_min_tokens": buckets[0][0] if buckets else None,
        "supported_max_tokens": buckets[-1][1] if buckets else None,
        "length_bins": len(buckets), "length_distribution": distribution,
        "capacity_limited_targets": limited,
        "text_tokens_min": min(lengths) if lengths else None,
        "text_tokens_max": max(lengths) if lengths else None,
        "text_tokens_distribution": dict(sorted(lengths.items())),
        "buckets": [{"min_tokens": lower, "max_tokens": upper,
                     "target_samples": quota, "samples": bucket_counts[(lower, upper)]}
                    for (lower, upper), quota in zip(buckets, quotas, strict=True)],
    }
    print(f"[prepare-pre {split}] complete samples={statistics['samples']} "
          f"buckets={counts} seconds={time.monotonic() - started:.1f}", flush=True)
    return statistics


def prepare_data(source_file, min_tokens, max_tokens, output_file, model_path,
                 length_bins=10, length_distribution="uniform", target_samples=None, seed=42):
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    path = Path(source_file).expanduser().resolve()
    for name, value in (("min_tokens", min_tokens), ("max_tokens", max_tokens),
                        ("target_samples", target_samples)):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be null or a positive integer")
    if min_tokens is not None and max_tokens is not None and min_tokens > max_tokens:
        raise ValueError("min_tokens must not exceed max_tokens")
    if type(length_bins) is not int or length_bins < 1:
        raise ValueError("length_bins must be a positive integer")
    if length_distribution not in ("uniform", "random"):
        raise ValueError("length_distribution must be uniform or random")
    data_path = Path(output_file).expanduser().resolve()
    output = data_path.parent
    metadata_path = output / f"preparation-{data_path.stem}.json"
    if data_path.exists():
        raise FileExistsError(f"output file already exists: {data_path}")
    started = time.monotonic()
    model_path = str(Path(model_path).expanduser())
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    capacities, split = _scan(path, tokenizer)
    capacity_min = min(capacities, default=None)
    capacity_max = max(capacities, default=None)
    lower = max(capacity_min, min_tokens if min_tokens is not None else 1) if capacities else None
    upper = min(capacity_max, max_tokens) if capacities and max_tokens is not None else capacity_max
    intersects = bool(capacities) and lower <= upper
    eligible = sum(capacity >= lower for capacity in capacities) if intersects else 0
    supported = f"[{lower}, {upper}]" if intersects else "none"
    requested = json.dumps([min_tokens, max_tokens])
    print(f"[prepare-pre {split}] source_token_capacity=[{capacity_min}, {capacity_max}] "
          f"requested_tokens={requested} supported={supported} "
          f"eligible={eligible}/{len(capacities)} excluded_below_min={len(capacities) - eligible}", flush=True)
    if capacities and not intersects:
        raise ValueError(
            f"{split}: requested token range {requested} has no intersection with "
            f"source capacity range [{capacity_min}, {capacity_max}]"
        )
    if eligible < len(capacities):
        print(f"[prepare-pre {split}] excluding {len(capacities) - eligible} source windows "
              f"shorter than {lower} tokens", flush=True)
    if eligible and min_tokens is not None and lower > min_tokens:
        print(f"[prepare-pre {split}] raising minimum from {min_tokens} to {lower} "
              "tokens to match the source capacity range", flush=True)
    if eligible and max_tokens is not None and upper < max_tokens:
        print(f"[prepare-pre {split}] reducing maximum from {max_tokens} to {upper} "
              "tokens to match the available source", flush=True)
    if not eligible:
        buckets = []
    else:
        size = upper - lower + 1
        count = min(length_bins, size)
        buckets = [(lower + index * size // count, lower + (index + 1) * size // count - 1)
                   for index in range(count)]
    assignments, quotas, limited = _assign(
        capacities, buckets, seed, split, length_distribution, target_samples,
    )
    resolved_target = sum(assignment is not None for assignment in assignments)
    print(f"[prepare-pre {split}] requested_target_samples={json.dumps(target_samples)} "
          f"resolved_target_samples={resolved_target} length_distribution={length_distribution} "
          f"bins={len(buckets)}/{length_bins} bin_ranges={buckets} "
          f"target_bin_counts={quotas}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.preparing-", dir=output.parent))
    try:
        statistics = _write_split(
            path, staging / data_path.name, capacities, assignments, quotas, limited, buckets,
            seed, length_distribution, split, target_samples, tokenizer,
        )
        metadata = {
            "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
            "dataset": output.name, "output_file": str(data_path),
            "split": split,
            "source_file": str(path),
            "tokenizer": model_path,
            "seed": seed, "length_bins": length_bins, "length_distribution": length_distribution,
            "min_tokens": min_tokens, "max_tokens": max_tokens,
            "target_samples": target_samples, "resolved_target_samples": resolved_target,
            "length_selection": f"seeded without-replacement source selection; {length_distribution} target bin selection; "
                                "equal-width integer bins and uniform random integer lengths within each bin; "
                                "descending targets matched to descending source token capacities; "
                                "capacity-limited targets move to their actual bucket",
            "text_construction": "one contiguous source substring per selected window from text + continuation; "
                                 "token offsets locate the crop and final text is retokenized to enforce its token range",
            "statistics": statistics, "elapsed_seconds": time.monotonic() - started,
        }
        (staging / metadata_path.name).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        output.mkdir(parents=True, exist_ok=True)
        (staging / data_path.name).replace(data_path)
        (staging / metadata_path.name).replace(metadata_path)
    finally:
        shutil.rmtree(staging)
    print(f"preparation complete: {data_path}; seconds={time.monotonic() - started:.1f}", flush=True)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-data", required=True, help="source Multisegment JSONL file")
    parser.add_argument("--output-file", required=True, help="output JSONL file path")
    parser.add_argument("--model-path", required=True, help="tokenizer path used by training")
    parser.add_argument("--min-tokens", type=optional_positive_int, default=None,
                        help="minimum actual sample tokens, inclusive; default null uses the source boundary")
    parser.add_argument("--max-tokens", type=optional_positive_int, default=None,
                        help="maximum actual sample tokens, inclusive; default null uses the source boundary")
    parser.add_argument("--length-bins", type=positive_int, default=10)
    parser.add_argument("--length-distribution", choices=("uniform", "random"), default="uniform")
    parser.add_argument("--target-samples", type=optional_positive_int, default=None,
                        help="number of samples to generate; null generates all samples meeting the length requirements")
    parser.add_argument("--seed", type=nonnegative_int, default=42,
                        help="random seed for source selection, length allocation and cropping, default 42")
    args = parser.parse_args(argv)
    return prepare_data(
        args.prepare_data, args.min_tokens, args.max_tokens, args.output_file, args.model_path,
        args.length_bins, args.length_distribution, args.target_samples, args.seed,
    )


if __name__ == "__main__":
    main()
