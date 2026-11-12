"""保留 FactQA 完整正文，按证据位置均匀选取后训练问答。"""

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

from tqdm import tqdm
from transformers import AutoTokenizer

from .cli import nonnegative_int, optional_positive_int


def _sample(line, path, line_number):
    """仅读取完整正文、段布局和 task QA，不复验上游构造元信息。"""
    try:
        sample = json.loads(line)
        layout = {segment["segment_id"]: segment["char_span"] for segment in sample["segments"]}
        candidates = [(sum(qa["evidence_char_span"]), {
            "question": qa["question"], "answer": qa["answer"],
            "segment_id": qa["segment_id"], "segment_char_span": layout[qa["segment_id"]],
        }) for qa in sample["qas"] if qa["role"] == "task"]
        return sample, candidates
    except (ValueError, TypeError, KeyError) as error:
        error.add_note(f"{path}:{line_number}")
        raise


def _select_qas(candidates, text_length, qas_per_sample, rng):
    count = len(candidates) if qas_per_sample is None else min(qas_per_sample, len(candidates))
    remaining = list(candidates)
    selected = []
    for index in range(count):
        # 证据中点与各份中心的距离同乘 2M，避免浮点误差改变同距判断。
        distances = [abs(item[0] * count - text_length * (2 * index + 1))
                     for item in remaining]
        minimum = min(distances)
        nearest = rng.choice([position for position, distance in enumerate(distances)
                              if distance == minimum])
        selected.append(remaining.pop(nearest))
    selected.sort(key=lambda item: item[0])
    return [item[1] for item in selected]


def _scan(path, min_tokens, max_tokens, tokenizer):
    rows, split = [], None
    print(f"[prepare-post] scanning complete source texts: {path}", flush=True)
    with path.open("rb") as stream:
        total = sum(1 for _ in stream)
    with (
        path.open(encoding="utf-8") as stream,
        tqdm(total=total, desc="[prepare-post] scan", unit="samples",
             mininterval=1, dynamic_ncols=True) as progress,
    ):
        for line_number, line in enumerate(stream, 1):
            sample, candidates = _sample(line, path, line_number)
            if not rows:
                split = sample["split"]
            length = len(tokenizer.encode(sample["text"], add_special_tokens=False))
            length_ok = ((min_tokens is None or length >= min_tokens)
                         and (max_tokens is None or length <= max_tokens))
            rows.append((length, len(candidates), length_ok))
            progress.update(1)
    return rows, split


def _write_split(path, destination, rows, selected, split, qas_per_sample, seed):
    lengths, qa_counts = [], Counter()
    with (
        path.open(encoding="utf-8") as source,
        destination.open("w", encoding="utf-8") as output,
        tqdm(total=len(selected), desc=f"[prepare-post {split}] build", unit="samples",
             mininterval=1, dynamic_ncols=True) as progress,
    ):
        for index, line in enumerate(source):
            if index not in selected:
                continue
            sample, candidates = _sample(line, path, index + 1)
            text = sample["text"]
            rng = random.Random(f"{seed}:prepare-post:{split}:{sample['trajectory_id']}:qas")
            qas = _select_qas(candidates, len(text), qas_per_sample, rng)
            output.write(json.dumps({"text": text, "qas": qas}, ensure_ascii=False) + "\n")
            lengths.append(rows[index][0])
            qa_counts[len(qas)] += 1
            progress.update(1)
    return {
        "samples": len(lengths),
        "qas": sum(count * samples for count, samples in qa_counts.items()),
        "qas_per_sample_distribution": dict(sorted(qa_counts.items())),
        "text_tokens_min": min(lengths, default=None),
        "text_tokens_max": max(lengths, default=None),
    }


def prepare_data(source_file, min_tokens, max_tokens, output_file, model_path,
                 target_samples=None, qas_per_sample=10, seed=42):
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    for name, value in (("min_tokens", min_tokens), ("max_tokens", max_tokens),
                        ("target_samples", target_samples), ("qas_per_sample", qas_per_sample)):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be null or a positive integer")
    if min_tokens is not None and max_tokens is not None and min_tokens > max_tokens:
        raise ValueError("min_tokens must not exceed max_tokens")
    path = Path(source_file).expanduser().resolve()
    data_path = Path(output_file).expanduser().resolve()
    if data_path.exists():
        raise FileExistsError(f"output file already exists: {data_path}")
    output = data_path.parent
    metadata_path = output / f"preparation-{data_path.stem}.json"
    started = time.monotonic()
    model_path = str(Path(model_path).expanduser())
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    rows, split = _scan(path, min_tokens, max_tokens, tokenizer)
    eligible = [index for index, row in enumerate(rows) if row[2] and row[1]]
    resolved_target = len(eligible) if target_samples is None else target_samples
    if resolved_target > len(eligible):
        raise ValueError(f"{split}: target_samples={resolved_target} exceeds the available "
                         f"sample count ({len(eligible)})")
    selected = set(eligible if resolved_target == len(eligible) else random.Random(
        f"{seed}:prepare-post:{split}:sources",
    ).sample(eligible, resolved_target))
    print(f"[prepare-post {split}] source_samples={len(rows)} eligible_samples={len(eligible)} "
          f"requested_tokens={json.dumps([min_tokens, max_tokens])} "
          f"requested_target_samples={json.dumps(target_samples)} "
          f"selected_samples={resolved_target} qas_per_sample={json.dumps(qas_per_sample)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.preparing-", dir=output.parent))
    try:
        statistics = _write_split(
            path, staging / data_path.name, rows, selected, split, qas_per_sample, seed,
        )
        statistics.update({
            "source_samples": len(rows), "eligible_samples": len(eligible),
            "excluded_by_length": sum(not row[2] for row in rows),
            "excluded_without_task_qas": sum(row[2] and not row[1] for row in rows),
            "unselected_samples": len(eligible) - resolved_target,
            "source_tokens_min": min((row[0] for row in rows), default=None),
            "source_tokens_max": max((row[0] for row in rows), default=None),
        })
        metadata = {
            "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
            "dataset": output.name, "output_file": str(data_path), "source_file": str(path),
            "split": split, "tokenizer": model_path, "seed": seed,
            "min_tokens": min_tokens, "max_tokens": max_tokens,
            "target_samples": target_samples, "resolved_target_samples": resolved_target,
            "qas_per_sample": qas_per_sample, "qa_role": "task",
            "source_selection": "seeded without-replacement sampling of eligible source rows",
            "qa_selection": "divide text into equal parts; select the nearest unused evidence midpoint "
                            "to each part center; seeded random tie-breaking; "
                            "output sorted by evidence midpoint",
            "text_construction": "complete source text retained without changes",
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
    print(f"preparation complete: {data_path}; samples={statistics['samples']} "
          f"qas={statistics['qas']}; seconds={time.monotonic() - started:.1f}", flush=True)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-data", required=True, help="source FactQA JSONL file")
    parser.add_argument("--output-file", required=True, help="output JSONL file path")
    parser.add_argument("--model-path", required=True, help="tokenizer path used by training")
    parser.add_argument("--min-tokens", type=optional_positive_int, default=None,
                        help="minimum actual body tokens, inclusive; default null sets no lower bound")
    parser.add_argument("--max-tokens", type=optional_positive_int, default=None,
                        help="maximum actual body tokens, inclusive; default null sets no upper bound")
    parser.add_argument("--target-samples", type=optional_positive_int, default=None,
                        help="sample count; null retains all eligible source rows")
    parser.add_argument("--qas-per-sample", type=optional_positive_int, default=10,
                        help="maximum task QAs per sample, default 10; null retains all task QAs")
    parser.add_argument("--seed", type=nonnegative_int, default=42,
                        help="random seed for source selection and QA tie-breaking, default 42")
    args = parser.parse_args(argv)
    return prepare_data(
        args.prepare_data, args.min_tokens, args.max_tokens, args.output_file, args.model_path,
        args.target_samples, args.qas_per_sample, args.seed,
    )


if __name__ == "__main__":
    main()
