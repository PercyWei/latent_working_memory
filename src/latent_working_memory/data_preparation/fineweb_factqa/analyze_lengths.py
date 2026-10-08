"""Census local FineWeb lengths after exact exclusion of explicitly used sources."""

import argparse
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from latent_working_memory.data_preparation.fineweb_factqa.storage import save_json as _save_json
from latent_working_memory.data_preparation.fineweb_factqa.config import load_config
from latent_working_memory.data_preparation.fineweb_source import (
    load_previous_sources,
    source_files,
)
from latent_working_memory.data_preparation.pretrain.dedup import source_key
from latent_working_memory.data_preparation.pretrain.sources import referenced_records


THRESHOLDS = (2048, 4096, 6144, 8192, 10240, 12288, 16384, 18432, 24576, 30720, 32768, 40960)
BIN_EDGES = (0, 1024, 2048, 4096, 8192, 12288, 16384, 18432, 24576, 32768, 40960, 65536)


def summarize_lengths(histogram: Counter) -> dict:
    lengths = np.array(sorted(histogram), dtype=np.int64)
    frequencies = np.array([histogram[int(n)] for n in lengths], dtype=np.int64)
    cumulative = frequencies.cumsum()
    total = int(cumulative[-1]) if len(cumulative) else 0
    quantiles = {}
    for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.98, 0.99, 0.995, 0.999, 1):
        if not total:
            break
        rank = (total - 1) * q
        lower, upper = int(np.floor(rank)), int(np.ceil(rank))
        left = int(lengths[np.searchsorted(cumulative, lower, side="right")])
        right = int(lengths[np.searchsorted(cumulative, upper, side="right")])
        quantiles[f"p{100 * q:g}"] = left + (right - left) * (rank - lower)
    return {
        "documents": total,
        "mean_chars": float(np.dot(lengths, frequencies) / total) if total else None,
        "quantiles_chars": quantiles,
        "thresholds": [
            {
                "minimum_chars": threshold,
                "documents": int(frequencies[lengths >= threshold].sum()),
                "fraction": float(frequencies[lengths >= threshold].sum() / total) if total else 0,
            }
            for threshold in THRESHOLDS
        ],
        "bins": [
            {
                "lower_inclusive": low,
                "upper_exclusive": high,
                "documents": int(
                    frequencies[
                        (lengths >= low) & (lengths < high if high is not None else True)
                    ].sum()
                ),
            }
            for low, high in zip(BIN_EDGES, (*BIN_EDGES[1:], None), strict=True)
        ],
    }


def analyze(config: dict, output: Path) -> dict:
    files = source_files(config["source_dir"])
    excluded_sources = load_previous_sources(config["previous_datasets"])
    contract = {
        "source_dir": config["source_dir"],
        "source_files": [str(path) for path in files],
        "previous_datasets": config["previous_datasets"],
        "excluded_sources": excluded_sources,
        "scope": "All rows in source_dir, all splits, no window-count limit",
        "exclusion": "Explicitly used sources: exact ID, canonical URL or whitespace-normalized text matches",
        "not_applied": [
            "near-duplicate clustering",
            "deduplication among remaining documents",
            "train-only filtering",
        ],
        "length": "Python len(original text), including whitespace; token estimate is chars / 4",
        "basic_min_stripped_chars": 64,
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists():
        if json.loads((output / "config.json").read_text()) != contract:
            raise ValueError("analysis inputs changed; use a new output directory")
    else:
        _save_json(output / "config.json", contract)
    used_ids, used_urls, used_texts = set(), set(), set()
    for record in referenced_records(excluded_sources):
        used_ids.add(record["id"])
        used_urls.add(source_key(record["url"]))
        used_texts.add(" ".join(record["text"].split()))
    total_counts, total_lengths = Counter(), Counter()
    started = time.perf_counter()
    for path in files:
        saved = output / "files" / f"{path.stem}.json"
        if saved.exists():
            result = json.loads(saved.read_text())
        else:
            counts, lengths = Counter(), Counter()
            with pq.ParquetFile(path) as parquet:
                for batch in parquet.iter_batches(batch_size=4096, columns=["id", "url", "text"]):
                    for record in batch.to_pylist():
                        counts["scanned"] += 1
                        if record["id"] in used_ids:
                            counts["excluded_id"] += 1
                        elif source_key(record["url"]) in used_urls:
                            counts["excluded_url"] += 1
                        elif " ".join(record["text"].split()) in used_texts:
                            counts["excluded_text"] += 1
                        else:
                            counts["remaining_before_basic"] += 1
                            if len(record["text"].strip()) < 64:
                                counts["basic_too_short"] += 1
                            else:
                                counts["remaining"] += 1
                                lengths[len(record["text"])] += 1
                    _save_json(
                        output / "progress.json",
                        {
                            "file": str(path),
                            "completed_files": len(list((output / "files").glob("*.json"))),
                            "current_file_counts": dict(counts),
                            "previous_files_counts": dict(total_counts),
                            "elapsed_this_run_seconds": time.perf_counter() - started,
                        },
                    )
            result = {
                "file": str(path),
                "counts": dict(counts),
                "length_counts": sorted(lengths.items()),
            }
            _save_json(saved, result)
        total_counts.update(result["counts"])
        total_lengths.update({length: n for length, n in result["length_counts"]})
        print(
            json.dumps(
                {
                    "file": str(path),
                    "counts": result["counts"],
                    "elapsed_this_run_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    assert total_counts["scanned"] == sum(
        total_counts[key]
        for key in ("excluded_id", "excluded_url", "excluded_text", "basic_too_short", "remaining")
    )
    result = {
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "counts": dict(total_counts),
        "distribution": summarize_lengths(total_lengths),
        "elapsed_this_run_seconds": time.perf_counter() - started,
    }
    _save_json(output / "summary.json", result)
    _save_json(output / "length-counts.json", sorted(total_lengths.items()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-datasets", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    config = load_config(args.config)
    config["previous_datasets"] = [str(path) for path in args.previous_datasets]
    print(json.dumps(analyze(config, args.output), indent=2))


if __name__ == "__main__":
    main()
