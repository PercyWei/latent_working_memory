"""Census FineWeb source lengths after conservative old-pool exact exclusion."""

import argparse
import itertools
import json
import time
from collections import Counter
from contextlib import closing
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from latent_working_memory.data_preparation.fineweb_qa.pipeline import _save_json
from latent_working_memory.data_preparation.pretrain.dedup import source_key
from latent_working_memory.data_preparation.pretrain.sources import parquet_records


THRESHOLDS = (2048, 4096, 6144, 8192, 10240, 12288, 16384, 18432, 24576, 30720, 32768, 40960)
BIN_EDGES = (0, 1024, 2048, 4096, 8192, 12288, 16384, 18432, 24576, 32768, 40960, 65536)


def summarize_lengths(histogram: Counter) -> dict:
    lengths = np.array(sorted(histogram), dtype=np.int64)
    frequencies = np.array([histogram[int(n)] for n in lengths], dtype=np.int64)
    cumulative = frequencies.cumsum()
    total = int(cumulative[-1])
    quantiles = {}
    for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.98, 0.99, 0.995, 0.999, 1):
        rank = (total - 1) * q
        lower, upper = int(np.floor(rank)), int(np.ceil(rank))
        left = int(lengths[np.searchsorted(cumulative, lower, side="right")])
        right = int(lengths[np.searchsorted(cumulative, upper, side="right")])
        quantiles[f"p{100 * q:g}"] = left + (right - left) * (rank - lower)
    return {
        "documents": total,
        "mean_chars": float(np.dot(lengths, frequencies) / total),
        "quantiles_chars": quantiles,
        "thresholds": [
            {
                "minimum_chars": threshold,
                "documents": int(frequencies[lengths >= threshold].sum()),
                "fraction": float(frequencies[lengths >= threshold].sum() / total),
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
    source = config["source"]
    files = [
        Path(source["raw_dir"]) / f"{i:03d}_00000.parquet" for i in range(source["file_count"])
    ]
    contract = {
        "source": source,
        "scope": "All rows in local sample-10BT, all splits, no window-count limit",
        "exclusion": "Entire old candidate pool, then exact ID, canonical URL or whitespace-normalized text matches",
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
    old_ids, old_urls, old_texts = set(), set(), set()
    old_count = 0
    with closing(parquet_records(files, source["data_seed"])) as records:
        for record, _ in itertools.islice(records, source["old_pool_documents"]):
            old_count += 1
            old_ids.add(record["id"])
            old_urls.add(source_key(record["url"]))
            old_texts.add(" ".join(record["text"].split()))
    if old_count != source["old_pool_documents"]:
        raise ValueError("source ended before reconstructing the old candidate pool")
    print(json.dumps({"stage": "old_pool_rebuilt", "documents": len(old_ids)}), flush=True)
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
                        if record["id"] in old_ids:
                            counts["excluded_id"] += 1
                        elif source_key(record["url"]) in old_urls:
                            counts["excluded_url"] += 1
                        elif " ".join(record["text"].split()) in old_texts:
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
    args = parser.parse_args()
    print(json.dumps(analyze(json.loads(args.config.read_text()), args.output), indent=2))


if __name__ == "__main__":
    main()
