"""仅在 CPU 上按全局样本 ID 合并 LaCT 原始预测并重新计分。"""

import argparse
import json
from pathlib import Path
import time

from provenance import capture_source
from records import (
    BATCHING, build_summary, load_samples, sample_index, timestamp, write_json,
)
from scheduling import collect_records, existing_protocol


def merge_records(directory, protocol, output, directories=None):
    samples = load_samples(protocol)
    expected = sample_index(samples)
    records, sources = collect_records(directory, protocol, expected, directories)
    if records.keys() != expected.keys():
        raise ValueError(f"Global predictions are incomplete: {len(records)}/{len(expected)} samples")
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        **protocol, "created_at": timestamp(), "source": capture_source(__file__, {}),
        "num_shards": None, "shard_index": None, "assigned_samples": len(expected),
        "batch_size": None, "batching": BATCHING, "source_shards": sources,
    }
    write_json(output / "plan.json", plan)
    started = time.perf_counter()
    execution = {"started_at": timestamp(), "status": "running"}
    write_json(output / "execution.json", execution)
    try:
        summary = build_summary(plan, samples, records, output)
        with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
            for sample_id in expected:
                stream.write(json.dumps(records[sample_id], ensure_ascii=False, allow_nan=False) + "\n")
        execution.update(
            status="completed", worker_seconds_total=sum(item["seconds"] for item in sources),
            peak_worker_cuda_memory_allocated_bytes=max(
                item["peak_cuda_memory_allocated_bytes"] for item in sources
            ),
            timing_interpretation="Worker times are cumulative process times, not parallel wall time.",
        )
        summary["worker_seconds_total"] = execution["worker_seconds_total"]
        write_json(output / "summary.json", summary)
    except BaseException as error:
        execution.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        execution.update(finished_at=timestamp(), merge_wall_seconds=time.perf_counter() - started)
        write_json(output / "execution.json", execution)
    return plan, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--eval-dir", type=Path)
    inputs.add_argument("--shard-dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.eval_dir is not None:
        directory = args.eval_dir.expanduser().resolve()
        protocol = existing_protocol(directory)
        directories = None
    else:
        directories = [path.expanduser().resolve() for path in args.shard_dirs]
        directory = directories[0]
        protocol = existing_protocol(directory)
    merge_records(directory, protocol, args.output_dir.expanduser().resolve(), directories)


if __name__ == "__main__":
    main()
