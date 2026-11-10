"""Merge completed LaCT row shards and rescore their original predictions."""

import argparse
import json
from pathlib import Path
import time

from evaluate import (
    TASKS,
    build_summary,
    read_samples,
    timestamp,
    validate_prediction,
    write_json,
)
from provenance import capture_source


PLAN_FIELDS = (
    "method", "inference_mode", "checkpoint", "checkpoint_config", "ruler_data",
    "lengths", "tasks", "samples_per_task_length", "prompt", "generation",
    "parameter_dtype", "compute_autocast_dtype", "upstream_revision", "scoring",
    "loaded_model", "runtime", "num_shards", "batch_size", "batching",
)


def merge(args):
    directories = [path.expanduser().resolve() for path in args.shard_dirs]
    plans = [json.loads((path / "plan.json").read_text()) for path in directories]
    reference = plans[0]
    num_shards = reference["num_shards"]
    indices = [plan["shard_index"] for plan in plans]
    if len(plans) != num_shards or sorted(indices) != list(range(num_shards)):
        raise ValueError("Provide exactly one completed directory for every shard index")
    for plan in plans:
        for field in PLAN_FIELDS:
            if plan[field] != reference[field]:
                raise ValueError(f"Shard plans differ: {field}")
    if reference["tasks"] != list(TASKS):
        raise ValueError("Shard tasks differ from the LaCT evaluation protocol")
    samples = {
        (length, task): read_samples(
            Path(reference["ruler_data"]) / str(length) / task / "test.jsonl"
        )
        for length in reference["lengths"]
        for task in TASKS
    }
    expected = {
        f"ruler/{length}/{task}/{position}": (length, task, position, row)
        for (length, task), rows in samples.items()
        for position, row in enumerate(rows)
    }
    sources = [
        {
            "shard_index": plan["shard_index"],
            "plan_path": str(path / "plan.json"),
            "execution_path": str(path / "execution.json"),
            "source": plan["source"],
            "cuda_visible_devices": plan["cuda_visible_devices"],
        }
        for path, plan in sorted(
            zip(directories, plans, strict=True), key=lambda pair: pair[1]["shard_index"]
        )
    ]
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        **reference,
        "created_at": timestamp(),
        "source": capture_source(__file__, {}),
        "shard_index": None,
        "assigned_samples": len(expected),
        "source_shards": sources,
        "cuda_visible_devices": ",".join(dict.fromkeys(
            entry["cuda_visible_devices"] for entry in sources
        )),
    }
    write_json(output / "plan.json", plan)
    started = time.perf_counter()
    records, executions = {}, []
    execution = {"started_at": timestamp(), "status": "running"}
    write_json(output / "execution.json", execution)
    try:
        for path, shard_plan in zip(directories, plans, strict=True):
            shard_index = shard_plan["shard_index"]
            shard_expected = {
                sample_id: sample
                for sample_id, sample in expected.items()
                if sample[2] % num_shards == shard_index
            }
            shard_execution = json.loads((path / "execution.json").read_text())
            attempts = shard_execution["attempts"]
            if attempts[-1]["status"] != "completed":
                raise ValueError(f"Shard has not completed successfully: {path}")
            seen = set()
            with (path / "predictions.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    validate_prediction(record, shard_expected, shard_index)
                    sample_id = record["id"]
                    if sample_id in records:
                        raise ValueError(f"Duplicate prediction across shards: {sample_id}")
                    attempt = record["execution_attempt"]
                    if not 0 <= attempt < len(attempts):
                        raise ValueError(f"Unknown execution attempt: {sample_id}")
                    if attempts[attempt]["execution_attempt"] != attempt:
                        raise ValueError(f"Execution attempt identity differs: {sample_id}")
                    seen.add(sample_id)
                    records[sample_id] = record
            if seen != shard_expected.keys():
                raise ValueError(f"Shard predictions are incomplete: {path}")
            executions.append(shard_execution)
        summary = build_summary(plan, samples, records, output)
        with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
            for sample_id in expected:
                stream.write(
                    json.dumps(records[sample_id], ensure_ascii=False, allow_nan=False) + "\n"
                )
        execution.update(
            status="completed",
            worker_seconds_total=sum(item["seconds"] for item in executions),
            peak_worker_cuda_memory_allocated_bytes=max(
                item["peak_cuda_memory_allocated_bytes"] for item in executions
            ),
            timing_interpretation=(
                "Worker times are cumulative process times, not parallel wall time. "
                "Whole-GPU peak memory and pool wall time belong to the launcher record."
            ),
        )
        summary["worker_seconds_total"] = execution["worker_seconds_total"]
        write_json(output / "summary.json", summary)
    except BaseException as error:
        execution.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        execution.update(
            finished_at=timestamp(), merge_wall_seconds=time.perf_counter() - started
        )
        write_json(output / "execution.json", execution)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    merge(parser.parse_args())


if __name__ == "__main__":
    main()
