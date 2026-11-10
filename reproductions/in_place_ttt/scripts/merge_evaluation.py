"""Merge completed In-Place TTT row shards with original OpenCompass scoring."""

import argparse
import json
from pathlib import Path
import time

from evaluate import TASKS, build_summary, load_expected, timestamp, validate_prediction, write_json
from provenance import capture_source


PLAN_FIELDS = (
    "upstream_revision", "checkpoint", "checkpoint_config", "inference_mode",
    "mlp_chunk_size", "compact_tails", "ruler_data", "lengths", "samples_per_task",
    "tasks", "num_shards", "tokens_to_generate", "prompt", "model_wrapper",
    "generation_path", "model_kwargs", "tokenizer_kwargs", "generation_kwargs",
    "postprocessor", "max_seq_len", "pytorch_cuda_alloc_conf", "device", "sample_timing",
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
        raise ValueError("Shard tasks differ from the In-Place TTT evaluation protocol")
    runtimes = [json.loads((path / "runtime.json").read_text()) for path in directories]
    if any(runtime != runtimes[0] for runtime in runtimes):
        raise ValueError("Shard model, backend, dtype or dependency settings differ")
    ordered = sorted(zip(directories, plans, strict=True), key=lambda item: item[1]["shard_index"])
    sources = [
        {
            "shard_index": plan["shard_index"],
            "cuda_visible_devices": plan["cuda_visible_devices"],
            "plan_path": str(path / "plan.json"),
            "execution_path": str(path / "execution.json"),
            "runtime_path": str(path / "runtime.json"),
            "source": plan["source"],
        }
        for path, plan in ordered
    ]
    devices = list(dict.fromkeys(plan["cuda_visible_devices"] for _, plan in ordered))
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        **reference,
        "created_at": timestamp(),
        "source": capture_source(__file__, {}),
        "shard_index": None,
        "assigned_samples": len(reference["lengths"]) * len(TASKS) * reference["samples_per_task"],
        "cuda_visible_devices": ",".join(devices),
        "output_dir": str(output),
        "source_shards": sources,
    }
    write_json(output / "plan.json", plan)
    write_json(output / "runtime.json", runtimes[0])
    execution = {"started_at": timestamp(), "status": "running"}
    write_json(output / "execution.json", execution)
    started = time.perf_counter()
    try:
        expected = load_expected(plan)
        completed, executions = {}, []
        for path, shard_plan in ordered:
            shard_index = shard_plan["shard_index"]
            shard_expected = {
                key: sample
                for key, sample in expected.items()
                if key[2] % num_shards == shard_index
            }
            if shard_plan["assigned_samples"] != len(shard_expected):
                raise ValueError(f"Shard assigned sample count differs: {path}")
            shard_execution = json.loads((path / "execution.json").read_text())
            attempts = shard_execution["attempts"]
            if attempts[-1]["status"] != "completed":
                raise ValueError(f"Shard has not completed successfully: {path}")
            if any(attempt["execution_attempt"] != index for index, attempt in enumerate(attempts)):
                raise ValueError(f"Shard execution attempt identities differ: {path}")
            seen = set()
            with (path / "predictions.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    key = validate_prediction(record, shard_expected, shard_index)
                    if key in completed:
                        raise ValueError(f"Duplicate prediction key: {key}")
                    if not 0 <= record["execution_attempt"] < len(attempts):
                        raise ValueError(f"Unknown execution attempt: {key}")
                    seen.add(key)
                    completed[key] = record
            if seen != shard_expected.keys():
                raise ValueError(f"Shard predictions are incomplete: {path}")
            executions.append(shard_execution)
        summary = build_summary(plan, expected, completed, output)
        with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
            for key in expected:
                stream.write(json.dumps(completed[key], ensure_ascii=False, allow_nan=False) + "\n")
        execution.update(
            status="completed",
            worker_seconds_total=sum(item["elapsed_seconds"] for item in executions),
            peak_worker_memory_allocated_bytes=max(
                item["peak_memory_allocated_bytes"] for item in executions
            ),
            timing_interpretation="Worker times are cumulative process times; pool wall belongs to the launcher record.",
        )
        summary["worker_seconds_total"] = execution["worker_seconds_total"]
        write_json(output / "summary.json", summary)
    except BaseException as error:
        execution.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        execution.update(finished_at=timestamp(), merge_wall_seconds=time.perf_counter() - started)
        write_json(output / "execution.json", execution)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    merge(parser.parse_args())


if __name__ == "__main__":
    main()
