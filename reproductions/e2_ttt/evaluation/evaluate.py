"""执行统一 launcher 分配的 E²-TTT 样本，保留原生生成、缓存和评分。"""

import argparse
import json
import logging
import os
from pathlib import Path
import random
import time

import e2_ttt  # noqa: F401 -- registers the original checkpoint model types
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import handle_non_serializable, set_torch_seed
import numpy as np
from optimized import optimize_model
import torch

from batch_harness import LengthSortedHFLM
from evaluation_runtime import WorkerProgress
from evaluation_state import (
    RULER_DATA, SQUAD_DATA, SWDE_DATA, benchmark_protocol, runtime_versions, sample_index,
    source_record, task_specs, timestamp, validate_sample_ids, write_json,
)
from local_tasks import LocalRuler, LocalSQuAD, LocalSWDE
from progress_caching import ProgressCachingLM


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-file", type=Path, required=True)
    parser.add_argument("--sample-ids-file", type=Path, required=True)
    parser.add_argument("--worker-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()
    for name in ("protocol_file", "sample_ids_file", "worker_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1"):
        parser.error("Use one assigned physical GPU: CUDA_VISIBLE_DEVICES=0 or 1")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or "LOCAL_RANK" in os.environ or "RANK" in os.environ:
        parser.error("Each evaluation worker must be a single process")
    return args


def evaluate_tasks(lm, tasks):
    random.seed(0)
    np.random.seed(1234)
    set_torch_seed(1234)
    for task in tasks:
        task.set_fewshot_seed(1234)
    return evaluator.evaluate(
        lm=lm, task_dict={task.config.task: task for task in tasks},
        log_samples=True, apply_chat_template=False, cache_requests=False, bootstrap_iters=0,
    )


def save_result(result, name, native_dir, predictions, specs):
    samples = result.pop("samples")
    serialized = json.loads(json.dumps(result, ensure_ascii=False, default=handle_non_serializable))
    write_json(native_dir / f"{name}.json", serialized)
    count = 0
    for task, rows in samples.items():
        for row in rows:
            index = row["doc"]["source_doc_id"]
            predictions.write(json.dumps(
                {**row, "id": f"{specs[task]['prefix']}/{index}", "task": task, "source_doc_id": index},
                ensure_ascii=False, default=handle_non_serializable,
            ) + "\n")
            count += 1
    predictions.flush()
    return count



def execute(args):
    progress = WorkerProgress(args.worker_dir, os.environ["CUDA_VISIBLE_DEVICES"])
    protocol = json.loads(args.protocol_file.read_text())["protocol"]
    assignments = json.loads(args.sample_ids_file.read_text())
    plan_path = args.worker_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    if (not isinstance(assignments, dict) or not assignments
            or not set(assignments) <= set(protocol["benchmarks"])):
        raise ValueError("Workers require pending IDs grouped by planned benchmark")
    if plan["protocol"] != protocol or plan["batch_size"] != args.batch_size or set(plan["outputs"]) != set(assignments):
        raise ValueError("Worker routing differs from its persisted plan")
    batch_size_by_length = plan["batch_size_by_length"]
    mode = protocol["inference_mode"]
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    if protocol["versions"] != runtime_versions():
        raise ValueError("Worker software versions differ from the frozen evaluation protocol")
    source = source_record(__file__, mode)
    plan["source"] = source
    write_json(plan_path, plan)
    for name, ids in assignments.items():
        child_protocol = benchmark_protocol(protocol, name)
        expected = sample_index(child_protocol)
        validate_sample_ids(ids, expected)
        if not ids:
            raise ValueError("Workers must not receive empty benchmark assignments")
        if mode == "original":
            effective_batches = {
                batch_size_by_length.get(str(expected[key][2]["length"]), args.batch_size)
                for key in ids
            } if name == "ruler" else {args.batch_size}
            if effective_batches != {1}:
                raise ValueError("Original inference requires effective batch size 1 for every assigned sample")
        folder = Path(plan["outputs"][name])
        child_plan = json.loads((folder / "plan.json").read_text())
        if child_plan["protocol"] != child_protocol or child_plan["sample_ids"] != ids:
            raise ValueError("Benchmark worker assignment differs from its persisted plan")
        child_plan["source"] = source
        write_json(folder / "plan.json", child_plan)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    state = {"started_at": timestamp(), "status": "running", "source": source,
             "benchmarks": {name: {"status": "pending", "completed_samples": 0} for name in assignments}}
    write_json(args.worker_dir / "execution.json", state)
    started = time.monotonic()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("E²-TTT workers require CUDA")
        torch.cuda.reset_peak_memory_stats(0)
        progress.set_stage("loading")
        harness = LengthSortedHFLM if mode == "optimized" else HFLM
        # Only this call loads checkpoint weights; every benchmark reuses lm.model.
        lm = harness(
            pretrained=protocol["checkpoint"], tokenizer=protocol["checkpoint"], dtype="bfloat16",
            device="cuda:0", batch_size=args.batch_size, max_length=2048,
            truncation=False, trust_remote_code=False,
        )
        if lm.world_size != 1:
            raise ValueError("The harness initialized more than one evaluation process")
        if mode == "optimized":
            optimize_model(lm.model)
        for name in ("squad_completion", "swde", "ruler"):
            if name not in assignments:
                continue
            progress.set_stage("preparing")
            ids = assignments[name]
            folder = Path(plan["outputs"][name])
            child_protocol = benchmark_protocol(protocol, name)
            expected, specs = sample_index(child_protocol), task_specs(child_protocol)
            selected = {task: [] for task in specs}
            for key in ids:
                task, index, _ = expected[key]
                selected[task].append(index)
            child_state = state["benchmarks"][name]
            child_state.update(status="running", started_at=timestamp())
            write_json(folder / "execution.json", child_state)
            write_json(args.worker_dir / "execution.json", state)
            native_dir = folder / "native_harness"
            native_dir.mkdir()
            benchmark_lm = lm
            if name == "ruler":
                benchmark_lm = harness(
                    pretrained=lm.model, tokenizer=lm.tokenizer, device="cuda:0", batch_size=args.batch_size,
                    max_length=max(protocol["ruler_lengths"]), truncation=False, add_bos_token=lm.add_bos_token,
                )
            # Activate one native cache at a time, preserving its original writes.
            cached = ProgressCachingLM(benchmark_lm, str(folder / "generation_cache.sqlite"), progress)
            try:
                with (folder / "predictions.jsonl").open("x") as predictions:
                    if name == "ruler":
                        for length in protocol["ruler_lengths"]:
                            total = sum(len(selected[task]) for task, spec in specs.items()
                                        if spec["length"] == length)
                            if total:
                                progress.start_group(
                                    "ruler", length, total,
                                    batch_size_by_length.get(str(length), args.batch_size),
                                )
                            tasks = [
                                LocalRuler(RULER_DATA / str(length) / spec["task_name"] / "test.jsonl",
                                           spec["task_name"], length, cached, selected[task])
                                for task, spec in specs.items() if spec["length"] == length and selected[task]
                            ]
                            if not tasks:
                                continue
                            benchmark_lm.batch_size_per_gpu = batch_size_by_length.get(str(length), args.batch_size)
                            result = evaluate_tasks(cached, tasks)
                            count = save_result(result, f"ruler_{length}", native_dir, predictions, specs)
                            child_state["completed_samples"] += count
                            progress.advance_completed(count)
                    else:
                        task_class, data = (LocalSQuAD, SQUAD_DATA) if name == "squad_completion" else (LocalSWDE, SWDE_DATA)
                        progress.start_group(name, None, len(ids), args.batch_size)
                        task = task_class(data / "data/validation-00000-of-00001.parquet", selected[name])
                        result = evaluate_tasks(cached, [task])
                        count = save_result(result, name, native_dir, predictions, specs)
                        child_state["completed_samples"] += count
                        progress.advance_completed(count)
            finally:
                cached.dbdict.close()
            if child_state["completed_samples"] != len(ids):
                raise ValueError("Native evaluator did not emit all assigned benchmark samples")
            child_state.update(status="completed", finished_at=timestamp())
            write_json(folder / "execution.json", child_state)
            write_json(args.worker_dir / "execution.json", state)
        torch.cuda.synchronize(0)
        state.update(status="completed", exit_code=0)
    except BaseException as error:
        progress.set_stage("failed", error=f"{type(error).__name__}: {error}")
        state.update(status="failed", exit_code=1, error=repr(error))
        for name, child_state in state["benchmarks"].items():
            if child_state["status"] == "running":
                child_state.update(status="failed", error=repr(error), finished_at=timestamp())
                write_json(Path(plan["outputs"][name]) / "execution.json", child_state)
        raise
    finally:
        state.update(finished_at=timestamp(), seconds=time.monotonic() - started,
                     peak_cuda_allocated_bytes=(torch.cuda.max_memory_allocated(0)
                                                if torch.cuda.is_initialized() else 0),
                     peak_cuda_reserved_bytes=(torch.cuda.max_memory_reserved(0)
                                               if torch.cuda.is_initialized() else 0))
        write_json(args.worker_dir / "execution.json", state)
    progress.set_stage("completed")


if __name__ == "__main__":
    execute(parse_args())
