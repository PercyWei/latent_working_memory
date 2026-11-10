"""Validate equal-length LaCT batching on distinct real inputs; not a benchmark score run."""

import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from incremental import IncrementalLaCTForCausalLM
from provenance import capture_source
import torch
from transformers import AutoTokenizer

from evaluate import (
    GENERATION_BUDGET, LENGTHS, PROJECT_ROOT, TASKS, load_checkpoint,
    read_samples, timestamp, write_json,
)


SCAN_ROWS_PER_TASK = 64


class DeviceMemory:
    """Sample the entire physical GPU; the thread never calls the model or PyTorch."""

    def __init__(self):
        self.peak_mib = 0
        self.samples = 0
        self.error = None
        self.stop = threading.Event()
        self.thread = None

    def sample(self):
        result = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        self.peak_mib = max(self.peak_mib, int(result.stdout.strip()))
        self.samples += 1

    def poll(self):
        try:
            while not self.stop.wait(0.2):
                self.sample()
        except Exception as error:
            self.error = error

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self.poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.sample()
        if self.error is not None:
            raise self.error


def trim_ids(ids, eos):
    return ids[:ids.index(eos) + 1] if eos in ids else ids


def generate(model, tokenizer, rows, budget=GENERATION_BUDGET, eos=None):
    lengths = {len(row["input_ids"]) for row in rows}
    if len(lengths) != 1:
        raise ValueError("Every actual model batch must contain equal-length inputs")
    input_ids = torch.tensor([row["input_ids"] for row in rows], device="cuda:0")
    eos = model.generation_config.eos_token_id if eos is None else eos
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.generate(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids), past_key_values=None,
            do_sample=False, num_beams=1, max_new_tokens=budget,
            use_cache=True, logits_to_keep=1, return_dict_in_generate=False,
            output_scores=False, eos_token_id=eos, pad_token_id=tokenizer.eos_token_id,
        )
    ids = output[:, input_ids.shape[1]:].cpu().tolist()
    return {row["id"]: trim_ids(tokens, eos) for row, tokens in zip(rows, ids, strict=True)}


def batches(rows, size):
    groups = defaultdict(list)
    for row in rows:
        groups[len(row["input_ids"])].append(row)
    return [group[start:start + size] for group in groups.values() for start in range(0, len(group), size)]


def select_samples(args, tokenizer, length):
    groups = defaultdict(list)
    for task in TASKS:
        source = args.ruler_data / str(length) / task / "test.jsonl"
        for position, row in enumerate(read_samples(source)[:SCAN_ROWS_PER_TASK]):
            ids = tokenizer.encode(row["input"] + row["answer_prefix"], truncation=False)
            if len(ids) + GENERATION_BUDGET > length:
                raise ValueError(f"{length}/{task}/{position}: input exceeds token budget")
            groups[len(ids)].append({
                "id": f"ruler/{length}/{task}/{position}", "length": length,
                "task": task, "sample_position": position, "source_index": row["index"],
                "input_ids": ids,
            })
    per_bucket = args.num_samples // args.length_buckets
    candidates = sorted(groups.items(), key=lambda pair: (-len(pair[1]), -pair[0]))
    eligible = [(n, rows) for n, rows in candidates if len(rows) >= per_bucket]
    if len(eligible) < args.length_buckets:
        raise ValueError(f"Not enough natural length buckets: {[(n, len(rows)) for n, rows in candidates[:10]]}")
    selected = [row for _, rows in eligible[:args.length_buckets] for row in rows[:per_bucket]]
    if len({row["id"] for row in selected}) != args.num_samples:
        raise ValueError("Selection repeated a source sample")
    return selected, {
        "scan_rows_per_task": SCAN_ROWS_PER_TASK,
        "selected_buckets": [
            {"input_tokens": n, "available_in_scan": len(rows), "selected": per_bucket}
            for n, rows in eligible[:args.length_buckets]
        ],
        "mixed_length_workload": args.length_buckets > 1,
        "samples": [{key: value for key, value in row.items() if key != "input_ids"} for row in selected],
    }


def measure(model, tokenizer, rows, size):
    gc.collect()
    torch.cuda.empty_cache()
    groups = batches(rows, size)
    shapes = {}
    for group in groups:
        shapes[(len(group), len(group[0]["input_ids"]))] = group
    torch.cuda.synchronize()
    warm_started = time.perf_counter()
    with DeviceMemory() as warm_memory:
        for group in shapes.values():
            generate(model, tokenizer, group)
        torch.cuda.synchronize()
    warm_seconds = time.perf_counter() - warm_started
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    with DeviceMemory() as memory:
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = {}
        for group in groups:
            outputs.update(generate(model, tokenizer, group))
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
    return {
        "batch_size": size, "status": "completed", "samples": len(rows),
        "actual_batch_sizes": [len(group) for group in groups],
        "warmup_seconds": warm_seconds, "generation_wall_seconds": seconds,
        "samples_per_second": len(rows) / seconds,
        "generated_tokens": sum(map(len, outputs.values())),
        "peak_device_memory_mib": max(warm_memory.peak_mib, memory.peak_mib),
        "peak_device_memory_mib_during_measurement": memory.peak_mib,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "token_ids": outputs,
    }


def controlled_eos(model, tokenizer, rows, reference, batch_size):
    for group in batches(rows, batch_size):
        if len(group) < 2:
            continue
        for offset in range(8):
            if any(len(reference[row["id"]]) <= offset for row in group):
                continue
            first_tokens = [reference[row["id"]][offset] for row in group]
            if len(set(first_tokens)) < 2:
                continue
            test_eos = first_tokens[0]
            inputs = [
                {**row, "input_ids": row["input_ids"] + reference[row["id"]][:offset]}
                for row in group
            ]
            singles = {}
            for row in inputs:
                singles.update(generate(model, tokenizer, [row], budget=8, eos=test_eos))
            combined = generate(model, tokenizer, inputs, budget=8, eos=test_eos)
            lengths = [len(combined[row["id"]]) for row in inputs]
            return {
                "purpose": "synthetic stopping test only; not the formal EOS setting",
                "temporary_eos_token_id": test_eos, "appended_reference_tokens": offset,
                "generation_budget": 8, "batch_size": len(inputs), "generated_lengths": lengths,
                "early_stop_and_other_rows_continue": min(lengths) == 1 and max(lengths) > 1,
                "all_token_ids_equal": singles == combined,
                "single_token_ids": singles, "batched_token_ids": combined,
            }
    return {"purpose": "synthetic stopping test", "covered": False, "error": "No distinct reference first-token choices"}


def legacy_comparison(path, checkpoint, length, reference, trials):
    saved = json.loads(path.read_text())
    identities = defaultdict(list)
    throughput = []
    for case, value in saved["cases"].items():
        if not case.startswith("lact-") or value["plan"]["checkpoint"] != str(checkpoint):
            continue
        if value["plan"]["length"] != length:
            continue
        for trial in value["trials"]:
            if trial["status"] != "completed":
                continue
            for row in trial["samples"]:
                if row["id"] in reference:
                    identities[row["id"]].append(row["token_ids"] == reference[row["id"]])
            if trial["workers"] == 8:
                throughput.append({
                    "case": case, "workers": 8, "batch_size_per_worker": 1,
                    "samples_per_second": trial["samples_per_second"],
                    "peak_device_memory_mib": trial["peak_device_memory_mib"],
                    "candidate_throughput_ratio": {
                        str(trial_now["batch_size"]): trial_now["samples_per_second"] / trial["samples_per_second"]
                        for trial_now in trials if trial_now["status"] == "completed"
                    },
                })
    return {
        "source": str(path), "overlap_ids": len(identities),
        "all_overlapping_token_ids_equal": all(all(values) for values in identities.values()),
        "eight_process_references": throughput,
        "interpretation": "Legacy throughput is a scale reference with different sample selection and timing scope; not a paired speedup.",
    }


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plan = {
        "created_at": timestamp(),
        "source": capture_source(__file__, {"incremental": IncrementalLaCTForCausalLM}),
        "checkpoint": str(args.checkpoint),
        "checkpoint_config": json.loads((args.checkpoint / "config.json").read_text()),
        "ruler_data": str(args.ruler_data), "lengths": args.lengths,
        "batch_sizes": args.batch_sizes, "num_samples_per_length": args.num_samples,
        "length_buckets": args.length_buckets, "generation_tokens": GENERATION_BUDGET,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "protocol": "Distinct real samples; exact token-length buckets; no padding; fresh cache per generate; EOS-aware complete IDs.",
    }
    write_json(args.output_dir / "plan.json", plan)
    started = time.perf_counter()
    results = []
    try:
        load_started = time.perf_counter()
        model, config = load_checkpoint(args.checkpoint, "incremental")
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_started
        write_json(args.output_dir / "runtime.json", {
            "load_seconds": load_seconds, "torch": torch.__version__, "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0), "model_class": type(model).__name__,
            "actual_fast_weight_heads": model.model.layers[0].attn.num_fw_heads,
            "chunk_size": config.lact_chunk_size,
        })
        for length in args.lengths:
            rows, selection = select_samples(args, tokenizer, length)
            write_json(args.output_dir / f"selection_{length}.json", selection)
            trials = []
            reference = None
            for size in args.batch_sizes:
                try:
                    measured = measure(model, tokenizer, rows, size)
                    if size == 1:
                        reference = measured["token_ids"]
                    measured["all_token_ids_equal_to_batch1"] = measured["token_ids"] == reference
                except torch.cuda.OutOfMemoryError as error:
                    measured = {"batch_size": size, "status": "oom", "error": str(error)}
                    gc.collect()
                    torch.cuda.empty_cache()
                    if size == 1:
                        raise
                trials.append(measured)
                write_json(args.output_dir / f"generation_{length}.json", trials)
                print(json.dumps({"length": length, **{k: v for k, v in measured.items() if k != "token_ids"}}, ensure_ascii=False), flush=True)
            largest_completed = max(trial["batch_size"] for trial in trials if trial["status"] == "completed")
            isolation_group = batches(rows, min(8, largest_completed))[0]
            reversed_ids = generate(model, tokenizer, list(reversed(isolation_group)))
            isolation = {
                "permuted_ids": [row["id"] for row in reversed(isolation_group)],
                "all_permuted_rows_equal_to_batch1": all(reversed_ids[row["id"]] == reference[row["id"]] for row in isolation_group),
                "batch1_replay_equal": generate(model, tokenizer, rows[:1])[rows[0]["id"]] == reference[rows[0]["id"]],
            }
            eos_test = controlled_eos(model, tokenizer, rows, reference, min(8, largest_completed))
            entry = {"length": length, "selection": selection, "trials": trials, "isolation": isolation, "controlled_eos": eos_test}
            if args.legacy_measurements is not None:
                entry["legacy"] = legacy_comparison(args.legacy_measurements, args.checkpoint, length, reference, trials)
            results.append(entry)
            write_json(args.output_dir / "results.json", results)
        passed = all(
            all(trial["all_token_ids_equal_to_batch1"] for trial in item["trials"] if trial["status"] == "completed")
            and item["isolation"]["all_permuted_rows_equal_to_batch1"]
            and item["isolation"]["batch1_replay_equal"]
            and item["controlled_eos"].get("early_stop_and_other_rows_continue", False)
            and item["controlled_eos"].get("all_token_ids_equal", False)
            for item in results
        )
        write_json(args.output_dir / "summary.json", {"passed": passed, "lengths": results})
        write_json(args.output_dir / "execution.json", {"finished_at": timestamp(), "elapsed_seconds": time.perf_counter() - started, "exit_code": 0 if passed else 1})
        if not passed:
            raise RuntimeError("Batch generation, isolation or stopping validation failed")
    except BaseException as error:
        write_json(args.output_dir / "execution.json", {"finished_at": timestamp(), "elapsed_seconds": time.perf_counter() - started, "exit_code": 1, "error": f"{type(error).__name__}: {error}"})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ruler-data", type=Path, default=PROJECT_ROOT / "data/benchmarks/ruler-v1/reproductions/lact-760m")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", choices=LENGTHS, default=[4096, 32768])
    parser.add_argument("--batch-sizes", type=int, nargs="+", choices=[1, 2, 4, 8, 16], default=[1, 2, 4, 8])
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--length-buckets", type=int, choices=[1, 2], default=2)
    parser.add_argument("--legacy-measurements", type=Path)
    args = parser.parse_args()
    args.batch_sizes = sorted(set([1, *args.batch_sizes]))
    if args.num_samples <= 0 or args.num_samples % (args.length_buckets * max(args.batch_sizes)):
        parser.error("num-samples must be divisible by length-buckets times the largest batch size")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Use one process on physical GPU 0: CUDA_VISIBLE_DEVICES=0")
    for name in ("checkpoint", "ruler_data", "output_dir", "legacy_measurements"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    run(args)


if __name__ == "__main__":
    main()
