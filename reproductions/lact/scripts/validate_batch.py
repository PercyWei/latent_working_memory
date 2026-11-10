"""Validate equal-length LaCT batching on distinct real inputs; not a benchmark score run."""

import argparse
from collections import defaultdict
import gc
from importlib.metadata import version
import json
import math
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


def run_boundary_checks(model, source_token_ids, batch_sizes):
    """Compare equal-length batch states with fresh B1 teacher-forced references."""
    sizes = list(batch_sizes)
    if not sizes or any(size < 2 for size in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError("Boundary checks require distinct batch sizes >= 2")
    if len(source_token_ids) < max(sizes):
        raise ValueError("Not enough distinct source sequences for boundary checks")
    if any(len(tokens) < 2053 for tokens in source_token_ids[:max(sizes)]):
        raise ValueError("Boundary sources must contain at least 2053 real tokens")
    if model.training:
        raise ValueError("Boundary checks require an evaluation model")
    if model.model.layers[0].attn.lact_chunk_size != 2048:
        raise ValueError("Boundary probes target the released 2048-token chunk")

    weight_fields = (
        "w0", "w1", "w2", "w0_norm", "w1_norm", "w2_norm",
        "w0_main", "w1_main", "w2_main",
        "dw0_momentum", "dw1_momentum", "dw2_momentum",
    )
    pending_fields = (
        "pending_k", "pending_v", "pending_lr0", "pending_lr1", "pending_lr2",
        "pending_momentum",
    )
    cases = []

    def forward(token_ids, total_seen, cache=None):
        return model(
            input_ids=token_ids,
            attention_mask=torch.ones(
                (token_ids.shape[0], total_seen), dtype=torch.long, device=token_ids.device
            ),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )

    def progress(cache, seen, batch_size):
        failed = []
        updates = (seen - 1) // 2048
        pending = seen - updates * 2048
        for layer_index, (block, layer) in enumerate(
            zip(model.model.layers, cache.layers, strict=True)
        ):
            ttt = layer.ttt
            heads = block.attn.num_fw_heads
            actual = {
                "seen": layer.seen,
                "ttt_seen": ttt.seen,
                "updates": ttt.n_updates,
                "pending": ttt.pending_length,
                "pending_fragments_length": sum(value.shape[1] for value in ttt.pending_k),
                "kv_batch": layer.keys.shape[0],
                "fast_weight_rows": ttt.w0.shape[0],
            }
            expected = {
                "seen": seen,
                "ttt_seen": seen,
                "updates": updates,
                "pending": pending,
                "pending_fragments_length": pending,
                "kv_batch": batch_size,
                "fast_weight_rows": batch_size * heads,
            }
            if actual != expected:
                failed.append({"layer": layer_index, "actual": actual, "expected": expected})
        return failed

    def compare_tensor(reports, name, actual, reference, location):
        report = reports.setdefault(name, {
            "comparisons": 0,
            "shape_match": True,
            "dtype_match": True,
            "finite": True,
            "exact": True,
            "max_abs": 0.0,
            "max_relative_l2": 0.0,
            "shape_pairs": set(),
            "dtype_pairs": set(),
            "worst_location": None,
        })
        report["comparisons"] += 1
        if actual is None or reference is None:
            same = actual is None and reference is None
            report["shape_match"] &= same
            report["exact"] &= same
            return same, True
        report["shape_pairs"].add((tuple(actual.shape), tuple(reference.shape)))
        report["dtype_pairs"].add((str(actual.dtype), str(reference.dtype)))
        shape_match = actual.shape == reference.shape
        dtype_match = actual.dtype == reference.dtype
        report["shape_match"] &= shape_match
        report["dtype_match"] &= dtype_match
        if not shape_match:
            report["exact"] = False
            return False, True
        if actual.numel() == 0:
            return dtype_match, True
        actual_float, reference_float = actual.float(), reference.float()
        difference = actual_float - reference_float
        values = torch.stack((
            difference.abs().amax(),
            difference.norm() / reference_float.norm().clamp_min(1e-12),
            (torch.isfinite(actual_float).all() & torch.isfinite(reference_float).all()).float(),
        )).cpu().tolist()
        max_abs, relative, finite_value = values
        finite = bool(finite_value)
        report["finite"] &= finite
        report["exact"] &= finite and max_abs == 0.0 and dtype_match
        if math.isfinite(max_abs) and max_abs > report["max_abs"]:
            report["max_abs"] = max_abs
            report["worst_location"] = location
        if math.isfinite(relative):
            report["max_relative_l2"] = max(report["max_relative_l2"], relative)
        return dtype_match, finite

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for prefix_length in (2047, 2048, 2049):
            for batch_size in sizes:
                tokens = torch.tensor(
                    [row[:prefix_length + 4] for row in source_token_ids[:batch_size]],
                    dtype=torch.long,
                    device="cuda:0",
                )
                batch_cache = None
                for step in range(5):
                    seen = prefix_length + step
                    current = tokens[:, :prefix_length] if step == 0 else tokens[:, seen - 1:seen]
                    batched = forward(current, seen, batch_cache)
                    batch_cache = batched.past_key_values
                    batched_argmax = batched.logits[:, -1].argmax(-1).cpu().tolist()
                    finite = bool(torch.isfinite(batched.logits).all())
                    progress_failures = progress(batch_cache, seen, batch_size)
                    states, references, argmax_equal = {}, [], []
                    structure_ok = True
                    for sample_index in range(batch_size):
                        reference_cache = None
                        for reference_step in range(step + 1):
                            reference_seen = prefix_length + reference_step
                            current = (
                                tokens[sample_index:sample_index + 1, :prefix_length]
                                if reference_step == 0
                                else tokens[sample_index:sample_index + 1, reference_seen - 1:reference_seen]
                            )
                            reference = forward(current, reference_seen, reference_cache)
                            reference_cache = reference.past_key_values
                        reference_argmax = int(reference.logits[0, -1].argmax())
                        references.append(reference_argmax)
                        argmax_equal.append(batched_argmax[sample_index] == reference_argmax)
                        finite &= bool(torch.isfinite(reference.logits).all())
                        for failure in progress(reference_cache, seen, 1):
                            progress_failures.append({"sample": sample_index, **failure})
                        for layer_index, (batch_layer, reference_layer) in enumerate(
                            zip(batch_cache.layers, reference_cache.layers, strict=True)
                        ):
                            location = {"sample": sample_index, "layer": layer_index}
                            pairs = [
                                (f"attention.{name}", getattr(batch_layer, name)[sample_index:sample_index + 1], getattr(reference_layer, name))
                                for name in ("keys", "values")
                            ]
                            for name in ("query_offsets", "key_offsets"):
                                offsets = getattr(batch_layer, name)
                                pairs.append((
                                    f"attention.{name}",
                                    offsets[sample_index:sample_index + 2] - offsets[sample_index],
                                    getattr(reference_layer, name),
                                ))
                            heads = reference_layer.ttt.w0.shape[0]
                            head_slice = slice(sample_index * heads, (sample_index + 1) * heads)
                            for name in weight_fields:
                                actual = getattr(batch_layer.ttt, name)
                                pairs.append((
                                    f"ttt.{name}",
                                    actual[head_slice] if actual is not None else None,
                                    getattr(reference_layer.ttt, name),
                                ))
                            for name in pending_fields:
                                actual = getattr(batch_layer.ttt, name)
                                expected = getattr(reference_layer.ttt, name)
                                if len(actual) != len(expected):
                                    structure_ok = False
                                    progress_failures.append({
                                        **location, "field": name,
                                        "batch_fragments": len(actual), "reference_fragments": len(expected),
                                    })
                                for left, right in zip(actual, expected):
                                    pairs.append((f"ttt.{name}", left[head_slice], right))
                            for name, actual, expected in pairs:
                                shape_ok, tensor_finite = compare_tensor(
                                    states, name, actual, expected, location
                                )
                                structure_ok &= shape_ok
                                finite &= tensor_finite
                        del reference, reference_cache, reference_layer, pairs
                    for report in states.values():
                        report["shape_pairs"] = [
                            {"batch_slice": list(left), "reference": list(right)}
                            for left, right in sorted(report["shape_pairs"])
                        ]
                        report["dtype_pairs"] = [list(pair) for pair in sorted(report["dtype_pairs"])]
                    cases.append({
                        "prefix_tokens": prefix_length,
                        "batch_size": batch_size,
                        "distinct_probe_sequences": len({
                            tuple(row[:prefix_length + 4])
                            for row in source_token_ids[:batch_size]
                        }),
                        "teacher_forced_steps": step,
                        "seen": seen,
                        "expected_updates": (seen - 1) // 2048,
                        "expected_pending_tokens": seen - ((seen - 1) // 2048) * 2048,
                        "batch_argmax": batched_argmax,
                        "reference_argmax": references,
                        "argmax_equal": argmax_equal,
                        "finite": finite,
                        "structure_ok": structure_ok,
                        "progress_ok": not progress_failures,
                        "progress_failures": progress_failures,
                        "states": states,
                        "primary_passed": finite and structure_ok and not progress_failures and all(argmax_equal),
                    })
                    del batched
                del batch_cache, tokens
    return {
        "data_kind": "real_token_prefixes_with_fixed_real_continuations",
        "formal_benchmark_result": False,
        "batch_sizes": sizes,
        "prefix_lengths": [2047, 2048, 2049],
        "continuation_tokens": 4,
        "cases": cases,
        "all_primary_checks_passed": all(row["primary_passed"] for row in cases),
        "interpretation": (
            "Every B1 reference is rebuilt with the same prefix and fixed real continuation, "
            "then compared and released immediately. Primary checks require finite logits/states, "
            "matching next-token argmax, shapes/dtypes, and the original chunk-update progress. "
            "State exactness and max absolute/relative L2 differences are reported separately."
        ),
    }


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
        batch_details = []
        for group in groups:
            batch_started = time.perf_counter()
            outputs.update(generate(model, tokenizer, group))
            batch_details.append({
                "batch_size": len(group), "input_tokens": len(group[0]["input_ids"]),
                "seconds": time.perf_counter() - batch_started,
                "sample_ids": [row["id"] for row in group],
            })
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
    return {
        "batch_size": size, "status": "completed", "samples": len(rows),
        "actual_batch_sizes": [len(group) for group in groups],
        "batch_details": batch_details,
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
        "boundary_only": args.boundary_only,
        "boundary_source": "Distinct real samples, contiguous final 2053-token windows; not formal benchmark prompts",
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "protocol": "Distinct real samples; exact token-length buckets; no padding; fresh cache per generate; EOS-aware complete IDs.",
    }
    write_json(args.output_dir / "plan.json", plan)
    started = time.perf_counter()
    results, selected = [], {}
    try:
        load_started = time.perf_counter()
        model, config = load_checkpoint(args.checkpoint, "incremental")
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_started
        write_json(args.output_dir / "runtime.json", {
            "load_seconds": load_seconds, "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": version("transformers"), "torch_num_threads": torch.get_num_threads(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "device_name": torch.cuda.get_device_name(0), "model_class": type(model).__name__,
            "actual_fast_weight_heads": model.model.layers[0].attn.num_fw_heads,
            "chunk_size": config.lact_chunk_size,
        })
        lengths = args.lengths[:1] if args.boundary_only else args.lengths
        for length in lengths:
            selection_started = time.perf_counter()
            rows, selection = select_samples(args, tokenizer, length)
            selection["selection_seconds"] = time.perf_counter() - selection_started
            selected[length] = rows
            write_json(args.output_dir / f"selection_{length}.json", selection)
            if args.boundary_only:
                break
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
            entry = {"length": length, "selection": selection, "trials": trials}
            if args.legacy_measurements is not None:
                entry["legacy"] = legacy_comparison(args.legacy_measurements, args.checkpoint, length, reference, trials)
            results.append(entry)
            write_json(args.output_dir / "results.json", results)

        # Extra shapes from stopping/state probes must not affect production timing warmup.
        for entry in results:
            rows = selected[entry["length"]]
            reference = entry["trials"][0]["token_ids"]
            largest_completed = max(trial["batch_size"] for trial in entry["trials"] if trial["status"] == "completed")
            isolation_group = batches(rows, min(8, largest_completed))[0]
            reversed_ids = generate(model, tokenizer, list(reversed(isolation_group)))
            entry["isolation"] = {
                "permuted_ids": [row["id"] for row in reversed(isolation_group)],
                "all_permuted_rows_equal_to_batch1": all(reversed_ids[row["id"]] == reference[row["id"]] for row in isolation_group),
                "batch1_replay_equal": generate(model, tokenizer, rows[:1])[rows[0]["id"]] == reference[rows[0]["id"]],
            }
            entry["controlled_eos"] = controlled_eos(model, tokenizer, rows, reference, min(8, largest_completed))
            write_json(args.output_dir / "results.json", results)

        completed_sizes = set(args.batch_sizes)
        for entry in results:
            completed_sizes &= {trial["batch_size"] for trial in entry["trials"] if trial["status"] == "completed"}
        boundary_sizes = sorted(size for size in completed_sizes if 1 < size <= 8)
        if not boundary_sizes:
            raise ValueError("No successful batch size in 2/4/8 for boundary checks")
        source_rows = batches(selected[lengths[0]], max(boundary_sizes))[0]
        boundary = run_boundary_checks(
            model, [row["input_ids"][-2053:] for row in source_rows], boundary_sizes
        )
        boundary["source_sample_ids"] = [row["id"] for row in source_rows]
        boundary["source_slice"] = "Contiguous final 2053 tokens of each real source input"
        write_json(args.output_dir / "boundary.json", boundary)
        generation_passed = all(
            all(trial["all_token_ids_equal_to_batch1"] for trial in item["trials"] if trial["status"] == "completed")
            and item["isolation"]["all_permuted_rows_equal_to_batch1"]
            and item["isolation"]["batch1_replay_equal"]
            and item["controlled_eos"].get("early_stop_and_other_rows_continue", False)
            and item["controlled_eos"].get("all_token_ids_equal", False)
            for item in results
        )
        passed = generation_passed and boundary["all_primary_checks_passed"]
        write_json(args.output_dir / "summary.json", {
            "passed": passed, "boundary_only": args.boundary_only, "lengths": results,
            "boundary": {"path": str(args.output_dir / "boundary.json"), "cases": len(boundary["cases"]),
                         "all_primary_checks_passed": boundary["all_primary_checks_passed"]},
        })
        write_json(args.output_dir / "execution.json", {"finished_at": timestamp(), "elapsed_seconds": time.perf_counter() - started, "exit_code": 0 if passed else 1})
        if not passed:
            raise RuntimeError("Batch generation, isolation, stopping or boundary validation failed")
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
    parser.add_argument("--boundary-only", action="store_true", help="Run the real GPU teacher-forced state checks without repeating generation timing")
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
