"""Validate real E²-TTT batched generation against the unchanged native B=1 path.

Run in the E²-TTT environment with the method directory, upstream and its
lm-evaluation-harness on PYTHONPATH. Each checkpoint is a separate invocation.
Only JSON/JSONL summaries are persisted; diagnostic tensors remain in RAM.
"""

import argparse
from collections import Counter, defaultdict, deque
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import time

import e2_ttt  # noqa: F401 -- register native checkpoint model classes
from lm_eval.api.instance import Instance
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils_hf import stop_sequences_criteria
import numpy as np
import torch
import transformers

from batch_harness import EqualLengthHFLM
from evaluate import PROJECT_ROOT, RULER_TASKS, source_record, timestamp, write_json
from local_tasks import LocalRuler, LocalSQuAD, LocalSWDE
from optimized import optimize_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ruler-data",
        type=Path,
        default=PROJECT_ROOT / "data/benchmarks/ruler-v1/reproductions/e2-ttt-1.3b",
    )
    parser.add_argument(
        "--squad-data", type=Path, default=PROJECT_ROOT / "data/benchmarks/based-squad"
    )
    parser.add_argument(
        "--swde-data", type=Path, default=PROJECT_ROOT / "data/benchmarks/based-swde"
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384])
    parser.add_argument("--ruler-samples", type=int, choices=[8, 16], default=16)
    parser.add_argument("--recall-samples", type=int, default=8)
    parser.add_argument("--probe-batch-size", type=int, default=4)
    parser.add_argument("--warmup-generations", type=int, default=1)
    parser.add_argument("--numeric-atol", type=float, default=0.03)
    parser.add_argument("--numeric-rtol", type=float, default=0.03)
    parser.add_argument("--isolation-atol", type=float, default=0.0)
    parser.add_argument("--isolation-rtol", type=float, default=0.0)
    args = parser.parse_args()
    for name in ("checkpoint", "output_dir", "ruler_data", "squad_data", "swde_data"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1"):
        parser.error("Use one physical GPU: CUDA_VISIBLE_DEVICES=0 or 1.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Use one process.")
    if (
        not args.batch_sizes
        or min(args.batch_sizes) < 1
        or len(set(args.batch_sizes)) != len(args.batch_sizes)
    ):
        parser.error("--batch-sizes must be distinct positive integers.")
    if max(args.batch_sizes) > args.ruler_samples:
        parser.error("The RULER sample group must cover the largest requested batch.")
    if not args.lengths or min(args.lengths) < 4096 or len(set(args.lengths)) != len(args.lengths):
        parser.error("Use distinct RULER lengths >=4096.")
    if not 3 <= args.probe_batch_size <= (args.ruler_samples + 1) // 2:
        parser.error("The probe requires >=3 rows and 2*probe_batch_size-1 real samples.")
    if args.recall_samples < 6 or args.warmup_generations < 0:
        parser.error("Use at least six recall samples and nonnegative warmup generations.")
    if min(args.numeric_atol, args.numeric_rtol, args.isolation_atol, args.isolation_rtol) < 0:
        parser.error("Comparison tolerances must be nonnegative.")
    if not (args.checkpoint / "config.json").is_file():
        parser.error(f"No config.json under {args.checkpoint}")
    if not torch.cuda.is_available():
        parser.error("The native method environment must provide CUDA.")
    return args


def append_json(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    stream.flush()


def encoded_input(lm, prompt, max_context):
    ids, mask = lm.tok_batch_encode([prompt], left_truncate_len=max_context, truncation=False)
    if not bool(mask.all()):
        raise ValueError("A singleton real prompt unexpectedly contains padding.")
    return ids[0].tolist()


def select_groups(args, lm):
    groups = []
    for length in args.lengths:
        for task_name in RULER_TASKS:
            task = LocalRuler(
                args.ruler_data / str(length) / task_name / "test.jsonl", task_name, length, lm
            )
            buckets = defaultdict(list)
            for doc in task.eval_docs:
                prompt = task.doc_to_text(doc)
                ids = encoded_input(lm, prompt, length - 128)
                item = {
                    "id": f"ruler/{length}/{task_name}/{doc['source_doc_id']}",
                    "doc": doc,
                    "prompt": prompt,
                    "input_ids": ids,
                }
                bucket = buckets[len(ids)]
                bucket.append(item)
                if len(bucket) == args.ruler_samples:
                    selected = bucket
                    break
            else:
                counts = Counter({n: len(rows) for n, rows in buckets.items()})
                raise ValueError(
                    f"No {args.ruler_samples}-sample exact-length group for {task_name}/{length}: {counts.most_common(5)}"
                )
            groups.append(
                {
                    "name": f"ruler/{length}/{task_name}",
                    "task": task,
                    "samples": selected,
                    "max_length": length,
                    "max_gen_toks": 128,
                }
            )
    for name, task_class, data_dir in (
        ("squad_completion", LocalSQuAD, args.squad_data),
        ("swde", LocalSWDE, args.swde_data),
    ):
        task = task_class(data_dir / "data/validation-00000-of-00001.parquet")
        buckets = defaultdict(list)
        for doc in task.eval_docs:
            prompt = task.doc_to_text(doc)
            item = {
                "id": f"{name}/{doc['source_doc_id']}",
                "doc": doc,
                "prompt": prompt,
                "input_ids": encoded_input(lm, prompt, 2000),
            }
            buckets[len(item["input_ids"])].append(item)
        primary_length = max(buckets, key=lambda length: len(buckets[length]))
        primary = buckets[primary_length]
        if len(primary) < 4:
            raise ValueError(f"{name}: no real post-truncation equal-length group with >=4 samples")
        others = [
            row for length, rows in buckets.items() if length != primary_length for row in rows
        ][:2]
        count = min(len(primary), args.recall_samples - len(others))
        selected = primary[:2] + others[:1] + primary[2:count] + others[1:]
        groups.append(
            {
                "name": name,
                "task": task,
                "samples": selected,
                "max_length": 2048,
                "max_gen_toks": 48,
            }
        )
    return groups


def requests_for(group, samples):
    requests = []
    for sample in samples:
        native = group["task"].construct_requests(
            sample["doc"],
            sample["prompt"],
            metadata=(group["name"], sample["doc"]["source_doc_id"], 1),
        )
        instances = [native] if isinstance(native, Instance) else native
        if len(instances) != 1:
            raise ValueError("Validation expects one native generation request per document.")
        requests.extend(instances)
    return requests


@contextmanager
def generation_measurements(lm, samples, stream, run_name, group_name):
    native_generate = lm._model_generate
    queues = defaultdict(deque)
    for sample in samples:
        queues[tuple(sample["input_ids"])].append(sample["id"])
    generated_ids, batches = {}, []

    def measured_generate(context, *positional, **kwargs):
        actual_batch, input_length = context.shape
        ids = [queues[tuple(row)].popleft() for row in context.tolist()]
        prefill = []
        active = []

        def before_forward(module, inputs, forward_kwargs):
            if not active:
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                active.append((start, end))
                start.record()

        def after_forward(module, inputs, forward_kwargs, output):
            if not prefill:
                active[0][1].record()
                prefill.append(active[0])

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        hooks = [
            lm.model.register_forward_pre_hook(before_forward, with_kwargs=True),
            lm.model.register_forward_hook(after_forward, with_kwargs=True),
        ]
        started = time.perf_counter()
        try:
            output = native_generate(context, *positional, **kwargs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
        finally:
            for hook in hooks:
                hook.remove()
        for sample_id, row in zip(ids, output[:, input_length:].tolist(), strict=True):
            generated_ids[sample_id] = {"raw_generated_ids": row, "stop_sequences": kwargs["stop"]}
        record = {
            "record_type": "generation_batch",
            "run": run_name,
            "group": group_name,
            "sample_ids": ids,
            "actual_batch_size": actual_batch,
            "input_tokens_per_row": input_length,
            "prefill_cuda_seconds": prefill[0][0].elapsed_time(prefill[0][1]) / 1000,
            "generation_wall_seconds": elapsed,
            "output_columns": output.shape[1] - input_length,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        batches.append(record)
        append_json(stream, record)
        return output

    lm._model_generate = measured_generate
    try:
        yield generated_ids, batches
    finally:
        lm._model_generate = native_generate


def completion_prefix(lm, raw_ids, stop_sequences):
    """Find the first singleton native stop, retaining its terminating token."""
    eos_ids = lm.model.generation_config.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
    criteria = stop_sequences_criteria(lm.tokenizer, stop_sequences, 0, 1)
    for count, token in enumerate(raw_ids, 1):
        if token in eos_ids:
            return raw_ids[:count], {"reason": "eos", "token_id": token, "tokens": count}
        prefix = torch.tensor([raw_ids[:count]], dtype=torch.long)
        for criterion in criteria:
            if criterion(prefix, scores=None):
                return raw_ids[:count], {
                    "reason": "until",
                    "sequence": criterion.sequence,
                    "tokens": count,
                }
    return raw_ids, {"reason": "generation_budget", "tokens": len(raw_ids)}


def run_generation(
    args, base_lm, groups, run_name, batch_size, optimized, predictions, measurements, original
):
    results, summaries = {}, {}
    wrapper_class = EqualLengthHFLM if optimized else HFLM
    for group in groups:
        lm = wrapper_class(
            pretrained=base_lm.model,
            tokenizer=base_lm.tokenizer,
            device="cuda:0",
            batch_size=batch_size,
            max_length=group["max_length"],
            truncation=False,
            add_bos_token=base_lm.add_bos_token,
        )
        warmup_buckets = defaultdict(list)
        for sample in group["samples"]:
            warmup_buckets[len(sample["input_ids"])].append(sample)
        warmup_samples = max(warmup_buckets.values(), key=len)[:batch_size]
        for _ in range(args.warmup_generations):
            lm.generate_until(requests_for(group, warmup_samples), disable_tqdm=True)
        with generation_measurements(
            lm, group["samples"], measurements, run_name, group["name"]
        ) as (token_ids, batches):
            harness_started = time.perf_counter()
            texts = lm.generate_until(requests_for(group, group["samples"]), disable_tqdm=True)
            harness_elapsed = time.perf_counter() - harness_started
        score_values, mismatched_ids, changed_scores = defaultdict(list), [], []
        for sample, text in zip(group["samples"], texts, strict=True):
            native_scores = group["task"].process_results(sample["doc"], [text])
            raw = token_ids[sample["id"]]
            completion_ids, cut = completion_prefix(
                lm, raw["raw_generated_ids"], raw["stop_sequences"]
            )
            result = {
                "text": text,
                "generated_ids": completion_ids,
                "raw_generated_ids": raw["raw_generated_ids"],
                "completion_cut": cut,
                "scores": native_scores,
            }
            results[sample["id"]] = result
            reference = original.get(sample["id"])
            if reference is not None:
                if result["generated_ids"] != reference["generated_ids"]:
                    mismatched_ids.append(sample["id"])
                if native_scores != reference["scores"]:
                    changed_scores.append(sample["id"])
            for name, value in native_scores.items():
                score_values[name].append(float(value))
            append_json(
                predictions,
                {
                    "run": run_name,
                    "requested_batch_size": batch_size,
                    "group": group["name"],
                    "sample_id": sample["id"],
                    "source_doc_id": sample["doc"]["source_doc_id"],
                    "prompt": sample["prompt"],
                    "input_tokens": len(sample["input_ids"]),
                    "target": group["task"].doc_to_target(sample["doc"]),
                    **result,
                    "ids_equal_original": None
                    if reference is None
                    else result["generated_ids"] == reference["generated_ids"],
                    "scores_equal_original": None
                    if reference is None
                    else native_scores == reference["scores"],
                },
            )
        summaries[group["name"]] = {
            "samples": len(group["samples"]),
            "native_mean_scores": {
                name: sum(values) / len(values) for name, values in score_values.items()
            },
            "actual_batch_sizes": [batch["actual_batch_size"] for batch in batches],
            "generation_wall_seconds": sum(batch["generation_wall_seconds"] for batch in batches),
            "harness_wall_seconds": harness_elapsed,
            "prefill_cuda_seconds": sum(batch["prefill_cuda_seconds"] for batch in batches),
            "peak_cuda_allocated_bytes": max(
                batch["peak_cuda_allocated_bytes"] for batch in batches
            ),
            "peak_cuda_reserved_bytes": max(batch["peak_cuda_reserved_bytes"] for batch in batches),
            "ids_differ_original": mismatched_ids,
            "scores_differ_original": changed_scores,
        }
        if original:
            reference_means = {
                name: sum(original[row["id"]]["scores"][name] for row in group["samples"])
                / len(group["samples"])
                for name in score_values
            }
            summaries[group["name"]]["native_mean_score_deltas"] = {
                name: summaries[group["name"]]["native_mean_scores"][name] - value
                for name, value in reference_means.items()
            }
    return results, summaries


def flatten_state(value, path, destination, row, batch_size, num_heads, recurrent=False):
    if isinstance(value, torch.Tensor):
        expected = batch_size * num_heads if recurrent else batch_size
        if value.ndim < 1 or value.shape[0] != expected:
            raise ValueError(
                f"Unexpected native cache shape at {path}: {tuple(value.shape)}; expected first dimension {expected}"
            )
        start, stop = (row * num_heads, (row + 1) * num_heads) if recurrent else (row, row + 1)
        selected = value[start:stop].detach()
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Nonfinite native cache at {path}")
        destination[path] = selected.cpu().clone()
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            flatten_state(
                item, f"{path}/{index}", destination, row, batch_size, num_heads, recurrent
            )
    elif isinstance(value, dict):
        for key, item in value.items():
            flatten_state(item, f"{path}/{key}", destination, row, batch_size, num_heads, recurrent)
    elif value is None or isinstance(value, (bool, int, float)):
        destination[path] = value
    else:
        raise TypeError(f"Unexpected native cache value at {path}: {type(value)}")


def snapshot(output, row, batch_size, num_heads, position):
    cache = output.past_key_values
    tensors, progress = {}, []
    for index in range(len(cache)):
        progress.append(cache.get_seq_length(index))
        for key, value in cache[index].items():
            flatten_state(
                value,
                f"layer/{index}/{key}",
                tensors,
                row,
                batch_size,
                num_heads,
                key == "recurrent_state",
            )
    logits = output.logits[row, -1].detach().float().cpu().clone()
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Nonfinite teacher-forced logits.")
    return {"position": position, "progress": progress, "state": tensors, "logits": logits}


def run_probe(model, token_rows, prefix_length, num_heads):
    inputs = torch.tensor(
        [row[:prefix_length] for row in token_rows], dtype=torch.long, device="cuda:0"
    )
    output = model(
        input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=True, logits_to_keep=1
    )
    snapshots = [snapshot(output, 0, len(token_rows), num_heads, prefix_length)]
    cache = output.past_key_values
    for position in range(prefix_length + 1, prefix_length + 3):
        inputs = torch.tensor(
            [[row[position - 1]] for row in token_rows], dtype=torch.long, device="cuda:0"
        )
        output = model(
            input_ids=inputs,
            attention_mask=torch.ones(
                (len(token_rows), position), dtype=torch.long, device="cuda:0"
            ),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = output.past_key_values
        snapshots.append(snapshot(output, 0, len(token_rows), num_heads, position))
    del output, cache
    return snapshots


def tensor_difference(actual, expected, atol, rtol):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {
            "structure_matches": False,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "actual_dtype": str(actual.dtype),
            "expected_dtype": str(expected.dtype),
        }
    delta = actual.float() - expected.float()
    count = delta.numel()
    return {
        "structure_matches": True,
        "equal": torch.equal(actual, expected),
        "allclose": torch.allclose(actual, expected, atol=atol, rtol=rtol),
        "max_abs": float(delta.abs().max()) if count else 0.0,
        "rms": float(delta.square().mean().sqrt()) if count else 0.0,
    }


def compare_probe(actual, expected, name, args, measurements, isolation=False):
    atol = args.isolation_atol if isolation else args.numeric_atol
    rtol = args.isolation_rtol if isolation else args.numeric_rtol
    structure_ok, numerics_ok = True, True
    for current, reference in zip(actual, expected, strict=True):
        paths_match = current["state"].keys() == reference["state"].keys()
        progress_match = (
            current["position"] == reference["position"]
            and current["progress"] == reference["progress"]
        )
        structure_ok &= paths_match and progress_match
        append_json(
            measurements,
            {
                "record_type": "cache_progress",
                "comparison": name,
                "position": current["position"],
                "progress": current["progress"],
                "reference_progress": reference["progress"],
                "state_paths_match": paths_match,
                "progress_matches": progress_match,
            },
        )
        for path in sorted(current["state"].keys() | reference["state"].keys()):
            value, target = current["state"].get(path), reference["state"].get(path)
            if isinstance(value, torch.Tensor) and isinstance(target, torch.Tensor):
                difference = tensor_difference(value, target, atol, rtol)
                structure_ok &= difference["structure_matches"]
                numerics_ok &= difference.get("allclose", False)
            else:
                difference = {
                    "structure_matches": type(value) is type(target),
                    "equal": value == target,
                    "allclose": value == target,
                }
                structure_ok &= difference["structure_matches"]
                numerics_ok &= difference["allclose"]
            append_json(
                measurements,
                {
                    "record_type": "cache_comparison",
                    "comparison": name,
                    "position": current["position"],
                    "path": path,
                    "atol": atol,
                    "rtol": rtol,
                    **difference,
                },
            )
        difference = tensor_difference(current["logits"], reference["logits"], atol, rtol)
        numerics_ok &= difference.get("allclose", False)
        append_json(
            measurements,
            {
                "record_type": "teacher_logits",
                "comparison": name,
                "position": current["position"],
                "argmax_equal": int(current["logits"].argmax())
                == int(reference["logits"].argmax()),
                "atol": atol,
                "rtol": rtol,
                **difference,
            },
        )
    return {
        "structure_passed": bool(structure_ok),
        "numeric_allclose": bool(numerics_ok),
        "required_passed": bool(structure_ok and (numerics_ok if isolation else True)),
        "isolation": isolation,
        "atol": atol,
        "rtol": rtol,
    }


def execute(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.checkpoint / "config.json").read_text())
    sources = {
        mode: {
            **source_record(argparse.Namespace(inference_mode=mode)),
            "entrypoint": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        }
        for mode in ("original", "optimized")
    }
    gpu = {
        "name": torch.cuda.get_device_name(0),
        "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    }
    plan = {
        "created_at": timestamp(),
        "source": sources,
        "gpu": gpu,
        "arguments": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "checkpoint_config": config,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "e2_ttt": e2_ttt.__version__,
            "cuda": torch.version.cuda,
        },
        "seeds": {"random": 0, "numpy": 1234, "torch": 1234},
        "generation": "native HFLM B1; optimized EqualLengthHFLM B1/Bn, original task prompts/stops/scoring",
        "sample_selection": "RULER: first exact-token-length bucket reaching requested 8/16 rows; recall: largest post-truncation length bucket (>=4 rows), plus up to two other-length rows interleaved, up to recall-samples total",
        "completion_ids": "raw IDs retained; comparison stops at first original EOS or singleton native until criterion, retaining its terminating token",
        "probe": "real fixed teacher tokens: prefill C-1 and 2C-1, then two one-token forwards; row0 fixed while other rows are permuted/replaced",
        "acceptance": "cache structure/progress/finite and same-B isolation required; cross-B numeric/greedy-ID differences reported, not automatic failures",
        "timing": "warmups excluded; native generation wall and first-forward CUDA event span measured with hooks; diagnostic CPU tensor copies excluded",
    }
    write_json(args.output_dir / "plan.json", plan)
    summary = {"completed": False, "source": sources, "gpu": gpu, "generation": {}, "probes": {}}
    started = time.perf_counter()
    random.seed(0)
    np.random.seed(1234)
    torch.manual_seed(1234)
    try:
        lm = HFLM(
            pretrained=str(args.checkpoint),
            tokenizer=str(args.checkpoint),
            dtype="bfloat16",
            device="cuda:0",
            batch_size=1,
            max_length=max(args.lengths),
            truncation=False,
            trust_remote_code=False,
        )
        lm.model.eval()
        groups = select_groups(args, lm)
        plan["selected_samples"] = {
            group["name"]: [
                {"id": row["id"], "input_tokens": len(row["input_ids"])} for row in group["samples"]
            ]
            for group in groups
        }
        chunk_size, num_heads = lm.model.config.ttt_chunk_size, lm.model.config.num_heads
        probe_samples = next(
            group["samples"]
            for group in groups
            if group["name"].startswith("ruler/")
            and min(len(row["input_ids"]) for row in group["samples"]) >= 2 * chunk_size + 1
        )
        token_rows = [sample["input_ids"] for sample in probe_samples]
        plan["probe_sample_ids"] = [
            sample["id"] for sample in probe_samples[: 2 * args.probe_batch_size - 1]
        ]
        size = args.probe_batch_size
        ids = [sample["id"] for sample in probe_samples]
        plan["probe_row_layouts"] = {
            "baseline": ids[:size],
            "permute-other-rows": [ids[0], *reversed(ids[1:size])],
            "replace-other-rows": [ids[0], *ids[size : 2 * size - 1]],
        }
        plan["probe_positions"] = [
            [chunk_size - 1, chunk_size, chunk_size + 1],
            [2 * chunk_size - 1, 2 * chunk_size, 2 * chunk_size + 1],
        ]
        write_json(args.output_dir / "plan.json", plan)
        with (
            (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as predictions,
            (args.output_dir / "measurements.jsonl").open("w", encoding="utf-8") as measurements,
            torch.inference_mode(),
        ):
            original, summary["generation"]["original-b1"] = run_generation(
                args, lm, groups, "original-b1", 1, False, predictions, measurements, {}
            )
            prefixes = [chunk_size - 1, 2 * chunk_size - 1]
            references = {
                prefix: run_probe(lm.model, token_rows[:1], prefix, num_heads)
                for prefix in prefixes
            }
            parameter_contract = {
                name: (value.data_ptr(), value.shape, value.dtype)
                for name, value in lm.model.state_dict().items()
            }
            model = optimize_model(lm.model)
            if model is not lm.model or parameter_contract != {
                name: (value.data_ptr(), value.shape, value.dtype)
                for name, value in model.state_dict().items()
            }:
                raise ValueError(
                    "Optimization changed model identity or checkpoint tensor storage/layout."
                )
            for batch_size in sorted(set([1, *args.batch_sizes])):
                name = f"optimized-b{batch_size}"
                try:
                    _, summary["generation"][name] = run_generation(
                        args,
                        lm,
                        groups,
                        name,
                        batch_size,
                        True,
                        predictions,
                        measurements,
                        original,
                    )
                except torch.cuda.OutOfMemoryError as exc:
                    summary["generation"][name] = {"cuda_oom": True, "error": str(exc)}
                    torch.cuda.empty_cache()
                write_json(args.output_dir / "summary.json", summary)
            size = args.probe_batch_size
            baseline_rows = token_rows[:size]
            permuted_rows = [token_rows[0], *reversed(token_rows[1:size])]
            replacement_rows = [token_rows[0], *token_rows[size : 2 * size - 1]]
            for prefix in prefixes:
                single = run_probe(model, token_rows[:1], prefix, num_heads)
                summary["probes"][f"optimized-b1/prefix-{prefix}"] = compare_probe(
                    single,
                    references[prefix],
                    f"optimized-b1-vs-original/prefix-{prefix}",
                    args,
                    measurements,
                )
                del single
                baseline = run_probe(model, baseline_rows, prefix, num_heads)
                summary["probes"][f"optimized-b{size}/prefix-{prefix}"] = compare_probe(
                    baseline,
                    references[prefix],
                    f"optimized-b{size}-vs-original/prefix-{prefix}",
                    args,
                    measurements,
                )
                for label, rows in (
                    ("permute-other-rows", permuted_rows),
                    ("replace-other-rows", replacement_rows),
                ):
                    changed = run_probe(model, rows, prefix, num_heads)
                    summary["probes"][f"{label}/prefix-{prefix}"] = compare_probe(
                        changed,
                        baseline,
                        f"{label}/prefix-{prefix}",
                        args,
                        measurements,
                        isolation=True,
                    )
                    del changed
                del baseline, references[prefix]
            summary["required_checks_passed"] = all(
                result["required_passed"] for result in summary["probes"].values()
            )
            summary["completed"] = True
            summary["elapsed_seconds"] = time.perf_counter() - started
            write_json(args.output_dir / "summary.json", summary)
            if not summary["required_checks_passed"]:
                raise ValueError(
                    "A required cache structure/progress or same-batch isolation check failed; inspect measurements.jsonl."
                )
    except Exception as exc:
        summary["error"] = repr(exc)
        summary["elapsed_seconds"] = time.perf_counter() - started
        write_json(args.output_dir / "summary.json", summary)
        raise


if __name__ == "__main__":
    execute(parse_args())
