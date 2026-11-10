"""Measure original E²-TTT generation phases and cache storage without changing computation."""

import argparse
from collections import defaultdict
from datetime import datetime
import gc
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

import e2_ttt  # noqa: F401 -- register official model classes
from fla.models.utils import Cache
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import handle_stop_sequences, postprocess_generated_text
from lm_eval.tasks.ruler.common_utils import process_results
import torch
import transformers


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LENGTHS = (512, 4096, 16384)
TASKS = ("niah_single_1", "niah_single_2")
UPSTREAM_REVISION = "0d1da370c7e3cea1642ba3af0477b7601849a1be"
STATE_NAMES = {
    "e2_ttt_mlp": (
        "w0", "w1", "buf_k", "buf_v", "buf_lr", "buf_log_decay",
        "buf_log_momentum", "momentum_w0", "momentum_w1",
    ),
    "e2_ttt_swiglu": (
        "w0", "w1", "w2", "buf_k", "buf_v", "buf_lr", "buf_log_decay",
        "buf_log_momentum", "momentum_w0", "momentum_w1", "momentum_w2",
    ),
}


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_record():
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()

    status = git("status", "--porcelain=v1", "--untracked-files=normal").splitlines()
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "git_status": status,
        "upstream_revision": UPSTREAM_REVISION,
        "entrypoint": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
    }


def cache_snapshot(cache, model_type, num_layers, parameter_storages):
    """Return CPU metadata only; storage excess is not a measured allocator saving."""
    names = STATE_NAMES[model_type]
    tensors = []
    storages = {}
    categories = defaultdict(lambda: {"logical_bytes": 0, "storage_ids": set()})

    def add(layer, category, name, tensor):
        if tensor is None:
            return
        logical = tensor.numel() * tensor.element_size()
        storage = tensor.untyped_storage()
        pointer = storage.data_ptr()
        key = (str(tensor.device), pointer)
        if logical == 0:
            storage_id = None
        else:
            if key not in storages:
                storages[key] = {
                    "id": len(storages), "bytes": storage.nbytes(), "categories": set(),
                    "aliases_model_parameter": key in parameter_storages,
                }
            record = storages[key]
            record["categories"].add(category)
            storage_id = record["id"]
            categories[category]["storage_ids"].add(storage_id)
        categories[category]["logical_bytes"] += logical
        tensors.append({
            "layer": layer, "category": category, "name": name,
            "shape": list(tensor.shape), "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype), "storage_offset": tensor.storage_offset(),
            "logical_bytes": logical, "storage_bytes": storage.nbytes(),
            "storage_id": storage_id,
            "storage_larger_than_tensor": storage.nbytes() > logical,
            "aliases_model_parameter": key in parameter_storages,
        })

    for layer in range(num_layers):
        state = cache[layer]
        for i, tensor in enumerate(state["attn_state"]):
            add(layer, "attention_kv", ("k", "v")[i], tensor)
        recurrent = state["recurrent_state"]
        if len(recurrent) != len(names):
            raise ValueError(f"Unexpected recurrent state length: {len(recurrent)} for {model_type}")
        for name, tensor in zip(names, recurrent, strict=True):
            category = (
                "ttt_buffers" if name.startswith("buf_")
                else "momentum" if name.startswith("momentum_")
                else "fast_weights"
            )
            add(layer, category, name, tensor)

    storage_records = list(storages.values())
    by_id = {row["id"]: row for row in storage_records}
    summary = {}
    for category, values in categories.items():
        total_storage = sum(by_id[i]["bytes"] for i in values["storage_ids"])
        summary[category] = {
            "logical_bytes": values["logical_bytes"],
            "unique_storage_bytes": total_storage,
            "unique_storage_bytes_excluding_model_parameters": sum(
                by_id[i]["bytes"] for i in values["storage_ids"]
                if not by_id[i]["aliases_model_parameter"]
            ),
            "storage_bytes_minus_logical_bytes": total_storage - values["logical_bytes"],
        }
    for record in storage_records:
        record["categories"] = sorted(record["categories"])
    return {
        "cuda_allocated_bytes": torch.cuda.memory_allocated(0),
        "cuda_reserved_bytes": torch.cuda.memory_reserved(0),
        "categories": summary,
        "unique_storage_bytes": sum(row["bytes"] for row in storage_records),
        "unique_storage_bytes_excluding_model_parameters": sum(
            row["bytes"] for row in storage_records if not row["aliases_model_parameter"]
        ),
        "storages": storage_records,
        "tensors": tensors,
        "interpretation": (
            "Storage excess describes retained backing storage, not measured releasable memory; "
            "storage shared across categories is counted once globally."
        ),
    }


class ForwardRecorder:
    def __init__(self, model):
        self.model = model
        self.calls = []
        self.snapshots = {}
        self.pending = None
        self.handles = []
        self.parameter_storages = {
            (str(parameter.device), parameter.untyped_storage().data_ptr())
            for parameter in model.parameters()
        }

    def before(self, module, args, kwargs):
        input_ids = kwargs["input_ids"]
        count = input_ids.shape[1]
        cache = kwargs.get("past_key_values")
        if count > 1:
            phase, tail = "prefill", None
        else:
            recurrent = cache[0]["recurrent_state"]
            names = STATE_NAMES[self.model.config.model_type]
            tail = recurrent[names.index("buf_k")].shape[-1]
            phase = "decode_chunk_update" if tail + count == self.model.config.ttt_chunk_size else "decode_read"
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        self.pending = {
            "index": len(self.calls), "phase": phase, "input_tokens": count,
            "tail_tokens_before": tail, "start": start, "end": end,
            "cpu_start": time.perf_counter(),
        }

    def after(self, module, args, kwargs, output):
        call = self.pending
        call["cpu_submission_seconds"] = time.perf_counter() - call.pop("cpu_start")
        call["end"].record()
        self.calls.append(call)
        if len(self.calls) <= 2:
            label = "after_prefill" if len(self.calls) == 1 else "after_first_decode"
            self.snapshots[label] = cache_snapshot(
                output.past_key_values, self.model.config.model_type,
                self.model.config.num_hidden_layers, self.parameter_storages,
            )
        self.pending = None

    def __enter__(self):
        self.handles = [
            self.model.register_forward_pre_hook(self.before, with_kwargs=True),
            self.model.register_forward_hook(self.after, with_kwargs=True),
        ]
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()

    def result(self):
        rows = []
        summary = defaultdict(lambda: {"calls": 0, "cuda_forward_span_seconds": 0.0, "cpu_submission_seconds": 0.0})
        for call in self.calls:
            row = {key: value for key, value in call.items() if key not in ("start", "end")}
            row["cuda_forward_span_seconds"] = call["start"].elapsed_time(call["end"]) / 1000
            rows.append(row)
            aggregate = summary[row["phase"]]
            aggregate["calls"] += 1
            aggregate["cuda_forward_span_seconds"] += row["cuda_forward_span_seconds"]
            aggregate["cpu_submission_seconds"] += row["cpu_submission_seconds"]
        return {
            "forwards": rows, "phases": dict(summary), "cache_snapshots": self.snapshots,
            "timing_interpretation": (
                "CUDA events measure whole-forward spans including launch gaps, not kernel-only totals; "
                "CPU submission durations may include synchronization performed by the original implementation."
            ),
        }


def generate(lm, input_ids, attention_mask, generation_tokens, instrument):
    gc.collect()
    torch.cuda.synchronize(0)
    torch.cuda.reset_peak_memory_stats(0)
    stop = handle_stop_sequences([], eos=lm.tok_decode(lm.eot_token_id, skip_special_tokens=False))
    started = time.perf_counter()
    if instrument:
        with ForwardRecorder(lm.model) as recorder:
            output = lm._model_generate(
                context=input_ids, attention_mask=attention_mask,
                max_length=input_ids.shape[1] + generation_tokens,
                stop=stop, do_sample=False, temperature=0.0,
            )
    else:
        output = lm._model_generate(
            context=input_ids, attention_mask=attention_mask,
            max_length=input_ids.shape[1] + generation_tokens,
            stop=stop, do_sample=False, temperature=0.0,
        )
    torch.cuda.synchronize(0)
    wall = time.perf_counter() - started
    output_ids = output[0].tolist()
    generated_ids = output_ids[input_ids.shape[1]:]
    prediction = postprocess_generated_text(
        generation=lm.tok_decode(generated_ids), stop=stop, think_end_token=None,
    )
    result = {
        "wall_seconds": wall,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(0),
        "input_ids": input_ids[0].tolist(), "output_ids": output_ids,
        "generated_ids": generated_ids, "generated_tokens": len(generated_ids),
        "prediction": prediction,
    }
    if instrument:
        result["diagnostic"] = recorder.result()
    return result


def read_sample(root, length, task):
    path = root / str(length) / task / "test.jsonl"
    with path.open(encoding="utf-8") as stream:
        next(stream)
        row = json.loads(next(stream))
    return row, path


def execute(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.checkpoint / "config.json").read_text())
    plan = {
        "created_at": timestamp(), "source": source_record(),
        "checkpoint": str(args.checkpoint), "checkpoint_config": config,
        "ruler_data": str(args.ruler_data), "lengths": list(LENGTHS),
        "tasks": list(TASKS), "row_index": 1, "generation_tokens": 128,
        "boundary_prefix_lengths": [511, 512], "boundary_generation_tokens": 4,
        "batch_size": 1, "dtype": "bfloat16", "do_sample": False,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "versions": {
            "torch": torch.__version__, "transformers": transformers.__version__,
            "e2_ttt": e2_ttt.__version__, "flash_linear_attention": version("flash-linear-attention"),
            "flash_attn": version("flash-attn"), "triton": version("triton"),
        },
        "protocol": (
            "Each case runs uninstrumented original HFLM generation, then original generation with whole-model hooks; "
            "wall baseline excludes hooks; compare complete input/output token IDs. "
            "No cache or update arithmetic is modified."
        ),
    }
    write_json(args.output_dir / "plan.json", plan)
    started = time.perf_counter()
    results = []
    try:
        lm = HFLM(
            pretrained=str(args.checkpoint), tokenizer=str(args.checkpoint),
            dtype="bfloat16", device="cuda:0", batch_size=1,
            max_length=max(LENGTHS), truncation=False, trust_remote_code=False,
        )
        runtime = {
            "model_class": f"{type(lm.model).__module__}.{type(lm.model).__qualname__}",
            "model_source": inspect.getfile(type(lm.model)),
            "cache_class": f"{Cache.__module__}.{Cache.__qualname__}",
            "cache_source": inspect.getfile(Cache),
            "device_name": torch.cuda.get_device_name(0),
            "device_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        }
        write_json(args.output_dir / "runtime.json", runtime)

        def encode(prompt, budget):
            ids, mask = lm.tok_batch_encode([prompt], left_truncate_len=None, truncation=False)
            if ids.shape[1] > budget:
                raise ValueError(f"Input length {ids.shape[1]} exceeds budget {budget}")
            return ids.to(lm.device), mask.to(lm.device)

        warmup_row, _ = read_sample(args.ruler_data, 512, TASKS[0])
        warm_ids, warm_mask = encode(warmup_row["input"] + warmup_row["answer_prefix"], 384)
        warmup = generate(lm, warm_ids, warm_mask, 128, instrument=False)
        write_json(args.output_dir / "warmup.json", warmup)
        for length in LENGTHS:
            for task in TASKS:
                row, path = read_sample(args.ruler_data, length, task)
                ids, mask = encode(row["input"] + row["answer_prefix"], length - 128)
                item = {
                    "case": f"{task}_{length}_row1", "kind": "ruler",
                    "source": str(path), "row_index": 1, "length": length,
                }
                # Each real shape is warmed separately; cold compilation is not a speed baseline.
                warmup = generate(lm, ids, mask, 128, instrument=False)
                item["shape_warmup_wall_seconds"] = warmup["wall_seconds"]
                for mode in ("original", "instrumented"):
                    item[mode] = generate(lm, ids, mask, 128, instrument=mode == "instrumented")
                    scored_row = {**row, "max_length": length}
                    item[mode]["native_score"] = process_results(scored_row, [item[mode]["prediction"]])[str(length)]
                item["input_ids_equal"] = item["original"]["input_ids"] == item["instrumented"]["input_ids"]
                item["output_ids_equal"] = item["original"]["output_ids"] == item["instrumented"]["output_ids"]
                results.append(item)
                write_json(args.output_dir / f"{item['case']}.json", item)
                print(json.dumps({"case": item["case"], "wall_seconds": item["original"]["wall_seconds"], "output_ids_equal": item["output_ids_equal"]}), flush=True)

        row, path = read_sample(args.ruler_data, 4096, TASKS[0])
        source_ids, source_mask = encode(row["input"] + row["answer_prefix"], 4096 - 128)
        for length in (511, 512):
            ids, mask = source_ids[:, :length], source_mask[:, :length]
            item = {
                "case": f"boundary_prefix_{length}", "kind": "boundary",
                "source": str(path), "row_index": 1,
                "description": "Real prompt token prefix; not a scored benchmark example.",
            }
            warmup = generate(lm, ids, mask, 4, instrument=False)
            item["shape_warmup_wall_seconds"] = warmup["wall_seconds"]
            for mode in ("original", "instrumented"):
                item[mode] = generate(lm, ids, mask, 4, instrument=mode == "instrumented")
            item["input_ids_equal"] = item["original"]["input_ids"] == item["instrumented"]["input_ids"]
            item["output_ids_equal"] = item["original"]["output_ids"] == item["instrumented"]["output_ids"]
            results.append(item)
            write_json(args.output_dir / f"{item['case']}.json", item)
        all_equal = all(row["input_ids_equal"] and row["output_ids_equal"] for row in results)
        covered_phases = sorted({
            phase for row in results for phase in row["instrumented"]["diagnostic"]["phases"]
        })
        coverage_complete = set(covered_phases) == {"prefill", "decode_read", "decode_chunk_update"}
        write_json(args.output_dir / "summary.json", {
            "completed": True, "all_input_output_ids_equal": all_equal,
            "covered_phases": covered_phases, "phase_coverage_complete": coverage_complete,
            "wall_interpretation": (
                "Uninstrumented wall covers HFLM _model_generate and final CUDA synchronization; "
                "CPU tokenization, prediction decoding, scoring and cache metadata collection are excluded."
            ),
            "cases": [{
                "case": row["case"], "kind": row["kind"],
                "prompt_tokens": len(row["original"]["input_ids"]),
                "generated_tokens": row["original"]["generated_tokens"],
                "original_wall_seconds": row["original"]["wall_seconds"],
                "original_peak_cuda_allocated_bytes": row["original"]["peak_cuda_allocated_bytes"],
                "output_ids_equal": row["output_ids_equal"],
                "phases": row["instrumented"]["diagnostic"]["phases"],
                "cache_categories_after_prefill": row["instrumented"]["diagnostic"]["cache_snapshots"]["after_prefill"]["categories"],
            } for row in results],
        })
        write_json(args.output_dir / "execution.json", {
            "finished_at": timestamp(), "elapsed_seconds": time.perf_counter() - started,
            "exit_code": 0 if all_equal else 1, "all_input_output_ids_equal": all_equal,
            "phase_coverage_complete": coverage_complete,
        })
        if not all_equal:
            raise ValueError("Instrumented and uninstrumented generation token IDs differ")
        if not coverage_complete:
            raise ValueError(f"Incomplete phase coverage with EOS preserved: {covered_phases}")
    except Exception as error:
        write_json(args.output_dir / "execution.json", {
            "finished_at": timestamp(), "elapsed_seconds": time.perf_counter() - started,
            "exit_code": 1, "error": repr(error), "completed_cases": len(results),
        })
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--ruler-data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        parser.error("This diagnostic requires physical GPU 0: CUDA_VISIBLE_DEVICES=0")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("This diagnostic requires one process")
    for name in ("checkpoint", "ruler_data", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


if __name__ == "__main__":
    execute(parse_args())
