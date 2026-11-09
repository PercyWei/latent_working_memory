"""配对检查 In-Place TTT 推理优化，并分别记录 prefill、decode 和显存。"""

import argparse
from datetime import datetime
import gc
import json
import logging
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

from evaluate import TASKS as EVALUATORS
from inference_model.hf_qwen3.modeling_qwen3 import TTTDynamicCache
from opencompass.utils.text_postprocessors import extract_non_reasoning_content
from optimized.model import CompactTTTDynamicCache
from provenance import capture_source
from runtime import MODEL_CLASSES, generate_one, load_model, runtime_info
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TASKS = ("niah_single_1", "niah_multiquery")
MODES = ("original", "tail", "chunked")


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d %H:%M:%S Asia/Shanghai")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_json(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def select_mode(wrapper, mode, chunk_size):
    wrapper.inference_mode = "original" if mode == "original" else "optimized"
    wrapper.model.set_optimization(mode != "original", chunk_size if mode == "chunked" else 0)


def cache_metadata(cache, config):
    seen = cache.get_seq_length()
    storages = {}
    for layer in cache.layers:
        for value in (layer.keys, layer.values):
            storage = value.untyped_storage()
            storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
    kv_consistent = all(layer.keys.shape[-2] == seen for layer in cache.layers)
    tails = []
    tail_storages = {}
    if config.ttt_mode:
        for layer_idx in config.ttt_layers:
            h, t, w = cache.ttt_states[layer_idx]
            fields = {}
            for name, value in (("h", h), ("t", t)):
                if value is None:
                    fields[name] = None
                else:
                    storage = value.untyped_storage()
                    size = storage.nbytes()
                    tail_storages[(str(value.device), storage.data_ptr())] = size
                    fields[name] = {
                        "shape": list(value.shape),
                        "logical_bytes": value.numel() * value.element_size(),
                        "storage_bytes": size,
                    }
            expected_tail = seen % config.ttt_chunk
            fields.update(
                layer=layer_idx,
                weight_shape=list(w.shape),
                consistent=(
                    (0 if h is None else h.shape[1]) == expected_tail
                    and (0 if t is None else t.shape[1]) == expected_tail
                ),
            )
            tails.append(fields)
    return {
        "seen": seen,
        "kv_storage_bytes": sum(storages.values()),
        "tail_storage_bytes": sum(tail_storages.values()),
        "consistent": kv_consistent and all(row["consistent"] for row in tails),
        "tails": tails,
    }


def snapshot_ttt(cache, config):
    if not config.ttt_mode:
        return []
    return [
        tuple(
            None if value is None else value.detach().cpu().clone() for value in cache.ttt_states[i]
        )
        for i in config.ttt_layers
    ]


def compare_states(reference, actual):
    equal = len(reference) == len(actual)
    finite = True
    max_abs = 0.0
    for old, new in zip(reference, actual, strict=True):
        for left, right in zip(old, new, strict=True):
            if left is None or right is None:
                equal = equal and left is None and right is None
                continue
            if left.shape != right.shape:
                equal = False
                continue
            finite = finite and bool(torch.isfinite(left).all() and torch.isfinite(right).all())
            equal = equal and torch.equal(left, right)
            if finite:
                max_abs = max(max_abs, (left.float() - right.float()).abs().max().item())
    return {"bitwise_equal": equal, "finite": finite, "max_abs": max_abs if finite else None}


class GenerationObserver:
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.events = []
        self.cache = None
        self.last_logits = None
        self.prefill_ids = None
        self.prefill_cache = None
        self.tokenizer_seconds = 0.0
        self.tokenizer_calls = []
        self.tokenizer_methods = {}

    def __enter__(self):
        def before(module, arguments, kwargs):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self.events.append([kwargs["input_ids"].shape[1], start, None])
            if self.prefill_ids is None:
                self.prefill_ids = kwargs["input_ids"]

        def after(module, arguments, kwargs, output):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.events[-1][2] = end
            self.cache = output.past_key_values
            self.last_logits = output.logits.detach()
            if self.prefill_cache is None:
                self.prefill_cache = cache_metadata(self.cache, module.config)

        self.handles = [
            self.wrapper.model.register_forward_pre_hook(before, with_kwargs=True),
            self.wrapper.model.register_forward_hook(after, with_kwargs=True),
        ]
        for name in ("encode", "batch_encode_plus"):
            original = getattr(self.wrapper.tokenizer, name)
            self.tokenizer_methods[name] = original

            def timed(*args, _original=original, _name=name, **kwargs):
                start = time.perf_counter()
                result = _original(*args, **kwargs)
                self.tokenizer_seconds += time.perf_counter() - start
                self.tokenizer_calls.append(_name)
                return result

            setattr(self.wrapper.tokenizer, name, timed)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        for name, original in self.tokenizer_methods.items():
            setattr(self.wrapper.tokenizer, name, original)


def measure(wrapper, prompt, length):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with GenerationObserver(wrapper) as observer:
        result = generate_one(wrapper, prompt, max_out_len=64, max_total_tokens=length)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        prefill_ms = observer.events[0][1].elapsed_time(observer.events[0][2])
        decode_ms = sum(start.elapsed_time(end) for _, start, end in observer.events[1:])
        final_cache = cache_metadata(observer.cache, wrapper.model.config)
        state = snapshot_ttt(observer.cache, wrapper.model.config)
        input_ids = observer.prefill_ids.detach().cpu().clone()
        logits_finite = bool(torch.isfinite(observer.last_logits).all())
        state_finite = all(
            torch.isfinite(value).all().item()
            for row in state
            for value in row
            if value is not None
        )
    result.update(
        seconds=seconds,
        prefill_seconds=prefill_ms / 1000,
        decode_seconds=decode_ms / 1000,
        forward_calls=len(observer.events),
        tokenizer_seconds=observer.tokenizer_seconds,
        tokenizer_calls=observer.tokenizer_calls,
        peak_allocated_bytes=allocated,
        peak_reserved_bytes=reserved,
        prefill_cache=observer.prefill_cache,
        final_cache=final_cache,
        last_logits_finite=logits_finite,
        ttt_state_finite=state_finite,
    )
    observer.cache = None
    return result, {"ttt": state, "input_ids": input_ids}


def compare_tokens(reference, actual):
    first = next((i for i, (a, b) in enumerate(zip(reference, actual)) if a != b), None)
    if first is None and len(reference) != len(actual):
        first = min(len(reference), len(actual))
    return {"equal": reference == actual, "first_different_position": first}


def boundary_checks(wrapper, modes, chunk_size, output, logger):
    tokenizer = wrapper.tokenizer
    seed = tokenizer.encode(
        " Alice stored the number 1729 beside a blue river.", add_special_tokens=False
    )
    continuation = tokenizer.encode(
        " The answer is remembered near the library.", add_special_tokens=False
    )[:4]
    records = []
    for prefix_len in dict.fromkeys((1022, 1024, 2046, chunk_size + 1022)):
        tokens = (seed * (prefix_len // len(seed) + 1))[:prefix_len]
        prefix = torch.tensor([tokens], device="cuda:0")
        references = []
        for mode in modes:
            select_mode(wrapper, mode, chunk_size)
            cache = None
            if wrapper.model.config.ttt_mode:
                cache = TTTDynamicCache() if mode == "original" else CompactTTTDynamicCache()
            for step in range(5):
                inputs = (
                    prefix
                    if step == 0
                    else torch.tensor([[continuation[step - 1]]], device="cuda:0")
                )
                with torch.no_grad():
                    result = wrapper.model(
                        input_ids=inputs,
                        attention_mask=torch.ones(
                            (1, prefix_len + step), device="cuda:0", dtype=torch.long
                        ),
                        past_key_values=cache,
                        use_cache=True,
                        logits_to_keep=1,
                        return_dict=True,
                    )
                cache = result.past_key_values
                logits = result.logits[0, -1].detach().float().cpu()
                state = snapshot_ttt(cache, wrapper.model.config)
                meta = cache_metadata(cache, wrapper.model.config)
                if mode == "original":
                    references.append((logits, state))
                old_logits, old_state = references[step]
                finite = bool(torch.isfinite(logits).all() and torch.isfinite(old_logits).all())
                record = {
                    "prefix_tokens": prefix_len,
                    "step": step,
                    "mode": mode,
                    "finite": finite,
                    "argmax_equal": int(logits.argmax()) == int(old_logits.argmax()),
                    "max_abs": (logits - old_logits).abs().max().item() if finite else None,
                    "state": compare_states(old_state, state),
                    "cache": meta,
                }
                append_json(output / "boundary.jsonl", record)
                records.append(record)
                del result
            del cache
        logger.info("Boundary prefix=%s finished", prefix_len)
    return records


def profile_attention(wrapper, prompt, output):
    select_mode(wrapper, "original", 0)
    ids = wrapper.tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda:0")
    signatures = {}
    original = F.scaled_dot_product_attention

    def observed(q, k, v, *args, **kwargs):
        signature = {
            "q": list(q.shape),
            "k": list(k.shape),
            "v": list(v.shape),
            "dtype": str(q.dtype),
            "mask_is_none": kwargs.get("attn_mask") is None,
            "enable_gqa": kwargs.get("enable_gqa", False),
        }
        key = json.dumps(signature, sort_keys=True)
        signatures.setdefault(key, {**signature, "calls": 0})["calls"] += 1
        return original(q, k, v, *args, **kwargs)

    F.scaled_dot_product_attention = observed
    try:
        with (
            torch.no_grad(),
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            ) as profiler,
        ):
            first = wrapper.model(
                input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, logits_to_keep=1
            )
            token = first.logits[:, -1].argmax(dim=-1, keepdim=True)
            second = wrapper.model(
                input_ids=token,
                attention_mask=torch.ones((1, ids.shape[1] + 1), dtype=torch.long, device="cuda:0"),
                past_key_values=first.past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            torch.cuda.synchronize()
        operators = [
            {"name": row.key, "calls": row.count}
            for row in profiler.key_averages()
            if "scaled_dot_product" in row.key or "flash_attention" in row.key
        ]
    finally:
        F.scaled_dot_product_attention = original
    write_json(
        output / "attention_profile.json",
        {"operators": operators, "inputs": list(signatures.values())},
    )
    del first, second
    gc.collect()
    torch.cuda.empty_cache()


def run(args, output, source, logger):
    plan = {
        "created_at": timestamp(),
        "purpose": "paired validation; not formal benchmark",
        "checkpoint": str(args.checkpoint.resolve()),
        "ruler_data": str(args.ruler_data.resolve()),
        "lengths": args.lengths,
        "tasks": list(TASKS),
        "samples_per_task_length": 1,
        "synthetic_prefix_tokens": list(
            dict.fromkeys((1022, 1024, 2046, args.mlp_chunk_size + 1022))
        ),
        "modes": list(MODES),
        "mlp_chunk_size": args.mlp_chunk_size,
        "generation": {"max_new_tokens": 64, "do_sample": False, "preserve_eos": True},
        "source": source,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "timing": "wall includes tokenization; CUDA events measure LM forwards without per-step synchronization",
    }
    write_json(output / "plan.json", plan)
    started = time.perf_counter()
    wrapper = load_model(
        args.checkpoint,
        inference_mode="optimized",
        mlp_chunk_size=args.mlp_chunk_size,
        max_seq_len=max(args.lengths),
    )
    torch.cuda.synchronize()
    config = wrapper.model.config
    modes = MODES if config.ttt_mode else ("original", "chunked")
    plan.update(
        effective_modes=list(modes),
        load_seconds=time.perf_counter() - started,
        runtime={
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "attention_implementation": config._attn_implementation,
            "ttt_mode": config.ttt_mode,
        },
        checkpoint_config=config.to_dict(),
    )
    plan["runtime_by_mode"] = {}
    for mode in modes:
        select_mode(wrapper, mode, args.mlp_chunk_size)
        plan["runtime_by_mode"][mode] = runtime_info(wrapper)
    write_json(output / "plan.json", plan)
    samples = {}
    for length in args.lengths:
        for task in TASKS:
            path = args.ruler_data / str(length) / task / "test.jsonl"
            with path.open() as stream:
                samples[length, task] = json.loads(next(stream))
    first_row = samples[args.lengths[0], TASKS[0]]
    first_prompt = first_row["input"] + first_row["answer_prefix"]
    if args.profile_attention:
        if args.lengths[0] != 4096:
            raise ValueError("Attention diagnosis uses a separate 4K prefill")
        logger.info("Profiling actual attention on 4K prefill and one decode step")
        profile_attention(wrapper, first_prompt, output)
    boundaries = boundary_checks(wrapper, modes, args.mlp_chunk_size, output, logger)
    for mode in modes:
        select_mode(wrapper, mode, args.mlp_chunk_size)
        logger.info("Full-budget warmup mode=%s", mode)
        result, _ = measure(wrapper, first_prompt, args.lengths[0])
        append_json(output / "warmup.jsonl", {"mode": mode, **result})
    generations = []
    first_by_mode = {}
    for length in args.lengths:
        for task_idx, task in enumerate(TASKS):
            row = samples[length, task]
            prompt = row["input"] + row["answer_prefix"]
            measured = {}
            states = {}
            order = modes if task_idx % 2 == 0 else tuple(reversed(modes))
            for mode in order:
                select_mode(wrapper, mode, args.mlp_chunk_size)
                logger.info("Generating length=%s task=%s mode=%s", length, task, mode)
                measured[mode], states[mode] = measure(wrapper, prompt, length)
                measured[mode]["native_score"] = EVALUATORS[task]().score(
                    [extract_non_reasoning_content(measured[mode]["raw_prediction"])],
                    [row["outputs"]],
                )
                logger.info(
                    "Finished mode=%s %.3fs prefill=%.3fs decode=%.3fs peak=%.2fGiB",
                    mode,
                    measured[mode]["seconds"],
                    measured[mode]["prefill_seconds"],
                    measured[mode]["decode_seconds"],
                    measured[mode]["peak_allocated_bytes"] / 1024**3,
                )
            comparisons = {
                mode: {
                    **compare_tokens(
                        measured["original"]["token_ids"], measured[mode]["token_ids"]
                    ),
                    "input_ids_equal": torch.equal(
                        states["original"]["input_ids"], states[mode]["input_ids"]
                    ),
                    "text_equal": measured["original"]["raw_prediction"]
                    == measured[mode]["raw_prediction"],
                    "native_score_equal": measured["original"]["native_score"]
                    == measured[mode]["native_score"],
                    "state": compare_states(states["original"]["ttt"], states[mode]["ttt"]),
                }
                for mode in modes
                if mode != "original"
            }
            record = {
                "length": length,
                "task": task,
                "sample_position": 0,
                "source_index": row["index"],
                "references": row["outputs"],
                "modes": measured,
                "comparisons": comparisons,
            }
            append_json(output / "generation.jsonl", record)
            generations.append(record)
            if length == args.lengths[0] and task_idx == 0:
                first_by_mode = {mode: measured[mode]["token_ids"] for mode in modes}
            logger.info("Comparisons: %s", json.dumps(comparisons))
            del states
    isolation = {}
    for mode in modes:
        select_mode(wrapper, mode, args.mlp_chunk_size)
        result, _ = measure(wrapper, first_prompt, args.lengths[0])
        isolation[mode] = compare_tokens(first_by_mode[mode], result["token_ids"])
    write_json(output / "isolation.json", isolation)
    by_mode = {}
    for mode in modes:
        items = [r["modes"][mode] for r in generations]
        mode_boundaries = [r for r in boundaries if r["mode"] == mode]
        passed = (
            all(
                r["finite"]
                and r["argmax_equal"]
                and r["state"]["finite"]
                and (mode != "tail" or r["state"]["bitwise_equal"])
                and r["cache"]["consistent"]
                for r in mode_boundaries
            )
            and all(
                r["prefill_cache"]["consistent"]
                and r["final_cache"]["consistent"]
                and r["last_logits_finite"]
                and r["ttt_state_finite"]
                for r in items
            )
            and (
                mode == "original"
                or all(
                    r["comparisons"][mode]["equal"]
                    and r["comparisons"][mode]["state"]["finite"]
                    and r["comparisons"][mode]["input_ids_equal"]
                    and r["comparisons"][mode]["text_equal"]
                    and r["comparisons"][mode]["native_score_equal"]
                    and (mode != "tail" or r["comparisons"][mode]["state"]["bitwise_equal"])
                    for r in generations
                )
            )
            and isolation[mode]["equal"]
        )
        timing = []
        for length in args.lengths:
            rows = [r["modes"][mode] for r in generations if r["length"] == length]
            timing.append(
                {
                    "length": length,
                    "samples": len(rows),
                    **{
                        key: sum(r[key] for r in rows) / len(rows)
                        for key in (
                            "seconds",
                            "prefill_seconds",
                            "decode_seconds",
                            "tokenizer_seconds",
                        )
                    },
                    "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in rows),
                }
            )
        by_mode[mode] = {
            "passed": passed,
            "timing": timing,
            "generation_equal": mode == "original"
            or all(r["comparisons"][mode]["equal"] for r in generations),
        }
    summary = {
        "finished_at": timestamp(),
        "passed": all(row["passed"] for row in by_mode.values()),
        "by_mode": by_mode,
        "real_samples": len(generations),
        "boundary_comparisons": len(boundaries),
        "formal_benchmark_completed": False,
        "plan_path": str(output / "plan.json"),
    }
    write_json(output / "summary.json", summary)
    if not summary["passed"]:
        raise RuntimeError("Optimization equivalence checks failed; inspect per-mode summary")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ruler-data",
        type=Path,
        default=PROJECT_ROOT / "data/benchmarks/ruler-v1/reproductions/in-place-ttt-qwen3-4b",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        choices=(4096, 8192, 16384, 32768, 65536, 131072, 262144),
        default=[4096, 32768, 131072, 262144],
    )
    parser.add_argument("--mlp-chunk-size", type=int, default=16384)
    parser.add_argument("--profile-attention", action="store_true")
    args = parser.parse_args()
    if len(set(args.lengths)) != len(args.lengths):
        raise ValueError("lengths must be unique")
    if os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0") not in {"0", "1"}:
        raise ValueError("Select a single physical GPU 0 or 1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    source = capture_source(
        __file__, {"original": MODEL_CLASSES["optimized"], "optimized": MODEL_CLASSES["optimized"]}
    )
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "validation.log")],
    )
    execution = {"started_at": timestamp(), "source": source, "status": "running"}
    write_json(output / "execution.json", execution)
    start = time.perf_counter()
    try:
        run(args, output, source, logging.getLogger(__name__))
        execution["status"] = "completed"
    except BaseException as error:
        execution.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        execution.update(finished_at=timestamp(), seconds=time.perf_counter() - start)
        write_json(output / "execution.json", execution)


if __name__ == "__main__":
    main()
