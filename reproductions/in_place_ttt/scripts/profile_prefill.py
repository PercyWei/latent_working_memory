"""单独诊断长 prefill，并比较原 SDPA 与仅 prefill 使用 cuDNN 的完整生成。"""

import argparse
from contextlib import contextmanager
import gc
import json
import logging
import os
from pathlib import Path
import time

import inference_model.hf_qwen3.modeling_qwen3 as upstream_model
from opencompass.utils.text_postprocessors import extract_non_reasoning_content
import optimized.model as optimized_model
from optimized.model import CompactTTTDynamicCache
from provenance import capture_source
from runtime import MODEL_CLASSES, load_model, runtime_info
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F

from validate_optimization import (
    EVALUATORS,
    PROJECT_ROOT,
    append_json,
    cache_metadata,
    compare_states,
    compare_tokens,
    measure,
    select_mode,
    snapshot_ttt,
    timestamp,
    write_json,
)


BACKENDS = ("sdpa", "cudnn_prefill")
TASKS = ("niah_single_1", "niah_multiquery")
MLP_CHUNK_SIZE = 16384


class AttentionRoute:
    def __init__(self, backend, spans=None):
        self.backend = backend
        self.spans = spans
        self.signatures = {}

    def __enter__(self):
        self.original = F.scaled_dot_product_attention

        def routed(q, k, v, *args, **kwargs):
            phase = "decode" if q.shape[-2] == 1 else "prefill"
            mask = kwargs.get("attn_mask", args[0] if args else None)
            if phase == "prefill" and (q.shape[-2] != k.shape[-2] or mask is not None):
                raise ValueError("This comparison requires a complete, unmasked square prefill")
            signature = {
                "phase": phase,
                "selected_backend": (
                    "CUDNN_ATTENTION"
                    if phase == "prefill" and self.backend == "cudnn_prefill"
                    else "original_default"
                ),
                "q_shape": list(q.shape),
                "k_shape": list(k.shape),
                "v_shape": list(v.shape),
                "q_stride": list(q.stride()),
                "k_stride": list(k.stride()),
                "v_stride": list(v.stride()),
                "dtype": str(q.dtype),
                "mask_is_none": mask is None,
                "enable_gqa": kwargs.get("enable_gqa", False),
                "is_causal": kwargs.get("is_causal", args[2] if len(args) > 2 else False),
            }
            key = json.dumps(signature, sort_keys=True)
            self.signatures.setdefault(key, {**signature, "calls": 0})["calls"] += 1

            def invoke():
                if phase == "prefill" and self.backend == "cudnn_prefill":
                    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                        return self.original(q, k, v, *args, **kwargs)
                return self.original(q, k, v, *args, **kwargs)

            if self.spans is None:
                return invoke()
            with self.spans.span("attention.sdpa"):
                return invoke()

        F.scaled_dot_product_attention = routed
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        F.scaled_dot_product_attention = self.original

    def result(self, layers, forward_calls):
        rows = list(self.signatures.values())
        counts = {
            phase: sum(row["calls"] for row in rows if row["phase"] == phase)
            for phase in ("prefill", "decode")
        }
        expected = {"prefill": layers, "decode": layers * (forward_calls - 1)}
        return {
            "signatures": rows,
            "counts": counts,
            "expected": expected,
            "valid": counts == expected,
        }


class ComponentSpans:
    def __init__(self, model):
        self.model = model
        self.phase = "prefill"
        self.events = []
        self.handles = []

    @contextmanager
    def span(self, name):
        label = f"component/{self.phase}/{name}"
        with torch.profiler.record_function(label):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            try:
                yield
            finally:
                end.record()
                self.events.append((self.phase, name, start, end))

    def attach(self, module, name):
        contexts = []

        def before(module, arguments):
            context = self.span(name)
            context.__enter__()
            contexts.append(context)

        def after(module, arguments, output):
            if contexts:
                contexts.pop().__exit__(None, None, None)

        self.handles.extend(
            (
                module.register_forward_pre_hook(before),
                module.register_forward_hook(after, always_call=True),
            )
        )

    def __enter__(self):
        def phase_before(module, arguments, kwargs):
            self.phase = "decode" if kwargs["input_ids"].shape[1] == 1 else "prefill"

        self.handles.append(self.model.register_forward_pre_hook(phase_before, with_kwargs=True))
        for layer in self.model.model.layers:
            self.attach(layer.self_attn, "attention.total")
            self.attach(layer.mlp, "mlp.ttt" if hasattr(layer.mlp, "ttt_conv") else "mlp.regular")
            self.attach(layer.input_layernorm, "norm.external")
            self.attach(layer.post_attention_layernorm, "norm.external")
        self.attach(self.model.model.norm, "norm.external")
        self.attach(self.model.model.rotary_emb, "rope.global")
        self.attach(self.model.model.embed_tokens, "embedding")
        self.attach(self.model.lm_head, "lm_head")
        self.original_contract = optimized_model.contract
        self.original_upstream_contract = upstream_model.contract

        def timed_contract(equation, *args, **kwargs):
            name = "ttt.read" if equation == "d h, c h -> c d" else "ttt.write"
            with self.span(name):
                return self.original_contract(equation, *args, **kwargs)

        optimized_model.contract = timed_contract
        upstream_model.contract = timed_contract
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        optimized_model.contract = self.original_contract
        upstream_model.contract = self.original_upstream_contract

    def result(self, phase):
        result = {}
        for event_phase, name, start, end in self.events:
            if event_phase == phase:
                row = result.setdefault(name, {"calls": 0, "span_seconds": 0.0})
                row["calls"] += 1
                row["span_seconds"] += start.elapsed_time(end) / 1000
        for parent, children, name in (
            ("attention.total", ("attention.sdpa",), "attention.other"),
            ("mlp.ttt", ("ttt.read", "ttt.write"), "mlp.ttt.other"),
        ):
            if parent in result:
                result[name] = {
                    "span_seconds": result[parent]["span_seconds"]
                    - sum(result.get(child, {}).get("span_seconds", 0.0) for child in children),
                    "derived": True,
                }
        return result


def profiler_result(profiler, phase):
    events = list(profiler.events())
    cpu = [event for event in events if str(event.device_type).endswith("CPU")]
    cuda = [event for event in events if str(event.device_type).endswith("CUDA")]
    annotations = [event for event in cuda if event.is_user_annotation]
    annotation_names = {event.name for event in annotations}
    kernels = [event for event in cuda if not event.is_user_annotation]
    pure_device_times = {}

    def pure_cpu_device_time(event):
        # Kineto also links CUDA user-annotation spans into CPU event.kernels.
        # Exclude those spans before following the CPU tree, rather than relying
        # on device_time_total, which can count an annotation and its children.
        key = id(event)
        if key not in pure_device_times:
            pure_device_times[key] = (
                0.0
                if event.is_async or not event.use_device
                else sum(
                    kernel.duration
                    for kernel in event.kernels
                    if kernel.name not in annotation_names
                )
                + sum(pure_cpu_device_time(child) for child in event.cpu_children)
            )
        return pure_device_times[key]

    def aggregate(rows):
        totals = {}
        for event in rows:
            key = (event.name, str(event.device_type), event.is_user_annotation)
            row = totals.setdefault(
                key,
                {
                    "name": event.name,
                    "device_type": str(event.device_type),
                    "is_user_annotation": event.is_user_annotation,
                    "calls": 0,
                    "self_cpu_seconds": 0.0,
                    "cpu_total_seconds": 0.0,
                    "self_device_seconds": 0.0,
                    "device_total_seconds": 0.0,
                    "reported_self_device_seconds": 0.0,
                    "reported_device_total_seconds": 0.0,
                },
            )
            is_cpu = str(event.device_type).endswith("CPU")
            own_device = (
                sum(
                    kernel.duration
                    for kernel in event.kernels
                    if kernel.name not in annotation_names
                )
                if is_cpu and not event.is_async and event.use_device
                else 0.0
                if is_cpu
                else event.self_device_time_total
            )
            inclusive_device = pure_cpu_device_time(event) if is_cpu else event.device_time_total
            row["calls"] += 1
            row["self_cpu_seconds"] += event.self_cpu_time_total / 1e6
            row["cpu_total_seconds"] += event.cpu_time_total / 1e6
            row["self_device_seconds"] += own_device / 1e6
            row["device_total_seconds"] += inclusive_device / 1e6
            row["reported_self_device_seconds"] += event.self_device_time_total / 1e6
            row["reported_device_total_seconds"] += event.device_time_total / 1e6
        return list(totals.values())

    component_rows = aggregate(
        event
        for event in cpu
        if event.is_user_annotation and event.name.startswith(f"component/{phase}/")
    )
    components = {row["name"].removeprefix(f"component/{phase}/"): row for row in component_rows}
    cpu_rows = aggregate(cpu)
    kernel_rows = aggregate(kernels)
    kernel_seconds = sum(event.self_device_time_total for event in kernels) / 1e6
    return {
        "component_kernel_inclusive": components,
        "top_cpu_operators_by_self_device_time": [
            row
            for row in sorted(cpu_rows, key=lambda e: e["self_device_seconds"], reverse=True)[:30]
        ],
        "top_gpu_kernels": [
            row
            for row in sorted(kernel_rows, key=lambda e: e["self_device_seconds"], reverse=True)[
                :30
            ]
        ],
        "sum_gpu_kernel_seconds": kernel_seconds if kernels else None,
        "gpu_kernel_timing_available": bool(kernels) and kernel_seconds > 0,
        "excluded_cuda_annotation_events": len(annotations),
        "excluded_cuda_annotation_seconds": sum(event.device_time_total for event in annotations)
        / 1e6,
        "excluded_cuda_annotation_names": sorted(annotation_names),
        "excluded_cpu_linked_annotation_seconds": sum(
            kernel.duration
            for event in cpu
            for kernel in event.kernels
            if kernel.name in annotation_names
        )
        / 1e6,
        "attention_operators": aggregate(
            event
            for event in events
            if not event.is_user_annotation
            and ("scaled_dot_product" in event.name or "flash_attention" in event.name)
        ),
        "timing_note": (
            "component仅取CPU record_function，device_total按CPU树递归累加实际关联kernel，"
            "先剔除CUDA user annotations；reported字段保留profiler未经修正的时间。"
            "GPU总量只加非annotation CUDA events，不再加CPU关联时间。顶层component互斥，"
            "SDPA与TTTread/write为子项；GPU活动总时长不是并发关键路径。"
        ),
    }


def error_record(error):
    return {"type": type(error).__name__, "message": str(error)}


def clean_cuda():
    gc.collect()
    torch.cuda.empty_cache()


def profile_one(wrapper, prompt, length, backend, expected):
    clean_cuda()
    tokens = wrapper.tokenizer(
        prompt, return_tensors="pt", add_special_tokens=True, truncation=False
    )
    tokens = {name: value.to(wrapper.model.device) for name, value in tokens.items()}
    tokens.pop("token_type_ids", None)
    phases = {}
    predictions = []
    token, cache = None, None
    with ComponentSpans(wrapper.model) as spans, AttentionRoute(backend, spans) as route:
        for phase in ("prefill", "decode"):
            torch.cuda.synchronize()
            started = time.perf_counter()
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
                if phase == "prefill":
                    result = wrapper.model(
                        **tokens,
                        past_key_values=CompactTTTDynamicCache(),
                        use_cache=True,
                        logits_to_keep=1,
                    )
                else:
                    result = wrapper.model(
                        input_ids=token,
                        attention_mask=torch.ones(
                            (1, tokens["input_ids"].shape[1] + 1),
                            dtype=torch.long,
                            device=wrapper.model.device,
                        ),
                        past_key_values=cache,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            cache = result.past_key_values
            logits = result.logits[0, -1].detach().float().cpu()
            predictions.append(int(logits.argmax()))
            phases[phase] = {
                **profiler_result(profiler, phase),
                "cuda_event_spans": spans.result(phase),
                "instrumented_wall_seconds": elapsed,
                "logits_finite": bool(torch.isfinite(logits).all()),
                "ttt_state_finite": all(
                    bool(torch.isfinite(value).all())
                    for index in wrapper.model.config.ttt_layers
                    for value in cache.ttt_states[index]
                    if value is not None
                ),
                "cache_class": f"{type(cache).__module__}.{type(cache).__qualname__}",
                "cache": cache_metadata(cache, wrapper.model.config),
            }
            if phase == "prefill":
                prefill_state = {
                    "ttt": snapshot_ttt(cache, wrapper.model.config),
                    "logits": logits,
                    "input_ids": tokens["input_ids"].detach().cpu().clone(),
                }
                token = result.logits[:, -1].argmax(dim=-1, keepdim=True)
            del result
    routing = route.result(len(wrapper.model.model.layers), 2)
    operator_names = {event["name"] for event in phases["prefill"]["attention_operators"]}
    wanted = (
        "aten::_scaled_dot_product_cudnn_attention"
        if backend == "cudnn_prefill"
        else "aten::_scaled_dot_product_flash_attention"
    )
    decode_operator_names = {event["name"] for event in phases["decode"]["attention_operators"]}
    record = {
        "length": length,
        "backend": backend,
        "phases": phases,
        "routing": routing,
        "first_two_token_ids": predictions,
        "unprofiled_first_two_token_ids": expected,
        "matches_unprofiled": expected is not None and predictions == expected,
        "prefill_operator_verified": wanted in operator_names,
        "decode_operator_verified": "aten::_scaled_dot_product_flash_attention"
        in decode_operator_names,
        "diagnostic_only": True,
        "note": "插桩wall不可用于后端加速比较；CUDA span含CPU launch gap，kernel时间单独报告。",
    }
    record["passed"] = (
        record["matches_unprofiled"]
        and record["prefill_operator_verified"]
        and record["decode_operator_verified"]
        and routing["valid"]
        and all(
            row["logits_finite"]
            and row["ttt_state_finite"]
            and row["cache"]["consistent"]
            and row["gpu_kernel_timing_available"]
            for row in phases.values()
        )
    )
    del cache, tokens
    clean_cuda()
    return record, prefill_state


def valid_measurement(result):
    return (
        result["last_logits_finite"]
        and result["ttt_state_finite"]
        and result["prefill_cache"]["consistent"]
        and result["final_cache"]["consistent"]
        and result["routing"]["valid"]
    )


def run(args, output, source, logger):
    plan = {
        "created_at": timestamp(),
        "purpose": "long prefill diagnosis and paired backend experiment; not formal benchmark",
        "checkpoint": str(args.checkpoint),
        "ruler_data": str(args.ruler_data),
        "lengths": args.lengths,
        "tasks": list(TASKS),
        "sample_position": 0,
        "backends": list(BACKENDS),
        "compact_tails": True,
        "mlp_chunk_size": MLP_CHUNK_SIZE,
        "generation": {"max_new_tokens": 64, "do_sample": False, "preserve_eos": True},
        "warmup": "每后端一次4K完整生成；长shape首次构建计入配对wall",
        "profiling": "全部端到端对照后，每长度/backend的NIAH1另做prefill+一步decode",
        "source": source,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    }
    write_json(output / "plan.json", plan)
    wrapper = load_model(args.checkpoint, "optimized", MLP_CHUNK_SIZE, max(args.lengths))
    if not wrapper.model.config.ttt_mode:
        raise ValueError("This diagnosis requires the TTT checkpoint")
    select_mode(wrapper, "chunked", MLP_CHUNK_SIZE)
    wrapper.model.set_attn_implementation("sdpa")
    plan.update(
        checkpoint_config=wrapper.model.config.to_dict(),
        runtime={
            **runtime_info(wrapper),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "sdp_enabled": {
                "flash": torch.backends.cuda.flash_sdp_enabled(),
                "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
                "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "math": torch.backends.cuda.math_sdp_enabled(),
            },
            "gpu": torch.cuda.get_device_name(0),
            "decode_backend": "original SDPA dispatch in both cases",
            "candidate_prefill_backend": "forced CUDNN_ATTENTION; no fallback",
        },
    )
    write_json(output / "plan.json", plan)
    samples = {}
    for length in dict.fromkeys((4096, *args.lengths)):
        for task in TASKS:
            with (args.ruler_data / str(length) / task / "test.jsonl").open() as stream:
                samples[length, task] = json.loads(next(stream))
    failures, generations, profiles, warmups = [], [], [], []
    if args.generation_results is not None:
        parent = args.generation_results
        parent_plan = json.loads((parent / "plan.json").read_text())
        parent_summary = json.loads((parent / "summary.json").read_text())
        if not parent_summary["completed"] or not parent_summary["process_success"]:
            raise ValueError(
                "The referenced execution must have completed without runtime failures"
            )
        for key in (
            "checkpoint",
            "ruler_data",
            "lengths",
            "tasks",
            "sample_position",
            "backends",
            "compact_tails",
            "mlp_chunk_size",
            "generation",
            "checkpoint_config",
            "runtime",
        ):
            if parent_plan[key] != plan[key]:
                raise ValueError(f"Referenced generation protocol differs: {key}")
        generations = [
            json.loads(line) for line in (parent / "generation.jsonl").read_text().splitlines()
        ]
        warmups = [json.loads(line) for line in (parent / "warmup.jsonl").read_text().splitlines()]
        cases = {(row["length"], row["task"]) for row in generations}
        if len(generations) != len(args.lengths) * len(TASKS) or cases != {
            (length, task) for length in args.lengths for task in TASKS
        }:
            raise ValueError("Referenced generation records do not cover the planned cases")
        plan["generation_results"] = {
            "directory": str(parent),
            "source": parent_plan["source"],
            "measured_in_this_execution": False,
        }
        plan["warmup"] = "复用父执行的4K预热记录；本次不运行生成预热"
        plan["profiling"] = "仅补做分项profiling，端到端耗时沿用所引用执行"
        write_json(output / "plan.json", plan)
        logger.info(
            "Reusing completed end-to-end records from %s; collecting profiles only", parent
        )
    prompt = samples[4096, TASKS[0]]["input"] + samples[4096, TASKS[0]]["answer_prefix"]
    for backend in BACKENDS if args.generation_results is None else ():
        logger.info("4K full-budget warmup backend=%s", backend)
        try:
            with AttentionRoute(backend) as route:
                result, state = measure(wrapper, prompt, 4096)
            result["routing"] = route.result(
                len(wrapper.model.model.layers), result["forward_calls"]
            )
            warmups.append({"backend": backend, **result})
            del state
        except Exception as error:
            failure = {
                "stage": "warmup",
                "backend": backend,
                "error": error_record(error),
                "routing_signatures": list(route.signatures.values()),
            }
            failures.append(failure)
            warmups.append(failure)
            append_json(output / "failures.jsonl", failure)
            logger.exception("Warmup failed backend=%s", backend)
            clean_cuda()
        append_json(output / "warmup.jsonl", warmups[-1])
    warmup_consistent = (
        len(warmups) == 2
        and all("token_ids" in row for row in warmups)
        and warmups[0]["token_ids"] == warmups[1]["token_ids"]
        and all(valid_measurement(row) for row in warmups)
    )
    expected = {
        (row["length"], backend): row["backends"][backend]["token_ids"][:2]
        for row in generations
        if row["task"] == TASKS[0]
        for backend in BACKENDS
    }
    for length in args.lengths if args.generation_results is None else ():
        for index, task in enumerate(TASKS):
            row = samples[length, task]
            prompt = row["input"] + row["answer_prefix"]
            measured, states = {}, {}
            order = BACKENDS if index % 2 == 0 else tuple(reversed(BACKENDS))
            for backend in order:
                logger.info("Generating length=%s task=%s backend=%s", length, task, backend)
                try:
                    with AttentionRoute(backend) as route:
                        result, states[backend] = measure(wrapper, prompt, length)
                    result["routing"] = route.result(
                        len(wrapper.model.model.layers), result["forward_calls"]
                    )
                    result["native_score"] = EVALUATORS[task]().score(
                        [extract_non_reasoning_content(result["raw_prediction"])], [row["outputs"]]
                    )
                    measured[backend] = result
                    if task == TASKS[0]:
                        expected[length, backend] = result["token_ids"][:2]
                    append_json(
                        output / "measurement.jsonl",
                        {"length": length, "task": task, "backend": backend, **result},
                    )
                    logger.info(
                        "Finished %.3fs prefill=%.3fs decode=%.3fs peak=%.2fGiB",
                        result["seconds"],
                        result["prefill_seconds"],
                        result["decode_seconds"],
                        result["peak_allocated_bytes"] / 1024**3,
                    )
                except Exception as error:
                    failure = {
                        "stage": "generation",
                        "length": length,
                        "task": task,
                        "backend": backend,
                        "error": error_record(error),
                        "routing_signatures": list(route.signatures.values()),
                    }
                    failures.append(failure)
                    measured[backend] = {"error": failure["error"]}
                    append_json(output / "failures.jsonl", failure)
                    logger.exception("Generation failed length=%s backend=%s", length, backend)
                    clean_cuda()
            comparison = None
            passed = False
            if all("token_ids" in measured[backend] for backend in BACKENDS):
                reference, actual = (measured[backend] for backend in BACKENDS)
                comparison = {
                    **compare_tokens(reference["token_ids"], actual["token_ids"]),
                    "input_ids_equal": torch.equal(
                        states["sdpa"]["input_ids"], states["cudnn_prefill"]["input_ids"]
                    ),
                    "text_equal": reference["raw_prediction"] == actual["raw_prediction"],
                    "native_score_equal": reference["native_score"] == actual["native_score"],
                    "final_state": compare_states(
                        states["sdpa"]["ttt"], states["cudnn_prefill"]["ttt"]
                    ),
                    "final_state_note": "生成历史发生分歧后，末状态差异不再是同输入的纯后端误差。",
                }
                passed = (
                    comparison["equal"]
                    and comparison["input_ids_equal"]
                    and comparison["text_equal"]
                    and comparison["native_score_equal"]
                    and comparison["final_state"]["finite"]
                    and all(valid_measurement(result) for result in measured.values())
                )
            record = {
                "length": length,
                "task": task,
                "source_index": row["index"],
                "references": row["outputs"],
                "order": list(order),
                "backends": measured,
                "comparison": comparison,
                "passed": passed,
            }
            generations.append(record)
            append_json(output / "generation.jsonl", record)
            logger.info("Pair finished passed=%s comparison=%s", passed, comparison)
            del states
            clean_cuda()
    for length in args.lengths:
        row = samples[length, TASKS[0]]
        prompt = row["input"] + row["answer_prefix"]
        states = {}
        for backend in BACKENDS:
            logger.info("Profiling length=%s backend=%s", length, backend)
            try:
                record, states[backend] = profile_one(
                    wrapper, prompt, length, backend, expected.get((length, backend))
                )
                profiles.append(record)
                append_json(output / "profiles.jsonl", record)
                logger.info(
                    "Profile finished length=%s backend=%s passed=%s",
                    length,
                    backend,
                    record["passed"],
                )
            except Exception as error:
                failure = {
                    "stage": "profile",
                    "length": length,
                    "backend": backend,
                    "error": error_record(error),
                }
                failures.append(failure)
                append_json(output / "failures.jsonl", failure)
                logger.exception("Profile failed length=%s backend=%s", length, backend)
                clean_cuda()
        if all(backend in states for backend in BACKENDS):
            reference, actual = (states[backend] for backend in BACKENDS)
            comparison = {
                "length": length,
                "input_ids_equal": torch.equal(reference["input_ids"], actual["input_ids"]),
                "prefill_logits_max_abs": (reference["logits"] - actual["logits"])
                .abs()
                .max()
                .item(),
                "prefill_argmax_equal": int(reference["logits"].argmax())
                == int(actual["logits"].argmax()),
                "prefill_state": compare_states(reference["ttt"], actual["ttt"]),
                "same_input_history": True,
            }
            write_json(output / f"prefill-comparison-{length}.json", comparison)
        del states
        clean_cuda()
    generation_consistent = warmup_consistent and all(row["passed"] for row in generations)
    profile_passed = len(profiles) == len(args.lengths) * len(BACKENDS) and all(
        row["passed"] for row in profiles
    )
    summary = {
        "finished_at": timestamp(),
        "completed": True,
        "process_success": not failures,
        "generation_consistent": generation_consistent,
        "warmup_generation_consistent": warmup_consistent,
        "profile_checks_passed": profile_passed,
        "candidate_eligible": not failures and generation_consistent and profile_passed,
        "generation_pairs": len(generations),
        "end_to_end_measured_in_this_execution": args.generation_results is None,
        "generation_results": plan.get("generation_results"),
        "profile_runs": len(profiles),
        "failures": failures,
        "formal_benchmark_completed": False,
        "note": "流程完成与候选通过分开；不更改生产默认，不以原生分数相同替代完整IDs一致。",
    }
    write_json(output / "summary.json", summary)
    logger.info("Summary: %s", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--generation-results",
        type=Path,
        help="复用同一协议的已完成端到端记录目录，仅补做分项 profiling",
    )
    parser.add_argument(
        "--ruler-data",
        type=Path,
        default=PROJECT_ROOT / "data/benchmarks/ruler-v1/reproductions/in-place-ttt-qwen3-4b",
    )
    parser.add_argument(
        "--lengths", nargs="+", type=int, choices=(131072, 262144), default=[131072, 262144]
    )
    args = parser.parse_args()
    if len(set(args.lengths)) != len(args.lengths):
        parser.error("lengths must be unique")
    if os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0") not in {"0", "1"}:
        parser.error("Select a single physical GPU 0 or 1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    source = capture_source(__file__, {"optimized": MODEL_CLASSES["optimized"]})
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.ruler_data = args.ruler_data.expanduser().resolve()
    if args.generation_results is not None:
        args.generation_results = args.generation_results.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "diagnosis.log")],
    )
    execution = {"started_at": timestamp(), "source": source, "status": "running"}
    write_json(output / "execution.json", execution)
    started = time.perf_counter()
    try:
        run(args, output, source, logging.getLogger(__name__))
        execution["status"] = "completed"
    except BaseException as error:
        execution.update(status="failed", error=error_record(error))
        logging.exception("Prefill diagnosis failed")
        raise
    finally:
        execution.update(finished_at=timestamp(), seconds=time.perf_counter() - started)
        write_json(output / "execution.json", execution)


if __name__ == "__main__":
    main()
