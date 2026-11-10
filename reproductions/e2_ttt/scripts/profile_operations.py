"""Attribute original E²-TTT forward operators; profiler times are not benchmark latency."""

import argparse
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path
import time

from fla.models.utils import Cache
from lm_eval.models.huggingface import HFLM
import torch

from profile_inference import STATE_NAMES, read_sample, source_record, timestamp, write_json


def event_summary(profiler):
    events = list(profiler.events())
    cpu = [event for event in events if event.device_type == torch.autograd.DeviceType.CPU]
    scopes = [event for event in cpu if event.name == "e2_forward"]
    if len(scopes) != 1:
        raise ValueError(f"Expected one e2_forward scope, got {len(scopes)}")

    def inside(event):
        while event is not None:
            if event is scopes[0]:
                return True
            event = event.cpu_parent
        return False

    operators = [event for event in cpu if not event.is_user_annotation and inside(event)]
    device = [
        event for event in events
        if event.device_type == torch.autograd.DeviceType.CUDA and not event.is_user_annotation
    ]
    if not device:
        raise ValueError("Profiler captured no actual CUDA activities")
    totals = {}
    intervals = []
    for event in device:
        start, end = event.time_range.start, event.time_range.end
        intervals.append((start, end))
        row = totals.setdefault(event.name, {"name": event.name, "calls": 0, "seconds": 0.0})
        row["calls"] += 1
        row["seconds"] += (end - start) / 1e6
    union, last = 0.0, None
    for start, end in sorted(intervals):
        union += max(0.0, end - max(start, last if last is not None else start))
        last = max(end, last if last is not None else end)
    scalar_names = ("aten::_local_scalar_dense", "aten::item", "aten::is_nonzero")
    return {
        "cpu_forward_scope_seconds": scopes[0].cpu_time_total / 1e6,
        "cpu_operator_self_seconds": sum(event.self_cpu_time_total for event in operators) / 1e6,
        "scalar_operator_counts": {name: sum(event.name == name for event in operators) for name in scalar_names},
        "cuda_activity_count": len(device),
        "cuda_activity_duration_sum_seconds": sum(row["seconds"] for row in totals.values()),
        "cuda_activity_interval_union_seconds": union / 1e6,
        "top_cuda_activities": sorted(totals.values(), key=lambda row: row["seconds"], reverse=True)[:20],
        "explicit_device_to_host_activities": [
            row for name, row in totals.items()
            if "dtoh" in name.lower() or "device -> host" in name.lower() or "device to host" in name.lower()
        ],
        "excluded_cuda_annotation_count": sum(
            event.device_type == torch.autograd.DeviceType.CUDA and event.is_user_annotation for event in events
        ),
        "interpretation": (
            "CUDA totals use actual nonannotation device events only; CPU-linked device_time_total is unused. "
            "Duration sum counts overlapping activity; interval union removes overlap. "
            "CPU scalar counts are restricted to descendants of e2_forward."
        ),
    }


def execute(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.checkpoint / "config.json").read_text())
    source = source_record()
    source["entrypoint"] = str(Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[3]))
    write_json(args.output_dir / "plan.json", {
        "created_at": timestamp(), "source": source, "checkpoint": str(args.checkpoint),
        "checkpoint_config": config, "ruler_data": str(args.ruler_data),
        "sample": {"task": "niah_single_1", "length": 4096, "row_index": 1},
        "captures": ["prefill_4k", "decode_read", "decode_chunk_update"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "batch_size": 1,
        "protocol": "Same-path warmup, fresh cache, single original forward; profiler overhead is not production latency.",
    })
    started = time.perf_counter()
    results = {}
    try:
        lm = HFLM(
            pretrained=str(args.checkpoint), tokenizer=str(args.checkpoint), dtype="bfloat16",
            device="cuda:0", batch_size=1, max_length=4096, truncation=False, trust_remote_code=False,
        )
        lm.model.eval()
        write_json(args.output_dir / "runtime.json", {
            "model_class": type(lm.model).__qualname__, "model_source": inspect.getfile(type(lm.model)),
            "cache_class": Cache.__qualname__, "cache_source": inspect.getfile(Cache),
            "torch": torch.__version__, "flash_linear_attention": version("flash-linear-attention"),
            "transformers": version("transformers"), "device_name": torch.cuda.get_device_name(0),
        })
        row, _ = read_sample(args.ruler_data, 4096, "niah_single_1")
        ids, mask = lm.tok_batch_encode([row["input"] + row["answer_prefix"]], truncation=False)
        if ids.shape[1] + 128 > 4096:
            raise ValueError("Real RULER input exceeds the 4K token budget")
        ids, mask = ids.to(lm.device), mask.to(lm.device)
        names = STATE_NAMES[config["model_type"]]

        def forward(token_ids, token_mask, cache=None):
            return lm.model(
                input_ids=token_ids, attention_mask=token_mask, past_key_values=cache,
                use_cache=True, logits_to_keep=1, return_dict=True,
            )

        def prepare(phase):
            prefix = ids[:, :511] if phase == "decode_chunk_update" else ids
            prefix_mask = mask[:, :prefix.shape[1]]
            if phase == "prefill_4k":
                return prefix, prefix_mask, None, None
            output = forward(prefix, prefix_mask)
            cache = output.past_key_values
            token = output.logits[:, -1:].argmax(-1)
            token_mask = torch.cat((prefix_mask, prefix_mask.new_ones((1, 1))), dim=1)
            tail = cache[0]["recurrent_state"][names.index("buf_k")].shape[-1]
            return token, token_mask, cache, tail

        with torch.no_grad():
            for phase in ("prefill_4k", "decode_read", "decode_chunk_update"):
                inputs = prepare(phase)
                output = forward(*inputs[:3])
                del inputs, output
                token_ids, token_mask, cache, before_tail = prepare(phase)
                torch.cuda.synchronize(0)
                capture_started = time.perf_counter()
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=False, profile_memory=False, with_stack=False,
                ) as profiler:
                    with torch.profiler.record_function("e2_forward"):
                        output = forward(token_ids, token_mask, cache)
                    torch.cuda.synchronize(0)
                after_tail = output.past_key_values[0]["recurrent_state"][names.index("buf_k")].shape[-1]
                if phase == "decode_chunk_update" and (before_tail != 511 or after_tail != 0):
                    raise ValueError(f"Boundary phase did not update: {before_tail} -> {after_tail}")
                results[phase] = {
                    "input_tokens": token_ids.shape[1], "tail_tokens_before": before_tail,
                    "tail_tokens_after": after_tail,
                    "profiler_wall_seconds": time.perf_counter() - capture_started,
                    **event_summary(profiler),
                }
                write_json(args.output_dir / "profile.json", results)
                del output, cache, token_ids, token_mask, profiler
        write_json(args.output_dir / "execution.json", {
            "finished_at": timestamp(), "exit_code": 0,
            "elapsed_seconds": time.perf_counter() - started,
        })
    except Exception as error:
        write_json(args.output_dir / "execution.json", {
            "finished_at": timestamp(), "exit_code": 1, "error": repr(error),
            "elapsed_seconds": time.perf_counter() - started, "completed_captures": list(results),
        })
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "ruler-data", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Use a single process on physical GPU 0: CUDA_VISIBLE_DEVICES=0")
    for name in ("checkpoint", "ruler_data", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


if __name__ == "__main__":
    execute(parse_args())
