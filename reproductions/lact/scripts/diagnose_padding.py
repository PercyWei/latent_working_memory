"""对照旧等长 B3 与新 padding B3 的首段数值差异，不修改模型计算。"""

import argparse
import gc
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from evaluate import load_checkpoint
import incremental.model as incremental_model
import incremental.padding as incremental_padding
import incremental.ttt as incremental_ttt
from incremental.padding import PaddedLaCTCache
from validate_padding import (
    compare, forward, pad, snapshot, source_record, source_samples, write_json,
)


def tensor_difference(actual, reference):
    if actual is None or reference is None:
        return {"structure": actual is None and reference is None, "exact": actual is None and reference is None}
    result = {
        "shape": list(actual.shape), "reference_shape": list(reference.shape),
        "dtype": str(actual.dtype), "reference_dtype": str(reference.dtype),
        "structure": actual.shape == reference.shape and actual.dtype == reference.dtype,
    }
    if not result["structure"]:
        return result
    left, right = actual.float(), reference.float()
    delta = left - right
    finite, exact, maximum, reference_maximum, relative = torch.stack((
        (torch.isfinite(left).all() & torch.isfinite(right).all()).float(),
        (actual == reference).all().float(), delta.abs().max(), right.abs().max(),
        delta.norm() / right.norm().clamp_min(1e-12),
    )).cpu().tolist()
    return {**result, "finite": bool(finite), "exact": bool(exact), "max_abs": maximum,
            "reference_max_abs": reference_maximum, "relative_l2": relative}


def comparison(actual, reference, layers, atol, rtol):
    measured, baseline = actual[0], reference[0]
    per_layer = []
    for index in range(layers):
        attention = {
            field: tensor_difference(measured[f"{index}/attention/{field}"], baseline[f"{index}/attention/{field}"])
            for field in ("keys", "values")
        }
        ttt = {
            name.split("/", 2)[-1]: tensor_difference(value, baseline[name])
            for name, value in measured.items() if name.startswith(f"{index}/ttt/")
        }
        per_layer.append({"layer": index, "attention": attention, "ttt": ttt})
    return {
        "summary": compare(actual, reference, atol, rtol),
        "logits": tensor_difference(measured["logits"], baseline["logits"]),
        "first_cache_difference_layer": next((
            item["layer"] for item in per_layer
            if any(not value.get("exact", False) for group in ("attention", "ttt") for value in item[group].values())
        ), None),
        "layers": per_layer,
    }


def capture_attention(layer, hidden, ids, mask, padded):
    """只在单层重放时包装现有算子，保存瞬时比较值并在退出时恢复。"""
    tensors, layouts, counts = {}, {}, {}
    hooks, replacements = [], []

    def record(name, value):
        if isinstance(value, torch.Tensor):
            layouts[name] = {"shape": list(value.shape), "dtype": str(value.dtype),
                             "stride": list(value.stride()), "contiguous": value.is_contiguous()}
            tensors[name] = value.detach().clone()
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                record(f"{name}/{index}", item)

    def event(name):
        index = counts.get(name, 0)
        counts[name] = index + 1
        return f"{name}/{index}"

    def module_hook(name):
        def hook(module, args, kwargs, output):
            prefix = event(name)
            record(f"{prefix}/input", args)
            record(f"{prefix}/output", output)
        return hook

    def replace(owner, name, label):
        original = getattr(owner, name)

        def wrapped(*args, **kwargs):
            prefix = event(label)
            record(f"{prefix}/input", args)
            for key, value in kwargs.items():
                record(f"{prefix}/keyword/{key}", value)
            output = original(*args, **kwargs)
            record(f"{prefix}/output", output)
            return output

        replacements.append((owner, name, original))
        setattr(owner, name, wrapped)

    try:
        for name in ("qkv", "rotary", "lr_proj", "ttt_norm", "o_proj"):
            hooks.append(getattr(layer, name).register_forward_hook(module_hook(name), with_kwargs=True))
        if layer.attn_qk_norm:
            for name in ("q_norm", "k_norm"):
                hooks.append(getattr(layer, name).register_forward_hook(module_hook(name), with_kwargs=True))
        if layer.use_momentum:
            hooks.append(layer.momentum_proj.register_forward_hook(module_hook("momentum_proj"), with_kwargs=True))
        if layer.learnable_ttt_scale:
            hooks.append(layer.ttt_scale_proj.register_forward_hook(module_hook("ttt_scale_proj"), with_kwargs=True))
        owner = incremental_padding if padded else incremental_model
        replace(owner, "flash_attn_varlen_func", "flash_attention")
        replace(owner, "l2_norm", "l2_norm")
        for name in ("_ttt_initial_norms", "_ttt_read", "_ttt_commit"):
            replace(incremental_ttt, name, name)
        replace(incremental_ttt.TTTState, "forward", "ttt_forward")
        record("hidden", hidden)
        if padded:
            cache = PaddedLaCTCache(1, ids.shape[0])
            step = cache.prepare_step(ids, mask)
            output = incremental_padding.attention_forward_padded(layer, hidden, cache.layers[0], step)
        else:
            output = incremental_model.attention_forward(layer, hidden, incremental_model.LayerCache())
        record("final_output", output)
    finally:
        for hook in hooks:
            hook.remove()
        for owner, name, original in reversed(replacements):
            setattr(owner, name, original)
    return tensors, layouts


def trace_attention(layer, hidden, ids, mask):
    baseline, baseline_layout = capture_attention(layer, hidden, ids, mask, False)
    measured, measured_layout = capture_attention(layer, hidden, ids, mask, True)
    fields = {}
    for name in baseline.keys() | measured.keys():
        if name not in baseline or name not in measured:
            fields[name] = {"structure": False, "present_old": name in baseline, "present_new": name in measured}
        else:
            fields[name] = {"old_layout": baseline_layout[name], "new_layout": measured_layout[name],
                            **tensor_difference(measured[name], baseline[name])}
    # 以新路径真实执行顺序寻找首差异，而非按字段名字排序。
    order = list(measured)
    return {
        "purpose": "Replay old and forced-new B1 attention on the exact same normalized hidden tensor",
        "structure": baseline.keys() == measured.keys(), "execution_order": order,
        "first_nonexact_field": next((name for name in order if not fields[name].get("exact", False)), None),
        "fields": fields,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ruler-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--teacher-steps", type=int, default=4, help="与原验证的源文本切片预算一致，不在本脚本执行decode")
    parser.add_argument("--trace-layer", type=int, default=0, help="使用相同归一化hidden重放的层编号")
    parser.add_argument("--numeric-atol", type=float, default=0.03)
    parser.add_argument("--numeric-rtol", type=float, default=0.03)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "1"} or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Use one process on physical GPU 0 or 1")
    if args.teacher_steps < 1 or min(args.numeric_atol, args.numeric_rtol) < 0:
        parser.error("Use a positive slice budget and nonnegative tolerances")
    for name in ("checkpoint", "ruler_data", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plan = {
        "source": source_record(),
        "environment": {"cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                        "torch_compile_disable": os.environ.get("TORCH_COMPILE_DISABLE", "0"),
                        "torch": str(torch.__version__), "cuda": torch.version.cuda},
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "purpose": "Prefill-only numeric diagnosis: old/new paths at identical batch shapes and padded batches; focus row has identical real tokens",
    }
    write_json(args.output_dir / "plan.json", plan)
    started = time.perf_counter()
    model, config = load_checkpoint(args.checkpoint, "incremental")
    if not 0 <= args.trace_layer < len(model.model.layers):
        raise ValueError("trace-layer must select an existing model layer")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    sources, _ = source_samples(args, tokenizer)
    chunk = config.lact_chunk_size
    needed = 2 * chunk + args.teacher_steps + 8
    if any(len(row["input_ids"]) < needed for row in sources[:3]):
        raise ValueError(f"Use sources containing at least {needed} real tokens")
    rows = [row["input_ids"][-needed:] for row in sources[:3]]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    plan.update(
        checkpoint_config=config.to_dict(), chunk_size=chunk, slice_tokens=needed,
        sources=[{"id": row["id"], "tokens": len(row["input_ids"])} for row in sources[:3]],
    )
    write_json(args.output_dir / "plan.json", plan)
    reports = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for length in (chunk - 1, 2 * chunk - 1):
            ids, mask = pad([rows[0][:length]], pad_id)
            trace_hidden = []
            trace_block = model.model.layers[args.trace_layer]
            handle = trace_block.attn_norm.register_forward_hook(
                lambda module, inputs, output, saved=trace_hidden: saved.append(output.detach().clone())
            )
            try:
                baseline_output = forward(model, ids, mask)
            finally:
                handle.remove()
            baseline = snapshot(baseline_output, 0, length, chunk, model)
            equal_baseline = None
            report = {
                "focus_tokens": length, "reference": "b1_unpadded", "modes": {},
                "single_layer_replay": {"layer": args.trace_layer,
                                        **trace_attention(trace_block.attn, trace_hidden[0], ids, mask)},
            }
            del trace_hidden
            for name, sizes, extra, force_new in (
                ("old_equal_b3", [length] * 3, 0, False),
                ("forced_new_b1", [length], 0, True),
                ("forced_new_equal_b3", [length] * 3, 0, True),
                ("new_padded_equal_b3", [length] * 3, 3, False),
                ("new_mixed_b3", [length, length + 1, length + 2], 0, False),
            ):
                ids, mask = pad([
                    row[:size] for row, size in zip(rows[:len(sizes)], sizes, strict=True)
                ], pad_id, extra)
                cache = PaddedLaCTCache(len(model.model.layers), len(sizes)) if force_new else None
                current = forward(model, ids, mask, cache)
                state = snapshot(current, 0, length, chunk, model)
                reference = equal_baseline if name == "forced_new_equal_b3" else baseline
                report["modes"][name] = {
                    "real_lengths": sizes, "physical_width": ids.shape[1],
                    "forced_new_path": force_new,
                    "reference": "old_equal_b3" if name == "forced_new_equal_b3" else "b1_unpadded",
                    **comparison(state, reference, len(model.model.layers), args.numeric_atol, args.numeric_rtol),
                }
                if name == "old_equal_b3":
                    equal_baseline = state
                del current, state
                gc.collect()
            reports.append(report)
            write_json(args.output_dir / "diagnosis.json", reports)
            print(json.dumps({"focus_tokens": length, "modes": {
                name: item["summary"] for name, item in report["modes"].items()
            }}, ensure_ascii=False), flush=True)
            del baseline, baseline_output, equal_baseline
            gc.collect()
    write_json(args.output_dir / "execution.json", {"seconds": time.perf_counter() - started, "completed": True})


if __name__ == "__main__":
    main()
