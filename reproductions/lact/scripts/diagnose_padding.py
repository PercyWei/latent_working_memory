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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ruler-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--teacher-steps", type=int, default=4, help="与原验证的源文本切片预算一致，不在本脚本执行decode")
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
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "purpose": "Prefill-only numeric diagnosis: old/new paths at identical batch shapes and padded batches; focus row has identical real tokens",
    }
    write_json(args.output_dir / "plan.json", plan)
    started = time.perf_counter()
    model, config = load_checkpoint(args.checkpoint, "incremental")
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
            baseline_output = forward(model, ids, mask)
            baseline = snapshot(baseline_output, 0, length, chunk, model)
            equal_baseline = None
            report = {"focus_tokens": length, "reference": "b1_unpadded", "modes": {}}
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
