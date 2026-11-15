"""单进程只运行一种完整 prompt 对照，记录真实 greedy 前两步与 128-token 结果。"""

import argparse
import json
import os
from pathlib import Path
import time

from evaluate import load_checkpoint
from incremental import IncrementalLaCTForCausalLM
from incremental.padding import PaddedLaCTCache
from provenance import capture_source
import torch
from transformers import AutoTokenizer

from latent_working_memory.benchmarks.vendor.ruler import postprocess_pred, string_match_all


MODES = (
    "old_B1", "old_equal_B3", "forced_new_B1", "forced_new_equal_B3",
    "new_B1_leftpad", "new_equal_B3_leftpad", "new_mixed_B3",
)
FOCUS_ID = "ruler/8192/niah_single_2/62"
GENERATION_TOKENS = 128


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_sample(data, sample_id, tokenizer):
    benchmark, length, task, position = sample_id.split("/")
    if benchmark != "ruler":
        raise ValueError(f"Expected a RULER sample ID: {sample_id}")
    path = data / length / task / "test.jsonl"
    with path.open() as stream:
        for index, line in enumerate(stream):
            if index == int(position):
                row = json.loads(line)
                return {"id": sample_id, "input_ids": tokenizer.encode(
                    row["input"] + row["answer_prefix"], truncation=False,
                ), "references": row["outputs"]}
    raise ValueError(f"Sample position not found: {sample_id}")


def make_inputs(mode, samples, pad_id):
    focus = samples[0]["input_ids"]
    rows = [row["input_ids"] for row in samples] if mode == "new_mixed_B3" else (
        [focus] * 3 if "equal_B3" in mode else [focus]
    )
    width = max(len(row) for row in rows)
    if mode.endswith("leftpad"):
        width = max(len(row["input_ids"]) for row in samples)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device="cuda:0")
    mask = torch.zeros_like(ids)
    for index, row in enumerate(rows):
        ids[index, -len(row):] = torch.tensor(row, device=ids.device)
        mask[index, -len(row):] = 1
    return ids, mask, rows


def cache_progress(cache):
    layers = []
    for index, layer in enumerate(cache.layers):
        separate = isinstance(layer.ttt, list)
        state = layer.ttt[0] if separate else layer.ttt
        keys = layer.keys[0] if separate else layer.keys[0:1]
        layers.append({
            "layer": index, "seen": layer.seen[0] if separate else layer.seen,
            "ttt_seen": state.seen, "updates": state.n_updates,
            "pending_length": state.pending_length, "kv_shape": list(keys.shape),
            "w0_shape": list(state.w0.shape), "w0_dtype": str(state.w0.dtype),
        })
    return {"physical_seen": cache.get_seq_length(), "layers": layers}


def describe_logits(logits, tokenizer):
    values, indices = logits.topk(10)
    return {
        "argmax": int(logits.argmax()), "top1_minus_top2": float(values[0] - values[1]),
        "top_candidates": [{"token_id": int(index), "logit": float(value),
                            "text": tokenizer.decode([int(index)], skip_special_tokens=False)}
                           for index, value in zip(indices, values, strict=True)],
    }


def compare_reference(result, logits, reference_dir):
    reference = json.loads((reference_dir / "result.json").read_text())
    previous = torch.load(reference_dir / "logits.pt", map_location="cpu", weights_only=True) if result["record_logits"] else []
    actual_ids, expected_ids = result["token_ids"], reference["token_ids"]
    divergence = next((index for index, (actual, expected) in enumerate(zip(actual_ids, expected_ids))
                       if actual != expected), None)
    if divergence is None and len(actual_ids) != len(expected_ids):
        divergence = min(len(actual_ids), len(expected_ids))
    steps = []
    for index, (actual, expected) in enumerate(zip(logits, previous, strict=True)):
        delta = actual - expected
        steps.append({
            "generated_index": index,
            "same_real_prefix": actual_ids[:index] == expected_ids[:index],
            "exact": bool(torch.equal(actual, expected)),
            "max_abs": float(delta.abs().max()),
            "relative_l2": float(delta.norm() / expected.norm().clamp_min(1e-12)),
            "argmax_equal": bool(actual.argmax() == expected.argmax()),
        })
    return {"reference_dir": str(reference_dir), "first_different_generated_index": divergence,
            "all_token_ids_equal": actual_ids == expected_ids,
            "score_equal": result["score"] == reference["score"], "steps": steps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ruler-data", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True,
                        help="原validate_padding.py的plan.json，包含三条selected样本")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, help="old_B1输出目录；其他mode必填")
    parser.add_argument("--record-logits", action=argparse.BooleanOptionalAction, default=True,
                        help="通过顶层hook记录真实生成前两步；关闭时完全不注册hook")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "1"} or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Use one fresh process on physical GPU 0 or 1 per mode")
    if (args.mode == "old_B1") != (args.reference_dir is None):
        parser.error("Only old_B1 runs without --reference-dir")
    for name in ("checkpoint", "ruler_data", "source_plan", "output_dir", "reference_dir"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    source_plan = json.loads(args.source_plan.read_text())
    selected = source_plan["selected"]
    ids = [row["id"] for row in selected]
    if len(ids) != 3 or len(set(ids)) != 3 or FOCUS_ID not in ids:
        raise ValueError(f"source-plan must contain three distinct samples including {FOCUS_ID}")
    ids = [FOCUS_ID, *(sample_id for sample_id in ids if sample_id != FOCUS_ID)]
    identity = {"checkpoint": str(args.checkpoint), "ruler_data": str(args.ruler_data),
                "focus_id": FOCUS_ID, "mixed_sample_ids": ids, "generation_tokens": GENERATION_TOKENS,
                "torch_compile_disable": os.environ.get("TORCH_COMPILE_DISABLE", "0")}
    if args.reference_dir is not None:
        baseline_plan = json.loads((args.reference_dir / "plan.json").read_text())
        if baseline_plan["mode"] != "old_B1" or baseline_plan["identity"] != identity:
            raise ValueError("Reference must be old_B1 with the same checkpoint, data, samples and compile setting")
        if baseline_plan["record_logits"] != args.record_logits:
            raise ValueError("Reference must use the same record-logits setting")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plan = {
        "identity": identity, "mode": args.mode, "record_logits": args.record_logits,
        "source": capture_source(__file__, {"incremental": IncrementalLaCTForCausalLM}),
        "environment": {"cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                        "torch": str(torch.__version__), "cuda": torch.version.cuda},
        "source_plan": str(args.source_plan),
        "reference_dir": str(args.reference_dir) if args.reference_dir else None,
        "equal_partners": "Three copies of the complete focus prompt",
        "sequence": "One greedy128 generation only; top-level forward hook records its actual first two steps"
                    if args.record_logits else "One greedy128 generation only; no forward hook or logits artifact",
    }
    write_json(args.output_dir / "plan.json", plan)
    started = time.perf_counter()
    torch.manual_seed(0)
    model, config = load_checkpoint(args.checkpoint, "incremental")
    if not config.use_muon or config.ttt_prenorm:
        raise ValueError("This diagnosis requires the Muon PostNorm checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    samples = [read_sample(args.ruler_data, sample_id, tokenizer) for sample_id in ids]
    lengths = {row["id"]: len(row["input_ids"]) for row in samples}
    if any(lengths[row["id"]] != row["tokens"] for row in selected):
        raise ValueError("Tokenized sample lengths differ from source-plan")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids, mask, rows = make_inputs(args.mode, samples, pad_id)
    force_new = args.mode.startswith("forced_new_")
    plan.update(checkpoint_config=config.to_dict(), selected=[
        {"id": row["id"], "tokens": lengths[row["id"]]} for row in samples
    ], physical_input_shape=list(input_ids.shape), actual_input_lengths=list(map(len, rows)),
                actual_sample_ids=ids if args.mode == "new_mixed_B3" else [FOCUS_ID] * len(rows),
                forced_new_cache=force_new)
    write_json(args.output_dir / "plan.json", plan)
    logits, steps = [], []

    def collect(module, inputs, kwargs, output):
        if len(logits) == 2:
            return
        value = output.logits[0, -1].detach().float().cpu()
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Focus logits contain non-finite values")
        progress = cache_progress(output.past_key_values)
        seen = len(rows[0]) + len(logits)
        updates = (seen - 1) // config.lact_chunk_size
        for layer in progress["layers"]:
            if (layer["seen"], layer["ttt_seen"], layer["updates"], layer["pending_length"]) != (
                seen, seen, updates, seen - updates * config.lact_chunk_size
            ):
                raise ValueError("Focus cache progress differs from the original real-token update schedule")
        logits.append(value)
        actual_input = kwargs["input_ids"]
        steps.append({"generated_index": len(logits) - 1,
                      "input_shape": list(actual_input.shape),
                      "focus_input_last_tokens": actual_input[0, -8:].cpu().tolist(),
                      "cache": progress})

    handle = model.register_forward_hook(collect, with_kwargs=True) if args.record_logits else None
    generation_kwargs = {}
    if force_new:
        generation_kwargs["past_key_values"] = PaddedLaCTCache(len(model.model.layers), input_ids.shape[0])
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated = model.generate(
                input_ids=input_ids, attention_mask=mask, do_sample=False, num_beams=1,
                max_new_tokens=GENERATION_TOKENS, use_cache=True, logits_to_keep=1,
                pad_token_id=pad_id, output_scores=False, **generation_kwargs,
            )
    finally:
        if handle is not None:
            handle.remove()
    torch.cuda.synchronize()
    tokens = generated[0, input_ids.shape[1]:].cpu().tolist()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    end = next((index + 1 for index, token in enumerate(tokens) if token in eos), len(tokens))
    tokens = tokens[:end]
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    result = {"mode": args.mode, "focus_id": FOCUS_ID, "record_logits": args.record_logits,
              "reference_dir": str(args.reference_dir) if args.reference_dir else None,
              "token_ids": tokens, "prediction": text,
              "references": samples[0]["references"],
              "score": string_match_all([postprocess_pred(text)], [samples[0]["references"]]),
              "steps": [{**step, **describe_logits(value, tokenizer)}
                        for step, value in zip(steps, logits, strict=True)],
              "seconds": time.perf_counter() - started}
    if args.record_logits:
        torch.save(logits, args.output_dir / "logits.pt")
    if args.reference_dir is not None:
        result["comparison"] = compare_reference(result, logits, args.reference_dir)
    write_json(args.output_dir / "result.json", result)
    print(json.dumps({name: result[name] for name in ("mode", "focus_id", "score", "seconds")}
                     | {"first_token_ids": tokens[:8], "comparison": result.get("comparison")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
