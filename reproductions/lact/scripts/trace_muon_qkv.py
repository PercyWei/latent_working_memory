"""最少hook记录首decode第0层QKV与O投影，完整生成后再重放同输入。"""

import argparse
import json
import os
from pathlib import Path
import time

from diagnose_muon_batch import (
    FOCUS_ID, GENERATION_TOKENS, cache_progress, compare_reference, describe_logits,
    load_checkpoint, make_inputs, read_sample, write_json,
)
from incremental import IncrementalLaCTForCausalLM
from provenance import capture_source
import torch
from transformers import AutoTokenizer

from latent_working_memory.benchmarks.vendor.ruler import postprocess_pred, string_match_all


MODES = ("old_B1", "old_equal_B3")


def projection_hook(name, logits, captured, metadata):
    def hook(module, args, output):
        # 顶层collect保持原调用位置；第二次模型forward进入时恰有一条logits。
        if len(logits) != 1:
            return
        value = args[0]
        metadata[name] = {
            "input": {"shape": list(value.shape), "stride": list(value.stride()), "dtype": str(value.dtype),
                      "pointer_mod256": value.data_ptr() % 256},
            "output": {"shape": list(output.shape), "stride": list(output.stride()), "dtype": str(output.dtype),
                       "pointer_mod256": output.data_ptr() % 256},
        }
        captured[name] = {"input": value.detach().cpu(), "output": output.detach().cpu()}
    return hook


def tensor_stats(actual, reference):
    left, right = actual.float(), reference.float()
    delta = left - right
    changed = (actual != reference).flatten().nonzero().flatten()
    return {
        "shape": list(actual.shape), "dtype": str(actual.dtype),
        "exact": bool(torch.equal(actual, reference)), "different_elements": int(changed.numel()),
        "first_different_flat_indices": changed[:16].tolist(),
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(delta.norm() / right.norm().clamp_min(1e-12)),
    }


def replay_projection(projection, captured, metadata):
    original_input, original_output = captured["input"], captured["output"]
    # 按现场stride恢复完整B输入；第一次重放不改变batch或值。
    restored = torch.empty_strided(metadata["input"]["shape"], metadata["input"]["stride"],
                                   device="cuda:0", dtype=original_input.dtype)
    restored.copy_(original_input)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        onsite_replay = projection(restored).detach().cpu()
        one = restored[0:1].repeat(1, 1, 1)
        three = restored[0:1].repeat(3, 1, 1)
        one_output = projection(one).detach().cpu()
        three_output = projection(three).detach().cpu()
    return {
        "restored_input_stride": list(restored.stride()),
        "restored_input_pointer_mod256": restored.data_ptr() % 256,
        "full_B_replay_vs_onsite": tensor_stats(onsite_replay, original_output),
        "repeat1_input_stride": list(one.stride()), "repeat3_input_stride": list(three.stride()),
        "repeat1_vs_onsite_focus": tensor_stats(one_output, original_output[0:1]),
        "repeat3_vs_onsite_focus": tensor_stats(three_output[0:1], original_output[0:1]),
        "repeat3_vs_repeat1_focus": tensor_stats(three_output[0:1], one_output),
        "onsite_all_input_rows_equal_focus": all(
            torch.equal(row, original_input[0]) for row in original_input
        ),
        "repeat3_all_output_rows_equal_focus": all(
            torch.equal(row, three_output[0]) for row in three_output
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ruler-data", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True,
                        help="原validate_padding.py的plan.json，包含三条selected样本")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True, help="此前相同mode的diagnose_muon_batch输出目录")
    parser.add_argument("--record-logits", action=argparse.BooleanOptionalAction, default=True,
                        help="通过顶层hook记录真实生成前两步；关闭时完全不注册hook")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "1"} or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Use one fresh process on physical GPU 0 or 1 per mode")
    if not args.record_logits:
        parser.error("Projection tracing requires --record-logits")
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
        if baseline_plan["mode"] != args.mode or baseline_plan["identity"] != identity:
            raise ValueError("Reference must use the same mode, checkpoint, data, samples and compile setting")
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
        "additional_observation": "Only layer0 QKV and O projection hooks at the first decode; raw dtype CPU copies after their actual calls",
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
    plan.update(checkpoint_config=config.to_dict(), selected=[
        {"id": row["id"], "tokens": lengths[row["id"]]} for row in samples
    ], physical_input_shape=list(input_ids.shape), actual_input_lengths=list(map(len, rows)),
                actual_sample_ids=[FOCUS_ID] * len(rows))
    write_json(args.output_dir / "plan.json", plan)
    logits, steps = [], []
    captured, metadata = {}, {}

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
    projections = {"qkv": model.model.layers[0].attn.qkv,
                   "o_proj": model.model.layers[0].attn.o_proj}
    projection_handles = [projection.register_forward_hook(
        projection_hook(name, logits, captured, metadata),
    ) for name, projection in projections.items()]
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated = model.generate(
                input_ids=input_ids, attention_mask=mask, do_sample=False, num_beams=1,
                max_new_tokens=GENERATION_TOKENS, use_cache=True, logits_to_keep=1,
                pad_token_id=pad_id, output_scores=False,
            )
    finally:
        if handle is not None:
            handle.remove()
        for projection_handle in projection_handles:
            projection_handle.remove()
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
    result["first_two_logits_match_reference_exactly"] = all(
        step["exact"] for step in result["comparison"]["steps"]
    )
    result["observer_matches_reference"] = (
        result["first_two_logits_match_reference_exactly"] and result["comparison"]["all_token_ids_equal"]
    )
    torch.save(captured, args.output_dir / "qkv.pt")
    replay = {}
    for name, projection in projections.items():
        replay[name] = replay_projection(projection, captured[name], metadata[name])
    write_json(args.output_dir / "qkv.json", {"modules": metadata, "replay": replay,
               "observer_matches_reference": result["observer_matches_reference"],
               "scope": "Actual first-decode full-B inputs/outputs; post-generation replay does not identify the causal effect on later tokens"})
    result["projection_trace"] = "qkv.pt"
    result["projection_replay"] = replay
    write_json(args.output_dir / "result.json", result)
    print(json.dumps({name: result[name] for name in ("mode", "focus_id", "score", "seconds")}
                     | {"first_token_ids": tokens[:8], "comparison": result.get("comparison"),
                        "observer_matches_reference": result["observer_matches_reference"], "projection_replay": replay},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
