"""以真实 prompt 的短前缀核对 In-Place TTT 提交边界及冻结行状态。"""

import argparse
import json
import os
from pathlib import Path
import time

from inference_model.hf_qwen3.modeling_qwen3 import TTTDynamicCache
from in_place_ttt.scripts.provenance import capture_source
from in_place_ttt.scripts.records import (
    RULER_DATA, SAMPLES_PER_TASK_LENGTH, TASKS, TOKENIZER_KWARGS,
    read_rows, timestamp, write_json,
)
from in_place_ttt.scripts.runtime import MODEL_CLASSES, load_model, runtime_info
import torch
from transformers import AutoTokenizer


def tensor_difference(reference, actual):
    assert reference.shape == actual.shape, (reference.shape, actual.shape)
    difference = actual.float() - reference.float()
    finite = bool(torch.isfinite(difference).all().item())
    return {
        "finite": finite,
        "max_abs": difference.abs().max().item() if finite else None,
        "mean_abs": difference.abs().mean().item() if finite else None,
    }


def compare_logits(reference, actual):
    result = tensor_difference(reference, actual)
    result.update(
        original_argmax=reference.argmax(-1).item(),
        optimized_argmax=actual.argmax(-1).item(),
    )
    result["argmax_equal"] = result["original_argmax"] == result["optimized_argmax"]
    return result


def cache_summary(cache, row, seen, ttt_layers, chunk):
    padded = hasattr(cache, "rows")
    sample = cache.rows[row] if padded else cache
    actual_seen = cache.seen[row] if padded else sample.get_seq_length()
    kv_lengths = [layer.get_seq_length() for layer in sample.layers]
    pending = []
    for index in ttt_layers:
        x, target, weight = sample.ttt_states[index]
        x_length = 0 if x is None else x.shape[1]
        target_length = 0 if target is None else target.shape[1]
        assert x_length == target_length == seen % chunk, (index, seen, x_length, target_length)
        assert weight is not None
        pending.append({
            "layer": index, "x_length": x_length, "target_length": target_length,
            "weight_shape": list(weight.shape), "weight_dtype": str(weight.dtype),
        })
    assert actual_seen == seen, (actual_seen, seen)
    assert kv_lengths and all(length == seen for length in kv_lengths), (seen, kv_lengths)
    return {"seen": actual_seen, "kv_lengths": kv_lengths, "pending": pending}


def compare_states(reference, actual, ttt_layers):
    result = []
    for index in ttt_layers:
        original = reference.ttt_states[index]
        optimized = actual.rows[0].ttt_states[index]
        values = {"layer": index}
        for name, left, right in zip(("pending_x", "pending_target", "weight"), original, optimized, strict=True):
            assert (left is None) == (right is None), (index, name)
            values[name] = None if left is None else tensor_difference(left, right)
        result.append(values)
    return result


def sample_tensors(sample, ttt_layers):
    tensors = []
    for layer in sample.layers:
        tensors.extend((layer.keys, layer.values))
    for index in ttt_layers:
        tensors.extend(value for value in sample.ttt_states[index] if value is not None)
    return tensors


def frozen_snapshot(cache, row, ttt_layers):
    # no_grad 保留版本计数；引用和版本同时检查，避免复制整套 fast weights。
    return {
        "seen": cache.seen[row],
        "tensors": [(value, value._version) for value in sample_tensors(cache.rows[row], ttt_layers)],
        "states": [cache.rows[row].ttt_states[index] for index in ttt_layers],
    }


def check_frozen(cache, row, snapshot, ttt_layers):
    assert cache.seen[row] == snapshot["seen"]
    after = sample_tensors(cache.rows[row], ttt_layers)
    assert len(after) == len(snapshot["tensors"])
    for actual, (previous, version) in zip(after, snapshot["tensors"], strict=True):
        assert actual is previous and actual._version == version
    for index, previous in zip(ttt_layers, snapshot["states"], strict=True):
        assert cache.rows[row].ttt_states[index] is previous


def forward(model, input_ids, attention_mask, cache):
    return model(
        input_ids=input_ids, attention_mask=attention_mask, past_key_values=cache,
        use_cache=True, logits_to_keep=1, return_dict=True,
    )


@torch.no_grad()
def run(plan, token_ids, report, output):
    original = load_model(plan["checkpoint"], "original", 0, 4096)
    optimized = load_model(plan["checkpoint"], "optimized", 0, 4096)
    report["runtime"] = {"original": runtime_info(original), "optimized": runtime_info(optimized)}
    ttt_layers = [layer.layer_idx for layer in original.model.model.layers if hasattr(layer.mlp, "ttt_conv")]
    chunk = plan["ttt_chunk"]
    assert ttt_layers and all(original.model.model.layers[index].mlp.ttt_chunk == chunk for index in ttt_layers)
    device = original.model.device
    next_tokens = {}
    for length in plan["prefix_lengths"]:
        ids = torch.tensor([token_ids[:length]], device=device)
        mask = torch.ones_like(ids)
        reference = forward(original.model, ids, mask, TTTDynamicCache())
        actual = forward(optimized.model, ids, mask, None)
        case = {"prefix_tokens": length, "stages": []}
        for stage, consumed in (("prefill", length), ("one_token_decode", length + 1)):
            case["stages"].append({
                "stage": stage,
                "logits": compare_logits(reference.logits[0, -1], actual.logits[0, -1]),
                "original_cache": cache_summary(reference.past_key_values, 0, consumed, ttt_layers, chunk),
                "optimized_cache": cache_summary(actual.past_key_values, 0, consumed, ttt_layers, chunk),
                "state_differences": compare_states(reference.past_key_values, actual.past_key_values, ttt_layers),
            })
            if stage == "prefill":
                # 两条实现输入同一个 token，避免 argmax 分歧变成不同历史的比较。
                token = reference.logits[:, -1].argmax(-1, keepdim=True)
                next_tokens[length] = token.item()
                case["decode_input_token"] = token.item()
                mask = torch.cat((mask, mask.new_ones((1, 1))), dim=1)
                reference = forward(original.model, token, mask, reference.past_key_values)
                actual = forward(optimized.model, token, mask, actual.past_key_values)
        report["boundaries"].append(case)
        write_json(output / "results.json", report)
        print(f"[boundaries] prefix={length} prefill/decode checked", flush=True)
        del reference, actual

    lengths = [chunk - 1, chunk + 1]
    width = max(lengths)
    pad_id = optimized.tokenizer.pad_token_id
    ids = torch.full((2, width), pad_id, device=device, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, length in enumerate(lengths):
        ids[row, -length:] = torch.tensor(token_ids[:length], device=device)
        mask[row, -length:] = 1
    actual = forward(optimized.model, ids, mask, None)
    cache = actual.past_key_values
    before = [cache_summary(cache, row, length, ttt_layers, chunk) for row, length in enumerate(lengths)]
    previous_weights = [[cache.rows[row].ttt_states[index][2] for index in ttt_layers] for row in range(2)]
    tokens = torch.tensor([[next_tokens[length]] for length in lengths], device=device)
    mask = torch.cat((mask, mask.new_ones((2, 1))), dim=1)
    actual = forward(optimized.model, tokens, mask, cache)
    assert actual.past_key_values is cache
    after = [cache_summary(cache, row, length + 1, ttt_layers, chunk) for row, length in enumerate(lengths)]
    for position, index in enumerate(ttt_layers):
        assert cache.rows[0].ttt_states[index][2] is not previous_weights[0][position]
        assert cache.rows[1].ttt_states[index][2] is previous_weights[1][position]
    snapshot = frozen_snapshot(cache, 0, ttt_layers)
    cache.finished[0] = True
    tokens = actual.logits[:, -1].argmax(-1, keepdim=True)
    mask = torch.cat((mask, mask.new_ones((2, 1))), dim=1)
    actual = forward(optimized.model, tokens, mask, cache)
    check_frozen(cache, 0, snapshot, ttt_layers)
    final = [cache_summary(cache, row, seen, ttt_layers, chunk) for row, seen in enumerate((chunk, chunk + 3))]
    report["mixed_boundary"] = {
        "prefix_lengths": lengths, "before": before, "after_one_token": after,
        "only_first_row_committed": True, "after_frozen_row_decode": final,
        "frozen_seen_kv_pending_weights_unchanged": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(TASKS), default="niah_single_1")
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        parser.error("Boundary validation requires CUDA_VISIBLE_DEVICES=0")
    if not 0 <= args.sample_index < SAMPLES_PER_TASK_LENGTH:
        parser.error("Sample index must be within the fixed RULER task")
    checkpoint = args.checkpoint.expanduser().resolve()
    config = json.loads((checkpoint / "config.json").read_text())
    if not config["ttt_mode"]:
        parser.error("Boundary validation requires the TTT checkpoint")
    chunk = config.get("ttt_chunk", 8192)
    source = RULER_DATA / "4096" / args.task / "test.jsonl"
    row = read_rows(source, SAMPLES_PER_TASK_LENGTH)[args.sample_index]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, **TOKENIZER_KWARGS)
    token_ids = tokenizer.encode(row["input"] + row["answer_prefix"], add_special_tokens=True, truncation=False)
    prefix_lengths = [chunk - 1, chunk, chunk + 1, 2 * chunk]
    if min(prefix_lengths) < 1 or max(prefix_lengths) > len(token_ids) or max(prefix_lengths) + 1 > 4096:
        parser.error("The real 4K prompt is too short for the configured TTT chunk boundaries")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "created_at": timestamp(), "source": capture_source(__file__, MODEL_CLASSES),
        "checkpoint": str(checkpoint), "checkpoint_config": config,
        "physical_gpu": 0, "simultaneous_model_copies": 2, "mlp_chunk_size": 0,
        "ruler_source": str(source), "sample_index": args.sample_index,
        "prompt_tokens": len(token_ids), "ttt_chunk": chunk, "prefix_lengths": prefix_lengths,
        "decode_input": "Original B1 argmax, teacher-forced into both implementations",
        "grad_mode": "torch.no_grad, matching GenerationMixin.generate; tensor versions retained",
        "numeric_policy": "Record logits/state differences; strictly check cache lengths and frozen states",
    }
    write_json(output / "plan.json", plan)
    report = {"started_at": timestamp(), "status": "running", "boundaries": []}
    write_json(output / "results.json", report)
    started = time.perf_counter()
    try:
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(0)
        run(plan, token_ids, report, output)
        torch.cuda.synchronize(0)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report.update(
            finished_at=timestamp(), seconds=time.perf_counter() - started,
            peak_memory_allocated_bytes=(torch.cuda.max_memory_allocated(0) if torch.cuda.is_initialized() else 0),
        )
        write_json(output / "results.json", report)
    print(f"[boundaries] completed: {output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
