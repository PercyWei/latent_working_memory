"""定位一个真实 LaCT 样本的 prefill 中间值差异及首次生成分歧。"""

import argparse
import gc
import json
import logging
import os
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from evaluate import (
    LENGTHS,
    PROJECT_ROOT,
    TASKS,
    load_checkpoint,
    read_samples,
    timestamp,
    write_json,
)


def tensor_metadata(value):
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "contiguous": value.is_contiguous(),
    }


def numerical_difference(reference, actual):
    shape_equal = reference.shape == actual.shape
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(actual).all())
    if not shape_equal or not finite:
        return {"shape_equal": shape_equal, "finite": finite, "max_abs": None, "mean_abs": None}
    difference = (actual.float() - reference.float()).abs()
    return {
        "shape_equal": True,
        "finite": True,
        "bitwise_equal": bool(torch.equal(reference, actual)),
        "max_abs": difference.max().item(),
        "mean_abs": difference.mean().item(),
    }


def compare_prefill(model, input_ids, output, logger):
    saved = {}
    records = []
    handles = []
    phase = "capture"

    def observe(name, tensor):
        metadata = tensor_metadata(tensor)
        if phase == "capture":
            if name in saved:
                raise RuntimeError(f"Hook fired more than once: {name}")
            saved[name] = (tensor.detach().clone(), metadata)
            return
        reference, reference_metadata = saved.pop(name)
        record = {
            "name": name,
            "original": reference_metadata,
            "incremental": metadata,
            **numerical_difference(reference, tensor),
        }
        del reference
        records.append(record)
        with (output / "prefill_layers.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        logger.info(
            "%s max_abs=%s mean_abs=%s original_stride=%s incremental_stride=%s",
            name,
            record["max_abs"],
            record["mean_abs"],
            reference_metadata["stride"],
            metadata["stride"],
        )

    def output_hook(name):
        def hook(module, arguments, result):
            observe(name, result)

        return hook

    def input_hook(name):
        def hook(module, arguments):
            observe(name, arguments[0])

        return hook

    for index, block in enumerate(model.model.layers):
        prefix = f"layer.{index}"
        handles.extend(
            [
                block.attn.qkv.register_forward_hook(output_hook(f"{prefix}.qkv")),
                block.attn.ttt_norm.register_forward_pre_hook(input_hook(f"{prefix}.ttt_raw")),
                block.attn.ttt_norm.register_forward_hook(output_hook(f"{prefix}.ttt_normalized")),
                block.attn.o_proj.register_forward_pre_hook(
                    input_hook(f"{prefix}.attention_plus_ttt")
                ),
                block.mlp.register_forward_hook(output_hook(f"{prefix}.mlp")),
            ]
        )
    mask = torch.ones_like(input_ids)
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            original = model(
                input_ids=input_ids,
                attention_mask=mask,
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
        original_logits = original.logits[0, -1].detach().float().cpu()
        del original
        logger.info("Captured %s original intermediate tensors", len(saved))
        phase = "compare"
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            incremental = model(
                input_ids=input_ids,
                attention_mask=mask,
                past_key_values=None,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
        incremental_logits = incremental.logits[0, -1].detach().float().cpu()
        del incremental
        if saved:
            raise RuntimeError(f"Incremental path did not execute hooks: {list(saved)}")
    finally:
        for handle in handles:
            handle.remove()
        saved.clear()
    if len(records) != 5 * len(model.model.layers):
        raise RuntimeError("Unexpected number of intermediate tensor comparisons")
    return records, original_logits, incremental_logits


def summarize_logits(logits, tokenizer):
    logits = logits.detach().float().cpu()
    values, indices = logits.topk(10)
    return {
        "argmax": int(logits.argmax()),
        "top1_top2_gap": (values[0] - values[1]).item(),
        "logit_token_1": logits[1].item(),
        "logit_token_13": logits[13].item(),
        "logit_1_minus_13": (logits[1] - logits[13]).item(),
        "top10": [
            {
                "token_id": token_id,
                "text": tokenizer.decode([token_id], skip_special_tokens=False),
                "logit": value,
            }
            for token_id, value in zip(indices.tolist(), values.tolist(), strict=True)
        ],
    }


def cache_progress(cache, model, consumed):
    chunk = model.config.lact_chunk_size
    updates = (consumed - 1) // chunk
    pending = consumed - updates * chunk
    layers = [
        {
            "layer": index,
            "seen": layer.seen,
            "ttt_seen": layer.ttt.seen,
            "n_updates": layer.ttt.n_updates,
            "pending_length": layer.ttt.pending_length,
            "consistent": (
                layer.seen == consumed
                and layer.ttt.seen == consumed
                and layer.ttt.n_updates == updates
                and layer.ttt.pending_length == pending
            ),
        }
        for index, layer in enumerate(cache.layers)
    ]
    return {
        "consistent": len(layers) == len(model.model.layers)
        and all(layer["consistent"] for layer in layers),
        "expected": {"seen": consumed, "n_updates": updates, "pending_length": pending},
        "layers": layers,
    }


def run(args, output, logger):
    checkpoint = args.checkpoint.expanduser().resolve()
    data = args.ruler_data.expanduser().resolve()
    source = args.generation_results.expanduser().resolve()
    sample_id = f"ruler/{args.length}/{args.task}/0"
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    matching = [record for record in records if record["id"] == sample_id]
    if len(matching) != 1:
        raise ValueError(f"Expected one generation record for {sample_id}, got {len(matching)}")
    generation = matching[0]
    first_difference = generation["comparison"]["first_different_generated_position"]
    if first_difference is None:
        raise ValueError("The selected generation has no differing token")
    original_ids = generation["original"]["token_ids"]
    incremental_ids = generation["incremental"]["token_ids"]
    if original_ids[:first_difference] != incremental_ids[:first_difference]:
        raise ValueError("Saved generations differ before the reported first difference")
    if original_ids[first_difference] == incremental_ids[first_difference]:
        raise ValueError("Saved generations agree at the reported first difference")
    row = read_samples(data / str(args.length) / args.task / "test.jsonl")[0]
    if row["index"] != generation["source_index"]:
        raise ValueError("The selected dataset row does not match the saved generation")
    plan = {
        "created_at": timestamp(),
        "checkpoint": str(checkpoint),
        "checkpoint_config": json.loads((checkpoint / "config.json").read_text()),
        "ruler_data": str(data),
        "generation_results": str(source),
        "sample_id": sample_id,
        "first_difference_zero_based": first_difference,
        "original_token": original_ids[first_difference],
        "incremental_token": incremental_ids[first_difference],
        "common_generated_prefix_token_ids": original_ids[:first_difference],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "parameter_dtype": "float32",
        "compute_autocast_dtype": "bfloat16",
        "hooks": ["qkv output", "ttt_norm input", "ttt_norm output", "o_proj input", "MLP output"],
        "purpose": "numerical diagnosis only; no precision or generation-rule changes",
    }
    write_json(output / "plan.json", plan)
    if not torch.cuda.is_available():
        raise RuntimeError("LaCT diagnosis requires its native CUDA environment")
    model, config = load_checkpoint(checkpoint, "incremental")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    input_ids = tokenizer(row["input"] + row["answer_prefix"], return_tensors="pt")["input_ids"].to(
        "cuda:0"
    )
    prefix_length = input_ids.shape[-1]
    if prefix_length != generation["input_tokens"]:
        raise ValueError("Tokenized prefix length does not match the saved generation")
    logger.info("Comparing complete-prefix intermediates: %s tokens", prefix_length)
    layer_records, original_logits, incremental_logits = compare_prefill(
        model, input_ids, output, logger
    )
    prefill = {
        "input_tokens": prefix_length,
        "first_nonidentical_intermediate": next(
            (record for record in layer_records if not record.get("bitwise_equal", False)), None
        ),
        "original": summarize_logits(original_logits, tokenizer),
        "incremental": summarize_logits(incremental_logits, tokenizer),
        "logits_difference": numerical_difference(original_logits, incremental_logits),
    }
    write_json(output / "prefill.json", prefill)
    gc.collect()
    torch.cuda.empty_cache()
    common_tokens = torch.tensor(
        [original_ids[:first_difference]], dtype=torch.long, device="cuda:0"
    )
    cache = None
    logger.info("Following %s common generated tokens with a fresh cache", first_difference)
    for position in range(first_difference + 1):
        new_tokens = input_ids if position == 0 else common_tokens[:, position - 1 : position]
        consumed = prefix_length + position
        mask = torch.ones((1, consumed), dtype=torch.long, device="cuda:0")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            incremental = model(
                input_ids=new_tokens,
                attention_mask=mask,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
        cache = incremental.past_key_values
        trace = {
            "generation_position_zero_based": position,
            "consumed_tokens": consumed,
            "argmax": int(incremental.logits[0, -1].argmax()),
            "saved_original_token": original_ids[position],
            "saved_incremental_token": incremental_ids[position],
            "forced_next_token": original_ids[position] if position < first_difference else None,
        }
        with (output / "incremental_trace.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(trace, ensure_ascii=False, allow_nan=False) + "\n")
        incremental_logits = incremental.logits[0, -1].detach().float().cpu()
        del incremental
    complete = torch.cat((input_ids, common_tokens), dim=-1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        original = model(
            input_ids=complete,
            attention_mask=torch.ones_like(complete),
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )
    original_logits = original.logits[0, -1].detach().float().cpu()
    del original
    progress = cache_progress(cache, model, complete.shape[-1])
    divergence = {
        "generation_position_zero_based": first_difference,
        "consumed_tokens": complete.shape[-1],
        "saved_original_token": original_ids[first_difference],
        "saved_incremental_token": incremental_ids[first_difference],
        "original": summarize_logits(original_logits, tokenizer),
        "incremental": summarize_logits(incremental_logits, tokenizer),
        "logits_difference": numerical_difference(original_logits, incremental_logits),
        "cache_progress": progress,
    }
    write_json(output / "divergence.json", divergence)
    logger.info("Divergence logits: %s", json.dumps(divergence, ensure_ascii=False))
    write_json(
        output / "summary.json",
        {
            "finished_at": timestamp(),
            "sample_id": sample_id,
            "first_nonidentical_intermediate": prefill["first_nonidentical_intermediate"],
            "prefill_logit_difference": prefill["logits_difference"],
            "generation_position_zero_based": first_difference,
            "original_argmax": divergence["original"]["argmax"],
            "incremental_argmax": divergence["incremental"]["argmax"],
            "original_top1_top2_gap": divergence["original"]["top1_top2_gap"],
            "incremental_top1_top2_gap": divergence["incremental"]["top1_top2_gap"],
            "original_logit_1_minus_13": divergence["original"]["logit_1_minus_13"],
            "incremental_logit_1_minus_13": divergence["incremental"]["logit_1_minus_13"],
            "cache_progress_consistent": progress["consistent"],
            "num_lact_heads": config.num_lact_heads,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
    )
    if not progress["consistent"]:
        raise RuntimeError("Cache progress is inconsistent; see divergence.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generation-results", type=Path, required=True)
    parser.add_argument(
        "--ruler-data",
        type=Path,
        default=PROJECT_ROOT / "data/benchmarks/ruler-v1/reproductions/lact-760m",
    )
    parser.add_argument("--length", type=int, choices=LENGTHS, default=16384)
    parser.add_argument("--task", choices=TASKS, default="niah_multiquery")
    args = parser.parse_args()
    if os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0") not in {"0", "1"}:
        raise ValueError("CUDA_VISIBLE_DEVICES must select physical GPU 0 or 1")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "diagnosis.log")],
    )
    logger = logging.getLogger(__name__)
    execution = {"started_at": timestamp(), "status": "running"}
    write_json(output / "execution.json", execution)
    start = time.perf_counter()
    try:
        run(args, output, logger)
        execution["status"] = "complete"
    except BaseException as error:
        execution["status"] = "failed"
        execution["error"] = f"{type(error).__name__}: {error}"
        logger.exception("Diagnosis failed")
        raise
    finally:
        execution["finished_at"] = timestamp()
        execution["wall_seconds"] = time.perf_counter() - start
        write_json(output / "execution.json", execution)


if __name__ == "__main__":
    main()
