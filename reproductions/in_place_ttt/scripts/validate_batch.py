"""用真实 4K／8K RULER 样本对照原始 B1 与优化批量推理。"""

import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import time

from opencompass.utils.text_postprocessors import extract_non_reasoning_content
import torch
from transformers import AutoTokenizer

from in_place_ttt.scripts.provenance import capture_source
from in_place_ttt.scripts.records import (
    GENERATION_BUDGET, TASKS, build_protocol, load_samples, timestamp, write_json,
)
from in_place_ttt.scripts.runtime import (
    MODEL_CLASSES, generate_batch, generate_one, load_model, runtime_info,
)


LENGTHS = (4096, 8192)


def select_samples(expected, tokenizer, length):
    candidates = []
    for sample_id, sample in expected.items():
        if sample[0] != length or sample[2] >= 20:
            continue
        row = sample[4]
        ids = tokenizer.encode(row["input"] + row["answer_prefix"], add_special_tokens=True,
                               truncation=False)
        if len(ids) + GENERATION_BUDGET > length:
            raise ValueError(f"Input plus generation budget exceeds {length}: {sample_id}")
        candidates.append({"id": sample_id, "sample": sample, "token_ids": ids})
    # 优先单键检索任务的自然等长样本；不修改真实文本。
    order = sorted(TASKS, key=lambda task: (task != "niah_single_1", not task.startswith("niah")))
    for task in order:
        groups = defaultdict(list)
        for item in candidates:
            if item["sample"][1] == task:
                groups[len(item["token_ids"])].append(item)
        for size in sorted(groups, reverse=True):
            rows = groups[size]
            if len(rows) < 2:
                continue
            a = rows[0]
            others = [item for item in rows[1:] if item["token_ids"] != a["token_ids"]]
            if not others:
                continue
            shorter = [item for item in candidates if len(item["token_ids"]) < size]
            same_task = [item for item in shorter if item["sample"][1] == task]
            pool = same_task or [item for item in shorter if item["sample"][1].startswith("niah")]
            pool = pool or shorter
            if pool:
                return [a, others[0], min(pool, key=lambda item: len(item["token_ids"]))]
    raise ValueError(f"No equal A/B plus shorter C in the first 20 rows per task at {length}")


def cache_metadata(cache, config, batch, outputs):
    optimized = hasattr(cache, "rows")
    rows = cache.rows if optimized else [cache]
    details = []
    for index, (row, item, result) in enumerate(zip(rows, batch, outputs, strict=True)):
        expected_seen = len(item["token_ids"]) + result["generated_tokens"] - 1
        seen = cache.seen[index] if optimized else row.get_seq_length()
        kv = [layer.keys.shape[-2] for layer in row.layers]
        kv_values = [layer.values.shape[-2] for layer in row.layers]
        tails = []
        if config.ttt_mode:
            for layer_index in config.ttt_layers:
                past_x, past_t, weight = row.ttt_states[layer_index]
                tails.append({
                    "layer": layer_index,
                    "x": 0 if past_x is None else past_x.shape[1],
                    "target": 0 if past_t is None else past_t.shape[1],
                    "weight_shape": None if weight is None else list(weight.shape),
                })
        passed = seen == expected_seen and len(kv) == config.num_hidden_layers
        passed = passed and all(value == expected_seen for value in [*kv, *kv_values])
        passed = passed and all(
            tail["x"] == tail["target"] == expected_seen % config.ttt_chunk for tail in tails
        )
        details.append({"id": item["id"], "seen": seen, "expected_seen": expected_seen,
                        "kv_lengths": kv, "ttt_tails": tails, "passed": passed})
    return {
        "rows": details,
        "passed": all(row["passed"] for row in details),
        "early_finish_exercised": len({result["generated_tokens"] for result in outputs}) > 1,
    }


def measure(wrapper, batch, length):
    captured = {}

    def capture(module, args, output):
        captured["cache"] = output.past_key_values

    handle = wrapper.model.register_forward_hook(capture)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    try:
        prompts = [item["sample"][4]["input"] + item["sample"][4]["answer_prefix"] for item in batch]
        with torch.no_grad():
            if wrapper.inference_mode == "original":
                outputs = [generate_one(wrapper, prompts[0], GENERATION_BUDGET, length)]
            else:
                outputs = generate_batch(wrapper, prompts, GENERATION_BUDGET, length,
                                         token_ids=[item["token_ids"] for item in batch])
        torch.cuda.synchronize()
        metrics = {
            "seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        }
        metadata = cache_metadata(captured["cache"], wrapper.model.config, batch, outputs)
        return outputs, metrics, metadata
    finally:
        handle.remove()


def compare(left, right):
    left_ids, right_ids = left["token_ids"], right["token_ids"]
    first = next((index for index, (a, b) in enumerate(zip(left_ids, right_ids)) if a != b), None)
    if first is None and len(left_ids) != len(right_ids):
        first = min(len(left_ids), len(right_ids))
    return {"token_ids_equal": left["token_ids"] == right["token_ids"],
            "first_different_token": first,
            "generated_tokens": left["generated_tokens"],
            "reference_generated_tokens": right["generated_tokens"],
            "raw_text_equal": left["raw_prediction"] == right["raw_prediction"],
            "text_equal": left["prediction"] == right["prediction"],
            "native_score_equal": left["native_score"] == right["native_score"]}


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["length"], row["task"]].append(row)
    scores = {f"{length}/{task}": TASKS[task]().score(
        [row["prediction"] for row in group], [row["gold"] for row in group],
    ) for (length, task), group in groups.items()}
    return {"samples": len(rows), "task_length_scores": scores,
            "task_length_macro_average": sum(value["score"] for value in scores.values()) / len(scores)}


def inspect_batch_difference(wrapper, rows, selected):
    """在首个等长 B2 分歧点比较同历史的 logits 和第一层投影。"""
    references = {row["id"]: row for row in rows if row["condition"] == "optimized_b1"}
    for row in rows:
        if row["condition"] == "equal_b2" and row["token_ids"] != references[row["id"]]["token_ids"]:
            break
    else:
        return None
    reference = references[row["id"]]
    first = compare(row, reference)["first_different_token"]
    batch = selected[row["length"]][:2]
    sample_index = next(i for i, item in enumerate(batch) if item["id"] == row["id"])

    def capture(items, index):
        step, saved = [-1], {}

        def begin(module, args):
            step[0] += 1

        def projection(module, args, output):
            if step[0] == first:
                saved["input"] = args[0][index, -1].detach().cpu().clone()
                saved["projection"] = output[index, -1].detach().cpu().clone()

        def logits(module, args, output):
            if step[0] == first:
                saved["logits"] = output.logits[index, -1].detach().float().cpu().clone()

        handles = [wrapper.model.register_forward_pre_hook(begin),
                   wrapper.model.model.layers[0].self_attn.q_proj.register_forward_hook(projection),
                   wrapper.model.register_forward_hook(logits)]
        try:
            outputs = generate_batch(
                wrapper, [item["sample"][4]["input"] + item["sample"][4]["answer_prefix"] for item in items],
                GENERATION_BUDGET, row["length"], token_ids=[item["token_ids"] for item in items],
            )
        finally:
            for handle in handles:
                handle.remove()
        saved["token_ids"] = outputs[index]["token_ids"]
        return saved

    single = capture([batch[sample_index]], 0)
    multiple = capture(batch, sample_index)
    top = torch.topk(single["logits"], 5).indices.tolist()
    top = list(dict.fromkeys(top + torch.topk(multiple["logits"], 5).indices.tolist()))
    return {
        "id": row["id"], "first_different_token": first,
        "common_generated_prefix_equal": single["token_ids"][:first] == multiple["token_ids"][:first],
        "reproduced_b1": single["token_ids"] == reference["token_ids"],
        "reproduced_b2": multiple["token_ids"] == row["token_ids"],
        "first_layer_q_proj_input_equal": torch.equal(single["input"], multiple["input"]),
        "first_layer_q_proj_max_abs": (single["projection"].float() - multiple["projection"].float()).abs().max().item(),
        "logits_max_abs": (single["logits"] - multiple["logits"]).abs().max().item(),
        "b1_argmax": single["logits"].argmax().item(), "b2_argmax": multiple["logits"].argmax().item(),
        "candidate_logits": [{"token_id": token, "text": wrapper.tokenizer.decode([token]),
                              "b1": single["logits"][token].item(), "b2": multiple["logits"][token].item()} for token in top],
    }


def run(args, plan, execution):
    protocol = build_protocol(args.checkpoint, LENGTHS, "original", 0)
    expected = load_samples(protocol)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    selected = {length: select_samples(expected, tokenizer, length) for length in LENGTHS}
    plan.update(protocol=protocol, selected_samples={str(length): [
        {"role": role, "id": item["id"], "input_tokens": len(item["token_ids"]),
         "source": item["sample"][3], "source_index": item["sample"][4]["index"]}
        for role, item in zip(("A", "B", "C"), batch, strict=True)
    ] for length, batch in selected.items()})
    write_json(args.output_dir / "plan.json", plan)
    rows, calls, runtimes, difference = [], [], {}, None
    with (args.output_dir / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for mode in ("original", "optimized"):
            wrapper = load_model(args.checkpoint, mode, mlp_chunk_size=0, max_seq_len=max(LENGTHS))
            runtimes[mode] = runtime_info(wrapper)
            execution["runtime"] = runtimes
            write_json(args.output_dir / "execution.json", execution)
            eos = wrapper.model.generation_config.eos_token_id
            eos = {eos} if isinstance(eos, int) else set(eos or [])
            for length, (a, b, c) in selected.items():
                cases = [(f"{mode}_b1", [item]) for item in (a, b, c)]
                if mode == "optimized":
                    cases += [("equal_b2", [a, b]), ("mixed_b2", [a, c]),
                              ("reverse_b2", [c, a]), ("replay_b1", [a])]
                for condition, batch in cases:
                    if len({item["id"] for item in batch}) != len(batch):
                        raise ValueError("Duplicate sample IDs within one generation batch")
                    outputs, metrics, metadata = measure(wrapper, batch, length)
                    calls.append({"condition": condition, "length": length,
                                  "sample_ids": [item["id"] for item in batch],
                                  "batch_input_width": max(len(item["token_ids"]) for item in batch),
                                  **metrics, "cache": metadata})
                    print(f"{mode} {length} {condition}: {metrics['seconds']:.3f}s", flush=True)
                    for item, result in zip(batch, outputs, strict=True):
                        task, gold = item["sample"][1], item["sample"][4]["outputs"]
                        prediction = extract_non_reasoning_content(result["raw_prediction"])
                        row = {"condition": condition, "id": item["id"], "length": length,
                               "task": task, **result, "prediction": prediction, "gold": gold,
                               "native_score": TASKS[task]().score([prediction], [gold]),
                               "eos_positions": [i for i, token in enumerate(result["token_ids"]) if token in eos],
                               "reached_budget": result["generated_tokens"] == GENERATION_BUDGET}
                        rows.append(row)
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
            if mode == "optimized":
                difference = inspect_batch_difference(wrapper, rows, selected)
            del wrapper
            gc.collect()
            torch.cuda.empty_cache()
    references = {mode: {row["id"]: row for row in rows if row["condition"] == f"{mode}_b1"}
                  for mode in ("original", "optimized")}
    comparisons = [{"condition": row["condition"], "id": row["id"],
                    **{mode: compare(row, references[mode][row["id"]]) for mode in references}}
                   for row in rows]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    aggregates = {condition: {"actual": aggregate(group), **{
        mode: aggregate([references[mode][row["id"]] for row in group]) for mode in references
    }} for condition, group in grouped.items()}
    write_json(args.output_dir / "summary.json", {
        "runtime": runtimes, "calls": calls, "comparisons": comparisons, "aggregates": aggregates,
        "first_equal_batch_difference": difference,
        "cache_checks_passed": all(call["cache"]["passed"] for call in calls),
        "note": "Small-sample validation; token, text and score differences are reported separately; not a benchmark speed measurement.",
    })
    if not all(call["cache"]["passed"] for call in calls):
        raise RuntimeError("Cache progress or sample-state checks failed; inspect summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1"):
        parser.error("Explicitly select one physical GPU 0 or 1 with CUDA_VISIBLE_DEVICES")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plan = {"created_at": timestamp(), "purpose": "paired small-sample batch validation",
            "checkpoint": str(args.checkpoint), "lengths": list(LENGTHS), "mlp_chunk_size": 0,
            "generation_budget": GENERATION_BUDGET, "source": capture_source(__file__, MODEL_CLASSES),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "timing": "Generation wall time with synchronization; excludes loading and scoring; no warmup."}
    write_json(args.output_dir / "plan.json", plan)
    execution = {"started_at": timestamp(), "status": "running"}
    write_json(args.output_dir / "execution.json", execution)
    try:
        run(args, plan, execution)
        execution["status"] = "completed"
    except BaseException as error:
        execution.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        execution["finished_at"] = timestamp()
        write_json(args.output_dir / "execution.json", execution)


if __name__ == "__main__":
    main()
