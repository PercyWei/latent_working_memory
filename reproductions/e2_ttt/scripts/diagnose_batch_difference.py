"""Replay one observed batch difference with a fixed native teacher prefix."""

import argparse
import json
import os
from pathlib import Path

from lm_eval.models.huggingface import HFLM
from optimized import optimize_model
import torch

from evaluate import PROJECT_ROOT, source_record, timestamp, write_json
from validate_batch import encoded_input


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def difference(actual, reference):
    delta = actual - reference
    flat = int(delta.abs().argmax())
    return {
        "equal": torch.equal(actual, reference),
        "max_abs": float(delta.abs().max()),
        "rms": float(delta.square().mean().sqrt()),
        "max_index": [int(index) for index in torch.unravel_index(torch.tensor(flat), delta.shape)],
        "reference_at_max": float(reference.flatten()[flat]),
        "actual_at_max": float(actual.flatten()[flat]),
    }


def replay(model, inputs, teachers, target_row, prefix_tokens, pad):
    captured = {}
    block = model.model.layers[0]

    def capture(name):
        def hook(module, arguments, output):
            captured[name] = output[target_row].detach().float().cpu().clone()

        return hook

    hooks = [
        block.attn_norm.register_forward_hook(capture("attn_norm")),
        block.attn.q_proj.register_forward_hook(capture("q_proj")),
    ]
    ids = torch.tensor(inputs, dtype=torch.long, device="cuda:0")
    try:
        output = model(
            input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, logits_to_keep=1
        )
    finally:
        for hook in hooks:
            hook.remove()
    captured["prefill_logits"] = output.logits[target_row, -1].detach().float().cpu().clone()
    cache = output.past_key_values
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else (eos or [])
    for step in range(prefix_tokens):
        tokens = [
            row[step] if step < len(row) and not any(token in eos for token in row[:step]) else pad
            for row in teachers
        ]
        ids = torch.tensor(tokens, dtype=torch.long, device="cuda:0").unsqueeze(1)
        output = model(
            input_ids=ids,
            attention_mask=torch.ones(
                (len(inputs), len(inputs[0]) + step + 1), dtype=torch.long, device="cuda:0"
            ),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = output.past_key_values
    captured["step_logits"] = output.logits[target_row, -1].detach().float().cpu().clone()
    captured["cache_progress"] = [cache.get_seq_length(index) for index in range(len(cache))]
    return captured


def logits_report(logits, tokenizer, candidates):
    top = logits.topk(5)

    def entries(ids):
        return [
            {"id": token, "text": tokenizer.decode([token]), "logit": float(logits[token])}
            for token in ids
        ]

    return {
        "argmax": int(logits.argmax()),
        "candidates": entries(candidates),
        "candidate_margin": float(logits[candidates[0]] - logits[candidates[1]]),
        "top5": entries(top.indices.tolist()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.validation_output, args.output_dir = (
        args.validation_output.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
    )
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1")
        or int(os.environ.get("WORLD_SIZE", "1")) != 1
    ):
        parser.error("Use one process with CUDA_VISIBLE_DEVICES=0 or 1.")
    validation = json.loads((args.validation_output / "plan.json").read_text())
    rows = read_jsonl(args.validation_output / "predictions.jsonl")
    selected = {row["sample_id"]: row for row in rows if row["run"] == args.run}
    original = next(
        row for row in rows if row["run"] == "original-b1" and row["sample_id"] == args.sample_id
    )
    observed = selected[args.sample_id]
    position = next(
        index
        for index, pair in enumerate(zip(original["generated_ids"], observed["generated_ids"]))
        if pair[0] != pair[1]
    )
    batch = next(
        row
        for row in read_jsonl(args.validation_output / "measurements.jsonl")
        if row["record_type"] == "generation_batch"
        and row["run"] == args.run
        and args.sample_id in row["sample_ids"]
    )
    batch_ids = batch["sample_ids"]
    target_row = batch_ids.index(args.sample_id)
    sources = {
        mode: {
            **source_record(argparse.Namespace(inference_mode=mode)),
            "entrypoint": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        }
        for mode in ("original", "optimized")
    }
    plan = {
        "created_at": timestamp(),
        "source": sources,
        "validation_output": str(args.validation_output),
        "validation_source": validation["source"],
        "checkpoint": validation["arguments"]["checkpoint"],
        "run": args.run,
        "sample_id": args.sample_id,
        "batch_sample_ids": batch_ids,
        "target_row": target_row,
        "first_difference_zero_based": position,
        "shared_teacher_tokens": original["generated_ids"][:position],
        "observed_candidate_ids": [
            original["generated_ids"][position],
            observed["generated_ids"][position],
        ],
        "input_tokens": batch["input_tokens_per_row"],
        "dtype": "bfloat16",
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "teacher_policy": "Target uses original shared prefix; other rows use observed raw IDs, pad after original EOS or exhausted raw IDs; native formulas/precision/backends unchanged",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "plan.json", plan)
    lm = HFLM(
        pretrained=plan["checkpoint"],
        tokenizer=plan["checkpoint"],
        dtype="bfloat16",
        device="cuda:0",
        batch_size=1,
        max_length=2048
        if observed["group"] in ("squad_completion", "swde")
        else int(observed["group"].split("/")[1]),
        truncation=False,
        trust_remote_code=False,
    )
    model = lm.model.eval()
    context_limit = (
        2000 if observed["group"] in ("squad_completion", "swde") else lm.max_length - 128
    )
    inputs = [encoded_input(lm, selected[sample]["prompt"], context_limit) for sample in batch_ids]
    if any(len(row) != plan["input_tokens"] for row in inputs):
        raise ValueError("Re-encoded prompts do not reproduce the observed equal-length batch.")
    teachers = [selected[sample]["raw_generated_ids"] for sample in batch_ids]
    teachers[target_row] = original["generated_ids"][:position]
    single_inputs, single_teacher = [inputs[target_row]], [teachers[target_row]]
    with torch.inference_mode():
        reference = replay(
            model, single_inputs, single_teacher, 0, position, lm.tokenizer.pad_token_id
        )
        contract = {
            name: (value.data_ptr(), value.shape, value.dtype)
            for name, value in model.state_dict().items()
        }
        if optimize_model(model) is not model or contract != {
            name: (value.data_ptr(), value.shape, value.dtype)
            for name, value in model.state_dict().items()
        }:
            raise ValueError(
                "Optimization changed checkpoint tensor storage/layout or model identity."
            )
        controls = {
            "optimized-b1": replay(
                model, single_inputs, single_teacher, 0, position, lm.tokenizer.pad_token_id
            ),
            args.run: replay(
                model, inputs, teachers, target_row, position, lm.tokenizer.pad_token_id
            ),
        }
    summary = {
        "source": sources,
        "plan": str(args.output_dir / "plan.json"),
        "gpu": plan["gpu"],
        "first_difference_zero_based": position,
        "original-b1": {
            "prefill": logits_report(
                reference["prefill_logits"], lm.tokenizer, plan["observed_candidate_ids"]
            ),
            "step": logits_report(
                reference["step_logits"], lm.tokenizer, plan["observed_candidate_ids"]
            ),
            "cache_progress": reference["cache_progress"],
        },
        "comparisons": {},
    }
    for name, current in controls.items():
        summary["comparisons"][name] = {
            "prefill": logits_report(
                current["prefill_logits"], lm.tokenizer, plan["observed_candidate_ids"]
            ),
            "step": logits_report(
                current["step_logits"], lm.tokenizer, plan["observed_candidate_ids"]
            ),
            "differences": {
                key: difference(current[key], reference[key])
                for key in ("attn_norm", "q_proj", "prefill_logits", "step_logits")
            },
            "cache_progress": current["cache_progress"],
            "cache_progress_matches": current["cache_progress"] == reference["cache_progress"],
        }
    summary["finished_at"] = timestamp()
    write_json(args.output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
