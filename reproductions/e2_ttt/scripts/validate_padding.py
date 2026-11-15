"""检查左 padding、异长状态时序及真实 RULER 生成；不执行完整 benchmark。"""

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import time

import torch
import e2_ttt  # noqa: F401 -- register the native checkpoint classes
from lm_eval.models.huggingface import HFLM
from optimized import optimize_model
from latent_working_memory.benchmarks.vendor.ruler import postprocess_pred, string_match_all


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_record():
    root = Path(__file__).resolve().parents[3]
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "status": git("status", "--porcelain=v1").splitlines()}


def pad(rows, pad_id, extra=0):
    width = max(map(len, rows)) + extra
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device="cuda:0")
    mask = torch.zeros_like(ids)
    for index, row in enumerate(rows):
        ids[index, -len(row):] = torch.tensor(row, device=ids.device)
        mask[index, -len(row):] = 1
    return ids, mask


def forward(model, ids, mask, cache=None):
    return model(input_ids=ids, attention_mask=mask, past_key_values=cache,
                 use_cache=True, logits_to_keep=1, return_dict=True)


def snapshot(output, row, seen, chunk, model):
    """Compare each padded row with the unchanged native B1 cache fields."""
    tensors, progress = {"logits": output.logits[row, -1].detach()}, []
    swiglu = model.config.model_type == "e2_ttt_swiglu"
    buffer_index = 3 if swiglu else 2

    def flatten(value, path):
        if isinstance(value, torch.Tensor):
            tensors[path] = value.detach()
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                flatten(item, f"{path}/{index}")
        elif isinstance(value, dict):
            for key, item in value.items():
                flatten(item, f"{path}/{key}")
        elif value is None or isinstance(value, (bool, int, float)):
            tensors[path] = value
        else:
            raise TypeError(f"Unexpected native state type at {path}: {type(value)}")

    cache = output.past_key_values
    for index in range(len(cache)):
        layer = cache[index]
        if "padding_state" in layer:
            padding = layer["padding_state"]
            state = padding["rows"][row]
            actual_seen = padding["seen"][row]
        else:
            if row != 0:
                raise ValueError("The unpadded reference must have batch size one")
            state = layer
            actual_seen = cache.get_seq_length(index)
        for name in ("attn_state", "recurrent_state"):
            flatten(state[name], f"{index}/{name}")
        buffer_length = state["recurrent_state"][buffer_index].shape[-1]
        actual = {"seen": actual_seen, "buffer": buffer_length,
                  "updates": (actual_seen - buffer_length) // chunk}
        expected = {"seen": seen, "buffer": seen % chunk, "updates": seen // chunk}
        progress.append({"layer": index, "actual": actual, "expected": expected})
    return tensors, progress

def compare(actual, expected, atol, rtol, include_logits=True):
    left, left_progress = actual
    right, right_progress = expected
    report = {"structure": left.keys() == right.keys(), "finite": True, "exact": True,
              "allclose": True, "max_abs": 0.0, "max_relative_l2": 0.0,
              "argmax_equal": bool(torch.equal(left["logits"].argmax(-1), right["logits"].argmax(-1)))
                              if include_logits else None,
              "worst_field": None, "fields": 0,
              "progress": left_progress == right_progress and all(
                  value["actual"] == value["expected"] for value in left_progress
              )}
    for name in left.keys() & right.keys():
        if name == "logits" and not include_logits:
            continue
        a, b = left[name], right[name]
        report["fields"] += 1
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            same = type(a) is type(b) and a == b
            report["structure"] &= same
            report["exact"] &= same
            report["allclose"] &= same
            continue
        if a is None or b is None:
            same = a is None and b is None
            report["structure"] &= same
            report["exact"] &= same
            report["allclose"] &= same
            continue
        if a.shape != b.shape or a.dtype != b.dtype:
            report.update(structure=False, exact=False, allclose=False, worst_field=name)
            continue
        if not a.numel():
            continue
        af, bf = a.float(), b.float()
        delta = af - bf
        finite, equal, close, maximum, relative = torch.stack((
            (torch.isfinite(af).all() & torch.isfinite(bf).all()).float(),
            (a == b).all().float(), torch.isclose(a, b, atol=atol, rtol=rtol).all().float(),
            delta.abs().max(), delta.norm() / bf.norm().clamp_min(1e-12),
        )).cpu().tolist()
        report["finite"] &= bool(finite)
        report["exact"] &= bool(equal)
        report["allclose"] &= bool(close)
        if finite and maximum > report["max_abs"]:
            report["max_abs"], report["worst_field"] = maximum, name
        if finite:
            report["max_relative_l2"] = max(report["max_relative_l2"], relative)
    return report


def source_samples(args, lm):
    candidates = []
    for task in ("niah_single_1", "niah_single_2"):
        path = args.ruler_data / str(args.length) / task / "test.jsonl"
        with path.open() as stream:
            for position, line in enumerate(stream):
                if position == 64:
                    break
                row = json.loads(line)
                encoded, mask = lm.tok_batch_encode([row["input"] + row["answer_prefix"]],
                                                     left_truncate_len=args.length - 128, truncation=False)
                if not bool(mask.all()):
                    raise ValueError("Singleton RULER input unexpectedly has padding")
                ids = encoded[0].tolist()
                candidates.append({"id": f"ruler/{args.length}/{task}/{position}",
                                   "input_ids": ids, "references": row["outputs"]})
    ordered = sorted(candidates, key=lambda row: len(row["input_ids"]))
    selected = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    if len({len(row["input_ids"]) for row in selected}) < 2:
        raise ValueError("The selected real prompts must contain different token lengths")
    return candidates[:3] + candidates[-2:], selected


def boundary_probe(model, tokenizer, sources, args, chunk, output):
    needed = 2 * chunk + args.teacher_steps + 8
    if any(len(row["input_ids"]) < needed for row in sources):
        raise ValueError(f"Use RULER sources containing at least {needed} tokens")
    rows = [row["input_ids"][-needed:] for row in sources]
    reports = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for multiple in (1, 2):
        prefixes = [multiple * chunk - 1, multiple * chunk, multiple * chunk + 1]
        variants = {"mixed": rows[:3], "pad_values": rows[:3],
                    "isolation": [rows[0], rows[3], rows[4]]}
        states, masks, results = {}, {}, {}
        for name, values in variants.items():
            ids, masks[name] = pad([value[:length] for value, length in zip(values, prefixes)], pad_id)
            if name == "pad_values":
                ids[masks[name] == 0] = (pad_id + 1) % model.config.vocab_size
            results[name] = forward(model, ids, masks[name])
            states[name] = results[name].past_key_values
        references = []
        for index in range(3):
            ids, mask = pad([rows[index][:prefixes[index]]], pad_id)
            references.append(forward(model, ids, mask))
        for name in ("padded_single", "padded_single_values"):
            ids, masks[name] = pad([rows[0][:prefixes[0]]], pad_id, extra=3)
            if name == "padded_single_values":
                ids[masks[name] == 0] = (pad_id + 1) % model.config.vocab_size
            results[name] = forward(model, ids, masks[name])
            states[name] = results[name].past_key_values
        seen = list(prefixes)
        for step in range(args.teacher_steps + 2):
            freeze = step == args.teacher_steps + 1
            if step:
                for name, values in {**variants, "padded_single": rows[:1], "padded_single_values": rows[:1]}.items():
                    valid = [0 if freeze and index == 0 else 1 for index in range(len(values))]
                    next_ids = [[value[prefixes[index] + step - 1]] for index, value in enumerate(values)]
                    ids = torch.tensor(next_ids, device="cuda:0")
                    if name in {"pad_values", "padded_single_values"}:
                        ids[torch.tensor(valid, device=ids.device) == 0] = (pad_id + 1) % model.config.vocab_size
                    masks[name] = torch.cat((masks[name], torch.tensor(valid, device=ids.device)[:, None]), dim=1)
                    results[name] = forward(model, ids, masks[name], states[name])
                    states[name] = results[name].past_key_values
                for index in range(3):
                    if freeze and index == 0:
                        continue
                    seen[index] += 1
                    ids = torch.tensor([[rows[index][prefixes[index] + step - 1]]], device="cuda:0")
                    references[index] = forward(model, ids, torch.ones((1, seen[index]), device=ids.device, dtype=torch.long),
                                                references[index].past_key_values)
            snapshots = [snapshot(results["mixed"], index, seen[index], chunk, model) for index in range(3)]
            comparisons = {}
            for index in range(3):
                valid_logits = not (freeze and index == 0)
                reference = snapshot(references[index], 0, seen[index], chunk, model)
                comparisons[f"b1/{index}"] = compare(snapshots[index], reference, args.numeric_atol, args.numeric_rtol, valid_logits)
                comparisons[f"pad_values/{index}"] = compare(
                    snapshot(results["pad_values"], index, seen[index], chunk, model), snapshots[index], 0, 0, valid_logits)
            comparisons["padded_single"] = compare(
                snapshot(results["padded_single"], 0, seen[0], chunk, model),
                snapshot(references[0], 0, seen[0], chunk, model), args.numeric_atol, args.numeric_rtol, not freeze)
            comparisons["focus_isolation"] = compare(
                snapshot(results["isolation"], 0, seen[0], chunk, model), snapshots[0], 0, 0, not freeze)
            comparisons["padded_single_values"] = compare(
                snapshot(results["padded_single_values"], 0, seen[0], chunk, model),
                snapshot(results["padded_single"], 0, seen[0], chunk, model), 0, 0, not freeze)
            physical = {name: cache.get_seq_length() for name, cache in states.items()}
            reports.append({"prefix_tokens": prefixes, "step": step, "seen": list(seen),
                            "row0_new_token_masked": freeze, "comparisons": comparisons,
                            "physical_widths": physical,
                            "physical_progress": all(physical[name] == masks[name].shape[1] for name in states),
                            "focus_partners_changed": [
                                rows[index][:prefixes[index]] != variants["isolation"][index][:prefixes[index]]
                                for index in (1, 2)
                            ]})
            write_json(output / "boundary.json", reports)
        del states, results, references, snapshots
        gc.collect()
    return reports


def generate(model, tokenizer, rows, budget):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ids, mask = pad([row["input_ids"] for row in rows], pad_id)
    output = model.generate(input_ids=ids, attention_mask=mask, do_sample=False,
                            num_beams=1, max_new_tokens=budget, use_cache=True,
                            logits_to_keep=1, pad_token_id=pad_id, output_scores=False)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    results = []
    for row, tokens in zip(rows, output[:, ids.shape[1]:].cpu().tolist(), strict=True):
        end = next((index + 1 for index, token in enumerate(tokens) if token in eos), len(tokens))
        tokens = tokens[:end]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        results.append({"id": row["id"], "token_ids": tokens, "prediction": text,
                        "score": string_match_all([postprocess_pred(text)], [row["references"]])})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ruler-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--teacher-steps", type=int, default=2)
    parser.add_argument("--generation-tokens", type=int, default=8)
    parser.add_argument("--numeric-atol", type=float, default=0.03)
    parser.add_argument("--numeric-rtol", type=float, default=0.03)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "1"} or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Use one process on physical GPU 0 or 1")
    if args.teacher_steps < 1 or args.generation_tokens < 1 or min(args.numeric_atol, args.numeric_rtol) < 0:
        parser.error("Use positive probe budgets and nonnegative tolerances")
    for name in ("checkpoint", "ruler_data", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plan = {"source": source_record(), "environment": {"cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "torch_compile_disable": os.environ.get("TORCH_COMPILE_DISABLE", "0"),
            "torch": str(torch.__version__), "cuda": torch.version.cuda}, "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "acceptance": "Structure, finite values, logical progress, exact pad-value/focus isolation; cross-B numerics and greedy differences reported separately",
            "generation": "Real unmodified RULER prompts and original tokenizer; requested budget is a validation budget, not a benchmark run"}
    write_json(args.output_dir / "plan.json", plan)
    started = time.perf_counter()
    try:
        lm = HFLM(pretrained=str(args.checkpoint), tokenizer=str(args.checkpoint),
                  dtype="bfloat16", device="cuda:0", batch_size=1,
                  max_length=args.length, truncation=False, trust_remote_code=False)
        model = optimize_model(lm.model)
        model.requires_grad_(False)
        model.eval()
        config, tokenizer = model.config, lm.tokenizer
        sources, real = source_samples(args, lm)
        plan.update(checkpoint_config=config.to_dict(),
                    selected=[{"id": row["id"], "tokens": len(row["input_ids"])} for row in real],
                    boundary_sources=[{"id": row["id"], "tokens": len(row["input_ids"])} for row in sources],
                    boundary_slice="Contiguous final 2*C + teacher_steps + 8 real tokens; prefixes are diagnostic inputs")
        write_json(args.output_dir / "plan.json", plan)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            boundary = boundary_probe(model, tokenizer, sources, args, config.ttt_chunk_size, args.output_dir)
            singles = [generate(model, tokenizer, [row], args.generation_tokens)[0] for row in real]
            mixed = generate(model, tokenizer, real, args.generation_tokens)
        checks = [value for case in boundary for value in case["comparisons"].values()]
        passed = all(value["structure"] and value["finite"] and value["progress"] for value in checks)
        passed &= all(case["physical_progress"] and all(case["focus_partners_changed"]) for case in boundary)
        passed &= all(value["exact"] for case in boundary for key, value in case["comparisons"].items()
                      if key.startswith("pad_values/") or key in {"focus_isolation", "padded_single_values"})
        summary = {"passed": passed, "cross_batch_numerics_allclose": all(value["allclose"] for value in checks),
                   "all_generation_ids_equal": all(a["token_ids"] == b["token_ids"] for a, b in zip(singles, mixed, strict=True)),
                   "all_generation_scores_equal": all(a["score"] == b["score"] for a, b in zip(singles, mixed, strict=True)),
                   "generation": {"single": singles, "mixed": mixed}, "boundary_cases": len(boundary),
                   "seconds": time.perf_counter() - started, "device": torch.cuda.get_device_name(0)}
        write_json(args.output_dir / "summary.json", summary)
        print(json.dumps({key: value for key, value in summary.items() if key != "generation"}), flush=True)
        if not passed:
            raise RuntimeError("Padding state, isolation or finite-value checks failed")
    except BaseException as error:
        write_json(args.output_dir / "failure.json", {"error": f"{type(error).__name__}: {error}", "seconds": time.perf_counter() - started})
        raise


if __name__ == "__main__":
    main()
