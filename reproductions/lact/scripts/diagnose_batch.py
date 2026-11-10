"""一次性 LaCT 数值诊断：固定 B1 续写，比较原 sweep 中相同的无 padding 分组。"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from evaluate import IncrementalLaCTForCausalLM, load_checkpoint, read_samples, timestamp, write_json
from provenance import capture_source


WEIGHT_FIELDS = (
    "w0", "w1", "w2", "w0_norm", "w1_norm", "w2_norm",
    "w0_main", "w1_main", "w2_main", "dw0_momentum", "dw1_momentum", "dw2_momentum",
)
PENDING_FIELDS = (
    "pending_k", "pending_v", "pending_lr0", "pending_lr1", "pending_lr2", "pending_momentum",
)


def tensor_difference(actual, reference):
    if actual is None or reference is None:
        return {"same_none": actual is None and reference is None,
                "exact": actual is None and reference is None}
    result = {
        "shape": list(actual.shape), "reference_shape": list(reference.shape),
        "dtype": str(actual.dtype), "reference_dtype": str(reference.dtype),
    }
    if actual.shape != reference.shape:
        return {**result, "exact": False}
    left, right = actual.float(), reference.float()
    difference = left - right
    values = torch.stack((
        difference.abs().amax(),
        difference.norm() / right.norm().clamp_min(1e-12),
        (torch.isfinite(left).all() & torch.isfinite(right).all()).float(),
    )).cpu().tolist()
    return {
        **result, "max_abs": values[0], "relative_l2": values[1],
        "finite": bool(values[2]),
        "exact": bool(values[2]) and values[0] == 0 and actual.dtype == reference.dtype,
    }


def cache_differences(batch_cache, single_cache, focus, model):
    layers = []
    for index, (batch, single, block) in enumerate(zip(
        batch_cache.layers, single_cache.layers, model.model.layers, strict=True
    )):
        heads = block.attn.num_fw_heads
        selected = slice(focus * heads, (focus + 1) * heads)
        fields = {
            f"attention.{name}": tensor_difference(
                getattr(batch, name)[focus:focus + 1], getattr(single, name)
            )
            for name in ("keys", "values")
        }
        for name in WEIGHT_FIELDS:
            value = getattr(batch.ttt, name)
            fields[f"ttt.{name}"] = tensor_difference(
                None if value is None else value[selected], getattr(single.ttt, name)
            )
        for name in PENDING_FIELDS:
            left, right = getattr(batch.ttt, name), getattr(single.ttt, name)
            fields[f"ttt.{name}.fragments"] = {"batch": len(left), "reference": len(right)}
            for fragment, (value, reference) in enumerate(zip(left, right, strict=True)):
                fields[f"ttt.{name}.{fragment}"] = tensor_difference(value[selected], reference)
        progress = {
            "batch": [batch.seen, batch.ttt.seen, batch.ttt.n_updates, batch.ttt.pending_length],
            "reference": [single.seen, single.ttt.seen, single.ttt.n_updates, single.ttt.pending_length],
        }
        layers.append({"layer": index, "progress": progress, "fields": fields})
    return layers


class LastTokenCapture:
    """只在内存保留 focus 的最后位置向量；不改变任何 forward 返回值。"""

    def __init__(self, model):
        self.model = model
        self.enabled = False
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) or name.endswith((
                ".mlp", ".attn_norm", ".mlp_norm", ".ttt_norm", ".q_norm", ".k_norm",
            )) or name == "model.norm":
                self.handles.append(module.register_forward_pre_hook(self.before(name)))
                self.handles.append(module.register_forward_hook(self.after(name)))

    def begin(self, batch_size, focus):
        self.batch_size, self.focus = batch_size, focus
        self.values, self.order, self.head_parts = {}, [], []
        self.samplewise_calls, self.current_rows = {}, {}
        self.enabled = True

    def take(self):
        self.enabled = False
        return self.values, self.order, torch.cat(self.head_parts, dim=0)

    def save(self, name, value, sample_row=None):
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                self.save(f"{name}.{index}", item, sample_row)
        elif isinstance(value, torch.Tensor) and value.ndim >= 2:
            if sample_row is not None:
                selected = value[sample_row:sample_row + 1]
                if value.ndim > 2:
                    selected = selected[:, -1:]
            elif ".ttt_norm." in name:
                layer_index = int(name.split(".")[2])
                heads = self.model.model.layers[layer_index].attn.num_fw_heads
                selected = value[self.focus * heads:(self.focus + 1) * heads, -1:]
            elif value.shape[0] == self.batch_size:
                selected = value[self.focus:self.focus + 1]
                if value.ndim > 2:
                    selected = selected[:, -1:]
            else:
                raise ValueError(f"Unexpected hook tensor shape: {name}: {tuple(value.shape)}")
            self.values[name] = selected.detach().cpu().clone()
            self.order.append(name)

    def before(self, name):
        def hook(module, inputs):
            if self.enabled:
                is_samplewise = (
                    self.batch_size > 1 and inputs[0].shape[0] == 1
                    and (name == "lm_head" or name.endswith((
                        ".lr_proj", ".momentum_proj.0", ".ttt_scale_proj",
                    )))
                )
                if is_samplewise:
                    row = self.samplewise_calls.get(name, 0)
                    self.samplewise_calls[name] = row + 1
                    self.current_rows[name] = row
                    if row == self.focus:
                        self.save(f"{name}.input", inputs, 0)
                else:
                    self.save(f"{name}.input", inputs)
                if name == "lm_head":
                    self.head_parts.append(inputs[0][:, -1:].detach().cpu().clone())
        return hook

    def after(self, name):
        def hook(module, inputs, output):
            if self.enabled:
                if name in self.current_rows:
                    if self.current_rows[name] == self.focus:
                        self.save(f"{name}.output", output, 0)
                else:
                    self.save(f"{name}.output", output)
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


def logit_report(actual, reference):
    actual, reference = actual.float().cpu(), reference.float().cpu()
    rows = {}
    for name, vector in (("batch", actual), ("reference", reference)):
        values, ids = vector.topk(5)
        rows[name] = {
            "top5_ids": ids.tolist(), "top5_logits": values.tolist(),
            "top2_margin": float(values[0] - values[1]),
        }
    candidates = sorted(set(rows["batch"]["top5_ids"][:2] + rows["reference"]["top5_ids"][:2]))
    rows["competing_tokens"] = [
        {"id": token, "batch_logit": float(actual[token]),
         "reference_logit": float(reference[token]), "difference": float(actual[token] - reference[token])}
        for token in candidates
    ]
    rows["difference"] = tensor_difference(actual, reference)
    rows["argmax_equal"] = rows["batch"]["top5_ids"][0] == rows["reference"]["top5_ids"][0]
    return rows


def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1"):
        raise ValueError("Set CUDA_VISIBLE_DEVICES=0 or 1 explicitly")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_plan = json.loads((args.sweep_dir / "plan.json").read_text())
    selection = json.loads((args.sweep_dir / "selection_32768.json").read_text())
    trials = json.loads((args.sweep_dir / "generation_32768.json").read_text())
    reference = next(row["token_ids"] for row in trials if row["batch_size"] == 1)
    original_trial = next(row for row in trials if row["batch_size"] == args.batch_size)
    checkpoint, data = Path(source_plan["checkpoint"]), Path(source_plan["ruler_data"])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    rows = []
    source_rows = {}
    for metadata in selection["samples"]:
        key = (metadata["length"], metadata["task"])
        if key not in source_rows:
            source_rows[key] = read_samples(data / str(key[0]) / key[1] / "test.jsonl")
        row = source_rows[key][metadata["sample_position"]]
        if row["index"] != metadata["source_index"]:
            raise ValueError(f"Selection source mismatch: {metadata['id']}")
        rows.append({**metadata, "input_ids": tokenizer.encode(row["input"] + row["answer_prefix"], truncation=False)})
    groups = defaultdict(list)
    for row in rows:
        groups[len(row["input_ids"])].append(row)
    batches = [group[start:start + args.batch_size] for group in groups.values()
               for start in range(0, len(group), args.batch_size)]
    group = next(group for group in batches if args.focus_id in [row["id"] for row in group])
    focus = [row["id"] for row in group].index(args.focus_id)
    write_json(args.output_dir / "plan.json", {
        "created_at": timestamp(),
        "source": capture_source(__file__, {"incremental": IncrementalLaCTForCausalLM}),
        "source_sweep": str(args.sweep_dir), "checkpoint": str(checkpoint),
        "focus_id": args.focus_id, "group_ids": [row["id"] for row in group],
        "batch_size": len(group), "decode_steps": args.decode_steps,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    })
    tokens = torch.tensor([row["input_ids"] for row in group], dtype=torch.long, device="cuda:0")
    input_tokens = tokens.shape[1]
    model, _ = load_checkpoint(checkpoint, "incremental")
    capture = LastTokenCapture(model)
    batch_cache = single_cache = None
    report = {
        "created_at": timestamp(), "formal_benchmark_result": False,
        "source_sweep": str(args.sweep_dir), "source_sweep_revision": source_plan["source"],
        "focus_id": args.focus_id, "batch_size": len(group), "focus_row": focus,
        "group_ids": [row["id"] for row in group], "input_tokens": input_tokens,
        "teacher_forced_continuation_tokens": args.decode_steps,
        "protocol": "Fresh B4/B1 caches; every row follows its original B1 continuation. Hooks capture only the focus last token. Cache statistics compare the entire focus state. No tensor dumps.",
        "steps": [],
    }

    def forward(input_ids, seen, cache, row):
        capture.begin(input_ids.shape[0], row)
        output = model(
            input_ids=input_ids, attention_mask=torch.ones(
                (input_ids.shape[0], seen), dtype=torch.long, device=input_ids.device
            ), past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True,
        )
        return output, capture.take()

    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for step in range(args.decode_steps + 1):
                current = tokens if step == 0 else torch.tensor(
                    [[reference[row["id"]][step - 1]] for row in group], dtype=torch.long, device="cuda:0"
                )
                seen = input_tokens + step
                batch, (batch_vectors, order, batch_head) = forward(current, seen, batch_cache, focus)
                batch_cache = batch.past_key_values
                single, (single_vectors, single_order, single_head) = forward(
                    current[focus:focus + 1], seen, single_cache, 0
                )
                single_cache = single.past_key_values
                if order != single_order:
                    raise ValueError("Batch and B1 called hooked modules in different orders")
                modules = {name: tensor_difference(batch_vectors[name], single_vectors[name]) for name in order}
                first_different = next((name for name in order if not modules[name]["exact"]), None)
                first_output = next((name for name in order if ".output" in name and not modules[name]["exact"]), None)
                # Same batch-derived hidden, head applied with M=B versus M=1; hooks are disabled here.
                batch_head = batch_head.to("cuda:0")
                single_head = single_head.to("cuda:0")
                replay = model.lm_head(batch_head)
                batch_hidden_m1 = model.lm_head(batch_head[focus:focus + 1])
                reference_hidden_m1 = model.lm_head(single_head)
                row_report = {
                    "phase": "prefill" if step == 0 else "decode", "generated_position": step,
                    "seen": seen, "fed_reference_token": None if step == 0 else reference[args.focus_id][step - 1],
                    "reference_generation_token": reference[args.focus_id][step],
                    "original_batch_generation_token": original_trial["token_ids"][args.focus_id][step],
                    "logits": logit_report(batch.logits[focus, -1], single.logits[0, -1]),
                    "fresh_b1_matches_saved_reference_argmax": int(single.logits[0, -1].argmax()) == reference[args.focus_id][step],
                    "first_different_last_token_module": first_different,
                    "first_different_last_token_module_output": first_output,
                    "last_token_modules": modules,
                    "lm_head_isolation": {
                        "raw_head_MB_replay_vs_actual": tensor_difference(replay[focus, -1], batch.logits[focus, -1]),
                        "raw_head_M1_replay_vs_actual": tensor_difference(batch_hidden_m1[0, -1], batch.logits[focus, -1]),
                        "same_batch_hidden_head_MB_vs_M1": logit_report(replay[focus, -1], batch_hidden_m1[0, -1]),
                        "head_M1_batch_hidden_vs_reference_hidden": logit_report(batch_hidden_m1[0, -1], reference_hidden_m1[0, -1]),
                        "reference_replay_vs_actual": tensor_difference(reference_hidden_m1[0, -1], single.logits[0, -1]),
                    },
                    "cache_layers": cache_differences(batch_cache, single_cache, focus, model),
                }
                report["steps"].append(row_report)
                write_json(args.output_dir / "diagnosis.json", report)
                print(json.dumps({"generated_position": step, "argmax_equal": row_report["logits"]["argmax_equal"],
                                  "first_different_module": first_different}, ensure_ascii=False), flush=True)
                del batch, single, batch_vectors, single_vectors, batch_head, single_head, replay, batch_hidden_m1, reference_hidden_m1
    finally:
        capture.close()
    report["completed_at"] = timestamp()
    write_json(args.output_dir / "diagnosis.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--focus-id", default="ruler/32768/niah_single_2/35")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--decode-steps", type=int, default=6)
    args = parser.parse_args()
    if args.batch_size < 2 or not 1 <= args.decode_steps < 128:
        parser.error("batch-size >= 2 and 1 <= decode-steps < 128 are required")
    run(args)


if __name__ == "__main__":
    main()
