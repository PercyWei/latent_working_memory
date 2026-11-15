"""在 CPU 上按执行顺序比较 Muon decode 追踪，区分采样值与完整单 token 输入。"""

import argparse
import json
from pathlib import Path
import re

import torch


MODULE_FIELD = re.compile(r"^(step\d+/.+/call\d+)/(input|output)(?:/\d+)*$")


def load_trace(directory):
    metadata = json.loads((directory / "trace.json").read_text())
    result = json.loads((directory / "result.json").read_text())
    tensors = torch.load(directory / "trace.pt", map_location="cpu", weights_only=True)
    return metadata, result, tensors


def focus_tensor(tensors, metadata, name):
    kind = "activations" if name in tensors["activations"] else "cache_samples"
    value = tensors[kind][name]
    layout = metadata["fields"][name]
    if kind == "activations" and layout["batch_rows"] == "all":
        value = value[0]
    return kind, value, layout


def compare_field(left, right, name):
    left_kind, a, left_layout = focus_tensor(left[2], left[0], name)
    right_kind, b, right_layout = focus_tensor(right[2], right[0], name)
    if left_kind != right_kind:
        raise ValueError(f"Trace field category differs: {name}")
    shape_equal = a.shape == b.shape
    dtype_equal = left_layout["source_dtype"] == right_layout["source_dtype"]
    positions_equal = left_layout["real_token_positions"] == right_layout["real_token_positions"]
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    exact = shape_equal and dtype_equal and positions_equal and torch.equal(a, b)
    maximum = relative = None
    if shape_equal and finite:
        delta = a.double() - b.double()
        maximum = delta.abs().max().item()
        relative = (delta.norm() / b.double().norm().clamp_min(1e-12)).item()
    step = left_layout["step"]
    return {
        "kind": left_kind,
        "step": step,
        "scope": "sampled_cache" if left_kind == "cache_samples" else (
            "sampled_prefill_activation" if step == 0 else "complete_decode_focus_activation"
        ),
        "left_shape": list(a.shape), "right_shape": list(b.shape),
        "shape_equal": shape_equal, "source_dtype_equal": dtype_equal,
        "positions_equal": positions_equal, "finite": finite,
        "exact": bool(exact), "max_abs": maximum, "relative_l2": relative,
        "layout_differences": {
            key: {"left": left_layout.get(key), "right": right_layout.get(key)}
            for key in left_layout.keys() | right_layout.keys()
            if left_layout.get(key) != right_layout.get(key)
        },
        "left_layout": left_layout, "right_layout": right_layout,
    }


def summarize_steps(order, fields):
    modules = {}
    for name in order:
        match = MODULE_FIELD.fullmatch(name)
        if match:
            prefix, role = match.groups()
            modules.setdefault(prefix, {"input": [], "output": []})[role].append(name)
    summaries = []
    for step in dict.fromkeys(fields[name]["step"] for name in order):
        names = [name for name in order if fields[name]["step"] == step]
        first_module = None
        # Prefill only samples positions; equality of those samples is not equality of full inputs.
        if step > 0:
            for prefix, roles in modules.items():
                if not prefix.startswith(f"step{step}/") or not roles["input"] or not roles["output"]:
                    continue
                inputs = [fields[name] for name in roles["input"]]
                complete = all(
                    field["exact"]
                    and field["left_layout"]["source_shape"][1] == 1
                    and field["right_layout"]["source_shape"][1] == 1
                    for field in inputs
                )
                different_outputs = [name for name in roles["output"] if not fields[name]["exact"]]
                if complete and different_outputs:
                    first_module = {"module": prefix, "inputs": roles["input"],
                                    "different_outputs": different_outputs}
                    break
        different = [name for name in names if not fields[name]["exact"]]
        summaries.append({
            "step": step, "label": "prefill" if step == 0 else "first_decode",
            "compared_fields": len(names), "different_fields": len(different),
            "first_observed_difference": next(iter(different), None),
            "first_activation_difference": next((
                name for name in different if fields[name]["kind"] == "activations"
            ), None),
            "first_cache_sample_difference": next((
                name for name in different if fields[name]["kind"] == "cache_samples"
            ), None),
            "first_layout_difference": next((name for name in names if fields[name]["layout_differences"]), None),
            "first_module_with_equal_complete_focus_tensor_inputs": first_module,
        })
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("output", type=Path, help="比较结果 JSON")
    args = parser.parse_args()
    args.left, args.right, args.output = (
        path.expanduser().resolve() for path in (args.left, args.right, args.output)
    )
    left, right = load_trace(args.left), load_trace(args.right)
    order = left[0]["execution_order"]
    if order != right[0]["execution_order"]:
        raise ValueError("Trace execution order differs; these traces do not have matching observation points")
    fields = {name: compare_field(left, right, name) for name in order}
    steps = summarize_steps(order, fields)
    observers = {"left": left[1]["observer_matches_reference"] is True,
                 "right": right[1]["observer_matches_reference"] is True}
    observer_pass = all(observers.values())
    report = {
        "left": str(args.left), "right": str(args.right),
        "observer_matches_reference": observers,
        "observer_pass": observer_pass, "attribution_allowed": observer_pass,
        "scope": {
            "prefill": "仅比较记录位置的采样值；未观察位置可能更早产生差异，采样相同不代表完整输入相同。",
            "decode": "比较首个 decode token 的完整 focus0 激活；all-batch 字段只比较 row0。",
            "cache": "仅比较记录的 KV、pending 和 fast-weight 样本，不能据此断言完整缓存相同。",
            "module_inputs": "完整输入相同仅指 decode 中记录的 focus0 位置参数张量；整个 batch 形状可不同，kwargs 未导出。",
            "precision": "trace.pt 的值已转 float32；原 dtype、shape、stride 来自 trace.json。relative_l2 的分母为 right 的范数。",
        },
        "results": {
            side: {key: run[1][key] for key in ("mode", "focus_id", "score", "token_ids")}
            for side, run in (("left", left), ("right", right))
        },
        "prefill_sample_positions": {"left": left[0]["prefill_sample_positions"],
                                     "right": right[0]["prefill_sample_positions"]},
        "cache_progress": {"left": left[0]["cache_progress"], "right": right[0]["cache_progress"]},
        "steps": steps, "execution_order": order, "fields": fields,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    if not observer_pass:
        print("WARNING: observer canary failed; comparison is exploratory, attribution is not allowed.")
    for step in steps:
        module = step["first_module_with_equal_complete_focus_tensor_inputs"]
        print(f"{step['label']}: {step['different_fields']}/{step['compared_fields']} fields differ; "
              f"first={step['first_observed_difference']}; "
              f"equal-input module={module['module'] if module else None}")
    print(args.output)


if __name__ == "__main__":
    main()
