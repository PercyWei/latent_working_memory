"""追踪旧无 padding B1/B3 的真实生成前两步，不重放或包装 compiled TTT 函数。"""

import argparse
import json
import os
from pathlib import Path
import time

from diagnose_muon_batch import (
    FOCUS_ID, GENERATION_TOKENS, compare_reference, describe_logits,
    load_checkpoint, make_inputs, read_sample, write_json,
)
from incremental import IncrementalLaCTForCausalLM
from provenance import capture_source
import torch
from transformers import AutoTokenizer

from latent_working_memory.benchmarks.vendor.ruler import postprocess_pred, string_match_all


class DecodeTrace:
    def __init__(self, model, prompt_length):
        self.model = model
        self.prompt_length = prompt_length
        self.chunk = model.config.lact_chunk_size
        self.positions = sorted({0, prompt_length - 1, *(
            multiple * self.chunk - 1 for multiple in range(1, 4)
            if multiple * self.chunk <= prompt_length
        )})
        self.step = -1
        self.counts = {}
        self.activations = {}
        self.cache_samples = {}
        self.layouts = {}
        self.order = []
        self.steps = []
        self.logits = []
        self.weight_refs = {}
        self.handles = []

    def store(self, target, name, value, selected, positions):
        # 所有CPU拷贝均推迟至完整generation结束后。
        target[name] = selected.detach().clone()
        self.layouts[name] = {
            "step": self.step, "source_shape": list(value.shape), "source_dtype": str(value.dtype),
            "source_stride": list(value.stride()), "sample_shape": list(selected.shape),
            "real_token_positions": positions,
        }
        self.order.append(name)

    def activation(self, name, value, fw_heads=None, all_batch=False):
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                self.activation(f"{name}/{index}", item, fw_heads, all_batch)
        elif isinstance(value, torch.Tensor):
            # 所有被hook模块的tensor均是[B,S,...]，TTT_norm例外为[B*H,S,D]。
            length = value.shape[1]
            local_positions = self.positions if self.step == 0 and length > 1 else [length - 1]
            real_positions = local_positions if length > 1 else [self.prompt_length - 1 + self.step]
            if all_batch:
                selected = value[:, local_positions]
            elif fw_heads is None:
                selected = value[0, local_positions]
            else:
                selected = value[:fw_heads, local_positions].transpose(0, 1)
            self.store(self.activations, name, value, selected, real_positions)
            self.layouts[name]["batch_rows"] = "all" if all_batch else "focus0"

    def module_hook(self, name, fw_heads=None, all_batch=False):
        def hook(module, args, kwargs, output):
            if self.step >= 2:
                return
            call = self.counts.get(name, 0)
            self.counts[name] = call + 1
            prefix = f"step{self.step}/{name}/call{call}"
            self.activation(f"{prefix}/input", args, fw_heads, all_batch)
            self.activation(f"{prefix}/output", output, fw_heads, all_batch)
        return hook

    def before_model(self, module, args, kwargs):
        self.step += 1
        self.counts = {}
        if self.step < 2:
            self.activation(f"step{self.step}/input_ids", kwargs["input_ids"])

    def after_model(self, module, args, kwargs, output):
        if self.step >= 2:
            return
        self.logits.append(output.logits[0, -1].detach().clone())
        progress = []
        for index, (block, layer) in enumerate(zip(self.model.model.layers, output.past_key_values.layers, strict=True)):
            state = layer.ttt
            heads = block.attn.num_fw_heads
            prefix = f"step{self.step}/cache/layer{index}"
            seen, pending = layer.seen, state.pending_length
            expected_seen = self.prompt_length + self.step
            updates = (expected_seen - 1) // self.chunk
            if (seen, state.seen, state.n_updates, pending) != (
                expected_seen, expected_seen, updates, expected_seen - updates * self.chunk,
            ):
                raise ValueError("Observed cache progress differs from the original real-token schedule")
            progress.append({"layer": index, "seen": seen, "ttt_seen": state.seen,
                             "updates": state.n_updates, "pending_length": pending,
                             "kv_real_start": seen - layer.keys.shape[1]})
            for name in ("keys", "values"):
                value = getattr(layer, name)
                start = seen - value.shape[1]
                positions = sorted({start, seen - 1, *(
                    position for position in self.positions if start <= position < seen
                )})
                selected = value[0, [position - start for position in positions]]
                self.store(self.cache_samples, f"{prefix}/{name}", value, selected, positions)
            for name in ("pending_k", "pending_v", "pending_lr0", "pending_lr1", "pending_lr2", "pending_momentum"):
                fragments = getattr(state, name)
                if not fragments:
                    continue
                selected = []
                for position in sorted({0, pending - 1}):
                    remaining = position
                    for fragment in fragments:
                        if remaining < fragment.shape[1]:
                            selected.append(fragment[:heads, remaining])
                            break
                        remaining -= fragment.shape[1]
                key = f"{prefix}/{name}"
                self.store(self.cache_samples, key, fragments[0], torch.stack(selected),
                           [seen - pending + position for position in sorted({0, pending - 1})])
                self.layouts[key]["fragment_shapes"] = [list(value.shape) for value in fragments]
                self.layouts[key]["fragment_strides"] = [list(value.stride()) for value in fragments]
            for name in ("w0", "w1", "w2", "w0_norm", "w1_norm", "w2_norm"):
                value = getattr(state, name)
                row_indices = sorted({0, value.shape[1] // 2, value.shape[1] - 1})
                column_indices = sorted({0, value.shape[2] // 2, value.shape[2] - 1})
                selected = value[:heads, row_indices][:, :, column_indices]
                key = f"{prefix}/{name}"
                self.store(self.cache_samples, key, value, selected, [])
                self.layouts[key].update(matrix_rows=row_indices, matrix_columns=column_indices)
                if name in {"w0", "w1", "w2"}:
                    # 保留只读引用，norm/max计算推迟到生成结束，不保存完整矩阵。
                    self.weight_refs[key] = value[:heads].detach()
        self.steps.append({"generated_index": self.step,
                           "physical_seen": output.past_key_values.get_seq_length(), "cache": progress})

    def install(self):
        self.handles.append(self.model.register_forward_pre_hook(self.before_model, with_kwargs=True))
        self.handles.append(self.model.register_forward_hook(self.after_model, with_kwargs=True))
        self.handles.append(self.model.model.embeddings.register_forward_hook(self.module_hook("embeddings"), with_kwargs=True))
        for index, block in enumerate(self.model.model.layers):
            prefix = f"layer{index}"
            for name in ("attn_norm", "mlp_norm", "mlp"):
                self.handles.append(getattr(block, name).register_forward_hook(
                    self.module_hook(f"{prefix}/{name}"), with_kwargs=True,
                ))
            for name in ("qkv", "rotary", "lr_proj", "ttt_norm", "o_proj"):
                self.handles.append(getattr(block.attn, name).register_forward_hook(
                    self.module_hook(f"{prefix}/{name}", block.attn.num_fw_heads if name == "ttt_norm" else None),
                    with_kwargs=True,
                ))
            optional = []
            if block.attn.attn_qk_norm:
                optional.extend(("q_norm", "k_norm"))
            if block.attn.use_momentum:
                optional.append("momentum_proj")
            if block.attn.learnable_ttt_scale:
                optional.append("ttt_scale_proj")
            for name in optional:
                self.handles.append(getattr(block.attn, name).register_forward_hook(
                    self.module_hook(f"{prefix}/{name}"), with_kwargs=True,
                ))
        self.handles.append(self.model.model.norm.register_forward_hook(self.module_hook("final_norm"), with_kwargs=True))
        self.handles.append(self.model.lm_head.register_forward_hook(self.module_hook("lm_head", all_batch=True), with_kwargs=True))

    def remove(self):
        for handle in self.handles:
            handle.remove()

    def export(self, directory):
        weights = {}
        for name, value in self.weight_refs.items():
            value = value.float()
            weights[name] = {"frobenius_per_head": value.norm(dim=(1, 2)).cpu().tolist(),
                             "max_abs_per_head": value.abs().amax(dim=(1, 2)).cpu().tolist()}
        logits = [value.float().cpu() for value in self.logits]
        torch.save({"activations": {name: value.float().cpu() for name, value in self.activations.items()},
                    "cache_samples": {name: value.float().cpu() for name, value in self.cache_samples.items()}},
                   directory / "trace.pt")
        torch.save(logits, directory / "logits.pt")
        write_json(directory / "trace.json", {"prefill_sample_positions": self.positions,
                   "execution_order": self.order, "fields": self.layouts, "cache_progress": self.steps,
                   "fast_weight_summaries": weights,
                   "scope": "Sparse prefill activations/cache samples; complete first-decode token activations. Only the first observed difference can be localized."})
        return logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ruler-data", type=Path, required=True)
    parser.add_argument("--mode", choices=("old_B1", "old_equal_B3"), required=True)
    parser.add_argument("--reference-dir", type=Path, required=True, help="此前相同mode的diagnose_muon_batch输出目录")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "1"} or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Use one fresh process on physical GPU 0 or 1 per mode")
    for name in ("checkpoint", "ruler_data", "reference_dir", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    previous_plan = json.loads((args.reference_dir / "plan.json").read_text())
    identity = previous_plan["identity"]
    if previous_plan["mode"] != args.mode or not previous_plan["record_logits"]:
        raise ValueError("Reference must use the same mode with record_logits=True")
    if (identity["checkpoint"], identity["ruler_data"], identity["focus_id"], identity["generation_tokens"],
        identity["torch_compile_disable"]) != (
        str(args.checkpoint), str(args.ruler_data), FOCUS_ID, GENERATION_TOKENS,
        os.environ.get("TORCH_COMPILE_DISABLE", "0"),
    ):
        raise ValueError("Reference checkpoint, input, generation budget and compile setting must match")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plan = {"identity": identity, "mode": args.mode, "record_logits": True,
            "source": capture_source(__file__, {"incremental": IncrementalLaCTForCausalLM}),
            "environment": {"cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                            "torch": str(torch.__version__), "cuda": torch.version.cuda},
            "reference_dir": str(args.reference_dir),
            "sequence": "First model call is greedy128; only nn.Module hooks observe the actual first two forwards; all GPU-to-CPU copies and cache reductions run after generation"}
    write_json(args.output_dir / "plan.json", plan)
    started = time.perf_counter()
    torch.manual_seed(0)
    model, config = load_checkpoint(args.checkpoint, "incremental")
    if not config.use_muon or config.ttt_prenorm:
        raise ValueError("This diagnosis requires Muon PostNorm")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    sample = read_sample(args.ruler_data, FOCUS_ID, tokenizer)
    expected_length = next(row["tokens"] for row in previous_plan["selected"] if row["id"] == FOCUS_ID)
    if len(sample["input_ids"]) != expected_length:
        raise ValueError("Focus token count differs from the original reference")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ids, mask, rows = make_inputs(args.mode, [sample], pad_id)
    plan.update(checkpoint_config=config.to_dict(), physical_input_shape=list(ids.shape),
                actual_sample_ids=[FOCUS_ID] * len(rows), actual_input_lengths=list(map(len, rows)))
    write_json(args.output_dir / "plan.json", plan)
    trace = DecodeTrace(model, len(sample["input_ids"]))
    trace.install()
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated = model.generate(
                input_ids=ids, attention_mask=mask, do_sample=False, num_beams=1,
                max_new_tokens=GENERATION_TOKENS, use_cache=True, logits_to_keep=1,
                pad_token_id=pad_id, output_scores=False,
            )
    finally:
        trace.remove()
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - started
    tokens = generated[0, ids.shape[1]:].cpu().tolist()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    end = next((index + 1 for index, token in enumerate(tokens) if token in eos), len(tokens))
    tokens = tokens[:end]
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    logits = trace.export(args.output_dir)
    result = {"mode": args.mode, "focus_id": FOCUS_ID, "record_logits": True,
              "reference_dir": str(args.reference_dir), "token_ids": tokens, "prediction": text,
              "references": sample["references"],
              "score": string_match_all([postprocess_pred(text)], [sample["references"]]),
              "steps": [describe_logits(value, tokenizer) for value in logits],
              "generation_seconds_including_load": generation_seconds,
              "seconds_including_export": time.perf_counter() - started}
    result["comparison"] = compare_reference(result, logits, args.reference_dir)
    result["first_two_logits_match_reference_exactly"] = all(step["exact"] for step in result["comparison"]["steps"])
    result["observer_matches_reference"] = result["first_two_logits_match_reference_exactly"] and result["comparison"]["all_token_ids_equal"]
    write_json(args.output_dir / "result.json", result)
    print(json.dumps({"mode": args.mode, "score": result["score"], "first_token_ids": tokens[:8],
                      "observer_matches_reference": result["observer_matches_reference"],
                      "comparison": result["comparison"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
