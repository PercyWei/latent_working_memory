"""Qwen optimization parity and isolated, warmed-up CUDA training benchmarks."""

import argparse
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist

from latent_working_memory.v1 import objectives
from latent_working_memory.v1.backbone import load_backbone
from latent_working_memory.v1.config import load_config
from latent_working_memory.v1.dynamic.config import DynamicConfig
from latent_working_memory.v1.dynamic.evaluation import evaluate_qa
from latent_working_memory.v1.dynamic.training import DynamicTrainer
from latent_working_memory.v1.engine import initialize_device
from latent_working_memory.v1.model import JointMemoryWriter
from latent_working_memory.v1.pretrain.training import PretrainTrainer
from validate_verl_dynamic_gpu import episode
from validate_verl_gpu import make_examples


def native_ce_chunk(hidden, weight, target, bias):
    return torch.nn.functional.cross_entropy(
        torch.nn.functional.linear(hidden, weight, bias).float(), target, reduction="none"
    )


def compare_metrics(expected, actual):
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            compare_metrics(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual)
        for a, b in zip(expected, actual):
            compare_metrics(a, b)
    elif isinstance(expected, float):
        assert abs(expected - actual) <= 0.002 + 0.001 * abs(expected), (expected, actual)
    else:
        assert expected == actual, (expected, actual)


def compare_gradients(reference, actual):
    norms, errors, maximum = [], [], 0.0
    for a, b in zip(reference.parameters, actual.parameters, strict=True):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            delta = (a.grad - b.grad).float()
            error, norm = delta.norm().item(), a.grad.float().norm().item()
            assert error <= 0.03 * norm + 1e-6, (a.shape, error, norm)
            norms.append(norm ** 2)
            errors.append(error ** 2)
            maximum = max(maximum, delta.abs().max().item())
        torch.testing.assert_close(a, b, rtol=1e-4, atol=2e-6)
        for key, value in reference.optimizer.state[a].items():
            observed = actual.optimizer.state[b][key]
            if key == "step":
                assert value.item() == observed.item()
            else:
                delta = (value - observed).float().norm().item()
                assert delta <= 0.04 * value.float().norm().item() + 1e-6
    return {"gradient_relative_l2": (sum(errors) / max(sum(norms), 1e-30)) ** .5,
            "gradient_max_absolute_error": maximum}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", choices=("sample", "mean", "full", "tokens", "updates"), required=True)
    parser.add_argument("--variant", choices=("baseline", "adam", "ce", "both"), required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--native-ce-control", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if args.native_ce_control:
        objectives._linear_loss_chunk = native_ce_chunk
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(args.verify)
    device = initialize_device(torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    config = replace(load_config(args.config), optimizer_fused=False, reader_loss_backend="torch")
    torch.manual_seed(config.model_seed)
    tokenizer, backbone = load_backbone(config, device, torch.bfloat16)
    writer = JointMemoryWriter(config.d_mem, config.num_layers, config.num_heads, config.ffn_dim, config.k_limit).to(device)
    recipe = DynamicConfig(
        capacities=(64,), global_batch_size=2, new_count=1, history_count=1,
        generation_tokens=8, eval_reads_per_kind=2, qa_activation_checkpointing=True,
        bptt_unit="updates" if args.task == "updates" else "tokens",
        bptt_span={"tokens": 192, "updates": 2}.get(args.task, 0),
    )
    pretrain = args.task in {"sample", "mean"}
    episodes = [episode(tokenizer, "short", 5), episode(tokenizer, "long", 8)]
    def trainer(bb, ww, fused):
        return (PretrainTrainer(replace(config, optimizer_fused=fused), bb, ww, device)
                if pretrain else DynamicTrainer(bb, ww, config, replace(recipe, optimizer_fused=fused, reader_loss_backend=bb.reader_loss_backend), device))
    reference = trainer(deepcopy(backbone), deepcopy(writer), False) if args.verify else None
    backbone.reader_loss_backend = "liger" if args.variant in {"ce", "both"} else "torch"
    actual = trainer(backbone, writer, args.variant in {"adam", "both"})
    def update(engine, step):
        if not pretrain:
            return engine.step(episodes, tokenizer, [42, 43], 64)
        examples = []
        for e in make_examples(step % 2, args.task):
            task = e.ae if e.ae is not None else e.lm
            task = replace(task, target_ids=(*task.target_ids, tokenizer.eos_token_id))
            examples.append(replace(e, ae=task if e.ae else None, lm=task if e.lm else None))
        return engine.step(examples)
    output = args.output_dir / f"rank-{dist.get_rank()}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "settings.json").write_text(json.dumps({
        "task": args.task, "variant": args.variant, "verify": args.verify,
        "warmup": args.warmup, "steps": args.steps,
        "deterministic": args.verify, "native_ce_control": args.native_ce_control,
        "input_model_config": config.to_dict(),
        "reader_loss_backend": backbone.reader_loss_backend,
        "optimizer_fused": bool(actual.optimizer.param_groups[0].get("fused", False)),
    }, indent=2) + "\n")
    records = []
    if args.verify:
        for step in range(2):
            expected, observed = update(reference, step), update(actual, step)
            (output / f"metrics-{step}.json").write_text(json.dumps({"reference": expected, "actual": observed}, indent=2) + "\n")
            compare_metrics(expected, observed)
            records.append({"step": step + 1, **compare_gradients(reference, actual)})
            (output / "steps.json").write_text(json.dumps(records, indent=2) + "\n")
        expected = evaluate_qa(reference.backbone, reference.writer, tokenizer, config, recipe, episodes, device, 64)
        observed = evaluate_qa(backbone, writer, tokenizer, config, recipe, episodes, device, 64)
        compare_metrics(expected, observed)
        result = {"passed": True, "steps": records, "dev_rows": len(observed[1])}
    else:
        for step in range(args.warmup):
            update(actual, step)
        for step in range(args.steps):
            dist.barrier()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            metrics = update(actual, step)
            torch.cuda.synchronize(device)
            elapsed = torch.tensor(time.perf_counter() - start, device=device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            records.append({"step": step + 1, "seconds": elapsed.item(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "input_tokens": metrics["input_tokens"], "target_tokens": metrics["target_tokens"]})
            (output / "steps.json").write_text(json.dumps(records, indent=2) + "\n")
        seconds = [r["seconds"] for r in records]
        result = {"passed": True, "mean_seconds": statistics.mean(seconds),
            "median_seconds": statistics.median(seconds), "stdev_seconds": statistics.stdev(seconds),
            "target_tokens_per_second": sum(r["target_tokens"] for r in records) / sum(seconds),
            "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in records)}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if dist.get_rank() == 0:
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
