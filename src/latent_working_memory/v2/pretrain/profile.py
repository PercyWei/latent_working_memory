"""真实基座的固定容量极限长度短测；只评估资源和梯度，不报告训练效果。"""

import argparse
from dataclasses import asdict
from itertools import accumulate
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer, set_seed

from latent_working_memory.v2.memory_codec import CodecConfig, MemoryCodec
from latent_working_memory.v2.pretrain.config import TrainingConfig
from latent_working_memory.v2.pretrain.data import Trajectory
from latent_working_memory.v2.pretrain.engine import ReconstructionEngine, initialize_device
from latent_working_memory.v2.pretrain.objective import ReconstructionTask


def main():
    parser = argparse.ArgumentParser(description="v2 real-model memory and throughput probe")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = initialize_device(args.device)
    if device.type != "cuda":
        parser.error("resource profiling requires CUDA")
    set_seed(42)
    model_config = CodecConfig(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    codec = MemoryCodec(model_config, torch.bfloat16).to(device)
    k = args.capacity
    cases = [
        ("warmup", "ae", (2 * k,)),
        ("warmup", "ae", (8 * k,)),
        ("warmup", "ae_lm", (8 * k,)),
        ("multiround", "ae_lm", (2 * k, 3 * k, 3 * k)),
        ("multiround", "ae_lm", (3 * k, 2 * k, k, k, k)),
    ]
    training_defaults = TrainingConfig()
    reader_limit = (
        training_defaults.micro_batch_decoder_tokens // 2
        - k
        - len(tokenizer.encode(training_defaults.ae_prompt, add_special_tokens=False))
    )
    if 5 * k <= reader_limit <= 7 * k:
        cases.append(("multiround", "ae_lm", (reader_limit - 4 * k, k, k, k, k)))
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Resource bounds must cover both tasks, even when a small random batch
    # happens to contain only LM. Probe each task separately at the ratio endpoints.
    cases = [
        (stage, objective, lengths, ratio)
        for stage, objective, lengths in cases
        for ratio in ((0.0, 1.0) if objective == "ae_lm" else (0.0,))
    ]
    for stage, objective, lengths, lm_ratio in cases:
        codec.set_stage(stage)
        config = TrainingConfig(
            objective=objective,
            lm_ratio=lm_ratio,
            global_batch_size=args.batch_size,
            micro_batch_size=args.micro_batch_size,
        )
        task = ReconstructionTask(codec, tokenizer, config).to(device)
        engine = ReconstructionEngine(task, device)
        engine.initialize()
        rows = [
            Trajectory(
                f"profile-{i}",
                0,
                torch.randint(
                    100, 10000, (sum(lengths) + k,), generator=torch.Generator().manual_seed(i)
                ),
                tuple(accumulate(lengths)),
                k,
            )
            for i in range(args.batch_size)
        ]
        task.validate_data({stage: {"profile": rows}})
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        steps = []
        for step in range(args.iterations):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            metrics = engine.step(rows, step)
            torch.cuda.synchronize(device)
            metrics["seconds"] = time.perf_counter() - start
            assert any(
                p.grad is not None and p.grad.abs().max().item() > 0
                for name, p in codec.backbone.named_parameters()
                if "lora_" in name and ".encoder." in name
            )
            assert all(
                p.grad is None
                for name, p in codec.backbone.named_parameters()
                if "lora_" not in name or ".decoder." in name
            )
            steps.append(metrics)
        result = {
            "stage": stage,
            "objective": objective,
            "lm_ratio": lm_ratio,
            "lengths": lengths,
            "batch_size": args.batch_size,
            "micro_batch_size": args.micro_batch_size,
            "steps": steps,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
            "end_allocated_gib": torch.cuda.memory_allocated(device) / 1024**3,
            "total_gib": torch.cuda.get_device_properties(device).total_memory / 1024**3,
        }
        results.append(result)
        args.output.write_text(
            json.dumps({"model": asdict(model_config), "cases": results}, indent=2) + "\n"
        )
        print(json.dumps(result), flush=True)
        codec.zero_grad(set_to_none=True)
        del engine, task
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
