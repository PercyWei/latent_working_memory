"""Local CPU paired benchmark: full-vocabulary logits versus target-position projection."""

import argparse
import gc
import json
import platform
from pathlib import Path
import statistics
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from latent_working_memory.v1.backbone import LatentMemoryBackbone, ReadTokens
from reader_reference import dense_read_batch


def measure(mode, workload):
    torch.set_num_threads(1)
    torch.manual_seed(20260914)
    base = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=151936,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=1024,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            attention_dropout=0.0,
        )
    )
    backbone = LatentMemoryBackbone(base, 1, 2, 16, 2, 4, ("q_proj", "v_proj"), 0.0)
    backbone.train()
    target_sizes = [256] * 4 if workload == "uniform" else [256, 128, 64, 32]
    capacities = [n // 2 for n in target_sizes]
    memories = [torch.randn(k, 16, requires_grad=True) for k in capacities]
    tasks = [
        ReadTokens((11, 12), tuple(4 + i % 128 for i in range(n)) + (2,)) for n in target_sizes
    ]
    weights = [0.25, 0.75, 1.0, 1.0]
    read = dense_read_batch if mode == "dense" else lambda b, m, t: b.read_batch(m, t)
    projected = []
    hook = base.get_output_embeddings().register_forward_pre_hook(
        lambda module, args: projected.append(tuple(args[0].shape))
    )

    def clear():
        backbone.zero_grad(set_to_none=True)
        for memory in memories:
            memory.grad = None

    def iteration():
        outputs = read(backbone, memories, tasks)
        loss = sum(w * o.mean_nll for w, o in zip(weights, outputs)) / sum(weights)
        loss.backward()
        return loss, outputs

    for _ in range(2):
        clear()
        loss, outputs = iteration()
        del loss, outputs
    times = []
    for _ in range(5):
        clear()
        start = time.perf_counter()
        loss, outputs = iteration()
        times.append(time.perf_counter() - start)
        del loss, outputs
    clear()
    gc.collect()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True
    ) as profile:
        loss, outputs = iteration()
    with TemporaryDirectory(prefix="reader-profile-") as tmp:
        path = Path(tmp) / "trace.json"
        profile.export_chrome_trace(str(path))
        trace = json.loads(path.read_text())
    memory_events = [
        event["args"]["Total Allocated"]
        for event in trace["traceEvents"]
        if event.get("name") == "[memory]"
    ]
    result = {
        "mode": mode,
        "workload": workload,
        "dtype": "float32",
        "vocab_size": 151936,
        "hidden_size": 64,
        "layers": 2,
        "target_sizes_including_eos": [len(t.target_ids) for t in tasks],
        "capacities": capacities,
        "projected_shape": projected[-1],
        "median_seconds": statistics.median(times),
        "trials_seconds": times,
        "peak_profiled_cpu_tensor_bytes": max(memory_events),
        "loss": float(loss.detach()),
        "nll": [o.token_nll.detach().tolist() for o in outputs],
    }
    hook.remove()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", choices=("dense", "packed"))
    parser.add_argument("--workload", choices=("uniform", "mixed"))
    args = parser.parse_args()
    if args.mode:
        print(json.dumps(measure(args.mode, args.workload)))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "platform": platform.platform(),
        "cpu_threads": 1,
        "scope": "reduced Qwen2 CPU model with full Qwen2.5 vocabulary; allocator traces exclude model tensors allocated before profiling",
        "results": [],
    }
    for workload in ("uniform", "mixed"):
        pair = []
        for mode in ("dense", "packed"):
            result = subprocess.run(
                [sys.executable, __file__, "--mode", mode, "--workload", workload],
                text=True,
                capture_output=True,
                check=True,
            )
            row = json.loads(result.stdout)
            report["results"].append(row)
            pair.append(row)
            (args.output_dir / f"{workload}-{mode}.stderr.log").write_text(result.stderr)
            print(
                json.dumps({k: v for k, v in row.items() if k not in ("nll", "trials_seconds")}),
                flush=True,
            )
        for a, b in zip(pair[0]["nll"], pair[1]["nll"], strict=True):
            torch.testing.assert_close(torch.tensor(a), torch.tensor(b), rtol=1e-5, atol=1e-6)
        (args.output_dir / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
