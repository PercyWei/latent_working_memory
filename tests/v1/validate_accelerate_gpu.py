"""Paired CUDA validation against adb3b57; run with torchrun on two allowed GPUs."""

import argparse
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import time

import torch
from accelerate.state import PartialState

from latent_working_memory.v1.backbone import ReadTokens, load_backbone
from latent_working_memory.v1.config import load_config
from latent_working_memory.v1.data import (
    Episode,
    EpisodeIndex,
    Read,
    Reference,
    Source,
    write_episodes,
)
from latent_working_memory.v1.model import JointMemoryWriter
from latent_working_memory.v1.pretrain.evaluation import evaluate_pretraining
from latent_working_memory.v1.pretrain.sampling import PretrainExample
from latent_working_memory.v1.pretrain.training import PretrainTrainer
from pretrain_reference import LegacyPretrainTrainer
from test_evaluation_batching import assert_results_equal


def make_examples(step, mode):
    examples = []
    for i, n in enumerate((64, 128, 256, 512, 768, 1024, 1536, 1960)):
        ids = tuple(100 + (j + i + step) % 31 for j in range(n))
        task = "ae" if (i + step) % 2 == 0 else "continuation"
        target = ids if task == "ae" else ids[: n // 2]
        read = Read(str(i), task, n, "prompt", (Reference("synthetic", ()),))
        source = Source(str(i), str(i), 0, n, {"boundary_variant": "semantic"})
        episode = Episode(str(i), ids, (n,), (source,), (read,))
        # Set actual model EOS in main; prompts are fixed valid model token IDs.
        tokens = ReadTokens((100, 101), target)
        capacities = [(n + r - 1) // r for r in ((2, 4, 8) if mode == "mean" else (4,))]
        for capacity in capacities:
            examples.append(
                PretrainExample(
                    episode,
                    tokens if task == "ae" else None,
                    tokens if task == "continuation" else None,
                    capacity,
                    1 / len(capacities),
                )
            )
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    config = load_config(args.config)
    torch.manual_seed(config.model_seed)
    tokenizer, reference_backbone = load_backbone(config, device, torch.bfloat16)
    reference_writer = JointMemoryWriter(
        config.d_mem, config.num_layers, config.num_heads, config.ffn_dim, config.k_limit
    ).to(device)
    actual_backbone, actual_writer = deepcopy(reference_backbone), deepcopy(reference_writer)
    actual = PretrainTrainer(config, actual_backbone, actual_writer, device)
    legacy = LegacyPretrainTrainer(config, reference_backbone, reference_writer, device)
    accelerator = actual.accelerator
    rank = accelerator.process_index
    output = args.output_dir / f"rank-{rank}"
    output.mkdir(parents=True, exist_ok=False)
    observations = []
    for step in range(args.steps):
        mode = "mean" if step % 4 == 3 else "sample"
        examples = []
        for e in make_examples(step, mode):
            task = e.ae if e.ae is not None else e.lm
            task = replace(task, target_ids=(*task.target_ids, tokenizer.eos_token_id))
            examples.append(replace(e, ae=task if e.ae else None, lm=task if e.lm else None))
        results = {}
        # Alternate execution order to reduce systematic cache/order effects.
        order = [("legacy", legacy), ("accelerate", actual)]
        if step % 2:
            order.reverse()
        for name, trainer in order:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            metrics = trainer.step(examples)
            torch.cuda.synchronize(device)
            results[name] = {
                "metrics": metrics,
                "seconds": time.perf_counter() - start,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
        assert_results_equal(results["legacy"]["metrics"], results["accelerate"]["metrics"])
        maximum_gradient_error = maximum_parameter_error = 0.0
        for reference, observed in zip(legacy.parameters, actual.parameters, strict=True):
            assert (reference.grad is None) == (observed.grad is None)
            if reference.grad is not None:
                torch.testing.assert_close(observed.grad, reference.grad, rtol=1e-3, atol=1e-6)
                maximum_gradient_error = max(
                    maximum_gradient_error, (observed.grad - reference.grad).abs().max().item()
                )
            torch.testing.assert_close(observed, reference, rtol=1e-4, atol=1e-6)
            maximum_parameter_error = max(
                maximum_parameter_error, (observed - reference).abs().max().item()
            )
        torch.testing.assert_close(
            actual.optimizer.state_dict(), legacy.optimizer.state_dict(), rtol=1e-3, atol=1e-6
        )
        row = {
            "step": step + 1,
            "mode": mode,
            "max_gradient_error": maximum_gradient_error,
            "max_parameter_error": maximum_parameter_error,
            "results": results,
        }
        observations.append(row)
        (output / "steps.json").write_text(json.dumps(observations, indent=2) + "\n")
        if rank == 0:
            print(json.dumps({k: v for k, v in row.items() if k != "results"}), flush=True)
    # Check the complete synchronous dev protocol against both resulting models.
    episodes = []
    for i in range(6):
        text = f"Document {i}. " + "The river flows past the quiet town. " * (i + 1)
        ids = tuple(tokenizer.encode(text, add_special_tokens=False))
        task = "ae" if i % 2 == 0 else "continuation"
        target = (
            ids
            if task == "ae"
            else tuple(tokenizer.encode("A short continuation.", add_special_tokens=False))
        )
        source = Source(
            str(i),
            str(i),
            0,
            len(ids),
            {"boundary_method": "random_token", "boundary_variant": "random"},
        )
        read = Read(
            str(i),
            task,
            len(ids),
            config.ae_prompt if task == "ae" else config.lm_prompt,
            (Reference(tokenizer.decode(target), ()),),
        )
        episodes.append(Episode(str(i), ids, (len(ids),), (source,), (read,)))
    write_episodes(episodes, output / "panel.jsonl")
    index = EpisodeIndex(output / "panel.jsonl")
    eval_config = replace(
        config, eval_examples=6, eval_generation_examples=2, eval_generation_every=1
    )
    reports = []
    for name, backbone, writer in [
        ("legacy", reference_backbone, reference_writer),
        ("accelerate", actual_backbone, actual_writer),
    ]:
        with accelerator.autocast():
            report = evaluate_pretraining(
                eval_config, tokenizer, backbone, writer, index, output / name, args.steps, 0
            )
        reports.append(report)
    assert_results_equal(*reports)
    left = [
        json.loads(line)
        for line in (output / "legacy" / f"dev-step-{args.steps:06d}.jsonl")
        .read_text()
        .splitlines()
    ]
    right = [
        json.loads(line)
        for line in (output / "accelerate" / f"dev-step-{args.steps:06d}.jsonl")
        .read_text()
        .splitlines()
    ]
    assert_results_equal(left, right)
    (output / "result.json").write_text(
        json.dumps(
            {
                "passed": True,
                "steps": args.steps,
                "world_size": accelerator.num_processes,
                "mixed_precision": accelerator.mixed_precision,
                "dev_records": len(left),
                "note": "Two frozen model copies coexist for paired validation; peak memory is not production footprint.",
            },
            indent=2,
        )
        + "\n"
    )
    PartialState().destroy_process_group()


if __name__ == "__main__":
    main()
