"""Real-model CUDA parity for the verl full-BPTT/TBPTT engine."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from latent_working_memory.v1.backbone import load_backbone
from latent_working_memory.v1.config import load_config
from latent_working_memory.v1.data import Episode, Read, Reference, Source
from latent_working_memory.v1.dynamic.config import DynamicConfig
from latent_working_memory.v1.dynamic.evaluation import evaluate_qa
from latent_working_memory.v1.dynamic.squad import QA_PROMPT
from latent_working_memory.v1.dynamic.training import DynamicTrainer
from latent_working_memory.v1.engine import initialize_device
from latent_working_memory.v1.model import JointMemoryWriter
from dynamic_reference import ReferenceDynamicTrainer
from test_evaluation_batching import assert_results_equal


def episode(tokenizer, name, paragraphs):
    ids, ends, sources, reads = [], [], [], []
    for i in range(paragraphs):
        text = (
            f"Document {name}, paragraph {i}. The answer is River. "
            + "The quiet town stands beside the river. " * 8
        )
        start = len(ids)
        ids.extend(tokenizer.encode(text, add_special_tokens=False))
        ends.append(len(ids))
        sources.append(
            Source(f"{name}:{i}", name, start, len(ids), {"context": text, "dedup_cluster": name})
        )
        reads.append(
            Read(
                f"{name}:q{i}",
                "qa",
                len(ids),
                QA_PROMPT.format(question="Which word is the answer?"),
                (Reference("River", ((start, len(ids)),)),),
            )
        )
    return Episode(name, tuple(ids), tuple(ends), tuple(sources), tuple(reads))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bptt-unit", choices=("tokens", "updates"), required=True)
    parser.add_argument("--bptt-span", type=int, required=True)
    parser.add_argument("--checkpointing", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    device = initialize_device(torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    config = load_config(args.config)
    torch.manual_seed(config.model_seed)
    tokenizer, baseline_backbone = load_backbone(config, device, torch.bfloat16)
    baseline_writer = JointMemoryWriter(
        config.d_mem, config.num_layers, config.num_heads, config.ffn_dim, config.k_limit
    ).to(device)
    backbone, writer = deepcopy(baseline_backbone), deepcopy(baseline_writer)
    recipe = DynamicConfig(
        capacities=(64,),
        global_batch_size=2,
        new_count=1,
        history_count=1,
        generation_tokens=8,
        eval_reads_per_kind=2,
        qa_activation_checkpointing=args.checkpointing,
        bptt_unit=args.bptt_unit,
        bptt_span=args.bptt_span,
    )
    reference = ReferenceDynamicTrainer(baseline_backbone, baseline_writer, config, recipe, device)
    actual = DynamicTrainer(backbone, writer, config, recipe, device)
    output = args.output_dir / f"rank-{dist.get_rank()}"
    output.mkdir(parents=True, exist_ok=False)
    episodes = [episode(tokenizer, "short", 5), episode(tokenizer, "long", 8)]
    optimizer_steps = []
    hook = actual.optimizer.register_step_post_hook(lambda *unused: optimizer_steps.append(1))
    records = []
    for step in range(2):
        expected = reference.step(episodes, tokenizer, [42, 43], 64)
        observed = actual.step(episodes, tokenizer, [42, 43], 64)
        assert_results_equal(expected, observed)
        grad_error = parameter_error = 0.0
        for x, y in zip(reference.parameters, actual.parameters, strict=True):
            assert (x.grad is None) == (y.grad is None)
            if x.grad is not None:
                torch.testing.assert_close(x.grad, y.grad, rtol=1e-4, atol=1e-6)
                grad_error = max(grad_error, (x.grad - y.grad).abs().max().item())
            torch.testing.assert_close(x, y, rtol=1e-4, atol=1e-6)
            parameter_error = max(parameter_error, (x - y).abs().max().item())
        torch.testing.assert_close(
            reference.optimizer.state_dict(), actual.optimizer.state_dict(), rtol=1e-4, atol=1e-6
        )
        assert len(optimizer_steps) == step + 1
        row = {
            "step": step + 1,
            "max_gradient_error": grad_error,
            "max_parameter_error": parameter_error,
            "metrics": observed,
        }
        records.append(row)
        (output / "steps.json").write_text(json.dumps(records, indent=2) + "\n")
        if dist.get_rank() == 0:
            print(json.dumps({k: v for k, v in row.items() if k != "metrics"}), flush=True)
    hook.remove()
    expected_metrics, expected_rows = evaluate_qa(
        baseline_backbone, baseline_writer, tokenizer, config, recipe, episodes, device, 64
    )
    metrics, rows = evaluate_qa(backbone, writer, tokenizer, config, recipe, episodes, device, 64)
    assert_results_equal(expected_metrics, metrics)
    assert_results_equal(expected_rows, rows)
    (output / "result.json").write_text(
        json.dumps(
            {
                "passed": True,
                "bptt_unit": args.bptt_unit,
                "bptt_span": args.bptt_span,
                "checkpointing": args.checkpointing,
                "optimizer_steps": len(optimizer_steps),
                "dev_rows": len(rows),
            },
            indent=2,
        )
        + "\n"
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
