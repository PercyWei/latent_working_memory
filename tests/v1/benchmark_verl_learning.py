"""Real-data learning and throughput comparison of replicated and native FSDP2 engines."""

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from importlib.metadata import version
from itertools import islice
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from tensordict import TensorDict
from verl.utils.tensordict_utils import assign_non_tensor, get_non_tensor_data
from verl.workers.config import FSDPEngineConfig, HFModelConfig

from latent_working_memory.data_preparation.pretrain.text_samples import TextSample
from latent_working_memory.v1.backbone import load_backbone
from latent_working_memory.v1.checkpoint import capture_rng_state, save_model_checkpoint
from latent_working_memory.v1.config import load_config
from latent_working_memory.v1.engine import initialize_device
from latent_working_memory.v1.model import GrowthValueNetwork, JointMemoryWriter
from latent_working_memory.v1.pretrain.fsdp_engine import (
    PretrainFSDPEngine,
    pretrain_batch,
    pretrain_loss,
)
from latent_working_memory.v1.pretrain.sampling import (
    PretrainExample,
    capacity_weights,
    read_tokens,
)
from latent_working_memory.v1.pretrain.training import (
    PretrainModel,
    PretrainTrainer,
    learning_rate_at,
)
from latent_working_memory.v1.training import trainable_model_state


BOUNDS = (128, 512, 1024, 2048)


def panel(data_root, split, per_cell, tokenizer, config, seed):
    """Fixed validation panel; shared text is tokenized only in memory."""
    selected = []
    counts = Counter()
    rng = random.Random(seed)
    for variant in ("semantic", "random"):
        with (data_root / variant / f"{split}.jsonl").open() as handle:
            while lines := list(islice(handle, 64)):
                rows = [TextSample(**json.loads(line)) for line in lines]
                ids_batch = tokenizer([row.text for row in rows], add_special_tokens=False)[
                    "input_ids"
                ]
                for row, ids in zip(rows, ids_batch, strict=True):
                    if not 64 <= len(ids) <= 2048:
                        continue
                    bound = next(b for b in BOUNDS if len(ids) <= b)
                    cell = variant, row.task, bound
                    if counts[cell] == per_cell:
                        continue
                    episode = row.to_episode_tokens(tuple(ids), config, variant)
                    ae, lm = read_tokens(episode, tokenizer)
                    weights = capacity_weights(config, len(ids), ae, lm, epoch=1)
                    if not weights:
                        continue
                    capacity = rng.choices(list(weights), weights=list(weights.values()))[0]
                    selected.append(PretrainExample(episode, ae, lm, capacity))
                    counts[cell] += 1
                if (
                    sum(
                        counts[v, t, b]
                        for v in (variant,)
                        for t in ("ae", "continuation")
                        for b in BOUNDS
                    )
                    == per_cell * 8
                ):
                    break
    expected = {
        (v, t, b): per_cell
        for v in ("semantic", "random")
        for t in ("ae", "continuation")
        for b in BOUNDS
    }
    if dict(counts) != expected:
        raise ValueError(f"insufficient eligible samples for the validation panel: {counts}")
    rng.shuffle(selected)
    return selected


def panel_description(samples):
    return [
        {
            "sample_id": e.episode.episode_id,
            "document_id": e.episode.sources[0].document_id,
            "source": e.episode.sources[0].provenance["boundary_variant"],
            "task": "ae" if e.ae else "continuation",
            "input_tokens": e.input_length,
            "target_tokens": len((e.ae or e.lm).target_ids),
            "capacity": e.capacity,
        }
        for e in samples
    ]


def native_loss(output, data, group):
    loss, _ = pretrain_loss(output, data, group)
    examples = get_non_tensor_data(data, "examples", None)
    selected = [examples[i] for i in data["sample_index"].tolist()]
    rows = [
        {
            "episode_id": e.episode.episode_id,
            "boundary_variant": e.episode.sources[0].provenance["boundary_variant"],
            "input_tokens": e.input_length,
            "ae_nll": a.mean_nll.detach().item() if a is not None else None,
            "lm_nll": b.mean_nll.detach().item() if b is not None else None,
        }
        for e, a, b in zip(selected, output.ae, output.lm, strict=True)
    ]
    return loss, {"records": rows}


def execute(engine, backend, samples, batch_size, training):
    if backend == "ddp":
        data = TensorDict({}, batch_size=[])
        assign_non_tensor(data, examples=tuple(samples))
        with engine.train_mode() if training else engine.eval_mode():
            output = engine.train_batch(data, None) if training else engine.infer_batch(data)
        metrics = output["metrics"]
        return metrics["loss"], metrics.get("grad_norm"), metrics["samples"]
    data = pretrain_batch(samples, batch_size, dist.get_rank(), dist.get_world_size())
    with engine.train_mode() if training else engine.eval_mode():
        output = (
            engine.train_batch(data, native_loss)
            if training
            else engine.infer_batch(data, native_loss)
        )
    loss = torch.tensor(sum(output["loss"]), device="cuda", dtype=torch.float64)
    dist.all_reduce(loss)
    rows = [row for batch in output["metrics"]["records"] for row in batch]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, rows)
    return loss.item(), output["metrics"].get("grad_norm"), [r for rows in gathered for r in rows]


def grouped_nll(rows):
    groups = defaultdict(list)
    for row in rows:
        task = "ae" if row["ae_nll"] is not None else "continuation"
        value = row["ae_nll"] if task == "ae" else row["lm_nll"]
        source = row["boundary_variant"]
        bound = next(b for b in BOUNDS if row["input_tokens"] <= b)
        for key in ("all", task, source, f"{source}/{task}/{bound}"):
            groups[key].append(value)
    return {
        key: {"nll": statistics.mean(values), "samples": len(values)}
        for key, values in sorted(groups.items())
    }


def timed_call(fn):
    dist.barrier()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    counters = torch.tensor(
        [
            time.perf_counter() - start,
            torch.cuda.max_memory_allocated(),
            torch.cuda.max_memory_reserved(),
        ],
        device="cuda",
        dtype=torch.float64,
    )
    dist.all_reduce(counters, op=dist.ReduceOp.MAX)
    seconds, allocated, reserved = counters.tolist()
    return result, {
        "seconds": seconds,
        "peak_allocated_bytes": int(allocated),
        "peak_reserved_bytes": int(reserved),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("ddp", "fsdp2"), required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=50)
    args = parser.parse_args()
    device = initialize_device(torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2 or args.steps <= 0 or args.steps % 2 or args.eval_every <= 0:
        raise ValueError("this paired protocol requires two GPUs and a positive even step budget")
    torch.set_num_threads(1)
    config = replace(
        load_config(args.config),
        model_seed=args.seed,
        batch_size=2,
        gradient_accumulation_steps=2,
        warmup_steps=min(20, args.steps // 10),
        optimizer_fused=False,
        reader_loss_backend="torch",
    )
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    torch.manual_seed(args.seed)
    tokenizer, backbone = load_backbone(config, device, torch.bfloat16)
    writer = JointMemoryWriter(
        config.d_mem, config.num_layers, config.num_heads, config.ffn_dim, config.k_limit
    ).to(device)
    value = GrowthValueNetwork(config.d_mem).to(device).requires_grad_(False)
    train = panel(args.data_root, "train", args.steps // 2, tokenizer, config, 20260916)
    dev = panel(args.data_root, "dev", 8, tokenizer, config, 20260917)
    for key in ("document_id", "dedup_cluster"):

        def identifiers(samples):
            return {
                getattr(e.episode.sources[0], key)
                if key == "document_id"
                else e.episode.sources[0].provenance[key]
                for e in samples
            }

        if identifiers(train) & identifiers(dev):
            raise ValueError(f"train/dev overlap: {key}")
    plan = {"train": panel_description(train), "dev": panel_description(dev)}
    if args.reference_dir:
        assert plan == json.loads((args.reference_dir / "panel.json").read_text())
    if args.backend == "ddp":
        trainer = PretrainTrainer(config, backbone, writer, device)
        engine = trainer.engine
        initial = {
            name: p.detach().cpu().clone()
            for name, p in engine.model.named_parameters()
            if p.requires_grad
        }
    else:
        hf_config = HFModelConfig(
            path=config.model_name_or_path,
            load_tokenizer=False,
            use_remove_padding=False,
            enable_gradient_checkpointing=False,
            override_config={"attn_implementation": "sdpa"},
        )
        engine_config = FSDPEngineConfig(
            strategy="fsdp2",
            use_dynamic_bsz=False,
            use_remove_padding=False,
            use_torch_compile=False,
            reshard_after_forward=True,
            mixed_precision={"param_dtype": "bf16", "reduce_dtype": "fp32"},
            wrap_policy={
                "transformer_layer_cls_to_wrap": backbone.language_model.get_base_model()._no_split_modules
            },
        )
        engine = PretrainFSDPEngine(
            PretrainModel(config, backbone, writer), hf_config, engine_config
        )
        engine.initialize()
        initial = get_model_state_dict(
            engine.module,
            options=StateDictOptions(
                full_state_dict=True, cpu_offload=True, ignore_frozen_params=True
            ),
        )
    if rank == 0:
        if args.reference_dir:
            torch.testing.assert_close(
                initial,
                torch.load(
                    args.reference_dir / "initial-state.pt", map_location="cpu", weights_only=True
                ),
                rtol=0,
                atol=0,
            )
        torch.save(initial, args.output_dir / "initial-state.pt")
        (args.output_dir / "panel.json").write_text(json.dumps(plan, indent=2) + "\n")
        (args.output_dir / "settings.json").write_text(
            json.dumps(
                {
                    "backend": args.backend,
                    "model": config.to_dict(),
                    "steps": args.steps,
                    "global_batch": 8,
                    "dev_samples": 128,
                    "eval_every": args.eval_every,
                    "data_root": str(args.data_root),
                    "precision": "bf16",
                    "software": {n: version(n) for n in ("torch", "transformers", "verl")},
                },
                indent=2,
            )
            + "\n"
        )
    del initial
    torch.cuda.empty_cache()
    updates, evaluations = [], []

    def evaluate(step):
        (loss, _, rows), timing = timed_call(
            lambda: execute(engine, args.backend, dev, config.batch_size, False)
        )
        report = {"global_step": step, "loss": loss, "groups": grouped_nll(rows), **timing}
        evaluations.append(report)
        if rank == 0:
            (args.output_dir / f"dev-{step:04d}.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(json.dumps({"event": "dev", **report}), flush=True)

    evaluate(0)
    for step in range(args.steps):
        batch = train[step * 8 : (step + 1) * 8]
        lr = learning_rate_at(config, step, args.steps)
        for group in engine.optimizer.param_groups:
            group["lr"] = lr
        (loss, norm, rows), timing = timed_call(
            lambda: execute(engine, args.backend, batch, config.batch_size, True)
        )
        if not math.isfinite(loss) or not math.isfinite(norm):
            raise FloatingPointError("non-finite loss or gradient norm")
        tokens = sum(len((e.ae or e.lm).target_ids) for e in batch)
        row = {
            "global_step": step + 1,
            "loss": loss,
            "gradient_norm": norm,
            "lr": lr,
            "input_tokens": sum(e.input_length for e in batch),
            "target_tokens": tokens,
            "groups": grouped_nll(rows),
            **timing,
        }
        updates.append(row)
        if rank == 0:
            with (args.output_dir / "train.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            if (step + 1) % 10 == 0:
                print(json.dumps({k: v for k, v in row.items() if k != "groups"}), flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            evaluate(step + 1)
    if args.backend == "fsdp2":
        state = engine.canonical_state(value)
    else:
        state = (trainable_model_state(backbone, writer, value), engine.optimizer.state_dict())
    if rank == 0:
        save_model_checkpoint(
            args.output_dir / "final.pt",
            "pretrain",
            config,
            *state,
            {"next_step": args.steps, "validation_panel": "panel.json"},
            capture_rng_state(),
        )
        measured = updates[min(10, len(updates) // 2) :]
        seconds = sum(r["seconds"] for r in measured)
        result = {
            "completed_steps": len(updates),
            "finite_training": True,
            "mean_step_seconds": seconds / len(measured),
            "samples_per_second": len(measured) * 8 / seconds,
            "target_tokens_per_second": sum(r["target_tokens"] for r in measured) / seconds,
            "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in updates),
            "peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in updates),
            "evaluations": evaluations,
        }
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"event": "complete", **result}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
