"""Paired CUDA validation of the native verl FSDP2 pretraining adapter."""

import argparse
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.utils._pytree import tree_map_only
from tensordict import TensorDict
from transformers import Qwen2Config, Qwen2ForCausalLM
from verl.workers.config import HFModelConfig, FSDPEngineConfig
from verl.workers.engine import BaseEngine
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
from verl.utils.tensordict_utils import assign_non_tensor

from latent_working_memory.v1.backbone import LatentMemoryBackbone, ReadTokens, load_backbone
from latent_working_memory.v1.checkpoint import (
    capture_rng_state,
    restore_rng_state,
    load_model_checkpoint,
    save_model_checkpoint,
)
from latent_working_memory.v1.config import ExperimentConfig, load_config
from latent_working_memory.v1.data import Episode, Read, Reference, Source
from latent_working_memory.v1.engine import initialize_device
from latent_working_memory.v1.model import GrowthValueNetwork, JointMemoryWriter
from latent_working_memory.v1.pretrain.fsdp_engine import (
    PretrainFSDPEngine,
    pretrain_batch,
    pretrain_loss,
)
from latent_working_memory.v1.pretrain.sampling import PretrainExample
from latent_working_memory.v1.pretrain.training import PretrainTrainer, PretrainModel
from latent_working_memory.v1.training import trainable_model_state


def examples(mode, eos_id):
    result = []
    for i, length in enumerate((16, 24, 32, 48)):
        ids = tuple(4 + j % 20 for j in range(length))
        task = "ae" if i % 2 == 0 else "continuation"
        read = Read(str(i), task, length, "prompt", (Reference("text", ()),))
        source = Source(str(i), str(i), 0, length, {"boundary_variant": "semantic"})
        episode = Episode(str(i), ids, (length,), (source,), (read,))
        tokens = ReadTokens((3,), (*(ids if task == "ae" else ids[: length // 2]), eos_id))
        ratios = (2, 4, 8) if mode == "mean" else (4,)
        for ratio in ratios:
            result.append(
                PretrainExample(
                    episode,
                    tokens if task == "ae" else None,
                    tokens if task != "ae" else None,
                    (length + ratio - 1) // ratio,
                    1 / len(ratios),
                )
            )
    return result


def tensors(engine):
    return {
        name: {
            "parameter": parameter.full_tensor().detach().cpu(),
            "gradient": None
            if parameter.grad is None
            else parameter.grad.full_tensor().detach().cpu(),
        }
        for name, parameter in zip(engine.parameter_names, engine.parameters, strict=True)
    }


def native_step(engine, batch, precision):
    data = pretrain_batch(
        batch, engine.model.experiment_config.batch_size, dist.get_rank(), dist.get_world_size()
    )
    torch.cuda.synchronize()
    start = time.perf_counter()
    with engine.train_mode():
        output = engine.train_batch(data, pretrain_loss)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        snapshot = tensors(engine)
    loss = torch.tensor(sum(output["loss"]), device="cuda", dtype=torch.float64)
    dist.all_reduce(loss)
    return {
        "loss": loss.item(),
        "gradient_norm": output["metrics"]["grad_norm"],
        "seconds": elapsed,
    }, snapshot


def compare(reference, snapshot):
    differences = []
    for name, parameter in reference.engine.model.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None
            continue
        actual = snapshot[name]
        assert (actual["gradient"] is None) == (parameter.grad is None)
        row = {
            "name": name,
            "parameter_max_abs": (actual["parameter"] - parameter.detach().cpu())
            .abs()
            .max()
            .item(),
        }
        if parameter.grad is not None:
            reference_grad = parameter.grad.detach().cpu().float()
            difference = actual["gradient"].float() - reference_grad
            row["gradient_max_abs"] = difference.abs().max().item()
            row["gradient_difference_l2"] = difference.norm().item()
            row["reference_gradient_l2"] = reference_grad.norm().item()
        differences.append(row)
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--reshard-after-forward", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--align-second-step",
        action="store_true",
        help="Diagnostic: start the second native step from the exact DDP model/optimizer state",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = initialize_device(torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    torch.manual_seed(42)
    rank = dist.get_rank()
    output = args.output_dir / f"rank-{rank}"
    output.mkdir(parents=True, exist_ok=False)
    dtype = torch.float32 if args.precision == "fp32" else torch.bfloat16
    if args.config:
        config = replace(
            load_config(args.config),
            batch_size=1,
            gradient_accumulation_steps=2,
            optimizer_fused=False,
            reader_loss_backend="torch",
        )
        _, backbone = load_backbone(config, device, dtype)
    else:
        config = replace(
            ExperimentConfig(),
            d_mem=8,
            num_layers=1,
            num_heads=2,
            ffn_dim=16,
            k_limit=32,
            batch_size=1,
            gradient_accumulation_steps=2,
            reader_lora_rank=2,
            reader_lora_alpha=4,
            gradient_checkpointing=False,
        )
        model = Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=256,
                attention_dropout=0.0,
                bos_token_id=1,
                eos_token_id=2,
                tie_word_embeddings=False,
                architectures=["Qwen2ForCausalLM"],
            )
        ).to(device=device, dtype=dtype)
        backbone = LatentMemoryBackbone(model, 1, 2, 8, 2, 4, ("q_proj", "v_proj"), 0.0).to(device)
    writer = JointMemoryWriter(
        config.d_mem, config.num_layers, config.num_heads, config.ffn_dim, config.k_limit
    ).to(device)
    value = GrowthValueNetwork(config.d_mem).to(device)
    initial = PretrainModel(config, deepcopy(backbone), deepcopy(writer))
    reference = PretrainTrainer(config, backbone, writer, device)
    if args.precision == "fp32":
        reference.engine.autocast = nullcontext
    hf_path = args.output_dir / "hf-config"
    if rank == 0:
        initial.config.save_pretrained(hf_path)
    dist.barrier()
    hf_config = HFModelConfig(
        path=str(hf_path.resolve()),
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
        reshard_after_forward=args.reshard_after_forward,
        mixed_precision={"param_dtype": args.precision, "reduce_dtype": "fp32"},
        wrap_policy={
            "transformer_layer_cls_to_wrap": initial.backbone.language_model.get_base_model()._no_split_modules
        },
    )
    engine = PretrainFSDPEngine(deepcopy(initial), hf_config, engine_config)
    engine.initialize()
    assert type(engine).train_batch is BaseEngine.train_batch
    assert type(engine).forward_backward_batch is FSDPEngine.forward_backward_batch
    assert type(engine).initialize is FSDPEngine.initialize
    errors = []

    def check(label, actual, expected, **tolerances):
        # Keep the original thresholds. Continue numeric comparisons so restore
        # and inference are exercised even when strict parity fails.
        try:
            torch.testing.assert_close(actual, expected, **tolerances)
        except AssertionError as error:
            errors.append({"check": label, "message": str(error)})

    rows = []
    for step, mode in enumerate(("sample", "mean", "sample")):
        batch = examples(mode, backbone.eos_token_id)
        if step == 1 and args.align_second_step:
            state = tree_map_only(
                torch.Tensor,
                lambda t: t.detach().cpu().clone(),
                trainable_model_state(backbone, writer, value),
            )
            optimizer = tree_map_only(
                torch.Tensor, lambda t: t.detach().cpu().clone(), reference.optimizer.state_dict()
            )
            engine.load_canonical_state(state, optimizer)
        expected = reference.step(batch)
        actual, snapshot = native_step(engine, batch, args.precision)
        row = {
            "step": step + 1,
            "mode": mode,
            "reference": {k: expected[k] for k in ("loss", "gradient_norm")},
            "native": actual,
        }
        (output / f"step-{step + 1}.json").write_text(json.dumps(row, indent=2) + "\n")
        row["differences"] = compare(reference, snapshot)
        for field in ("loss", "gradient_norm"):
            check(
                f"step-{step + 1}/{field}",
                torch.tensor(actual[field]),
                torch.tensor(expected[field]),
                rtol=1e-3,
                atol=1e-6,
            )
        (output / f"step-{step + 1}.json").write_text(json.dumps(row, indent=2) + "\n")
        for name, parameter in reference.engine.model.named_parameters():
            if not parameter.requires_grad:
                continue
            check(
                f"step-{step + 1}/{name}/parameter",
                snapshot[name]["parameter"],
                parameter.detach().cpu(),
                rtol=1e-4,
                atol=1e-6,
            )
            if parameter.grad is not None:
                check(
                    f"step-{step + 1}/{name}/gradient",
                    snapshot[name]["gradient"],
                    parameter.grad.cpu(),
                    rtol=1e-3,
                    atol=1e-6,
                )
        state = engine.canonical_state(value)
        if rank == 0:
            # Framework schedulers add initial_lr; compare optimizer state and
            # effective hyperparameters, not that scheduler bookkeeping key.
            ref_optimizer = tree_map_only(
                torch.Tensor, lambda t: t.cpu(), reference.optimizer.state_dict()
            )
            check(
                f"step-{step + 1}/optimizer",
                state[1]["state"],
                ref_optimizer["state"],
                rtol=1e-3,
                atol=1e-6,
            )
            for left, right in zip(
                state[1]["param_groups"], ref_optimizer["param_groups"], strict=True
            ):
                check(
                    f"step-{step + 1}/optimizer-group",
                    {k: left[k] for k in right},
                    right,
                    rtol=0,
                    atol=0,
                )
        if step == 1:
            rng = capture_rng_state()
            if rank == 0:
                save_model_checkpoint(
                    args.output_dir / "checkpoint.pt",
                    "pretrain",
                    config,
                    *state,
                    {"next_step": 2},
                    capture_rng_state(),
                )
            dist.barrier()
        rows.append(row)
    resumed = PretrainFSDPEngine(deepcopy(initial), hf_config, engine_config)
    resumed.initialize()
    checkpoint = load_model_checkpoint(args.output_dir / "checkpoint.pt")
    resumed.load_canonical_state(checkpoint.model_state, checkpoint.optimizer_state)
    restore_rng_state(rng)
    resumed_metrics, resumed_snapshot = native_step(resumed, batch, args.precision)
    check("resume/parameters-and-gradients", resumed_snapshot, snapshot, rtol=0, atol=0)
    for field in ("loss", "gradient_norm"):
        check(f"resume/{field}", resumed_metrics[field], actual[field], rtol=0, atol=0)
    resumed_state = resumed.canonical_state(value)
    if rank == 0:
        check("resume/model-and-optimizer-state", resumed_state, state, rtol=0, atol=0)
    eval_results = []
    for mode in ("sample", "mean"):
        batch = examples(mode, backbone.eos_token_id)
        data = pretrain_batch(batch, config.batch_size, rank, dist.get_world_size())
        with resumed.eval_mode():
            outputs = resumed.infer_batch(data, pretrain_loss)
        actual_nll = torch.tensor(sum(outputs["loss"]), device=device, dtype=torch.float64)
        dist.all_reduce(actual_nll)
        reference_data = TensorDict({}, batch_size=[])
        assign_non_tensor(reference_data, examples=tuple(batch))
        with reference.engine.eval_mode():
            expected_nll = reference.engine.infer_batch(reference_data)["metrics"]["loss"]
        check(
            f"evaluation/{mode}/nll",
            actual_nll.cpu().float(),
            torch.tensor(expected_nll),
            rtol=1e-3,
            atol=1e-6,
        )
        eval_results.append({"mode": mode, "native": actual_nll.item(), "reference": expected_nll})
    (output / "result.json").write_text(
        json.dumps(
            {
                "passed": not errors,
                "errors": errors,
                "precision": args.precision,
                "second_step_aligned": args.align_second_step,
                "reshard_after_forward": args.reshard_after_forward,
                "steps": rows,
                "resume_exact": not any(e["check"].startswith("resume/") for e in errors),
                "evaluation_nll": eval_results,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps({"rank": rank, "passed": not errors, "failed_checks": len(errors)}), flush=True
    )
    failed = torch.tensor(int(bool(errors)), device=device)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    dist.destroy_process_group()
    if failed.item():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
