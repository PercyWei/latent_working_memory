"""独立评估 QA 答案 NLL 与 greedy 生成；逐题重新初始化记忆并执行局部 TTT。"""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import time

import torch
from transformers import set_seed

from latent_working_memory.data_preparation.personamem.audit import scores
from latent_working_memory.runtime import initialize_device
from latent_working_memory.v4.checkpoint import read_checkpoint
from latent_working_memory.v4.data import dataset_identity, load_evaluation_episodes
from latent_working_memory.v4.evaluation_records import (
    EvaluationHistory,
    evaluation_config_path,
    load_evaluation_settings,
)
from latent_working_memory.v4.evaluation_runtime import (
    load_evaluation_inputs,
    load_evaluation_model,
)
from latent_working_memory.v4.tracking import (
    publish_evaluations,
    validate_evaluation_publication,
)


_UNSPECIFIED = object()


def _check_test_split(statistics, run):
    documents = set(statistics["raw"]["group_ids"])
    selection = run["data"]["selection"]
    for split in ("train", "dev"):
        if documents.intersection(selection[split]["raw"]["group_ids"]):
            raise ValueError(f"raw test documents overlap the training run's {split} split")


def evaluate(
    checkpoint_path, output_dir, dataset_dir=None, max_new_tokens=None, device="cuda", tracking=None,
    defer_publication=False, max_test_samples=_UNSPECIFIED,
):
    output_dir = Path(output_dir).expanduser().resolve()
    evaluation_path = evaluation_config_path(output_dir)
    evaluation, seed = load_evaluation_settings(output_dir)
    if dataset_dir is None:
        if evaluation is None:
            raise ValueError(f"{evaluation_path} is missing; specify --test-data")
        dataset_dir = Path(evaluation["test_file"]).parent
    if max_new_tokens is None:
        max_new_tokens = evaluation["max_new_tokens"] if evaluation is not None else 128
    if max_test_samples is _UNSPECIFIED:
        max_test_samples = evaluation["max_test_samples"] if evaluation is not None else None
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if tracking not in {None, "online", "disabled"}:
        raise ValueError("tracking must be online or disabled")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("v4 evaluation runs in one process; do not launch it with torchrun")
    set_seed(seed)
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = read_checkpoint(checkpoint_path)
    run, step = checkpoint["run"], checkpoint["cursor"]["step"]
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    test_file = dataset_dir / "test.jsonl"
    if tracking is None:
        tracking = "online" if run["training"]["swanlab_project"] is not None else "disabled"
    if tracking == "online":
        validate_evaluation_publication(run, checkpoint_path, step)

    config, tokenizer, backbone_config = load_evaluation_inputs(checkpoint)
    training_dir = Path(run["training"]["output_dir"]).expanduser().resolve()
    model_window = backbone_config.max_position_embeddings
    episodes, references, statistics = load_evaluation_episodes(
        test_file,
        tokenizer,
        config.pending_size + config.recent_size + 1,
        model_window,
        seed,
        max_test_samples,
    )
    _check_test_split(statistics, run)

    evaluation_config = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "split": "test",
        "checkpoint": {"path": str(checkpoint_path), "step": step},
        "training_run": str(training_dir),
        "seed": seed,
        "model": asdict(config),
        "test_file": str(test_file),
        "dataset": dataset_identity(episodes),
        "device": str(device),
        "tracking": tracking,
        "model_window": model_window,
        "max_new_tokens": max_new_tokens,
        "max_test_samples": max_test_samples,
        "eos_token_id": tokenizer.eos_token_id,
        "protocol": {
            "nll": "answer-token weighted; includes assistant template ending tokens",
            "generation": "greedy; each NLL and generation call starts with independent memory",
            "em_f1": "single-reference QA mean; lowercase, remove ASCII punctuation and English "
            "articles a/an/the, collapse whitespace; token-overlap F1",
            "task": "held-out document-QA test with single-reference scoring",
            "selection": "after length filtering; QA hash sampling with the evaluation seed",
            "write_loss": "teacher-forced path only; average across layers and write events",
        },
    }
    history = EvaluationHistory(output_dir, evaluation_config)
    with history.lock():
        summary = history.cached_summary()
        if summary is not None:
            print(f"test reusing cached summary: {output_dir / 'summary.json'}", flush=True)
            if tracking == "online" and not defer_publication:
                publish_evaluations(run, [(checkpoint_path, summary, output_dir)])
            return summary

        device = initialize_device(device)
        evaluation_config["device"] = str(device)
        model = load_evaluation_model(checkpoint, device, config)
        del checkpoint
        slots = [layer.initial_slots for layer in model.memory_layers]
        evaluation_config["memory"] = {
            "layers": len(slots),
            "slots_per_layer": config.num_slots,
            "dimension": config.memory_dim,
            "total_slots": len(slots) * config.num_slots,
            "elements": sum(value.numel() for value in slots),
            "bytes": sum(value.numel() * value.element_size() for value in slots),
            "dtype": str(slots[0].dtype),
        }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        nll_sum, target_tokens, em_sum, f1_sum = 0.0, 0, 0.0, 0.0
        writes, write_loss_before, write_loss = 0, 0.0, 0.0
        generated_tokens, capped_budgets = 0, 0
        budgets, endings = [], Counter()

        with (output_dir / "details.jsonl").open("w", encoding="utf-8") as stream:
            for index, episode in enumerate(episodes, 1):
                prompt_length = len(episode.prompt_ids)
                budget = min(max_new_tokens, model_window - prompt_length)
                # 两次调用各自创建 initial_state；生成路径从未接收参考答案。
                with torch.no_grad():
                    model.evaluation_phase = "nll"
                    output = model(
                        torch.tensor(
                            episode.prompt_ids + episode.target_ids, dtype=torch.long, device=device
                        ),
                        differentiable=False,
                        target_starts=[prompt_length],
                    )
                    if output["target_tokens"] != len(episode.target_ids):
                        raise ValueError("model target count differs from the reference answer tokens")
                    nll = float(output["loss"])
                    model.evaluation_phase = "generation"
                    generated = model.generate(
                        torch.tensor(episode.prompt_ids, dtype=torch.long, device=device),
                        budget,
                        tokenizer.eos_token_id,
                    )[prompt_length:].tolist()
                prediction = tokenizer.decode(generated, skip_special_tokens=True)
                reference = references[episode.id]
                quality = scores(prediction, reference["answer"])
                finish_reason = (
                    "eos"
                    if generated[-1] == tokenizer.eos_token_id
                    else "model_window"
                    if budget < max_new_tokens
                    else "max_new_tokens"
                )
                record = {
                    "id": episode.id,
                    "group_id": episode.group_id,
                    "question": reference["question"],
                    "reference": reference["answer"],
                    "prediction": prediction,
                    "prompt_tokens": prompt_length,
                    "target_tokens": output["target_tokens"],
                    "generated_tokens": len(generated),
                    "generated_token_ids": generated,
                    "generation_budget": budget,
                    "finish_reason": finish_reason,
                    "nll": nll,
                    **quality,
                    "write_events": output["write_events"],
                    "write_loss_before": output["write_loss_before"],
                    "write_loss": output["write_loss"],
                }
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                nll_sum += nll * output["target_tokens"]
                target_tokens += output["target_tokens"]
                em_sum += quality["em"]
                f1_sum += quality["f1"]
                writes += output["write_events"]
                write_loss_before += output["write_loss_before"] * output["write_events"]
                write_loss += output["write_loss"] * output["write_events"]
                generated_tokens += len(generated)
                capped_budgets += int(budget < max_new_tokens)
                budgets.append(budget)
                endings[finish_reason] += 1
                print(
                    f"test {index}/{len(episodes)} | NLL {nll_sum / target_tokens:.4f} | "
                    f"EM {em_sum / index:.4f} | F1 {f1_sum / index:.4f} | "
                    f"elapsed {time.perf_counter() - started:.1f}s",
                    flush=True,
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        resources = {
            "evaluation_seconds": time.perf_counter() - started,
            "scope": "NLL and generation loop; excludes model and data loading",
            "write_timing": model.write_timing(device),
        }
        if device.type == "cuda":
            resources.update(
                peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(device),
            )
        summary = {
            **evaluation_config,
            "data_statistics": statistics,
            "samples": len(episodes),
            "documents": statistics["groups"],
            "target_tokens": target_tokens,
            "metrics": {
                "nll": nll_sum / target_tokens,
                "em": em_sum / len(episodes),
                "f1": f1_sum / len(episodes),
                "write_loss_before": write_loss_before / writes,
                "write_loss": write_loss / writes,
            },
            "write_events": writes,
            "generation": {
                "max_new_tokens": max_new_tokens,
                "budget_min": min(budgets),
                "budget_max": max(budgets),
                "model_window_capped_samples": capped_budgets,
                "generated_tokens": generated_tokens,
                "finish_reasons": dict(endings),
            },
            "resources": resources,
        }
        history.save_summary(summary)
        if tracking == "online" and not defer_publication:
            publish_evaluations(run, [(checkpoint_path, summary, output_dir)])
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="global_step_<step> 目录")
    parser.add_argument(
        "--test-data", type=Path,
        help="默认 null，沿用已保存的评估目录；无评估配置时必填，读取 test.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="评估产物目录")
    parser.add_argument("--max-new-tokens", type=int, help="沿用已有评估配置；无配置时默认 128")
    parser.add_argument(
        "--max-test-samples", type=lambda value: None if value == "null" else int(value),
        default=_UNSPECIFIED,
        help="过滤后 test QA 样本上限；沿用已有配置，无配置时默认 null（全量）",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tracking", choices=("online", "disabled"))
    parser.add_argument("--defer-publication", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    evaluate(
        args.checkpoint,
        args.output_dir,
        args.test_data,
        args.max_new_tokens,
        args.device,
        args.tracking,
        args.defer_publication,
        args.max_test_samples,
    )


if __name__ == "__main__":
    main()
