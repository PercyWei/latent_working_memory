"""补充固定 QA 请求的基座与预训练 Reader 空记忆对照，不重新训练或写入记忆。"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from latent_working_memory.devices import validate_device
from latent_working_memory.v1.backbone import ReadTokens, load_backbone
from latent_working_memory.v1.checkpoint import load_model_checkpoint
from latent_working_memory.v1.dynamic.evaluation import (
    aggregate_qa,
    answer_scores,
    write_evaluation,
)
from latent_working_memory.v1.training import precision_context


EMPTY_CONDITIONS = ("no_memory_base", "no_memory_pretrain")
ORIGINAL_CONDITIONS = frozenset(
    ("memory", "no_memory", "wrong_memory", "gold_paragraph", "gold_paragraph_base")
)


def read_identity(row):
    return row["capacity"], row["episode_id"], row["read_id"], row["prefix_end"]


def original_requests(rows):
    groups = {}
    for row in rows:
        group = groups.setdefault(read_identity(row), {})
        if row["condition"] in group:
            raise ValueError("duplicate QA condition at the same read")
        group[row["condition"]] = row
    if not groups or any(set(group) != ORIGINAL_CONDITIONS for group in groups.values()):
        raise ValueError("supplement requires exactly the original five conditions for every read")
    requests = []
    for group in groups.values():
        reference = group["no_memory"]
        for row in group.values():
            for field in ("question", "references", "target_tokens", "document_id", "kind"):
                if row[field] != reference[field]:
                    raise ValueError(f"conditions disagree on {field}")
        requests.append(reference)
    return requests


@torch.no_grad()
def evaluate_empty_memory(
    backbone, tokenizer, rows, generation_tokens, read_context_tokens, device
):
    requests = original_requests(rows)
    backbone.eval()
    empty = torch.empty(
        (0, backbone.d_mem), device=device, dtype=backbone.memory_projection.weight.dtype
    )
    cache, added = {}, []
    for index, row in enumerate(requests):
        tokens = ReadTokens(
            tuple(tokenizer.encode(row["question"], add_special_tokens=False)),
            tuple(tokenizer.encode(row["references"][0], add_special_tokens=False))
            + (tokenizer.eos_token_id,),
        )
        if len(tokens.target_ids) != row["target_tokens"]:
            raise ValueError("reference tokenization differs from the original evaluation")
        if (
            1 + len(tokens.prompt_ids) + max(len(tokens.target_ids), generation_tokens)
            > read_context_tokens
        ):
            raise ValueError("empty-memory QA exceeds read context")
        for condition in EMPTY_CONDITIONS:
            key = condition, tokens.prompt_ids, tokens.target_ids
            if key not in cache:
                use_lora = condition == "no_memory_pretrain"
                with precision_context(device):
                    result = backbone.read_batch([empty], [tokens], use_reader_lora=use_lora)[0]
                    generated = backbone.greedy_students(
                        [empty], [tokens.prompt_ids], [generation_tokens], use_reader_lora=use_lora
                    )[0]
                cache[key] = {
                    "nll_sum": float(result.token_nll.sum()),
                    "target_tokens": result.target_length,
                    "prediction": tokenizer.decode(generated, skip_special_tokens=True),
                    "hit_limit": len(generated) == generation_tokens
                    and generated[-1] != tokenizer.eos_token_id,
                }
            prediction = cache[key]
            em, f1 = answer_scores(prediction["prediction"], row["references"])
            added.append(dict(row, condition=condition, **prediction, em=em, f1=f1))
        if (index + 1) % 10 == 0 or index + 1 == len(requests):
            print(
                json.dumps(
                    {
                        "completed_reads": index + 1,
                        "reads": len(requests),
                        "unique_condition_requests": len(cache),
                    }
                ),
                flush=True,
            )
    return added


def supplement_report(report, output_dir, dataset, device):
    if output_dir.exists():
        raise FileExistsError("use a new supplemental evaluation directory")
    info = json.loads((report.parent / "evaluation.json").read_text())
    if info["split"] != "test":
        raise ValueError("supplemental baselines require a fixed test report")
    rows = [json.loads(line) for line in report.with_suffix(".jsonl").read_text().splitlines()]
    original_requests(rows)
    prefix = {"squad": "squad:", "personamem": "personamem-v2:"}[dataset]
    if any(not row["document_id"].startswith(prefix) for row in rows):
        raise ValueError("dataset label differs from report document identities")
    checkpoint = load_model_checkpoint(info["checkpoint"])
    if checkpoint.phase != "dynamic" or checkpoint.progress["next_step"] != info["checkpoint_step"]:
        raise ValueError("report does not match its dynamic checkpoint")
    initial_path = Path(checkpoint.progress["initial_checkpoint"])
    initial = load_model_checkpoint(initial_path)
    if (
        initial.phase != "pretrain"
        or replace(initial.config, gradient_checkpointing=False) != checkpoint.config
    ):
        raise ValueError("pretraining initialization differs from the dynamic model contract")
    config = checkpoint.config
    del checkpoint
    tokenizer, backbone = load_backbone(
        config, device, torch.bfloat16 if device.type == "cuda" else torch.float32
    )
    backbone.load_trainable_state_dict(initial.model_state["backbone"])
    del initial
    begin = time.perf_counter()
    added = evaluate_empty_memory(
        backbone,
        tokenizer,
        rows,
        info["config"]["generation_tokens"],
        config.read_context_tokens,
        device,
    )
    combined = sorted(rows + added, key=lambda row: (*read_identity(row), row["condition"]))
    output_dir.mkdir(parents=True)
    write_evaluation(output_dir, report.stem, aggregate_qa(combined), combined)
    metadata = dict(
        info,
        dataset=dataset,
        supplemental_baselines={
            "source_report": str(report.resolve()),
            "pretrain_checkpoint": str(initial_path.resolve()),
            "conditions": list(EMPTY_CONDITIONS),
            "reads": len(added) // len(EMPTY_CONDITIONS),
            "seconds": time.perf_counter() - begin,
        },
    )
    (output_dir / "evaluation.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("squad", "personamem"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    validate_device(device)
    supplement_report(args.report, args.output_dir, args.dataset, device)


if __name__ == "__main__":
    main()
