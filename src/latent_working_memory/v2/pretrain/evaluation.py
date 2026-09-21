"""固定轨迹的各次压缩后的损失、一次压缩对照和自由重构。"""

from collections import defaultdict
import math

import torch
import torch.distributed as dist

from latent_working_memory.v2.pretrain.engine import precision_context


def summarize_reads(records, rounds_key):
    groups = defaultdict(list)
    for row in records:
        for read in row[rounds_key]:
            for name in (
                "all",
                f"round/{read['round']}",
                f"depth/{row['depth']}",
                f"ratio/{row['ratio_bin']}",
            ):
                groups[name].append(read)
    result = {}
    for name, reads in groups.items():
        for objective in ("ae", "lm"):
            valid = [r for r in reads if r[objective] is not None]
            if valid:
                # Within each group report token-weighted NLL and its exact denominator.
                count = sum(r[f"{objective}_tokens"] for r in valid)
                result[f"{name}/{objective}_nll"] = (
                    sum(r[objective] * r[f"{objective}_tokens"] for r in valid) / count
                )
                result[f"{name}/{objective}_tokens"] = count
    if records:
        # evaluate() always measures both objectives, including AE-only training runs.
        for objective in ("ae", "lm"):
            result[f"trajectory_{objective}"] = sum(
                sum(r[objective] for r in row[rounds_key]) / len(row[rounds_key]) for row in records
            ) / len(records)

    result["trajectories"] = len(records)
    return result


def summarize(records):
    result = summarize_reads(records, "rounds")
    result.update(
        {
            f"independent_prefix/{key}": value
            for key, value in summarize_reads(records, "independent_prefix").items()
        }
    )
    if records:
        for objective in ("ae", "lm"):
            result[f"one_shot_{objective}"] = sum(
                row["one_shot"][objective] for row in records
            ) / len(records)
            result[f"final_minus_one_shot_{objective}"] = sum(
                row["rounds"][-1][objective] - row["one_shot"][objective] for row in records
            ) / len(records)
    return result


def generate_record(task, memory, reference, tokenizer):
    generated = task.codec.generate(
        memory,
        task.ae_prompt,
        len(reference) + 1,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    ).tolist()
    reached_eos = tokenizer.eos_token_id in generated
    if reached_eos:
        generated = generated[: generated.index(tokenizer.eos_token_id)]
    return {
        "prediction": tokenizer.decode(generated, skip_special_tokens=True),
        "reference": tokenizer.decode(reference, skip_special_tokens=True),
        "final_round_exact_match": generated == reference,
        "hit_limit": not reached_eos and len(generated) >= len(reference) + 1,
    }


@torch.no_grad()
def evaluate(task, rows, tokenizer, generation_samples=0):
    device = task.ae_prompt.device
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    was_training = task.training
    task.eval()
    original_write_alignment = task.codec.write_alignment
    if task.codec.stage != "multiround":
        # Static training never optimizes Aw. Use the learned Ar weights for this read-only
        # evaluation, equivalent to copying Ar at the single-write -> multi-write transition.
        task.codec.write_alignment = task.codec.read_alignment
    records = []
    try:
        for i in range(rank, len(rows), world):
            row = rows[i]
            with precision_context(device):
                output = task(row, read_task="both")
                control = task(row, read_task="both", write_mode="independent_prefix")
                record = {
                    "index": i,
                    "sample_id": row.sample_id,
                    "document_id": row.document_id,
                    "source_char_start": row.source_char_start,
                    "depth": len(row.write_ends),
                    "ratio_bin": math.ceil(row.write_ends[-1] / row.capacity),
                    "rounds": output["rounds"],
                    "one_shot": control["rounds"][-1],
                    "independent_prefix": control["rounds"],
                }
                if i < generation_samples:
                    ids = row.token_ids.to(device)
                    memory, previous = None, 0
                    for end in row.write_ends:
                        memory = task.codec.write(memory, ids[previous:end], row.capacity)
                        previous = end
                    reference = ids[:previous].tolist()
                    record["generation"] = generate_record(task, memory, reference, tokenizer)
                    independent_memory = task.codec.write(None, ids[:previous], row.capacity)
                    record["independent_generation"] = generate_record(
                        task, independent_memory, reference, tokenizer
                    )
                records.append(record)
        if world > 1:
            gathered = [None] * world
            dist.all_gather_object(gathered, records)
            records = [row for rank_rows in gathered for row in rank_rows]
        records.sort(key=lambda row: row["index"])
        metrics = summarize(records)
        for field, prefix in (
            ("generation", "generation"),
            ("independent_generation", "independent_prefix/generation"),
        ):
            generated = [r[field] for r in records if field in r]
            for name in ("final_round_exact_match", "hit_limit"):
                values = [r[name] for r in generated]
                if values:
                    metrics[f"{prefix}/{name}"] = sum(values) / len(values)
            metrics[f"{prefix}/samples"] = len(generated)
        return metrics, records
    finally:
        task.codec.write_alignment = original_write_alignment
        task.train(was_training)
