"""Fixed-panel QA evaluation and paired statistics."""

from bisect import bisect_left
from collections import Counter, defaultdict
import json
import math
import random
import re
import string

import torch
import torch.distributed as dist

from latent_working_memory.v1.backbone import ReadTokens
from latent_working_memory.v1.dynamic_data import write_boundaries
from latent_working_memory.v1.dynamic_training import read_schedule
from latent_working_memory.v1.training import precision_context


def answer_scores(prediction, references):
    def normalize(text):
        text = "".join(c for c in text.lower() if c not in string.punctuation)
        return re.sub(r"\s+", " ", re.sub(r"\b(a|an|the)\b", " ", text)).strip()

    pred = normalize(prediction)
    em, f1 = 0.0, 0.0
    for reference in references:
        gold = normalize(reference)
        em = max(em, float(pred == gold))
        p, g = pred.split(), gold.split()
        common = sum((Counter(p) & Counter(g)).values())
        score = 2 * common / (len(p) + len(g)) if common else 0.0
        f1 = max(f1, score)
    return em, f1


def encode_episode(backbone, writer, episode, capacity):
    state, previous = None, 0
    for end in write_boundaries(episode, capacity):
        f = backbone.text_features([episode.input_ids[previous:end]], [previous])[0]
        state = (
            writer(writer.initialize_state(f.dtype), f, first_slots=capacity)
            if state is None
            else writer(state, f)
        )
        previous = end
    return state


def evaluation_schedule(episode, tokenizer, recipe, model_config, capacity):
    schedule = read_schedule(
        episode,
        tokenizer,
        recipe,
        model_config,
        f"{recipe.seed}:eval:{episode.episode_id}",
        capacity,
        generation=True,
    )
    rng = random.Random(f"{recipe.seed}:eval-subset:{episode.episode_id}")
    candidates = {"arrival": [], "delayed": []}
    for end, jobs in schedule.items():
        for read, _ in jobs:
            evidence_end = max(b for ref in read.references for _, b in ref.evidence_spans)
            candidates["arrival" if end == evidence_end else "delayed"].append((end, read.read_id))
    selected = set()
    for values in candidates.values():
        selected.update(rng.sample(values, min(len(values), recipe.eval_reads_per_kind)))
    return {
        end: [(read, tokens) for read, tokens in jobs if (end, read.read_id) in selected]
        for end, jobs in schedule.items()
    }


@torch.no_grad()
def evaluate_qa(
    backbone,
    writer,
    tokenizer,
    model_config,
    recipe,
    episodes,
    device,
    capacity,
    conditions=("memory", "no_memory", "wrong_memory", "gold_paragraph", "gold_paragraph_base"),
    generate=True,
    read_plans=None,
):
    if not episodes or not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("non-empty episodes and unique conditions required")
    if set(conditions) - {
        "memory",
        "no_memory",
        "wrong_memory",
        "gold_paragraph",
        "gold_paragraph_base",
    }:
        raise ValueError("unsupported QA condition")
    if "wrong_memory" in conditions and len({e.sources[0].document_id for e in episodes}) < 2:
        raise ValueError("wrong-memory comparison requires two independent documents")
    backbone.eval()
    writer.eval()
    records = []
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    for index in range(rank, len(episodes), world_size):
        episode = episodes[index]
        schedule = evaluation_schedule(episode, tokenizer, recipe, model_config, capacity)
        if read_plans is not None:
            actual = [[end, read.read_id] for end, jobs in schedule.items() for read, _ in jobs]
            if actual != read_plans[episode.episode_id]:
                raise ValueError("evaluation reads differ from the shared plan")
        cached = {}
        with precision_context(device):
            wrong = None
            if "wrong_memory" in conditions:
                donor = next(
                    e
                    for e in episodes[index + 1 :] + episodes[: index + 1]
                    if e.sources[0].document_id != episode.sources[0].document_id
                )
                # Validate the donor's write budget too; donor QA targets are not used.
                if any(
                    b - a + 1 > model_config.write_context_tokens
                    for a, b in zip(
                        (0, *write_boundaries(donor, capacity)[:-1]),
                        write_boundaries(donor, capacity),
                    )
                ):
                    raise ValueError("donor paragraph exceeds write context budget")
                wrong = encode_episode(backbone, writer, donor, capacity).values
            state, previous = None, 0
            boundaries = tuple(schedule)
            for end in boundaries:
                f = backbone.text_features([episode.input_ids[previous:end]], [previous])[0]
                state = (
                    writer(writer.initialize_state(f.dtype), f, first_slots=capacity)
                    if state is None
                    else writer(state, f)
                )
                previous = end
                for read, tokens in schedule[end]:
                    for condition in conditions:
                        memory = state.values[:0]
                        condition_tokens = tokens
                        use_lora = condition != "gold_paragraph_base"
                        if condition == "memory":
                            memory = state.values
                        elif condition == "wrong_memory":
                            memory = wrong
                        elif condition in {"gold_paragraph", "gold_paragraph_base"}:
                            evidence_span = read.references[0].evidence_spans[0]
                            source = next(
                                source
                                for source in episode.sources
                                if (source.token_start, source.token_end) == evidence_span
                            )
                            prompt = (
                                "Text:\n"
                                + source.provenance["context"]
                                + "\n\n"
                                + read.prompt.replace(
                                    "information stored in memory", "provided text", 1
                                )
                            )
                            condition_tokens = ReadTokens(
                                tuple(tokenizer.encode(prompt, add_special_tokens=False)),
                                tokens.target_ids,
                            )
                        if (
                            1
                            + len(memory)
                            + len(condition_tokens.prompt_ids)
                            + max(len(condition_tokens.target_ids), recipe.generation_tokens)
                            > model_config.read_context_tokens
                        ):
                            raise ValueError(
                                f"{condition} QA read exceeds context budget: {read.read_id}"
                            )
                        cache_key = condition, read.read_id
                        if condition == "memory" or cache_key not in cached:
                            result = backbone.read_batch(
                                [memory], [condition_tokens], use_reader_lora=use_lora
                            )[0]
                            scores = {
                                "nll_sum": float(result.token_nll.sum()),
                                "target_tokens": result.target_length,
                            }
                            if generate:
                                generated = backbone.greedy_students(
                                    [memory],
                                    [condition_tokens.prompt_ids],
                                    [recipe.generation_tokens],
                                    use_reader_lora=use_lora,
                                )[0]
                                prediction = tokenizer.decode(generated, skip_special_tokens=True)
                                em, f1 = answer_scores(
                                    prediction, [r.text for r in read.references]
                                )
                                scores.update(
                                    prediction=prediction,
                                    em=em,
                                    f1=f1,
                                    hit_limit=len(generated) == recipe.generation_tokens
                                    and generated[-1] != tokenizer.eos_token_id,
                                )
                            if condition != "memory":
                                cached[cache_key] = scores
                        else:
                            scores = cached[cache_key]
                        evidence_end = max(
                            b for ref in read.references for _, b in ref.evidence_spans
                        )
                        records.append(
                            {
                                "episode_id": episode.episode_id,
                                "read_id": read.read_id,
                                "prefix_end": end,
                                "condition": condition,
                                "delay_tokens": end - evidence_end,
                                "delay_writes": boundaries.index(end)
                                - bisect_left(boundaries, evidence_end),
                                "capacity": capacity,
                                "input_tokens": len(episode.input_ids),
                                "compression_ratio": end / capacity,
                                "final_compression_ratio": len(episode.input_ids) / capacity,
                                "kind": "arrival" if end == evidence_end else "delayed",
                                "question": read.prompt,
                                "references": [r.text for r in read.references],
                                **scores,
                            }
                        )
    if world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, records)
        records = [row for rank_rows in gathered for row in rank_rows]
    records = sorted(
        records,
        key=lambda row: (
            row["episode_id"],
            row["prefix_end"],
            row["read_id"],
            row["condition"],
        ),
    )
    metrics = {}
    for condition in conditions:
        for kind in ("all", "arrival", "delayed"):
            rows = [
                r
                for r in records
                if r["condition"] == condition and (kind == "all" or r["kind"] == kind)
            ]
            if rows:
                metrics[f"{condition}/{kind}"] = qa_summary(rows)
    return metrics, records


def evaluate_panel(
    backbone,
    writer,
    tokenizer,
    model_config,
    recipe,
    data,
    panel,
    device,
    generate=True,
    read_plans=None,
):
    records = []
    for capacity, texts in panel.items():
        episodes = [text.episode(data) for text in texts]
        _, rows = evaluate_qa(
            backbone,
            writer,
            tokenizer,
            model_config,
            recipe,
            episodes,
            device,
            capacity,
            generate=generate,
            read_plans=read_plans[str(capacity)] if read_plans is not None else None,
        )
        metadata = {episode.episode_id: text for episode, text in zip(episodes, texts, strict=True)}
        for row in rows:
            text = metadata[row["episode_id"]]
            row.update(
                target_ratio=text.ratio,
                document_id=text.document_id,
                paragraph_start=text.paragraph_start,
                paragraph_end=text.paragraph_end,
            )
        records.extend(rows)
    return aggregate_qa(records), records


def qa_summary(rows):
    result = {
        "reads": len(rows),
        "texts": len({r["episode_id"] for r in rows}),
        "target_tokens": sum(r["target_tokens"] for r in rows),
        "nll": sum(r["nll_sum"] for r in rows) / sum(r["target_tokens"] for r in rows),
    }
    generated = [r for r in rows if "prediction" in r]
    result["generations"] = len(generated)
    if generated:
        result.update(
            em=sum(r["em"] for r in generated) / len(generated),
            f1=sum(r["f1"] for r in generated) / len(generated),
            hit_limit_rate=sum(r["hit_limit"] for r in generated) / len(generated),
        )
    return result


def aggregate_qa(records):
    buckets = defaultdict(list)
    for row in records:
        prefixes = ["overall", f"k{row['capacity']}"]
        if "target_ratio" in row:
            prefixes.extend(
                [
                    f"k{row['capacity']}/target-r{row['target_ratio']}",
                    f"k{row['capacity']}/length-le{2 ** (row['input_tokens'] - 1).bit_length()}",
                    f"k{row['capacity']}/actual-r-le{math.ceil(row['compression_ratio'])}",
                    f"delay-tokens-le{0 if not row['delay_tokens'] else 2 ** (row['delay_tokens'] - 1).bit_length()}",
                    f"delay-updates-le{0 if not row['delay_writes'] else 2 ** (row['delay_writes'] - 1).bit_length()}",
                ]
            )
        for prefix in prefixes:
            for kind in ("all", row["kind"]):
                buckets[f"{prefix}/{row['condition']}/{kind}"].append(row)
    result = {key: qa_summary(rows) for key, rows in buckets.items()}
    paired = defaultdict(dict)
    for row in records:
        paired[row["capacity"], row["episode_id"], row["read_id"], row["prefix_end"]][
            row["condition"]
        ] = row
    for condition in ("no_memory", "wrong_memory", "gold_paragraph", "gold_paragraph_base"):
        pairs = [
            (v["memory"], v[condition]) for v in paired.values() if "memory" in v and condition in v
        ]
        if pairs:
            values = {
                "reads": len(pairs),
                "nll_difference": sum(
                    a["nll_sum"] / a["target_tokens"] - b["nll_sum"] / b["target_tokens"]
                    for a, b in pairs
                )
                / len(pairs),
            }
            generated = [(a, b) for a, b in pairs if "prediction" in a and "prediction" in b]
            for key in ("em", "f1"):
                if generated:
                    values[f"{key}_difference"] = sum(a[key] - b[key] for a, b in generated) / len(
                        generated
                    )
            result[f"paired/memory-minus-{condition}"] = values
    return result


def write_evaluation(output_dir, name, metrics, rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output_dir / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
