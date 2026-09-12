"""Prepare shared evaluation reads and validate the complete dynamic curriculum."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from latent_working_memory.v1.checkpoint import load_model_checkpoint
from latent_working_memory.v1.dynamic_config import load_dynamic_config
from latent_working_memory.v1.dynamic_data import DynamicTextSampler, TrainingText
from latent_working_memory.v1.dynamic_evaluation import evaluation_schedule
from latent_working_memory.v1.dynamic_training import read_schedule
from latent_working_memory.v1.squad import SquadDataset


def evaluation_identity(recipe):
    return {
        key: asdict(recipe)[key]
        for key in (
            "capacities",
            "ratios",
            "new_count",
            "history_count",
            "max_visits",
            "seed",
            "generation_tokens",
            "eval_texts_per_ratio",
            "eval_reads_per_kind",
        )
    }


def prepare_dynamic(index_path, recipe, model_config, output_dir):
    data = SquadDataset(index_path)
    if max(recipe.capacities) > model_config.k_limit:
        raise ValueError("capacity exceeds checkpoint slot limit")
    sampler = DynamicTextSampler(data, recipe, model_config.write_context_tokens)
    report = []
    for epoch in range(recipe.epochs):
        for micro in range(recipe.micro_epochs_per_epoch):
            capacity, texts, info = sampler.micro_epoch(epoch, micro, recipe.epochs)
            reads = 0
            for offset, text in enumerate(texts):
                schedule = read_schedule(
                    text.episode(data),
                    data.tokenizer,
                    recipe,
                    model_config,
                    f"{recipe.seed}:read:{epoch}:{micro}:{offset}",
                    capacity,
                )
                reads += sum(map(len, schedule.values()))
            report.append(
                {k: v for k, v in info.items() if k != "texts"}
                | {
                    "input_tokens": sum(t.input_tokens for t in texts),
                    "updates": sum(t.updates for t in texts),
                    "reads": reads,
                }
            )
    plan = {
        "data_index": data.index,
        "selection": evaluation_identity(recipe),
        "panels": {},
        "reads": {},
    }
    for split in ("dev", "test"):
        panel = sampler.evaluation_texts(split, recipe.eval_texts_per_ratio)
        plan["panels"][split] = {k: [asdict(t) for t in texts] for k, texts in panel.items()}
        plan["reads"][split] = {}
        for capacity, texts in panel.items():
            plans = plan["reads"][split][capacity] = {}
            for text in texts:
                episode = text.episode(data)
                schedule = evaluation_schedule(
                    episode, data.tokenizer, recipe, model_config, capacity
                )
                plans[episode.episode_id] = [
                    [end, read.read_id] for end, jobs in schedule.items() for read, _ in jobs
                ]
                for jobs in schedule.values():
                    for read, tokens in jobs:
                        span = read.references[0].evidence_spans[0]
                        source = next(
                            s for s in episode.sources if (s.token_start, s.token_end) == span
                        )
                        prompt = (
                            "Text:\n"
                            + source.provenance["context"]
                            + "\n\n"
                            + read.prompt.replace(
                                "information stored in memory", "provided text", 1
                            )
                        )
                        if (
                            1
                            + len(data.tokenizer.encode(prompt, add_special_tokens=False))
                            + max(len(tokens.target_ids), recipe.generation_tokens)
                            > model_config.read_context_tokens
                        ):
                            raise ValueError(f"gold paragraph exceeds read context: {read.read_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (("evaluation-plan", plan), ("data-validation", report)):
        path = output_dir / f"{name}.json"
        with path.open("x") as f:
            json.dump(value, f, indent=2)
            f.write("\n")
    return report


def load_evaluation_plan(path, data, recipe):
    plan = json.loads(path.read_text())
    expected = json.loads(json.dumps(evaluation_identity(recipe)))
    if plan["data_index"] != data.index or plan["selection"] != expected:
        raise ValueError("shared evaluation data or selection configuration differs")
    panels = {
        split: {int(k): [TrainingText(**t) for t in texts] for k, texts in groups.items()}
        for split, groups in plan["panels"].items()
    }
    return plan, panels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = load_model_checkpoint(args.checkpoint)
    rows = prepare_dynamic(
        args.index, load_dynamic_config(args.config), checkpoint.config, args.output_dir
    )
    print(
        json.dumps(
            {
                key: sum(r[key] for r in rows)
                for key in ("used_samples", "input_tokens", "updates", "reads")
            }
        )
    )


if __name__ == "__main__":
    main()
