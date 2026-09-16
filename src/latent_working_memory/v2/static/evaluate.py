"""Static greedy generation with correct/empty/wrong memory and raw-text controls."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import string

import torch
from transformers import AutoTokenizer

from latent_working_memory.devices import validate_device
from latent_working_memory.v2.gmsa_checkpoint import load_weights
from latent_working_memory.v2.gmsa_config import GMSAConfig
from latent_working_memory.v2.gmsa import GMSA
from latent_working_memory.v2.static.data import StaticDataset


def normalize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def score(prediction, references):
    predicted = normalize(prediction).split()
    scores = []
    for reference in references:
        target = normalize(reference).split()
        common = sum((Counter(predicted) & Counter(target)).values())
        f1 = (
            2 * common / (len(predicted) + len(target))
            if predicted and target
            else float(predicted == target)
        )
        scores.append((float(predicted == target), f1))
    return max(s[0] for s in scores), max(s[1] for s in scores)


def main():
    parser = argparse.ArgumentParser(description="GMSA 静态逐条生成评估")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("autoencoding", "finetune"), required=True)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--max-target-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    validate_device(device)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("evaluation requires an empty output directory")
    config = GMSAConfig(**json.loads((args.checkpoint / "model.json").read_text()))
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    dataset = StaticDataset(
        args.data, tokenizer, args.stage, args.max_context_tokens, args.max_target_tokens
    )
    model = GMSA(config, torch.bfloat16 if device.type == "cuda" else torch.float32).to(device)
    load_weights(model, args.checkpoint)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            indent=2,
        )
        + "\n"
    )
    aggregates = {}
    with torch.no_grad(), (args.output_dir / "predictions.jsonl").open("w") as stream:
        for index, row in enumerate(dataset.rows):
            prompt = row["prompt_ids"].to(device)[None]
            context = row["context_ids"].to(device)[None]
            for ratio in config.compression_ratios:
                memories = model.encode(context, torch.ones_like(context, dtype=torch.bool), ratio)
                variants = {
                    "memory": model.reader_prefix(
                        memories, prompt, torch.ones_like(prompt, dtype=torch.bool)
                    )[0]
                }
                embedding = model.decoder.get_input_embeddings()
                variants["no_memory"] = embedding(prompt[0])
                variants["full_context"] = embedding(torch.cat((context[0], prompt[0])))
                # Wrong source must be distinct; use a deterministic next-row donor.
                donor = None
                for offset in range(1, len(dataset)):
                    candidate = dataset.rows[(index + offset) % len(dataset)]
                    if not torch.equal(candidate["context_ids"], row["context_ids"]) and (
                        (len(candidate["context_ids"]) + ratio - 1) // ratio == len(memories[0])
                    ):
                        donor = candidate
                        break
                if donor is not None:
                    donor_ids = donor["context_ids"].to(device)[None]
                    wrong = model.encode(
                        donor_ids, torch.ones_like(donor_ids, dtype=torch.bool), ratio
                    )
                    variants["wrong_memory"] = model.reader_prefix(
                        wrong, prompt, torch.ones_like(prompt, dtype=torch.bool)
                    )[0]
                for condition, prefix in variants.items():
                    if len(prefix) + args.max_new_tokens > model.max_positions:
                        raise ValueError("evaluation generation budget exceeds decoder window")
                    generated = model.decoder.generate(
                        inputs_embeds=prefix[None],
                        attention_mask=torch.ones(
                            (1, len(prefix)), device=device, dtype=torch.long
                        ),
                        do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        repetition_penalty=args.repetition_penalty,
                    )[0]
                    prediction = tokenizer.decode(generated, skip_special_tokens=True)
                    em, f1 = score(prediction, dataset.references[index])
                    record = dict(
                        index=index,
                        ratio=ratio,
                        condition=condition,
                        prediction=prediction,
                        references=dataset.references[index],
                        em=em,
                        f1=f1,
                        slots=len(memories[0])
                        if condition.endswith("memory") and condition != "no_memory"
                        else 0,
                        hit_limit=tokenizer.eos_token_id not in generated.tolist(),
                    )
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    values = aggregates.setdefault(f"r{ratio}/{condition}", [0, 0.0, 0.0])
                    values[0] += 1
                    values[1] += em
                    values[2] += f1
    summary = {key: dict(reads=n, em=em / n, f1=f1 / n) for key, (n, em, f1) in aggregates.items()}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
