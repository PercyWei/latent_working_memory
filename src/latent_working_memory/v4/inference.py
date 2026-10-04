"""从 v4 checkpoint 运行单条流式 greedy 生成。"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from latent_working_memory.v4.config import ModelConfig
from latent_working_memory.v4.engine import initialize_device
from latent_working_memory.v4.model import StreamingMemoryLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = initialize_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig(**checkpoint["run"]["model"])
    config = replace(
        config, revision=checkpoint["run"]["resolved_model_revision"] or config.revision
    )
    tokenizer = AutoTokenizer.from_pretrained(
        Path(checkpoint["run"]["training"]["output_dir"]) / "tokenizer"
    )
    model = StreamingMemoryLM(config).to(device).eval()
    model.load_trainable_state_dict(checkpoint["memory"])
    prompt = args.prompt_file.read_text()
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids:
        raise ValueError("prompt must tokenize to at least one token")
    with torch.no_grad():
        output = model.generate(
            torch.tensor(ids, dtype=torch.long, device=device),
            args.max_new_tokens,
            tokenizer.eos_token_id,
        )
    generated = output[len(ids) :].tolist()
    result = {
        "prompt": prompt,
        "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
        "generated_token_ids": generated,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
