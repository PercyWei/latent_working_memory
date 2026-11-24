"""用官方 ICAE v2 预训练或指令微调权重评估完整 LoCoMo 对话。"""

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from peft import LoraConfig
from safetensors.torch import load_file

from latent_working_memory.benchmarks.data import load_samples
from latent_working_memory.prompts import QA_SYSTEM_PROMPT
from reproductions.evaluation_runtime import ROOT
from reproductions.icae.source.icae.code.icae_v2.modeling_icae_multi_span import (
    ICAE,
    ModelArguments,
    TrainingArguments,
)
from reproductions.soft_context_locomo import add_evaluation_arguments, checkpoint_identity, run_evaluation


UPSTREAM_COMMIT = "469a46886a92dd5e76b2d12a8bac0fb7ed7d4cdd"
CHECKPOINT_FILES = {
    "pretrained": "mistral_7b_pretrained_icae.safetensors",
    "finetuned": "mistral_7b_ft_icae.safetensors",
}


class ICAELoCoMo:
    def __init__(self, model, stage):
        self.model = model
        self.stage = stage
        self.decoder = model.icae.get_base_model()
        self.device = model.memory_token_embed.weight.device
        self.window = self.decoder.config.max_position_embeddings
        # Official Mistral ICAE frame: BOS [INST] ... [/INST]. Mistral v0.2 has
        # no system role, so the shared system instruction opens the user turn.
        self.prefix_ids = [1, 733, 16289, 28793] + model.tokenizer.encode(
            QA_SYSTEM_PROMPT + "\n\n", add_special_tokens=False
        )

    def token_tensor(self, ids):
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    @torch.inference_mode()
    def encode_history(self, text):
        body_ids = self.model.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        ids = [self.model.bos_id] + body_ids
        memory_tokens = self.model.compute_num_segments(len(ids)) * self.model.mem_size
        prefix_length = len(self.prefix_ids) + memory_tokens
        if prefix_length >= self.window:
            raise ValueError(f"ICAE reader prefix requires {prefix_length} tokens, window {self.window}")
        slots = self.model._compress(self.token_tensor(ids))
        prefix = self.model.tokens_to_embeddings(self.token_tensor(self.prefix_ids))
        embeddings = torch.cat((prefix, slots.to(prefix).unsqueeze(0)), dim=1)
        with self.model.icae.disable_adapter():
            # The base model returns the legacy tuple cache in Transformers 4.40.
            # Per-question extensions allocate new KV tensors and preserve this prefix.
            cache = self.decoder.model(inputs_embeds=embeddings, use_cache=True).past_key_values
        stats = {"context_tokens": len(body_ids), "memory_tokens": memory_tokens, "raw_history_tokens": 0}
        return (cache, prefix_length), stats

    @torch.inference_mode()
    def answer(self, memory, question_prompt, max_new_tokens):
        cache, prefix_length = memory
        ids = self.model.tokenizer.encode(question_prompt, add_special_tokens=False)
        if self.stage == "finetuned":
            ids = [self.model.ft_token_id] + ids
        ids += [733, 28748, 16289, 28793]  # [/INST] in the official release.
        total_length = prefix_length + len(ids) + max_new_tokens
        if total_length > self.window:
            raise ValueError(f"ICAE QA requires {total_length} tokens, window {self.window}")
        inputs = self.model.tokens_to_embeddings(self.token_tensor(ids))
        generated = []
        with self.model.icae.disable_adapter():
            for _ in range(max_new_tokens):
                outputs = self.decoder(inputs_embeds=inputs, past_key_values=cache, use_cache=True)
                # Added padding/memory/operation rows are never answer tokens.
                token = outputs.logits[:, -1, : self.model.vocab_size - 1].argmax(dim=-1)
                token_id = token.item()
                if token_id == self.model.eos_id:
                    break
                generated.append(token_id)
                inputs = self.decoder.get_input_embeddings()(token.unsqueeze(1))
                cache = outputs.past_key_values
        return self.model.tokenizer.decode(generated, skip_special_tokens=True).strip()


def load_adapter(model_path, checkpoint, stage):
    if not torch.cuda.is_available():
        raise RuntimeError("ICAE v2 evaluation requires CUDA and FlashAttention 2")
    model_args = ModelArguments(model_name_or_path=str(model_path), lora_r=512, train=False)
    inference_args = TrainingArguments(
        output_dir=str(checkpoint.parent), bf16=True, fixed_mem_size=128, mean_compression_rate=4,
        report_to=[],
    )
    lora = LoraConfig(r=512, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = ICAE(model_args, inference_args, lora)
    # Official safetensors contain the complete writer LoRA and slot embeddings,
    # while all frozen weights come from the base model directory.
    weights = load_file(str(checkpoint))
    expected = {name for name, param in model.named_parameters() if param.requires_grad}
    if set(weights) != expected:
        raise ValueError(
            f"ICAE adapter keys differ: missing={sorted(expected - weights.keys())}, "
            f"unexpected={sorted(weights.keys() - expected)}"
        )
    model.load_state_dict(weights, strict=False)
    return ICAELoCoMo(model.to("cuda").eval(), stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_evaluation_arguments(parser)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/reproductions/icae",
                        help="评估产物父目录，默认 artifacts/reproductions/icae")
    parser.add_argument("--output-id", help="运行目录名，默认 <checkpoint 文件主名>_<YYYYMMDD-HHMM>")
    parser.add_argument("--model-path", type=Path, required=True, help="本地 Mistral-7B-Instruct-v0.2")
    parser.add_argument("--checkpoint", type=Path, required=True, help="官方 v2 safetensors 文件")
    parser.add_argument("--stage", choices=CHECKPOINT_FILES, required=True, help="权重的训练阶段")
    args = parser.parse_args()
    model_path = args.model_path.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().absolute()
    if args.output_id is None:
        if args.resume:
            parser.error("--resume requires the original --output-id")
        current = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M")
        args.output_id = f"{checkpoint.stem}_{current}"
    if not args.output_id.strip() or args.output_id in {".", ".."} or any(
        separator in args.output_id for separator in ("/", "\\")
    ):
        parser.error("--output-id must be one nonempty directory name")
    if checkpoint.name != CHECKPOINT_FILES[args.stage]:
        parser.error(f"--stage {args.stage} requires {CHECKPOINT_FILES[args.stage]}")
    if not model_path.is_dir() or not checkpoint.is_file():
        parser.error("--model-path must be a local directory and --checkpoint must be a local file")
    settings = {
        "upstream_commit": UPSTREAM_COMMIT,
        "model_path": str(model_path),
        "model_files": checkpoint_identity(model_path),
        "checkpoint": str(checkpoint),
        "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
        "stage": args.stage,
        "slots_per_segment": 128,
        "mean_compression_rate": 4,
        "reader_template": "Mistral [INST], shared QA instruction, slots, optional trained [FT], question, [/INST]",
        "dtype": "bfloat16",
        "attention_implementation": "flash_attention_2",
    }
    run_dir = args.output_root.expanduser().resolve() / "eval" / args.output_id
    args.output_dir = run_dir / "locomo"
    samples = load_samples("locomo", args.benchmark_data)
    if not args.resume:
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            parser.error(f"{run_dir} already exists; select a new --output-id or use --resume")
    print(f"[evaluate] output-id={args.output_id} | LoCoMo: {args.output_dir}", flush=True)
    return run_evaluation(
        args, samples, "icae", settings, lambda: load_adapter(model_path, checkpoint, args.stage)
    )


if __name__ == "__main__":
    main()
