"""用公开的 Llama-2 AutoCompressor 在完整对话上做 LoCoMo QA。"""

import argparse
import subprocess
from pathlib import Path

import torch
from auto_compressor import LlamaAutoCompressorModel
from torch.nn.functional import pad
from transformers import AutoTokenizer

from latent_working_memory.benchmarks.data import load_samples
from latent_working_memory.prompts import QA_SYSTEM_PROMPT
from reproductions.soft_context_locomo import (
    add_evaluation_arguments,
    checkpoint_identity,
    run_evaluation,
)

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source"
OFFICIAL_DIR = SOURCE_ROOT / "AutoCompressors"


class LoCoMoAutoCompressor:
    def __init__(self, model_path, segment_length):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model, info = LlamaAutoCompressorModel.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, output_loading_info=True,
        )
        if info["missing_keys"] or info["unexpected_keys"] or info["mismatched_keys"]:
            raise ValueError(f"incompatible AutoCompressor state dict: {info}")
        if not model.config.accumulate_summary or model.config.summary_length != 50:
            raise ValueError("expected the Llama-2-7b-6k checkpoint with 50 accumulated summary vectors per segment")
        self.model = model.eval().cuda()
        self.segment_length = segment_length
        self.window = model.config.max_position_embeddings

    @torch.inference_mode()
    def encode_history(self, text):
        input_ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        summary = None
        slots = self.model.config.summary_length
        for index, segment in enumerate(input_ids.split(self.segment_length, dim=1)):
            write_length = index * slots + segment.size(1) + slots
            if write_length > self.window:
                raise ValueError(f"AutoCompressor write needs {write_length} positions (window {self.window})")
            summary = self.model(
                segment.to(self.model.device), softprompt=summary, output_softprompt=True, use_cache=False,
            ).softprompt

        # Llama 版将 softprompt 当作连续的输入嵌入。把 system 放在槽之前，
        # 各问题复用同一个只读前缀；写入阶段仅接触正文。
        system_ids = self.tokenizer(QA_SYSTEM_PROMPT + "\n\n", return_tensors="pt").input_ids
        system = self.model.get_input_embeddings()(system_ids.to(self.model.device))
        prefix = torch.cat((system, summary), dim=1)
        if prefix.size(1) > self.window:
            raise ValueError(f"AutoCompressor reader prefix needs {prefix.size(1)} positions (window {self.window})")
        prefix_cache = self.model.model(inputs_embeds=prefix, use_cache=True).past_key_values
        return list(prefix_cache), {
            "context_tokens": input_ids.size(1),
            "memory_tokens": summary.size(1),
            "raw_history_tokens": 0,
        }

    @torch.inference_mode()
    def answer(self, memory, question_prompt, max_new_tokens):
        question = self.tokenizer(
            question_prompt + "Answer:", add_special_tokens=False, return_tensors="pt",
        ).input_ids.to(self.model.device)
        # 上游每层缓存为 (KV 存储, 有效长度)。预留本题所需空间，使官方
        # 注意力能一次写入整道问题。每道题从相同的前缀长度起读，覆盖旧答案的后缀。
        prefix_length = memory[0][1]
        read_length = prefix_length + question.size(1) + max_new_tokens
        if read_length > self.window:
            raise ValueError(f"AutoCompressor reader needs {read_length} positions (window {self.window})")
        extra = read_length - memory[0][0].size(1)
        if extra > 0:
            memory[:] = [(pad(kv, (0, 0, 0, 0, 0, 0, 0, extra)), prefix_length) for kv, _ in memory]

        inputs = question
        cache = memory
        generated = []
        for _ in range(max_new_tokens):
            output = self.model.model(input_ids=inputs, past_key_values=cache, use_cache=True)
            inputs = self.model.lm_head(output.last_hidden_state[:, -1:]).argmax(dim=-1)
            token = inputs.item()
            if token == self.tokenizer.eos_token_id:
                break
            generated.append(token)
            cache = output.past_key_values
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_evaluation_arguments(parser)
    parser.add_argument("--output-dir", type=Path, required=True, help="本次 LoCoMo 运行的结果目录")
    parser.add_argument("--model-path", type=Path, required=True,
                        help="princeton-nlp/AutoCompressor-Llama-2-7b-6k 的完整本地目录")
    parser.add_argument("--segment-length", type=int, default=2048,
                        help="每次写入的最大原文 token 数，默认 2048，每段输出 50 个槽")
    args = parser.parse_args()
    if args.segment_length <= 0:
        parser.error("--segment-length must be positive")
    revision = subprocess.check_output(["git", "-C", str(OFFICIAL_DIR), "rev-parse", "HEAD"], text=True).strip()
    if revision != (SOURCE_ROOT / "revision").read_text().strip():
        parser.error("AutoCompressors checkout differs from source/revision")
    model_path = args.model_path.expanduser().resolve(strict=True)
    settings = {
        "model_path": str(model_path), "model_files": checkpoint_identity(model_path),
        "upstream_revision": revision,
        "segment_length": args.segment_length, "summary_length": 50,
        "weights": "princeton-nlp/AutoCompressor-Llama-2-7b-6k",
        "prompt": "BOS + QA system + accumulated summary vectors + task and question + Answer:",
        "dtype": "bfloat16",
    }
    samples = load_samples("locomo", args.benchmark_data)
    run_evaluation(
        args, samples, "autocompressors", settings,
        lambda: LoCoMoAutoCompressor(model_path, args.segment_length),
    )


if __name__ == "__main__":
    main()
