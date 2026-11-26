"""ICAE v2 的正文压缩、共享前缀和问答批处理。"""

from pathlib import Path
import torch
from transformers import AutoConfig, AutoTokenizer

from benchmarks.documents import PreparedConversation, PreparedQuestion
from benchmarks.locomo.reader import read_prefixes
from benchmarks.locomo.records import file_identity
from latent_working_memory.prompts import QA_SYSTEM_PROMPT
from reproductions.icae.formatting import icae_input_format
from .model import (
    UPSTREAM_COMMIT,
    load_adapter as load_official,
)


class ICAEAdapter:
    def __init__(self, adapter):
        self.adapter = adapter

    def prepare_contexts(self, documents):
        encoded = self.adapter.encode_histories([history for history, _ in documents])
        contexts = []
        for (history, questions), (state, stats) in zip(documents, encoded, strict=True):
            prepared = []
            for sample in questions:
                ids = self.adapter.question_ids(sample.prompt.removeprefix(history))
                if state[1] + len(ids) + sample.max_new_tokens > self.adapter.window:
                    raise ValueError(f"ICAE reader exceeds position window: {sample.id}")
                prepared.append(PreparedQuestion(sample, ids))
            contexts.append(PreparedConversation(state, prepared, stats))
        return contexts

    @torch.inference_mode()
    def answer_batch(self, states, questions, complete):
        model = self.adapter.model
        embeddings = [model.tokens_to_embeddings(self.adapter.token_tensor(q.inputs))[0] for q in questions]

        def completed(index, result):
            tokens = result["generated_token_ids"]
            result.update(prediction=model.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                          generated_tokens=len(tokens), decode_batch_size=len(questions))
            complete(questions[index], result)

        caches, lengths = zip(*states)
        with model.icae.disable_adapter():
            read_prefixes(self.adapter.decoder, caches, lengths, embeddings,
                          [q.sample.max_new_tokens for q in questions], self.adapter.eos_id,
                          completed, vocab_size=model.vocab_size - 1)


def add_arguments(parser):
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage", choices=("pretrained", "finetuned"), required=True)
    parser.add_argument("--fixed-mem-size", type=int, default=128,
                        help="每块记忆槽数，取正整数，默认 128，须与 checkpoint 匹配")
    parser.add_argument("--mean-compression-rate", type=int, default=4,
                        help="官方平均压缩率参数，取正整数，默认 4")


def configuration(args, system_prompt=QA_SYSTEM_PROMPT):
    if args.fixed_mem_size <= 0:
        raise ValueError("--fixed-mem-size must be positive")
    if args.mean_compression_rate <= 0:
        raise ValueError("--mean-compression-rate must be positive")
    checkpoint = args.checkpoint.expanduser().absolute()
    # 官方 v2 在基座词表后依次添加 PAD、memory slots、AE、LM、FT。
    base_config = AutoConfig.from_pretrained(args.model_path.expanduser(), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path.expanduser(), use_fast=False, local_files_only=True)
    input_format = icae_input_format(
        tokenizer, args.stage, base_config.vocab_size + 1 + args.fixed_mem_size + 2, system_prompt,
    )
    return {"model_path": str(args.model_path.expanduser().resolve()), "checkpoint": str(checkpoint),
            "stage": args.stage, "fixed_mem_size": args.fixed_mem_size,
            "mean_compression_rate": args.mean_compression_rate,
            "upstream_commit": UPSTREAM_COMMIT,
            "input_format": input_format,
            "model_files": file_identity(args.model_path), "checkpoint_files": file_identity(checkpoint)}


def load_adapter(settings):
    adapter = load_official(Path(settings["model_path"]), Path(settings["checkpoint"]),
                            settings["stage"], settings["mean_compression_rate"], settings["fixed_mem_size"],
                            settings.get("system_prompt", QA_SYSTEM_PROMPT))
    if adapter.input_format != settings["input_format"]:
        raise ValueError("loaded ICAE input format differs from the evaluation plan")
    return ICAEAdapter(adapter)


def load_longmemeval_adapter(settings):
    return load_adapter(settings)


def summarize_metrics(records, conversations):
    return {
        "mean_memory_tokens": sum(s["memory_tokens"] for s in conversations.values()) / len(conversations),
        "mean_retention_ratio": sum(s["memory_tokens"] / s["context_tokens"]
                                    for s in conversations.values()) / len(conversations),
    }
