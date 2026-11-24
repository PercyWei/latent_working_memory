"""各方法共用的 prompt、数据身份和可对照计算设置。"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from benchmarks.data import (
    DEFAULT_RULER_LENGTHS as DEFAULT_RULER_LENGTHS,
    LOCOMO_COMMIT,
    RULER_COMMIT,
)
from latent_working_memory.prompts import QA_SYSTEM_PROMPT, qa_messages


def protocol(benchmark, lengths, memory, use_qa_system=False):
    """记录评估的输入、生成与评分协议，以及实际使用的 system 消息。"""
    if benchmark not in {"ruler", "locomo"}:
        raise ValueError("benchmark must be ruler or locomo")
    return {
        "decoding": "greedy; base model tokenizer chat template; enable_thinking=False",
        "memory": memory,
        "input": "complete benchmark prompt; no truncation or training length filtering",
        **({"system_prompt": QA_SYSTEM_PROMPT} if use_qa_system else {}),
        "generation_budget": "RULER task budget; LoCoMo 50 tokens",
        "reference": "scoring only; never passed to memory writes or generation",
        **(
            {"lengths": list(lengths), "answer_prefix": "omitted in chat protocol"}
            if benchmark == "ruler"
            else {"categories": [1, 2, 3, 4], "history_order": "chronological"}
        ),
    }


def data_identity(samples, data_dir, benchmark):
    """绑定实际读取的原生样本，保留目录来源但不靠路径判断内容一致。"""
    if benchmark not in {"ruler", "locomo"}:
        raise ValueError("benchmark must be ruler or locomo")
    fingerprint = hashlib.blake2b(digest_size=32)
    for sample in samples:
        fingerprint.update(
            json.dumps(asdict(sample), ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        fingerprint.update(b"\n")
    return {
        "directory": str(Path(data_dir).expanduser().resolve()),
        "source_commit": RULER_COMMIT if benchmark == "ruler" else LOCOMO_COMMIT,
        "samples": len(samples),
        "fingerprint": fingerprint.hexdigest(),
    }


def encode_prompt(sample, tokenizer, model_window, benchmark, use_qa_system=False):
    """使用指定的聊天协议编码完整题面，保留任务生成预算。"""
    if benchmark not in {"ruler", "locomo"}:
        raise ValueError("benchmark must be ruler or locomo")
    messages = (qa_messages(sample.prompt) if use_qa_system
                else [{"role": "user", "content": sample.prompt}])
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    limit = min(sample.metadata["length"], model_window) if benchmark == "ruler" else model_window
    if len(ids) + sample.max_new_tokens > limit:
        raise ValueError(
            f"{sample.id}: prompt ({len(ids)}) + generation budget "
            f"({sample.max_new_tokens}) exceeds limit {limit}"
        )
    return ids


def comparison_config(summary):
    """提取完整上下文与压缩方法都需一致的质量评估条件。"""
    return {
        "benchmark": summary["benchmark"],
        "model": {
            "model_name_or_path": summary["model"]["model_name_or_path"],
            "backbone_dtype": summary["model"]["backbone_dtype"],
        },
        "model_window": summary["model_window"],
        "data": {name: value for name, value in summary["data"].items() if name != "directory"},
        "protocol": {
            name: value for name, value in summary["protocol"].items() if name != "memory"
        },
    }
