"""不依赖下载的真实 Qwen3 与小型 teacher 缓存。"""

import math
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from latent_working_memory.v4_1.config import (
    ExperimentConfig, ModelConfig, TeacherConfig, TrainingConfig,
)
from latent_working_memory.v4_1.data import Episode
from latent_working_memory.v4_1.teacher.cache import build_cache_identity, initialize_cache
from latent_working_memory.v4_1.teacher import attention
from latent_working_memory.v4_1.teacher.prepare import TeacherPreparer, plan_preparation
from latent_working_memory.v4_1.teacher.replay import load_teacher


@pytest.fixture
def reference_teacher_attention(monkeypatch):
    """CPU 小模型只替换算子，以相同布局验证生产 wrapper 与采样流程。"""
    def single(query, keys, values, kv_layout="NHD", causal=False, sm_scale=None, **kwargs):
        if kv_layout == "HND":
            keys, values = keys.transpose(0, 1), values.transpose(0, 1)
        groups = query.shape[1] // keys.shape[1]
        keys, values = (tensor.repeat_interleave(groups, dim=1) for tensor in (keys, values))
        scores = torch.einsum("qhd,khd->hqk", query.float(), keys.float()) * sm_scale
        if causal:
            visible = torch.arange(len(keys))[None] <= (
                torch.arange(len(query))[:, None] + len(keys) - len(query)
            )
            scores = scores.masked_fill(~visible, -torch.inf)
        output = torch.einsum("hqk,khd->qhd", scores.softmax(-1), values.float())
        return output.to(query.dtype), scores.logsumexp(-1).transpose(0, 1) / math.log(2)

    class Ragged:
        def __init__(self, workspace, kv_layout, backend):
            self.kv_layout = kv_layout

        def plan(self, qo_indptr, kv_indptr, num_heads, num_kv_heads, head_dim, **options):
            self.qo, self.kv, self.options = qo_indptr.tolist(), kv_indptr.tolist(), options

        def run(self, query, keys, values, return_lse):
            results = [single(query[ql:qr], keys[kl:kr], values[kl:kr],
                              kv_layout=self.kv_layout, **self.options)
                       for ql, qr, kl, kr in zip(self.qo, self.qo[1:], self.kv, self.kv[1:])]
            return tuple(torch.cat(parts) for parts in zip(*results))

    def merge(output_a, mass_a, output_b, mass_b):
        masses = torch.stack((mass_a, mass_b)) * math.log(2)
        weights = masses.softmax(0)[..., None]
        output = output_a.float() * weights[0] + output_b.float() * weights[1]
        return output.to(output_a.dtype), masses.logsumexp(0) / math.log(2)

    monkeypatch.setattr(attention, "single_prefill_with_kv_cache", single)
    monkeypatch.setattr(attention, "BatchPrefillWithRaggedKVCacheWrapper", Ragged)
    monkeypatch.setattr(attention, "merge_state", merge)


@pytest.fixture
def tiny_setup(tmp_path, reference_teacher_attention):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model_dir = tmp_path / "model"
    vocabulary = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    vocabulary.update({f"word{index}": index + 3 for index in range(20)})
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] + ': ' + message['content'] }}"
        "{% if message['role'] == 'assistant' %}{{ '<eos>' }}{% else %}{{ '\\n' }}"
        "{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    )
    tokenizer.save_pretrained(model_dir)
    backbone = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=256, eos_token_id=1, pad_token_id=0, bos_token_id=1,
    ))
    backbone.save_pretrained(model_dir)
    config = ExperimentConfig(
        model=ModelConfig(
            model_name_or_path=str(model_dir), backbone_dtype="float32",
            pending_size=8, recent_size=4, num_slots=4, memory_dim=8,
            decoder_hidden_dim=16, inner_steps=2, inner_lr_init=0.01,
            lora_rank=2, lora_alpha=2, am_std_floor=1.0,
            gradient_checkpointing=True, prefill_chunk_size=5, loss_chunk_size=2,
        ),
        teacher=TeacherConfig(num_prompts=2, response_max_tokens=3, pool_size=8, am_sample_size=4),
        training=TrainingConfig(
            teacher_cache_dir=str(tmp_path / "teacher"), output_dir=str(tmp_path / "run"),
            max_steps=2, warmup_steps=1, micro_batch_size=1,
            gradient_accumulation_steps=2, tracking="disabled",
        ),
    )
    teacher = load_teacher(config.model, torch.device("cpu"))
    cache = initialize_cache(config.training.teacher_cache_dir, build_cache_identity(config, tokenizer))
    episode = Episode("document:0", "document", tuple(range(3, 23)), (5, 6, 7), (8, 9, 1))
    preparer = TeacherPreparer(teacher, config, cache)
    jobs = plan_preparation({episode.document_id: episode.context_ids}, config, cache, 1)
    preparer.prepare_document(jobs[0][0])
    try:
        yield SimpleNamespace(
            config=config, tokenizer=tokenizer, teacher=teacher, cache=cache, episode=episode,
        )
    finally:
        torch.set_num_threads(threads)
