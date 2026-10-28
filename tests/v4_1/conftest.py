"""不依赖下载的真实 Qwen3 与小型 teacher 缓存。"""

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
from latent_working_memory.v4_1.teacher.prepare import TeacherPreparer
from latent_working_memory.v4_1.teacher.replay import load_teacher


@pytest.fixture
def tiny_setup(tmp_path):
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
            max_steps=2, warmup_steps=1, gradient_accumulation_steps=2, tracking="disabled",
        ),
    )
    teacher = load_teacher(config.model, torch.device("cpu"))
    cache = initialize_cache(config.training.teacher_cache_dir, build_cache_identity(config, tokenizer))
    episode = Episode("document:0", "document", tuple(range(3, 23)), (5, 6, 7), (8, 9, 1))
    preparer = TeacherPreparer(teacher, config, cache)
    preparer.prepare_document(episode.document_id, episode.context_ids)
    try:
        yield SimpleNamespace(
            config=config, tokenizer=tokenizer, teacher=teacher, cache=cache, episode=episode,
        )
    finally:
        torch.set_num_threads(threads)
