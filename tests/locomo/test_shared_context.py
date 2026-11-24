"""关键回归：同正文批量回答等同独立计算，缓存不被前题污染。"""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json
import os
import tempfile

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GenerationConfig, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from benchmarks.data import BenchmarkSample, locomo_question_prompt
from benchmarks.protocol import encode_prompt
from benchmarks.locomo.engine import evaluate_conversations, shard_conversations
from benchmarks.locomo.records import read_progress
from latent_working_memory.full_context.locomo import FullContextLoCoMo
from latent_working_memory.v3_1_1.config import ComputeConfig, ModelConfig, ModelStructureConfig, ModelLoraConfig
from latent_working_memory.v3_1_1.evaluation.locomo_adapter import MemoryAdapter
from latent_working_memory.v3_1_1.modeling.compressor import MemoryModel
from latent_working_memory.v3_1_1.rollout import execute_batch


@pytest.fixture(scope="module")
def setup():
    torch.set_num_threads(1)
    torch.manual_seed(311)
    device = torch.device(os.environ.get("LOCOMO_TEST_DEVICE", "cpu"))
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=2048, eos_token_id=1, pad_token_id=0, attention_dropout=0,
        ))
        model.save_pretrained(folder)
        vocab = {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, **{f"t{i}": i for i in range(3, 64)}}
        backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
                                           eos_token="[EOS]", unk_token="[UNK]")
        tokenizer.chat_template = (
            "{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' ' }}{% endfor %}"
            "{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
        )
        tokenizer.save_pretrained(folder)
        history = "t3 t4 t5 t6 t7 t8 " * 8
        samples = [BenchmarkSample(
            f"c1/{i}", history + locomo_question_prompt(text, 4), ("t3",), 5,
            {"conversation_id": "c1", "question": text, "category": 4, "category_name": "single-hop"},
        ) for i, text in enumerate(("t9", "t10", "t11 t12"))]
        yield folder, model.eval().to(device), tokenizer, samples, device


def collect(adapter, samples, batch_size):
    result = {}
    evaluate_conversations(samples, adapter, batch_size,
                           lambda sample, answer, stats: result.update({sample.id: answer["generated_token_ids"]}))
    return result


def test_full_context_matches_independent_generation(setup):
    _, model, tokenizer, samples, device = setup
    generation = GenerationConfig(do_sample=False, eos_token_id=1, pad_token_id=0, use_cache=True)
    adapter = FullContextLoCoMo(tokenizer, model, generation, 2048, device)
    expected = {}
    with torch.no_grad():
        for sample in samples:
            ids = torch.tensor([encode_prompt(sample, tokenizer, 2048, "locomo", True)], device=device)
            output = model.generate(ids, generation_config=generation, max_new_tokens=sample.max_new_tokens,
                                    use_model_defaults=False)[0, ids.shape[1]:].tolist()
            expected[sample.id] = output
    assert collect(adapter, samples, 2) == expected
    assert collect(adapter, list(reversed(samples)), 1) == expected
    prepared = adapter.prepare_context("t3 t4 " * 8, samples)
    original = [layer.keys.clone() for layer in prepared.state[0].layers]
    adapter.answer_batch(prepared.state, prepared.questions[:2], lambda *args: None)
    assert all(torch.equal(old, layer.keys) for old, layer in zip(original, prepared.state[0].layers))


def test_memory_reader_matches_full_reprefill(setup):
    folder, _, _, samples, device = setup
    model_config = ModelConfig(
        structure=ModelStructureConfig(memory_slots=4, pending_stride=16, pending_size=16,
                                       recent_size=4, boundary_hidden_size=8),
        lora=ModelLoraConfig(lora_rank=2, lora_alpha=4),
    )
    model = MemoryModel(str(folder), asdict(model_config), asdict(ComputeConfig()), device).eval()
    config = SimpleNamespace(model=model_config, recent_size=lambda: 4,
                             training=SimpleNamespace(objective=SimpleNamespace(memory_cost_power=1)))
    adapter = MemoryAdapter(model, config)
    history = samples[0].prompt.removesuffix(locomo_question_prompt(samples[0].metadata["question"], 4))
    with torch.no_grad(), model.forward_context():
        path = execute_batch(model, [adapter.encoder.context(history)], 16, pending_size=16,
                             recent_size=4, collect_boundary=True)[0]
        prepared = adapter.prepare_context(history, samples)
        expected = {}
        for question in prepared.questions:
            tokens = model.generate(path.blocks, path.retained_ids + tuple(question.inputs),
                                    question.sample.max_new_tokens, adapter.encoder.reader_prefix_ids)
            expected[question.sample.id] = list(tokens)
    assert collect(adapter, samples, 2) == expected
    assert collect(adapter, list(reversed(samples)), 1) == expected


def test_resume_keeps_conversation_shards_and_repairs_last_line(tmp_path, setup):
    _, _, _, samples, _ = setup
    other = [BenchmarkSample("c2/0", samples[0].prompt, ("t3",), 5,
                             {**samples[0].metadata, "conversation_id": "c2"})]
    all_samples = samples + other
    progress = tmp_path / "progress"
    progress.mkdir()
    row = {"id": samples[1].id, "metadata": samples[1].metadata, "references": ["t3"]}
    path = progress / "worker-0.jsonl"
    path.write_text(json.dumps(row) + "\n" + '{"id":')
    records = read_progress(tmp_path, all_samples, repair=True)
    assert list(records) == [samples[1].id]
    assert path.read_text().endswith("\n")
    pending = [sample for sample in all_samples if sample.id not in records]
    shards = shard_conversations(pending, 4)
    assert sorted(len(shard) for shard in shards) == [1, 2]
    assert all(len({s.metadata["conversation_id"] for s in shard}) == 1 for shard in shards)
