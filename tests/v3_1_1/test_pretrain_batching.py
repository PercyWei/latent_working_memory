"""真实 Qwen3 上比较预训练批量前向与逐条前向的损失、梯度和更新。"""

from dataclasses import asdict

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from latent_working_memory.v3_1_1.config import (
    ComputeConfig,
    ExperimentConfig,
    ModelConfig,
    ModelLoraConfig,
    ModelStructureConfig,
    OptimizerConfig,
    SchedulerConfig,
    SwanLabConfig,
    TrainingConfig,
    TrainingDataConfig,
    TrainingLoopConfig,
)
from latent_working_memory.v3_1_1.data import PretrainExample
from latent_working_memory.v3_1_1.modeling.compressor import MemoryModel
from latent_working_memory.v3_1_1.training.engine import TrainingEngine, document_loss


@pytest.fixture(scope="module")
def tiny_source(tmp_path_factory):
    folder = tmp_path_factory.mktemp("qwen3-pretrain-batch")
    torch.manual_seed(310)
    Qwen3ForCausalLM(Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=1024, eos_token_id=1,
        pad_token_id=0, attention_dropout=0.0,
    )).save_pretrained(folder)
    vocabulary = {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2,
                  **{f"t{i}": i for i in range(3, 64)}}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", eos_token="[EOS]",
        unk_token="[UNK]",
    ).save_pretrained(folder)
    return folder


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_models(folder, gradient_checkpointing):
    config = ExperimentConfig(
        stage="pretrain", model_name_or_path=str(folder),
        model=ModelConfig(
            structure=ModelStructureConfig(memory_slots=4),
            lora=ModelLoraConfig(lora_rank=2, lora_alpha=4),
        ),
        compute=ComputeConfig(gradient_checkpointing=gradient_checkpointing),
        training=TrainingConfig(
            data=TrainingDataConfig(dataset_dir=str(folder)),
            loop=TrainingLoopConfig(
                micro_batch_size=3, gradient_accumulation_steps=2, max_steps=3,
            ),
            optimizer=OptimizerConfig(gradient_clip=0.5),
            scheduler=SchedulerConfig(warmup_steps=0),
            swanlab=SwanLabConfig(tracking=False),
        ),
    )
    torch.manual_seed(311)
    expected = MemoryModel(
        config.model_name_or_path, asdict(config.model), asdict(config.compute),
        torch.device("cpu"),
    )
    with torch.no_grad():
        for name, parameter in expected.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 0.025)
    actual = MemoryModel(
        config.model_name_or_path, asdict(config.model), asdict(config.compute),
        torch.device("cpu"),
    )
    actual.load_state_dict(expected.state_dict())
    return expected, actual, config


def example(index, task, length):
    text = tuple(3 + (index + offset) % 61 for offset in range(length))
    target = text + (1,) if task == "ae" else (19, 23, 29, 1)
    return PretrainExample(
        id=f"sample-{index}", document_id=f"sample-{index}",
        input_ids=text, target_ids=target, task=task,
        prompt_ids=tuple(range(3, 5 + index % 3)),
    )


def gradients(model):
    return {name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in model.named_parameters() if parameter.requires_grad}


def compare_gradients(expected, actual):
    assert expected.keys() == actual.keys()
    for name in expected:
        left, right = expected[name], actual[name]
        assert (left is None) == (right is None), name
        if left is not None:
            assert torch.isfinite(right).all(), name
            torch.testing.assert_close(right, left, rtol=3e-4, atol=3e-6, msg=name)


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_batch_loss_and_gradients_match_single_examples(tiny_source, gradient_checkpointing):
    expected, actual, _ = make_models(tiny_source, gradient_checkpointing)
    rows = [example(0, "ae", 279), example(1, "lm", 7), example(2, "ae", 13)]
    frozen = {name: parameter.detach().clone() for name, parameter in actual.named_parameters()
              if not parameter.requires_grad}
    writer_shapes, reader_shapes, reader_layers = [], [], []

    def writer_call(module, args, kwargs):
        writer_shapes.append(tuple(kwargs["inputs_embeds"].shape))

    def reader_call(module, args, kwargs):
        reader_shapes.append(tuple(kwargs["inputs_embeds"].shape))
        assert module.training

    handles = [
        actual.writer.get_base_model().model.register_forward_pre_hook(writer_call, with_kwargs=True),
        actual.decoder.model.register_forward_pre_hook(reader_call, with_kwargs=True),
        actual.decoder.model.layers[0].register_forward_pre_hook(
            lambda module, args: reader_layers.append(module.training),
        ),
    ]
    old_memories = [expected.compress(row.input_ids) for row in rows]
    memories = list(actual.compress_batch([row.input_ids for row in rows]))
    for memory in old_memories + memories:
        memory.retain_grad()
    old_losses = torch.stack([
        expected.answer_nll((memory,), row.prompt_ids, row.target_ids)
        for memory, row in zip(old_memories, rows, strict=True)
    ])
    losses = actual.answer_nll_batch(
        [(memory,) for memory in memories],
        [row.prompt_ids for row in rows], [row.target_ids for row in rows],
    )
    torch.testing.assert_close(losses, old_losses, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(torch.stack(memories), torch.stack(old_memories),
                               rtol=3e-4, atol=3e-6)
    old_losses.sum().backward()
    losses.sum().backward()
    for handle in handles:
        handle.remove()
    assert writer_shapes[0][0] == reader_shapes[0][0] == len(rows)
    assert len(reader_layers) == (2 if gradient_checkpointing else 1)
    assert all(reader_layers)
    assert actual.training and actual.decoder.training
    for old_memory, memory in zip(old_memories, memories, strict=True):
        assert memory.grad is not None and memory.grad.abs().sum() > 0
        torch.testing.assert_close(memory.grad, old_memory.grad, rtol=3e-4, atol=3e-6)
    compare_gradients(gradients(expected), gradients(actual))
    for name, parameter in actual.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None
            torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
    actual.eval()
    with torch.no_grad():
        evaluation_losses = actual.answer_nll_batch(
            [(memory,) for memory in memories],
            [row.prompt_ids for row in rows], [row.target_ids for row in rows],
        )
    assert not actual.decoder.training and not evaluation_losses.requires_grad
    torch.testing.assert_close(evaluation_losses, losses.detach(), rtol=2e-5, atol=2e-6)


def sequential_update(engine, rows):
    """逐条目标作为独立参考，仅按实际样本数归约一次。"""
    engine.model.train()
    engine.optimizer.zero_grad(set_to_none=True)
    losses = []
    for row in rows:
        with engine.model.forward_context():
            loss, _ = document_loss(engine.model, row, engine.experiment)
        losses.append(float(loss.detach()))
        loss.backward()
    for parameter in engine.parameters:
        if parameter.grad is not None:
            parameter.grad.div_(len(rows))
    raw_gradients = gradients(engine.model)
    torch.nn.utils.clip_grad_norm_(engine.parameters, engine.config.optimizer.gradient_clip)
    engine.optimizer.step()
    engine.scheduler.step()
    return sum(losses) / len(rows), raw_gradients


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_sorted_engine_accumulation_and_single_tail_match_sequential(
    tiny_source, gradient_checkpointing, monkeypatch,
):
    expected, actual, config = make_models(tiny_source, gradient_checkpointing)
    reference, engine = TrainingEngine(expected, config), TrainingEngine(actual, config)
    rows = [example(index, task, length) for index, (task, length) in enumerate(
        [("ae", 13), ("ae", 7), ("lm", 8), ("lm", 19), ("ae", 12), ("lm", 5)],
    )]
    captured, shapes, compressed = [], [], []
    synchronize = engine._synchronize_gradients
    compress = actual.compress_batch

    def capture(samples):
        synchronize(samples)
        captured.append(gradients(engine.model))

    def record_compression(input_ids):
        compressed.append(list(input_ids))
        return compress(input_ids)

    monkeypatch.setattr(engine, "_synchronize_gradients", capture)
    monkeypatch.setattr(actual, "compress_batch", record_compression)
    handle = actual.writer.get_base_model().model.register_forward_pre_hook(
        lambda module, args, kwargs: shapes.append(kwargs["inputs_embeds"].shape[0]),
        with_kwargs=True,
    )
    tail = example(8, "ae", 9)
    cases = (
        (rows, [[rows[0], rows[4]], [rows[3]], [rows[1]], [rows[2], rows[5]]]),
        ([tail], [[tail]]),
    )
    for batch, groups in cases:
        original = list(batch)
        compressed.clear()
        loss, raw_gradients = sequential_update(reference, batch)
        metrics = engine.train_batch(batch, None)
        assert batch == original
        assert compressed == [[row.input_ids for row in group] for group in groups]
        assert metrics["samples"] == len(batch)
        assert metrics["loss"] == pytest.approx(loss, rel=2e-5, abs=2e-6)
        assert metrics["lm_fraction"] == sum(row.task == "lm" for row in batch) / len(batch)
        compare_gradients(raw_gradients, captured[-1])
        for (name, left), (_, right) in zip(
            expected.named_parameters(), actual.named_parameters(), strict=True,
        ):
            torch.testing.assert_close(right, left, rtol=3e-4, atol=3e-6, msg=name)
        assert engine.scheduler.state_dict() == reference.scheduler.state_dict()
    handle.remove()
    assert 2 in shapes and shapes[-1] == 1
