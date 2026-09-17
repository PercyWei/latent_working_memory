"""固定采样、累计监督、递归梯度与 verl DDP 的真实小模型验证。"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from latent_working_memory.v2.compression import SlotCompression
from latent_working_memory.v2.memory_codec import CodecConfig, MemoryCodec
from latent_working_memory.v2 import memory_codec
from latent_working_memory.v2.pretrain.checkpoint import codec_state
from latent_working_memory.v2.pretrain.config import TrainingConfig
from latent_working_memory.v2.pretrain.data import (
    Trajectory,
)
from latent_working_memory.v2.pretrain.engine import ReconstructionEngine, initialize_device
from latent_working_memory.v2.pretrain.evaluation import evaluate
from latent_working_memory.v2.pretrain.objective import ReconstructionTask


def make_task(path, method="mean", checkpointing=False, feature_layer="last", dropout=0):
    config = CodecConfig(
        str(path),
        encoder_layers=2,
        alignment_layers=1,
        lora_rank=2,
        lora_alpha=4,
        lora_dropout=dropout,
        compression=method,
        spectral_bottleneck=8,
        gradient_checkpointing=checkpointing,
        feature_layer=feature_layer,
        attention_implementation="eager",
    )
    return ReconstructionTask(
        MemoryCodec(config),
        AutoTokenizer.from_pretrained(path),
        TrainingConfig(objective="ae_lm", global_batch_size=2),
    )


def example(lengths=(2, 3, 4)):
    ends, total = [], 0
    for length in lengths:
        total += length
        ends.append(total)
    return Trajectory("test", 0, torch.tensor([4, 5, 6, 7] * 10)[: total + 2], tuple(ends), 2)


def test_group_boundaries_and_weighted_initialization():
    hidden = torch.arange(35, dtype=torch.float32).reshape(7, 5).requires_grad_()
    expected = torch.stack([hidden[:2].mean(0), hidden[2:4].mean(0), hidden[4:].mean(0)])
    for method in ("mean", "weighted"):
        module = SlotCompression(5, method)
        actual = module(hidden, 3)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
    assert (hidden.grad.abs().sum(1) > 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA pooling repeatability")
def test_cuda_mean_pooling_is_repeatable_through_backward():
    device = initialize_device("cuda")
    hidden = torch.randn(1298, 2560, device=device, requires_grad=True)
    module = SlotCompression(2560, "mean")
    expected = module(hidden, 512)
    expected.square().sum().backward()
    gradient = hidden.grad.clone()
    for _ in range(5):
        hidden.grad = None
        actual = module(hidden, 512)
        actual.square().sum().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(hidden.grad, gradient, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["mean", "weighted", "spectral"])
def test_full_bptt_frozen_decoder_and_raw_update(tiny_base, method):
    task = make_task(tiny_base, method)
    task.codec.set_stage("multiround")
    row = example()
    memories, input_lengths = [], []
    original = task.codec.write

    def capture(previous, ids, capacity):
        input_lengths.append(len(ids))
        result = original(previous, ids, capacity)
        result.retain_grad()
        memories.append(result)
        return result

    task.codec.write = capture
    result = task(row)
    assert input_lengths == [2, 3, 4]
    assert [r["ae_tokens"] for r in result["rounds"]] == [3, 6, 10]
    # Use only a final read: first memory must still receive a gradient through later writes.
    task.codec.read_loss(memories[-1], task.ae_prompt, torch.tensor([4, 5, 2])).backward()
    assert memories[0].grad.abs().sum() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in task.codec.write_alignment.layers.parameters()
    )
    assert all(
        p.grad is None
        for name, p in task.codec.backbone.named_parameters()
        if "lora_" not in name or ".encoder." not in name
    )


@pytest.mark.parametrize("feature_layer", ["last", "mean"])
def test_checkpointed_read_matches_full_graph(tiny_base, feature_layer):
    reference = make_task(tiny_base, "weighted", False, feature_layer)
    actual = make_task(tiny_base, "weighted", True, feature_layer)
    actual.load_state_dict(reference.state_dict())
    reference(example())["loss"].backward()
    actual(example())["loss"].backward()
    for (name, a), (_, b) in zip(
        reference.named_parameters(), actual.named_parameters(), strict=True
    ):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=2e-6, msg=name)


def test_chunked_head_matches_native_causal_loss_and_gradients(tiny_base):
    reference = make_task(tiny_base, "weighted")
    actual = deepcopy(reference)
    actual.codec.config = replace(
        actual.codec.config, gradient_checkpointing=True, lm_head_chunk_size=2
    )

    def native_loss(memory, prompt, targets):
        codec = reference.codec
        embed = codec.backbone.get_input_embeddings()
        prefix = torch.cat((codec.align(memory), embed(prompt)))
        inputs = torch.cat((prefix, embed(targets)))[None]
        labels = torch.cat((targets.new_full((len(prefix),), -100), targets))[None]
        with codec.use_adapter(None) as backbone:
            return backbone(
                inputs_embeds=inputs,
                labels=labels,
                attention_mask=torch.ones(inputs.shape[:2], dtype=torch.bool),
                use_cache=False,
            ).loss

    reference.codec.read_loss = native_loss
    expected, observed = reference(example())["loss"], actual(example())["loss"]
    expected.backward()
    observed.backward()
    torch.testing.assert_close(observed, expected)
    for (name, a), (_, b) in zip(actual.named_parameters(), reference.named_parameters()):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=2e-6, msg=name)


@pytest.mark.parametrize("objective", ["ae", "ae_lm"])
@pytest.mark.parametrize("method", ["mean", "weighted"])
def test_microbatch_matches_serial_with_unequal_lengths_and_depths(tiny_base, objective, method):
    reference = make_task(tiny_base, method, checkpointing=True)
    reference.config = replace(reference.config, objective=objective, lm_weight=0.3)
    actual = deepcopy(reference)
    actual.config = replace(actual.config, micro_batch_size=2)
    actual.codec.config = replace(actual.codec.config, lm_head_chunk_size=2)
    rows = [example((2, 3, 4)), example((2, 3, 3, 4)), example((4, 2, 5))]
    engines = [ReconstructionEngine(task, "cpu") for task in (reference, actual)]
    for engine in engines:
        engine.initialize()
    expected, observed = [engine.step(rows) for engine in engines]
    assert observed["max_microbatch_size"] == 2
    assert observed["microbatches"] == 2 and observed["batched_samples"] == 2
    for key in ("loss", "ae", "lm"):
        if expected[key] is not None:
            assert observed[key] == pytest.approx(expected[key], rel=2e-5)
    for key in ("samples", "source_tokens", "target_tokens"):
        assert observed[key] == expected[key]
    for (name, a), (_, b) in zip(actual.named_parameters(), reference.named_parameters()):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=5e-4, atol=3e-7, msg=name)


def test_microbatch_encoder_budget_splits_before_allocating(tiny_base):
    task = make_task(tiny_base)
    task.config = replace(task.config, micro_batch_size=2, micro_batch_encoder_tokens=10)
    engine = ReconstructionEngine(task, "cpu")
    engine.initialize()
    metrics = engine.step([example((9,)), example((8,))])
    assert metrics["samples"] == metrics["microbatches"] == 2
    assert metrics["max_microbatch_size"] == 1 and metrics["batched_samples"] == 0


def test_microbatch_decoder_budget_splits_long_histories(tiny_base):
    task = make_task(tiny_base)
    task.config = replace(task.config, micro_batch_size=2, micro_batch_decoder_tokens=20)
    engine = ReconstructionEngine(task, "cpu")
    engine.initialize()
    metrics = engine.step([example(), example((4, 2, 5))])
    assert metrics["microbatches"] == 2 and metrics["max_microbatch_size"] == 1


@pytest.mark.parametrize("objective", ["ae", "ae_lm"])
def test_warmup_freezes_write_alignment_and_updates_read_alignment(tiny_base, objective):
    task = make_task(tiny_base)
    task.config = replace(task.config, objective=objective)
    task.codec.set_stage("warmup")
    engine = ReconstructionEngine(task, torch.device("cpu"))
    engine.initialize()
    before = deepcopy(codec_state(task.codec))
    metrics = engine.step([example((12,))])
    assert (metrics["lm"] is None) == (objective == "ae")
    assert all(p.grad is None for p in task.codec.write_alignment.parameters())
    assert any(
        not torch.equal(value, codec_state(task.codec)[name])
        for name, value in before.items()
        if name.startswith("read_alignment.layers.")
    )
    task.codec.initialize_write_alignment()
    for a, b in zip(
        task.codec.read_alignment.parameters(), task.codec.write_alignment.parameters(), strict=True
    ):
        torch.testing.assert_close(a, b)
        assert a is not b


def test_joint_encoder_is_causal_and_objective_averages_rounds(tiny_base):
    task = make_task(tiny_base)
    task.eval()
    encoded = []
    hook = task.codec.backbone.get_base_model().model.register_forward_hook(
        lambda module, args, output: encoded.append(output.last_hidden_state.detach().clone())
    )
    task.codec.write(None, torch.tensor([4, 5, 6]), 2)
    task.codec.write(None, torch.tensor([4, 5, 7]), 2)
    hook.remove()
    torch.testing.assert_close(encoded[0][:, :2], encoded[1][:, :2], rtol=0, atol=0)
    assert not torch.equal(encoded[0][:, 2], encoded[1][:, 2])
    task.config = replace(task.config, lm_weight=0.3)
    result = task(example())
    expected = sum(r["ae"] + 0.3 * r["lm"] for r in result["rounds"]) / 3
    assert result["loss"].item() == pytest.approx(expected)


def test_full_encoder_and_physically_independent_alignment(tiny_base):
    codec = MemoryCodec(
        CodecConfig(
            str(tiny_base),
            encoder_layers=None,
            alignment_layers=1,
            lora_rank=2,
            lora_alpha=4,
            gradient_checkpointing=False,
        )
    )
    decoder = codec.backbone.get_base_model().model
    assert len(decoder.layers) == 2
    assert "encoder" not in dict(codec.named_children())
    assert "decoder" not in dict(codec.named_children())
    assert not codec.backbone.is_gradient_checkpointing
    components = [codec.backbone, codec.read_alignment, codec.write_alignment]
    addresses = [{p.data_ptr() for p in module.parameters()} for module in components]
    for i, current in enumerate(addresses):
        assert all(not current.intersection(other) for other in addresses[i + 1 :])
    torch.testing.assert_close(
        codec.read_alignment.layers[0].self_attn.q_proj.weight,
        decoder.layers[0].self_attn.q_proj.base_layer.weight,
    )
    assert codec.read_alignment.embed_tokens is None
    assert not codec.read_alignment.norm.weight.requires_grad


def test_only_one_pretrained_load_and_no_meta_alignment_weights(tiny_base, monkeypatch):
    original = memory_codec.AutoModelForCausalLM.from_pretrained
    calls = []

    def load(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(memory_codec.AutoModelForCausalLM, "from_pretrained", load)
    codec = make_task(tiny_base).codec
    assert calls == [str(tiny_base)]
    assert set(codec.backbone.peft_config) == {"encoder", "decoder"}
    for module in (codec.read_alignment, codec.write_alignment):
        assert len(module.layers) == 1 and module.embed_tokens is None
        assert all(not p.is_meta for p in module.parameters())
        assert all(not p.is_meta for p in module.buffers())
        assert not any("lora_" in name for name, _ in module.named_parameters())


def test_adapter_selection_and_disabled_reader_are_isolated(tiny_base):
    task = make_task(tiny_base)
    task.eval()
    codec = task.codec
    ids = torch.tensor([4, 5, 6, 7])
    memory = torch.randn(2, codec.width)
    target = torch.tensor([5, 6, 2])
    initial_write = codec.write(None, ids, 2)
    initial_read = codec.read_loss(memory, task.ae_prompt, target)
    with torch.no_grad():
        for name, p in codec.backbone.named_parameters():
            if "lora_B" in name:
                p.fill_(0.1)
    assert not torch.allclose(codec.write(None, ids, 2), initial_write)
    torch.testing.assert_close(codec.read_loss(memory, task.ae_prompt, target), initial_read)
    flags = {name: p.requires_grad for name, p in codec.backbone.named_parameters()}
    with codec.use_adapter("decoder") as backbone:
        decoder_logits = backbone(input_ids=ids[None], use_cache=False).logits
    with codec.use_adapter(None) as backbone:
        base_logits = backbone(input_ids=ids[None], use_cache=False).logits
    assert not torch.allclose(decoder_logits, base_logits)
    assert codec.backbone.active_adapter == "encoder"
    assert flags == {name: p.requires_grad for name, p in codec.backbone.named_parameters()}


def test_nested_adapter_context_restores_state_after_exception(tiny_base):
    codec = make_task(tiny_base, dropout=0.1).codec
    # Preserve heterogeneous submodule modes, as well as the stage's trainable parameters.
    codec._lora_layers[0].lora_dropout["encoder"].eval()
    modes = [m.training for m in codec.backbone.modules()]
    flags = [p.requires_grad for p in codec.backbone.parameters()]
    with pytest.raises(RuntimeError, match="test interruption"):
        with codec.use_adapter(None):
            assert all(layer.disable_adapters for layer in codec._lora_layers)
            with codec.use_adapter("decoder", training=True):
                assert all(not layer.disable_adapters for layer in codec._lora_layers)
                assert codec.backbone.active_adapter == "decoder"
            assert all(layer.disable_adapters for layer in codec._lora_layers)
            raise RuntimeError("test interruption")
    assert codec.backbone.active_adapter == "encoder"
    assert all(not layer.disable_adapters for layer in codec._lora_layers)
    assert modes == [m.training for m in codec.backbone.modules()]
    assert flags == [p.requires_grad for p in codec.backbone.parameters()]


def test_bf16_alignment_preserves_backbone_dtype_and_memory_gradient(tiny_base):
    codec = MemoryCodec(
        CodecConfig(str(tiny_base), lora_rank=2, lora_alpha=4, gradient_checkpointing=True),
        torch.bfloat16,
    )
    observed = []
    hook = codec.backbone.get_base_model().model.register_forward_pre_hook(
        lambda module, args, kwargs: observed.append(kwargs["inputs_embeds"].dtype),
        with_kwargs=True,
    )
    ids = torch.tensor([4, 5, 6, 7])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        memory = codec.write(None, ids, 2)
        memory.retain_grad()
        updated = codec.write(memory, ids, 2)
        loss = codec.read_loss(updated, ids[:1], ids)
    loss.backward()
    hook.remove()
    assert observed and set(observed) == {torch.bfloat16}
    assert memory.grad is not None and torch.isfinite(memory.grad).all()
    assert memory.grad.abs().sum() > 0


def test_shared_checkpointed_backbone_matches_independent_encoder_and_decoder(tiny_base):
    actual = make_task(tiny_base, "weighted", checkpointing=True, dropout=0.1)
    reference = deepcopy(actual)
    reference.codec.config = replace(reference.codec.config, gradient_checkpointing=False)
    independent_decoder = deepcopy(reference.codec.backbone)
    independent_decoder.requires_grad_(False)
    independent_decoder.base_model.disable_adapter_layers()
    independent_decoder.eval()

    @contextmanager
    def independent_path(adapter, training=False):
        if adapter == "encoder":
            reference.codec.backbone.train(training)
            yield reference.codec.backbone
        else:
            yield independent_decoder

    reference.codec.use_adapter = independent_path
    torch.manual_seed(123)
    expected = reference(example())["loss"]
    expected.backward()
    torch.manual_seed(123)
    observed = actual(example())["loss"]
    observed.backward()
    torch.testing.assert_close(observed, expected)
    for (name, a), (_, b) in zip(
        actual.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=2e-6, msg=name)
    assert all(p.grad is None for p in independent_decoder.parameters())


def test_static_evaluation_uses_learned_alignment_without_mutating_checkpoint(tiny_base):
    task = make_task(tiny_base)
    task.codec.set_stage("warmup")
    with torch.no_grad():
        next(task.codec.read_alignment.layers.parameters()).add_(0.01)
    reference = deepcopy(task)
    reference.codec.initialize_write_alignment()
    reference.codec.set_stage("multiround")
    original = task.codec.write_alignment
    state = deepcopy(codec_state(task.codec))
    tokenizer = AutoTokenizer.from_pretrained(tiny_base)
    metrics, _ = evaluate(task, [example()], tokenizer)
    expected, _ = evaluate(reference, [example()], tokenizer)
    assert metrics == expected
    assert task.codec.write_alignment is original
    assert task.codec.stage == "warmup" and task.training
    torch.testing.assert_close(codec_state(task.codec), state, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["mean", "weighted", "spectral"])
def test_qwen3_sdpa_recurrent_checkpointing(tiny_base, tmp_path, method, monkeypatch):
    path = tmp_path / "qwen3"
    base = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    base.save_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(tiny_base)
    tokenizer.save_pretrained(path)
    codec = MemoryCodec(
        CodecConfig(
            str(path),
            encoder_layers=2,
            alignment_layers=1,
            lora_rank=2,
            lora_alpha=4,
            compression=method,
            spectral_bottleneck=8,
            attention_implementation="sdpa",
            gradient_checkpointing=True,
        )
    )
    task = ReconstructionTask(codec, tokenizer, TrainingConfig(objective="ae_lm"))
    reference = deepcopy(task)
    reference.codec.config = replace(reference.codec.config, gradient_checkpointing=False)

    def legacy_mask(module, args, kwargs):
        kwargs.pop("attention_mask", None)
        return args, kwargs

    for module in (
        reference.codec.backbone.get_base_model().model,
        reference.codec.read_alignment,
        reference.codec.write_alignment,
    ):
        module.register_forward_pre_hook(legacy_mask, with_kwargs=True)
    expected = reference(example())["loss"]
    expected.backward()
    sdpa = torch.nn.functional.scaled_dot_product_attention
    calls = []

    def capture(*args, **kwargs):
        calls.append((kwargs.get("attn_mask"), kwargs.get("is_causal")))
        return sdpa(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", capture)
    output = task(example())
    output["loss"].backward()
    assert calls and all(mask is None and causal for mask, causal in calls)
    torch.testing.assert_close(output["loss"], expected)
    for (name, actual), (_, previous) in zip(
        task.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert (actual.grad is None) == (previous.grad is None), name
        if actual.grad is not None:
            torch.testing.assert_close(actual.grad, previous.grad, rtol=2e-4, atol=2e-6, msg=name)
    assert torch.isfinite(output["loss"])
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in codec.write_alignment.layers.parameters()
    )


def _ddp_worker(rank, rendezvous, base, destination):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    torch.manual_seed(100)
    task = make_task(base, "weighted")
    task.config = replace(task.config, gradient_clip=100, micro_batch_size=2)
    engine = ReconstructionEngine(task, torch.device("cpu"))
    engine.initialize()
    # Unequal local trajectory counts, then a one-sample final batch with an empty rank.
    for rows in (
        [
            example(),
            example((3, 3, 3, 3)),
            example((4, 3, 2)),
            example((2, 2, 3, 3)),
            example((2, 2, 2, 2, 2)),
        ],
        [example()],
    ):
        engine.step(rows)
    if rank == 0:
        torch.save(codec_state(task.codec), destination)
    dist.destroy_process_group()


def test_verl_ddp_matches_global_mean_with_uneven_and_empty_ranks(tiny_base, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(100)
    task = make_task(tiny_base, "weighted")
    task.config = replace(task.config, gradient_clip=100)
    engine = ReconstructionEngine(task, torch.device("cpu"))
    engine.initialize()
    for rows in (
        [
            example(),
            example((3, 3, 3, 3)),
            example((4, 3, 2)),
            example((2, 2, 3, 3)),
            example((2, 2, 2, 2, 2)),
        ],
        [example()],
    ):
        engine.step(rows)
    destination = tmp_path / "ddp.pt"
    mp.spawn(
        _ddp_worker,
        args=(str(tmp_path / "rendezvous"), str(tiny_base), str(destination)),
        nprocs=2,
        join=True,
    )
    observed = torch.load(destination, weights_only=True)
    for name, value in codec_state(task.codec).items():
        torch.testing.assert_close(observed[name], value, rtol=5e-4, atol=3e-7, msg=name)
