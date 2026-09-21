"""固定采样、累计监督、递归梯度与 verl DDP 的真实小模型验证。"""

from copy import deepcopy
from dataclasses import replace
import random

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from verl.utils.tensordict_utils import get_non_tensor_data

from latent_working_memory.v2.compression import SlotCompression
from latent_working_memory.v2.memory_codec import CodecConfig, MemoryCodec
from latent_working_memory.v2 import memory_codec
from latent_working_memory.v2.pretrain.checkpoint import codec_state, restore_codec
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
        TrainingConfig(objective="ae_lm", lm_ratio=0.5, global_batch_size=2),
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
@pytest.mark.parametrize("checkpointing", [False, True])
def test_full_bptt_frozen_decoder_and_raw_update(tiny_base, method, checkpointing):
    task = make_task(tiny_base, method, checkpointing=checkpointing)
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
    result = task(row, read_task="both")
    assert input_lengths == [2, 3, 4]
    assert [r["ae_tokens"] for r in result["rounds"]] == [3, 6, 10]
    # Use only a final read: first memory must still receive a gradient through later writes.
    task.codec.read_loss(memories[-1], task.ae_prompt, torch.tensor([4, 5, 2])).backward()
    assert memories[0].grad.abs().sum() > 0
    assert all(p.grad is None for p in task.codec.decoder.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in task.codec.write_alignment.layers.parameters()
    )
    assert all(
        p.grad is None
        for name, p in task.codec.encoder.named_parameters()
        if "lora_" not in name or ".encoder." not in name
    )


@pytest.mark.parametrize("feature_layer", ["last", "mean"])
def test_checkpointed_read_matches_full_graph(tiny_base, feature_layer):
    reference = make_task(tiny_base, "weighted", False, feature_layer)
    actual = make_task(tiny_base, "weighted", True, feature_layer)
    actual.load_state_dict(reference.state_dict())
    reference(example(), read_task="both")["loss"].backward()
    actual(example(), read_task="both")["loss"].backward()
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
        embed = codec.decoder.get_input_embeddings()
        prefix = torch.cat((codec.align(memory), embed(prompt)))
        inputs = torch.cat((prefix, embed(targets)))[None]
        labels = torch.cat((targets.new_full((len(prefix),), -100), targets))[None]
        return codec.decoder(
            inputs_embeds=inputs,
            labels=labels,
            attention_mask=torch.ones(inputs.shape[:2], dtype=torch.bool),
            use_cache=False,
        ).loss

    reference.codec.read_loss = native_loss
    expected, observed = (
        reference(example(), read_task="both")["loss"],
        actual(example(), read_task="both")["loss"],
    )
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
    reference.config = replace(
        reference.config, objective=objective, lm_ratio=0.5 if objective == "ae_lm" else 0
    )
    actual = deepcopy(reference)
    actual.config = replace(actual.config, micro_batch_size=2)
    actual.codec.config = replace(actual.codec.config, lm_head_chunk_size=2)
    rows = [example((2, 3, 4)), example((2, 3, 3, 4)), example((4, 2, 5))]
    engines = [ReconstructionEngine(task, "cpu") for task in (reference, actual)]
    for engine in engines:
        engine.initialize()
    expected, observed = [engine.step(rows, step=4) for engine in engines]
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
    metrics = engine.step([example((9,)), example((8,))], step=0)
    assert metrics["samples"] == metrics["microbatches"] == 2
    assert metrics["max_microbatch_size"] == 1 and metrics["batched_samples"] == 0


@pytest.mark.parametrize("lm_ratio,expected_groups", [(0.0, 2), (1.0, 1)])
def test_microbatch_decoder_budget_splits_long_histories(tiny_base, lm_ratio, expected_groups):
    task = make_task(tiny_base)
    task.config = replace(
        task.config, micro_batch_size=2, micro_batch_decoder_tokens=26, lm_ratio=lm_ratio
    )
    engine = ReconstructionEngine(task, "cpu")
    engine.initialize()
    metrics = engine.step([example(), example((4, 2, 5))], step=0)
    assert metrics["microbatches"] == expected_groups
    assert metrics["max_microbatch_size"] == 3 - expected_groups


@pytest.mark.parametrize("objective", ["ae", "ae_lm"])
def test_warmup_freezes_write_alignment_and_updates_read_alignment(tiny_base, objective):
    task = make_task(tiny_base)
    task.config = replace(
        task.config, objective=objective, lm_ratio=0.5 if objective == "ae_lm" else 0
    )
    task.codec.set_stage("warmup")
    engine = ReconstructionEngine(task, torch.device("cpu"))
    engine.initialize()
    before = deepcopy(codec_state(task.codec))
    metrics = engine.step([example((12,))], step=0)
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
    hook = task.codec.encoder.get_base_model().model.register_forward_hook(
        lambda module, args, output: encoded.append(output.last_hidden_state.detach().clone())
    )
    task.codec.write(None, torch.tensor([4, 5, 6]), 2)
    task.codec.write(None, torch.tensor([4, 5, 7]), 2)
    hook.remove()
    torch.testing.assert_close(encoded[0][:, :2], encoded[1][:, :2], rtol=0, atol=0)
    assert not torch.equal(encoded[0][:, 2], encoded[1][:, 2])
    task.config = replace(task.config, lm_ratio=0.3)
    result = task(example(), read_task="both")
    expected = sum(0.7 * r["ae"] + 0.3 * r["lm"] for r in result["rounds"]) / 3
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
    decoder = codec.decoder.model
    assert len(decoder.layers) == len(codec.encoder.get_base_model().model.layers) == 2
    assert not codec.encoder.is_gradient_checkpointing
    assert not codec.decoder.is_gradient_checkpointing
    components = [codec.encoder, codec.decoder, codec.read_alignment, codec.write_alignment]
    addresses = [{p.data_ptr() for p in module.parameters()} for module in components]
    for i, current in enumerate(addresses):
        assert all(not current.intersection(other) for other in addresses[i + 1 :])
    torch.testing.assert_close(
        codec.read_alignment.layers[0].self_attn.q_proj.weight,
        decoder.layers[0].self_attn.q_proj.weight,
    )
    assert codec.read_alignment.embed_tokens is None
    assert not codec.read_alignment.norm.weight.requires_grad


def test_two_pretrained_loads_and_no_meta_alignment_weights(tiny_base, monkeypatch):
    original = memory_codec.AutoModelForCausalLM.from_pretrained
    calls = []

    def load(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(memory_codec.AutoModelForCausalLM, "from_pretrained", load)
    codec = make_task(tiny_base).codec
    assert calls == [str(tiny_base), str(tiny_base)]
    assert set(codec.encoder.peft_config) == {"encoder"}
    assert not any("lora_" in name for name, _ in codec.decoder.named_parameters())
    for module in (codec.read_alignment, codec.write_alignment):
        assert len(module.layers) == 1 and module.embed_tokens is None
        assert all(not p.is_meta for p in module.parameters())
        assert all(not p.is_meta for p in module.buffers())
        assert not any("lora_" in name for name, _ in module.named_parameters())


def test_encoder_lora_does_not_change_reader_or_parameter_contract(tiny_base):
    task = make_task(tiny_base)
    task.eval()
    codec = task.codec
    ids = torch.tensor([4, 5, 6, 7])
    memory = torch.randn(2, codec.width)
    target = torch.tensor([5, 6, 2])
    initial_write = codec.write(None, ids, 2)
    initial_read = codec.read_loss(memory, task.ae_prompt, target)
    flags = {name: p.requires_grad for name, p in codec.named_parameters()}
    with torch.no_grad():
        for name, p in codec.encoder.named_parameters():
            if "lora_B" in name:
                p.normal_(std=0.1)
    assert not torch.allclose(codec.write(None, ids, 2), initial_write)
    torch.testing.assert_close(codec.read_loss(memory, task.ae_prompt, target), initial_read)
    assert codec.encoder.active_adapter == "encoder"
    assert flags == {name: p.requires_grad for name, p in codec.named_parameters()}
    assert all(not p.requires_grad for p in codec.decoder.parameters())


def test_codec_checkpoint_contains_only_adapters_and_interfaces(tiny_base):
    codec = make_task(tiny_base).codec
    state = codec_state(codec)
    assert any(name.startswith("encoder.") and "lora_" in name for name in state)
    assert not any(name.startswith(("backbone.", "decoder.")) for name in state)
    restored = make_task(tiny_base).codec
    restore_codec(restored, state)
    torch.testing.assert_close(codec_state(restored), state, rtol=0, atol=0)
    old_state = {name.replace("encoder.", "backbone.", 1): p for name, p in state.items()}
    with pytest.raises(ValueError, match="differ from this architecture"):
        restore_codec(restored, old_state)


def test_bf16_alignment_preserves_backbone_dtype_and_memory_gradient(tiny_base):
    codec = MemoryCodec(
        CodecConfig(str(tiny_base), lora_rank=2, lora_alpha=4, gradient_checkpointing=True),
        torch.bfloat16,
    )
    observed = []
    hook = codec.encoder.get_base_model().model.register_forward_pre_hook(
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


@pytest.mark.parametrize("stage", ["warmup", "multiround"])
@pytest.mark.parametrize("feature_layer", ["last", "mean"])
def test_layer_checkpoint_matches_full_graph_with_nonzero_lora_and_dropout(
    tiny_base, stage, feature_layer
):
    actual = make_task(
        tiny_base, "weighted", checkpointing=True, feature_layer=feature_layer, dropout=0.1
    )
    actual.codec.set_stage(stage)
    with torch.no_grad():
        for name, p in actual.codec.encoder.named_parameters():
            if "lora_B" in name:
                p.normal_(std=0.05)
    reference = make_task(
        tiny_base, "weighted", checkpointing=False, feature_layer=feature_layer, dropout=0.1
    )
    reference.codec.set_stage(stage)
    reference.load_state_dict(actual.state_dict())
    counts = {"encoder": 0, "decoder": 0}

    def count(name):
        def capture(module, args):
            counts[name] += 1

        return capture

    handles = [
        actual.codec.encoder.get_base_model()
        .model.layers[0]
        .register_forward_pre_hook(count("encoder")),
        actual.codec.decoder.model.layers[0].register_forward_pre_hook(count("decoder")),
    ]
    rows = [example((9,))] if stage == "warmup" else [example(), example((3, 2, 2, 2))]
    torch.manual_seed(123)
    expected = reference(rows, read_task="both")["loss"]
    expected.backward()
    torch.manual_seed(123)
    observed = actual(rows, read_task="both")["loss"]
    forward_counts = counts.copy()
    observed.backward()
    for handle in handles:
        handle.remove()
    assert all(counts[name] > forward_counts[name] > 0 for name in counts)
    torch.testing.assert_close(observed, expected)
    for (name, a), (_, b) in zip(
        actual.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=2e-6, msg=name)
    assert all(p.grad is None for p in actual.codec.decoder.parameters())
    for task in (actual, reference):
        torch.optim.AdamW([p for p in task.parameters() if p.requires_grad], lr=1e-4).step()
    torch.testing.assert_close(
        codec_state(actual.codec), codec_state(reference.codec), rtol=2e-4, atol=2e-6
    )


def test_generation_uses_cache_and_restores_training_without_unfreezing_decoder(tiny_base):
    task = make_task(tiny_base, checkpointing=True)
    codec = task.codec
    ids = torch.tensor([4, 5, 6, 7])
    memory = codec.write(None, ids, 2)
    caches = []
    handle = codec.decoder.model.register_forward_pre_hook(
        lambda module, args, kwargs: caches.append(kwargs.get("use_cache")), with_kwargs=True
    )
    generated = codec.generate(memory, task.ae_prompt, 3, None, 0)
    handle.remove()
    assert len(generated) == 3 and len(caches) == 3 and all(caches)
    assert codec.training and codec.encoder.training and codec.decoder.training
    codec.read_loss(memory, task.ae_prompt, torch.tensor([4, 5, 2])).backward()
    assert all(p.grad is None and not p.requires_grad for p in codec.decoder.parameters())
    assert any(p.grad is not None for p in codec.encoder.parameters())


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
@pytest.mark.parametrize("mixed", [False, True])
def test_qwen3_sdpa_recurrent_checkpointing(tiny_base, tmp_path, method, mixed, monkeypatch):
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
    task = ReconstructionTask(codec, tokenizer, TrainingConfig(objective="ae_lm", lm_ratio=0.5))
    reference = deepcopy(task)
    reference.codec.config = replace(reference.codec.config, gradient_checkpointing=False)
    reference.codec.encoder.gradient_checkpointing_disable()
    reference.codec.decoder.gradient_checkpointing_disable()

    def legacy_mask(module, args, kwargs):
        kwargs.pop("attention_mask", None)
        return args, kwargs

    for module in (
        reference.codec.encoder.get_base_model().model,
        reference.codec.decoder.model,
        reference.codec.read_alignment,
        reference.codec.write_alignment,
    ):
        module.register_forward_pre_hook(legacy_mask, with_kwargs=True)
    rows = [example(), example((3, 2, 2, 2, 3))] if mixed else example()
    tasks = ["ae", "lm"] if mixed else "both"
    expected = reference(rows, read_task=tasks)["loss"]
    expected.backward()
    sdpa = torch.nn.functional.scaled_dot_product_attention
    calls = []

    def capture(*args, **kwargs):
        calls.append((kwargs.get("attn_mask"), kwargs.get("is_causal")))
        return sdpa(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", capture)
    output = task(rows, read_task=tasks)
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


def _ddp_worker(rank, rendezvous, base, destination, independent):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    torch.manual_seed(100)
    task = make_task(base, "weighted", checkpointing=True)
    if independent:
        task.codec.set_stage("independent_prefix")
    task.config = replace(task.config, gradient_clip=100, micro_batch_size=2)
    engine = ReconstructionEngine(task, torch.device("cpu"))
    engine.initialize()
    # Unequal local trajectory counts, then a one-sample final batch with an empty rank.
    for step, rows in enumerate(
        (
            [
                example(),
                example((3, 3, 3, 3)),
                example((4, 3, 2)),
                example((2, 2, 3, 3)),
                example((2, 2, 2, 2, 2)),
            ],
            [example()],
        ),
        start=1,
    ):
        engine.step(rows, step)
    if rank == 0:
        torch.save(codec_state(task.codec), destination)
    dist.destroy_process_group()


@pytest.mark.parametrize("independent", [False, True])
def test_verl_ddp_matches_global_mean_with_uneven_and_empty_ranks(tiny_base, tmp_path, independent):
    torch.set_num_threads(1)
    torch.manual_seed(100)
    task = make_task(tiny_base, "weighted", checkpointing=True)
    if independent:
        task.codec.set_stage("independent_prefix")
    task.config = replace(task.config, gradient_clip=100)
    engine = ReconstructionEngine(task, torch.device("cpu"))
    engine.initialize()
    for step, rows in enumerate(
        (
            [
                example(),
                example((3, 3, 3, 3)),
                example((4, 3, 2)),
                example((2, 2, 3, 3)),
                example((2, 2, 2, 2, 2)),
            ],
            [example()],
        ),
        start=1,
    ):
        engine.step(rows, step)
    destination = tmp_path / "ddp.pt"
    mp.spawn(
        _ddp_worker,
        args=(str(tmp_path / "rendezvous"), str(tiny_base), str(destination), independent),
        nprocs=2,
        join=True,
    )
    observed = torch.load(destination, weights_only=True)
    for name, value in codec_state(task.codec).items():
        torch.testing.assert_close(observed[name], value, rtol=5e-4, atol=3e-7, msg=name)


@pytest.mark.parametrize(
    "lm_ratio,tasks",
    [
        (0.0, ["ae", "ae", "ae"]),
        (0.5, ["ae", "lm", "ae"]),
        (1.0, ["lm", "lm", "lm"]),
    ],
)
def test_sampled_reads_match_manual_losses_gradients_and_counts(
    tiny_base, monkeypatch, lm_ratio, tasks
):
    actual = make_task(tiny_base, "weighted", checkpointing=True)
    actual.config = replace(actual.config, lm_ratio=lm_ratio, gradient_clip=100)
    reference = deepcopy(actual)
    rows = [example(), example((2, 2, 3, 3)), example((4, 3, 2))]
    expected = [reference(row, read_task=name) for row, name in zip(rows, tasks)]
    expected_loss = torch.stack([result["loss"] for result in expected]).mean()
    expected_loss.backward()
    calls = []
    original = actual.codec.read_loss

    def capture(memory, prompt, targets):
        calls.append("ae" if prompt is actual.ae_prompt else "lm")
        return original(memory, prompt, targets)

    monkeypatch.setattr(actual.codec, "read_loss", capture)
    engine = ReconstructionEngine(actual, "cpu")
    engine.initialize()
    observed = engine.step(rows, step=4)
    assert calls == [name for row, name in zip(rows, tasks) for _ in row.write_ends]
    assert observed["loss"] == pytest.approx(expected_loss.item())
    assert observed["samples"] == observed["ae_samples"] + observed["lm_samples"] == 3
    for name in ("ae", "lm"):
        selected = [r for r, task_name in zip(expected, tasks) if task_name == name]
        assert observed[f"{name}_samples"] == len(selected)
        if selected:
            assert observed[name] == pytest.approx(
                sum(r["loss"].item() for r in selected) / len(selected)
            )
        else:
            assert observed[name] is None
        assert observed[f"{name}_tokens"] == sum(
            t[f"{name}_tokens"] for r in expected for t in r["rounds"]
        )
    assert observed["target_tokens"] == observed["ae_tokens"] + observed["lm_tokens"]
    for (name, a), (_, b) in zip(
        actual.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=5e-4, atol=3e-7, msg=name)


def test_task_sampling_is_independent_of_dropout_rng_and_repeatable(tiny_base, monkeypatch):
    engine = ReconstructionEngine(make_task(tiny_base), "cpu")
    engine.initialize()
    captured = []

    def capture(data, loss_function):
        captured.append(get_non_tensor_data(data, "read_tasks", None))
        return {"metrics": {}}

    monkeypatch.setattr(engine, "train_batch", capture)
    rows = [example()] * 24
    before_python, before_torch = random.getstate(), torch.get_rng_state()
    engine.step(rows, step=8)
    assert random.getstate() == before_python
    torch.testing.assert_close(torch.get_rng_state(), before_torch, rtol=0, atol=0)
    random.random()
    torch.rand(10)
    engine.step(rows, step=8)
    engine.step(rows, step=9)
    assert captured[0] == captured[1] and captured[1] != captured[2]
    assert set(captured[0]) == {"ae", "lm"}


@pytest.mark.parametrize("ratio", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_lm_ratio_is_rejected(ratio):
    with pytest.raises(ValueError, match="lm_ratio"):
        TrainingConfig(objective="ae_lm", lm_ratio=ratio)


def test_ae_only_requires_zero_lm_ratio():
    with pytest.raises(ValueError, match="AE-only"):
        TrainingConfig(objective="ae", lm_ratio=0.5)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_mixed_tasks_and_depths_compact_without_dummy_writes(tiny_base, monkeypatch, checkpointing):
    actual = make_task(tiny_base, "weighted", checkpointing=checkpointing)
    # Unequal prompt lengths expose padding inserted between prompt and target.
    actual.ae_prompt = torch.tensor([4, 5])
    actual.lm_prompt = torch.tensor([4, 5, 6, 7, 4])
    reference = deepcopy(actual)
    rows = [example(), example((2, 2, 2, 2, 3)), example((7,))]
    tasks = ["ae", "lm", "ae"]
    expected = [reference(row, read_task=name) for row, name in zip(rows, tasks)]
    expected_loss = torch.stack([x["loss"] for x in expected]).mean()
    expected_loss.backward()
    calls = []
    original_write, original_read = actual.codec.write_batch, actual.codec.read_loss_batch

    def write(previous, tokens, capacity):
        calls.append(("write", len(tokens), [len(x) for x in tokens]))
        return original_write(previous, tokens, capacity)

    def read(memory, prompts, targets):
        calls.append(("read", len(prompts), [len(x) for x in prompts]))
        return original_read(memory, prompts, targets)

    monkeypatch.setattr(actual.codec, "write_batch", write)
    monkeypatch.setattr(actual.codec, "read_loss_batch", read)
    observed = actual(rows, read_task=tasks)
    observed["loss"].backward()
    assert observed["batch_sizes"] == [3, 2, 2, 1, 1]
    assert [c[:2] for c in calls] == [
        (name, size) for size in [3, 2, 2, 1, 1] for name in ("write", "read")
    ]
    assert calls[1][2] == [2, 5, 2]
    assert calls[0][2] == [2, 2, 7] and calls[-2][2] == [3]
    torch.testing.assert_close(observed["loss"], expected_loss)
    for records, expected_row in zip(observed["rounds"], expected, strict=True):
        assert len(records) == len(expected_row["rounds"])
        for record, expected_record in zip(records, expected_row["rounds"], strict=True):
            for key in record:
                if key in ("ae", "lm") and record[key] is not None:
                    assert record[key] == pytest.approx(expected_record[key], rel=2e-5)
                else:
                    assert record[key] == expected_record[key]
    for (name, a), (_, b) in zip(
        actual.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=5e-4, atol=3e-7, msg=name)


def test_four_mixed_trajectories_share_one_group_and_budget_only_active_rows(tiny_base):
    task = make_task(tiny_base)
    task.config = replace(task.config, micro_batch_size=4, micro_batch_decoder_tokens=53)
    rows = [example((2, 2, 2, 2, 2)), example((2,)), example((2,)), example((2,))]
    # Step 4 samples AE, LM, AE, LM. Late steps keep only the first trajectory.
    engine = ReconstructionEngine(task, "cpu")
    engine.initialize()
    metrics = engine.step(rows, step=4)
    assert metrics["max_microbatch_size"] == 4 and metrics["microbatches"] == 1
    assert metrics["mean_active_microbatch_size"] == pytest.approx(8 / 5)
    assert metrics["ae_samples"] == metrics["lm_samples"] == 2


@pytest.mark.parametrize("tasks", [["ae", "ae"], ["ae", "lm"]])
def test_independent_prefix_matches_separate_losses_and_gradients(tiny_base, monkeypatch, tasks):
    actual = make_task(tiny_base, "weighted", checkpointing=True)
    actual.codec.set_stage("independent_prefix")
    reference = deepcopy(actual)
    rows = [example(), example((3, 2, 2, 2))]
    expected = []
    for row, name in zip(rows, tasks, strict=True):
        q = len(row.token_ids) - row.write_ends[-1]
        reads = []
        for end in row.write_ends:
            memory = reference.codec.write(None, row.token_ids[:end], row.capacity)
            target = row.token_ids[:end] if name == "ae" else row.token_ids[end : end + q]
            target = torch.cat((target, target.new_tensor([reference.eos_id])))
            reads.append(
                reference.codec.read_loss(memory, getattr(reference, f"{name}_prompt"), target)
            )
        expected.append(torch.stack(reads).mean())
    expected_loss = torch.stack(expected).mean()
    expected_loss.backward()
    calls = []
    original = actual.codec.write_batch

    def capture(previous, tokens, capacity):
        assert previous is None
        calls.append([t.tolist() for t in tokens])
        return original(previous, tokens, capacity)

    monkeypatch.setattr(actual.codec, "write_batch", capture)
    result = actual(rows, read_task=tasks, write_mode="independent_prefix")
    result["loss"].backward()
    assert [[len(t) for t in batch] for batch in calls] == [[2, 3], [5, 5], [9, 7], [9]]
    for step, batch in enumerate(calls):
        active = [row for row in rows if step < len(row.write_ends)]
        assert batch == [row.token_ids[: row.write_ends[step]].tolist() for row in active]
    torch.testing.assert_close(result["loss"], expected_loss)
    for (name, a), (_, b) in zip(
        actual.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=5e-4, atol=3e-7, msg=name)
    assert all(
        p.grad is None and not p.requires_grad for p in actual.codec.write_alignment.parameters()
    )


def test_independent_engine_budgets_complete_prefixes(tiny_base):
    task = make_task(tiny_base)
    task.codec.set_stage("independent_prefix")
    # Final prefixes need 9+9 positions, while recurrent inputs need only 6+6.
    task.config = replace(task.config, micro_batch_size=2, micro_batch_encoder_tokens=12)
    engine = ReconstructionEngine(task, "cpu")
    engine.initialize()
    result = engine.step([example(), example()], 4)
    assert result["microbatches"] == 2
    assert result["max_microbatch_size"] == 1
    assert result["ae_tokens"] == 3 + 6 + 10
    assert result["lm_tokens"] == 3 * 3


def test_evaluation_pairs_each_independent_prefix_and_restores_codec(tiny_base):
    task = make_task(tiny_base)
    task.codec.set_stage("independent_prefix")
    task.eval()
    original = task.codec.write_alignment
    before = deepcopy(codec_state(task.codec))
    row = example()
    metrics, records = evaluate(task, [row], AutoTokenizer.from_pretrained(tiny_base), 1)
    record = records[0]
    q = len(row.token_ids) - row.write_ends[-1]
    for step, end in enumerate(row.write_ends):
        prefix = replace(row, token_ids=row.token_ids[: end + q], write_ends=(end,))
        expected = task(prefix, read_task="both")["rounds"][0]
        observed = record["independent_prefix"][step]
        for key in ["ae", "lm", "seen_tokens", "ae_tokens", "lm_tokens"]:
            assert observed[key] == pytest.approx(expected[key])
    assert record["one_shot"] == record["independent_prefix"][-1]
    assert record["rounds"][0] == record["independent_prefix"][0]
    assert metrics["independent_prefix/trajectory_ae"] == pytest.approx(
        sum(r["ae"] for r in record["independent_prefix"]) / len(row.write_ends)
    )
    assert metrics["independent_prefix/generation/samples"] == 1
    assert "independent_generation" in record
    assert task.codec.write_alignment is original and not task.training
    torch.testing.assert_close(codec_state(task.codec), before, rtol=0, atol=0)
