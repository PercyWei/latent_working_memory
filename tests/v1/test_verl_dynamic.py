"""Compare verl recurrent engine with the pre-refactor full/TBPTT implementation."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from verl.workers.engine import BaseEngine, EngineRegistry

from latent_working_memory.v1.dynamic.training import DynamicTrainer, DynamicEngine
from latent_working_memory.v1.model import JointMemoryWriter
from dynamic_reference import ReferenceDynamicTrainer
from test_dynamic import example, small_recipe
from test_evaluation_batching import assert_results_equal
from test_reader_projection import make_backbone


def compare(config, tokenizer, architecture, unit, span, checkpointing, trailing=False):
    torch.manual_seed(42)
    reference_backbone = make_backbone(architecture)
    reference_writer = JointMemoryWriter(8, 1, 2, 16, 32)
    backbone, writer = deepcopy(reference_backbone), deepcopy(reference_writer)
    recipe = replace(
        small_recipe(bptt_unit=unit, bptt_span=span),
        global_batch_size=2,
        new_count=0 if trailing else 1,
        history_count=1,
        max_visits=1 if trailing else 5,
        qa_activation_checkpointing=checkpointing,
    )
    reference = ReferenceDynamicTrainer(
        reference_backbone, reference_writer, config, recipe, torch.device("cpu")
    )
    actual = DynamicTrainer(backbone, writer, config, recipe, torch.device("cpu"))
    assert isinstance(actual.engine, BaseEngine)
    assert type(actual.engine).train_batch is BaseEngine.train_batch
    assert EngineRegistry.get_engine_cls("lwm_dynamic", "replicated") is DynamicEngine
    if dist.is_initialized():
        assert actual.engine.world_size == 2
        assert isinstance(actual.engine.module, torch.nn.parallel.DistributedDataParallel)
    episodes = [example(tokenizer, "short", 4), example(tokenizer, "long", 8)]
    if trailing:
        episodes = [replace(e, reads=(e.reads[1],)) for e in episodes]
    captured_states = []
    handle = actual.engine.model.register_forward_pre_hook(
        lambda module, args: captured_states.append(args[3])
    )
    try:
        for capacity in (4, 4):
            expected = reference.step(episodes, tokenizer, [42, 43], capacity)
            observed = actual.step(episodes, tokenizer, [42, 43], capacity)
            assert_results_equal(expected, observed)
            for a, b in zip(reference.parameters, actual.parameters, strict=True):
                assert (a.grad is None) == (b.grad is None)
                if a.grad is not None:
                    torch.testing.assert_close(a.grad, b.grad, rtol=3e-5, atol=2e-7)
                torch.testing.assert_close(a, b, rtol=3e-5, atol=2e-7)
            torch.testing.assert_close(
                reference.optimizer.state_dict(),
                actual.optimizer.state_dict(),
                rtol=3e-5,
                atol=2e-7,
            )
        # State crosses only segment boundaries here; every carried state is detached.
        states = [s for s in captured_states if s is not None]
        local_indices = range(actual.engine.rank, len(episodes), actual.engine.world_size)
        if any(len(observed["sample_metrics"][i]["bptt_segments"]) > 1 for i in local_indices):
            assert states
        assert all(s.values.grad_fn is None and not s.values.requires_grad for s in states)
    finally:
        handle.remove()


@pytest.mark.parametrize("architecture", ["llama", "qwen2"])
@pytest.mark.parametrize("unit,span", [("tokens", 0), ("tokens", 1), ("updates", 2)])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_recurrent_engine_matches_reference(
    tiny_config, tokenizer, architecture, unit, span, checkpointing
):
    compare(tiny_config, tokenizer, architecture, unit, span, checkpointing)


def test_recurrent_engine_distributed_unequal_segments_and_trailing_no_reads(
    tmp_path, tiny_config, tokenizer
):
    mp.spawn(
        _worker, args=(str(tmp_path / "rendezvous"), tiny_config, tokenizer), nprocs=2, join=True
    )


def _worker(rank, rendezvous, config, tokenizer):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        for unit, span in (("tokens", 0), ("tokens", 1), ("updates", 2)):
            compare(config, tokenizer, "qwen2", unit, span, False)
            compare(config, tokenizer, "qwen2", unit, span, True, trailing=True)
    finally:
        dist.destroy_process_group()
