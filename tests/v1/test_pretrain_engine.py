"""Paired old/new optimizer updates, including uneven DDP capacity expansion."""

from copy import deepcopy
from dataclasses import replace
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from latent_working_memory.v1.backbone import ReadTokens
from latent_working_memory.v1.data import Episode, Read, Reference, Source
from latent_working_memory.v1.model import JointMemoryWriter
from latent_working_memory.v1.pretrain.sampling import PretrainExample
from latent_working_memory.v1.pretrain.training import PretrainTrainer
from pretrain_reference import LegacyPretrainTrainer
from test_reader_projection import make_backbone
from test_evaluation_batching import assert_results_equal


def comparison_examples(mode, task_kind="mixed"):
    examples = []
    for i, length in enumerate((2, 4, 8, 2, 4, 8, 4, 8)):
        ids = tuple(4 + (i + j) % 6 for j in range(length))
        task = ("ae" if i % 2 == 0 else "continuation") if task_kind == "mixed" else task_kind
        target = ids if task == "ae" else (7, 8, 9)
        read = Read(str(i), task, length, "prompt", (Reference("reference", ()),))
        source = Source(str(i), str(i), 0, length, {"boundary_variant": "semantic"})
        episode = Episode(str(i), ids, (length,), (source,), (read,))
        tokens = ReadTokens((11, 12), (*target, 2))
        capacities = sorted({max(1, (length + r - 1) // r) for r in (2, 4, 8)})
        if mode == "sample":
            capacities = capacities[:1]
        for capacity in capacities:
            examples.append(
                PretrainExample(
                    episode,
                    tokens if task == "ae" else None,
                    tokens if task == "continuation" else None,
                    capacity,
                    1 / len(capacities),
                )
            )
    return examples


def compare_trainers(config, architecture, mode, device):
    torch.manual_seed(42)
    reference_backbone = make_backbone(architecture).to(device)
    reference_writer = JointMemoryWriter(8, 1, 2, 16, 32).to(device)
    actual_backbone, actual_writer = deepcopy(reference_backbone), deepcopy(reference_writer)
    legacy = LegacyPretrainTrainer(config, reference_backbone, reference_writer, device)
    actual = PretrainTrainer(config, actual_backbone, actual_writer, device)
    if dist.is_initialized():
        assert actual.engine.world_size == 2
        assert isinstance(actual.model, torch.nn.parallel.DistributedDataParallel)
    # Different tasks on successive updates exercise DDP unused-parameter bookkeeping.
    for task in ("mixed", "ae", "continuation"):
        examples = comparison_examples(mode, task)
        expected_metrics = legacy.step(examples)
        actual_metrics = actual.step(examples)
        assert_results_equal(actual_metrics, expected_metrics)
        for reference, observed in zip(legacy.parameters, actual.parameters, strict=True):
            assert (reference.grad is None) == (observed.grad is None)
            if reference.grad is not None:
                torch.testing.assert_close(observed.grad, reference.grad, rtol=3e-5, atol=2e-7)
            torch.testing.assert_close(observed, reference, rtol=3e-5, atol=2e-7)
        torch.testing.assert_close(
            actual.optimizer.state_dict(), legacy.optimizer.state_dict(), rtol=3e-5, atol=2e-7
        )
        assert all(p.grad is None for p in actual_backbone.parameters() if not p.requires_grad)
    return actual


@pytest.mark.parametrize("architecture", ["llama", "qwen2"])
@pytest.mark.parametrize("mode", ["sample", "mean"])
def test_single_device_matches_legacy_updates(tiny_config, architecture, mode):
    compare_trainers(replace(tiny_config, batch_size=2), architecture, mode, torch.device("cpu"))


def test_two_rank_updates_match_legacy_sum_and_uneven_microbatches(tmp_path, tiny_config):
    mp.spawn(
        _compare_worker,
        args=(str(tmp_path / "rendezvous"), replace(tiny_config, batch_size=2)),
        nprocs=2,
        join=True,
    )


def _compare_worker(rank, rendezvous, config):
    torch.set_num_threads(1)
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_WORLD_SIZE="2",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="29500",
    )
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    for architecture in ("llama", "qwen2"):
        for mode in ("sample", "mean"):
            compare_trainers(config, architecture, mode, torch.device("cpu"))
    dist.destroy_process_group()


def test_nonfinite_gradients_stop_before_optimizer_update(tiny_config):
    torch.manual_seed(42)
    backbone = make_backbone("qwen2")
    writer = JointMemoryWriter(8, 1, 2, 16, 32)
    trainer = PretrainTrainer(tiny_config, backbone, writer, torch.device("cpu"))
    before = [p.detach().clone() for p in trainer.parameters]
    hook = writer.output_projection.weight.register_hook(lambda gradient: gradient * float("nan"))
    try:
        with pytest.raises(RuntimeError, match="non-finite"):
            trainer.step(comparison_examples("sample"))
    finally:
        hook.remove()
    for old, current in zip(before, trainer.parameters, strict=True):
        torch.testing.assert_close(old, current, rtol=0, atol=0)
    assert not trainer.optimizer.state_dict()["state"]
