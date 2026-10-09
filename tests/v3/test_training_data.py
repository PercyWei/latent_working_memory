from dataclasses import dataclass
import random

import pytest
import torch

from latent_working_memory.v3.pretrain_data import PretrainExample
from latent_working_memory.v3.training_data import make_data_loader
from .test_objective import trajectory


@dataclass(frozen=True)
class Example:
    sample_id: str
    values: tuple[float, ...]


def expected_order(rows, seed, epoch):
    indices = list(range(len(rows)))
    random.Random(f"{seed}:v3:{epoch}").shuffle(indices)
    return [rows[index] for index in indices]


@pytest.mark.parametrize("epoch", [0, 1, 7])
@pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
def test_native_loader_matches_original_epoch_order_and_keeps_true_tail(epoch, batch_size):
    rows = [Example(f"row-{index}", (float(index),)) for index in range(17)]
    loader = make_data_loader(rows, batch_size, seed=42)
    loader.sampler.set_epoch(epoch)
    batches = list(loader)
    actual = [row for batch in batches for row in batch]

    assert actual == expected_order(rows, 42, epoch)
    assert len(loader) == (len(rows) + batch_size - 1) // batch_size
    assert [len(batch) for batch in batches] == [
        min(batch_size, len(rows) - start) for start in range(0, len(rows), batch_size)
    ]
    assert all(
        actual_row is expected_row
        for actual_row, expected_row in zip(actual, expected_order(rows, 42, epoch), strict=True)
    )


def test_collator_preserves_complete_pretraining_factqa_and_generic_objects():
    rows = [
        PretrainExample("p0", "doc", "cluster", "ae", (1, 2, 3), (1, 2, 3)),
        trajectory(n=3),
        Example("generic", (0.1, 0.2)),
    ]
    loader = make_data_loader(rows, 3, seed=42)
    actual = next(iter(loader))
    assert all(row is rows[rows.index(row)] for row in actual)
    assert actual == expected_order(rows, 42, 0)


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("epoch", [0, 3])
def test_mid_epoch_restoration_uses_native_loader_position_once(batch_size, epoch):
    rows = list(range(17))
    loader = make_data_loader(rows, batch_size, seed=42)
    loader.sampler.set_epoch(epoch)
    iterator = iter(loader)
    first_batch = next(iterator)
    saved = loader.state_dict()

    restored = make_data_loader(rows, batch_size, seed=42)
    restored.load_state_dict(saved)
    restored.sampler.set_epoch(epoch)
    remaining = [row for batch in restored for row in batch]

    assert first_batch + remaining == expected_order(rows, 42, epoch)
    assert remaining == [row for batch in iterator for row in batch]


@pytest.mark.parametrize("sample_count,batch_size", [(17, 16), (16, 8), (1, 16)])
@pytest.mark.parametrize("epoch", [0, 3])
def test_checkpoint_immediately_after_last_batch_resumes_next_epoch(
    sample_count, batch_size, epoch
):
    rows = list(range(sample_count))
    loader = make_data_loader(rows, batch_size, seed=42)
    loader.sampler.set_epoch(epoch)
    iterator = iter(loader)
    for _ in range(len(loader)):
        next(iterator)
    # 尚未向 iterator 请求 StopIteration；这是训练循环保存 epoch-end checkpoint 的时机。
    saved = loader.state_dict()

    restored = make_data_loader(rows, batch_size, seed=42)
    restored.load_state_dict(saved)
    restored.sampler.set_epoch(epoch + 1)
    actual = [row for batch in restored for row in batch]
    assert actual == expected_order(rows, 42, epoch + 1)
    assert len(actual) == sample_count


def test_native_loader_can_restore_after_epoch_iterator_has_already_finished():
    rows = list(range(17))
    loader = make_data_loader(rows, 8, seed=42)
    list(loader)
    restored = make_data_loader(rows, 8, seed=42)
    restored.load_state_dict(loader.state_dict())
    restored.sampler.set_epoch(1)
    assert [row for batch in restored for row in batch] == expected_order(rows, 42, 1)


def test_loader_creation_iteration_snapshot_and_restoration_preserve_global_torch_rng():
    before = torch.get_rng_state().clone()
    loader = make_data_loader(list(range(17)), 8, seed=42)
    iterator = iter(loader)
    next(iterator)
    saved = loader.state_dict()
    restored = make_data_loader(list(range(17)), 8, seed=42)
    restored.load_state_dict(saved)
    list(restored)
    loader.sampler.set_epoch(1)
    list(loader)
    assert torch.equal(torch.get_rng_state(), before)


def test_independent_rank_loaders_produce_the_same_global_batches():
    rows = list(range(17))
    first = make_data_loader(rows, 16, seed=42)
    second = make_data_loader(rows, 16, seed=42)
    first.sampler.set_epoch(5)
    second.sampler.set_epoch(5)
    assert list(first) == list(second)


@pytest.mark.parametrize("epoch", [0, 7])
def test_validation_loader_preserves_source_order_and_true_tail(epoch):
    rows = list(range(17))
    loader = make_data_loader(rows, 16, seed=42, shuffle=False)
    loader.sampler.set_epoch(epoch)
    assert list(loader) == [rows[:16], rows[16:]]


def test_snapshot_before_first_batch_restores_the_complete_epoch():
    rows = list(range(17))
    loader = make_data_loader(rows, 16, seed=42)
    loader.sampler.set_epoch(3)
    saved = loader.state_dict()
    restored = make_data_loader(rows, 16, seed=42)
    restored.load_state_dict(saved)
    restored.sampler.set_epoch(3)
    assert [row for batch in restored for row in batch] == expected_order(rows, 42, 3)
