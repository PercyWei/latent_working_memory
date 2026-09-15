"""CPU checks of the model/batch adapter, independent of CUDA FSDP execution."""

import pytest
import torch
from verl.workers.engine.utils import prepare_micro_batches

from latent_working_memory.v1.model import JointMemoryWriter
from latent_working_memory.v1.pretrain.fsdp_engine import PretrainFSDPEngine, pretrain_batch
from latent_working_memory.v1.pretrain.training import PretrainModel
from test_reader_projection import make_backbone
from validate_verl_fsdp import examples


@pytest.mark.parametrize("mode", ["sample", "mean"])
@pytest.mark.parametrize("micro_batch_size", [1, 2])
def test_native_batch_and_model_adapter(tiny_config, mode, micro_batch_size):
    samples = examples(mode, 2)
    visits = []
    for rank in (0, 1):
        data = pretrain_batch(samples, micro_batch_size, rank, 2)
        batches, _ = prepare_micro_batches(data, same_micro_num_in_dp=False)
        # Only call the input/forward adapter; no FSDP engine initialization.
        engine = object.__new__(PretrainFSDPEngine)
        engine._autocast_dtype = torch.float32
        engine.ulysses_device_mesh = None
        engine.module = PretrainModel(
            tiny_config, make_backbone("qwen2"), JointMemoryWriter(8, 1, 2, 16, 32)
        )
        for batch in batches:
            loss, info = engine.forward_step(
                batch, lambda output, data, group: (output.loss, {}), forward_only=False
            )
            assert torch.isfinite(loss)
            loss.backward()
            assert engine.module.backbone.memory_projection.weight.grad is not None
            assert engine.module.writer.output_projection.weight.grad is not None
            assert info["loss"] == loss.item()
            visits.extend((rank, i) for i in batch["sample_index"].tolist())
    assert len(visits) == len(set(visits)) == len(samples)


def test_native_fixed_batch_rejects_partial_rank_batches():
    with pytest.raises(ValueError, match="complete batches"):
        pretrain_batch(examples("sample", 2)[:3], 1, 0, 2)
