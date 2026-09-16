"""CPU checks of the model/batch adapter, independent of CUDA FSDP execution."""

import pytest
import torch
import torch.distributed as dist
from verl.workers.engine.utils import prepare_micro_batches

from latent_working_memory.v1.model import JointMemoryWriter, GrowthValueNetwork
from latent_working_memory.v1.training import trainable_model_state
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


def test_checkpoint_keeps_optimizer_parameters_with_transient_frozen_flags(tmp_path, tiny_config):
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1
    )
    try:
        model = PretrainModel(
            tiny_config, make_backbone("qwen2"), JointMemoryWriter(8, 1, 2, 16, 32)
        )
        value = GrowthValueNetwork(8)
        engine = object.__new__(PretrainFSDPEngine)
        engine.model = engine.module = model
        engine.parameters = list(model.trainable_parameters())
        names = {id(p): name for name, p in model.named_parameters()}
        engine.parameter_names = [names[id(p)] for p in engine.parameters]
        engine.optimizer = torch.optim.AdamW(engine.parameters, lr=1e-4)
        samples = examples("sample", 2)
        model(samples, len(samples)).loss.backward()
        engine.optimizer.step()
        expected = trainable_model_state(model.backbone, model.writer, value)
        for name, p in zip(engine.parameter_names, engine.parameters, strict=True):
            if name.startswith("backbone.language_model."):
                p.requires_grad_(False)
        actual, optimizer = engine.canonical_state(value)
        assert actual["backbone"]["reader_lora"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            optimizer["state"], engine.optimizer.state_dict()["state"], rtol=0, atol=0
        )
    finally:
        dist.destroy_process_group()
