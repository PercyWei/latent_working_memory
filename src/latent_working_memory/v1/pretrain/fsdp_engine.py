"""Pretraining model/loss adapter for verl's native FSDP2 engine.

Used by the validation entry point until CUDA parity and checkpoint tests pass.
Batch execution, backward, sharding and optimizer stepping belong to verl.
"""

from contextlib import nullcontext
import math

from peft import get_peft_model_state_dict
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.utils._pytree import register_dataclass
from tensordict import TensorDict
from verl.trainer.config import CheckpointConfig
from verl.utils.tensordict_utils import assign_non_tensor, get_non_tensor_data
from verl.utils.fsdp_utils import get_fsdp_full_state_dict
from verl.workers.config import FSDPOptimizerConfig
from verl.workers.config.optimizer import build_optimizer
from verl.workers.engine import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

from latent_working_memory.v1.pretrain.training import PretrainOutput


register_dataclass(PretrainOutput)


def pretrain_loss(model_output, data, dp_group):
    # pretrain_forward already applies sample/capacity weights and the global
    # sample denominator. FSDP averages DP gradients, so compensate exactly once.
    return model_output.loss * dist.get_world_size(dp_group), {}


def pretrain_batch(examples, micro_batch_size, rank, world_size):
    ordered = sorted(
        examples,
        key=lambda e: max(e.input_length, len(e.lm.target_ids) if e.lm else 0) + e.capacity,
    )
    local = ordered[rank::world_size]
    if len(ordered) % (world_size * micro_batch_size):
        raise ValueError("native fixed microbatches require equal, complete batches per rank")
    target_lengths = torch.tensor(
        [len(e.ae.target_ids if e.ae else e.lm.target_ids) for e in local]
    )
    data = TensorDict(
        {
            "sample_index": torch.arange(len(local)),
            "loss_mask": torch.arange(int(target_lengths.max()))[None, :] < target_lengths[:, None],
        },
        batch_size=[len(local)],
    )
    assign_non_tensor(
        data,
        examples=tuple(local),
        sample_count=round(sum(e.loss_weight for e in examples)),
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=micro_batch_size,
    )
    return data


@EngineRegistry.register(model_type="lwm_pretrain", backend="fsdp2", device="cuda")
class PretrainFSDPEngine(FSDPEngine):
    def __init__(self, model, model_config, engine_config):
        if engine_config.strategy != "fsdp2":
            raise ValueError("the pretraining adapter supports fsdp2")
        if (
            model_config.lora_rank
            or model_config.use_remove_padding
            or model_config.use_fused_kernels
        ):
            raise ValueError("the adapter owns Reader LoRA and uses the existing padded objective")
        if model_config.use_liger:
            raise ValueError(
                "the supplied model has already been loaded; Liger is not enabled here"
            )
        self.model = model
        config = model.experiment_config
        super().__init__(
            model_config,
            engine_config,
            FSDPOptimizerConfig(
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                clip_grad=config.gradient_clip,
                total_training_steps=1,
                lr_scheduler_type="constant",
                override_optimizer_config={"fused": True} if config.optimizer_fused else None,
            ),
            CheckpointConfig(),
        )

    def _build_module(self):
        # Preserve the existing frozen weights, projections, LoRA and Writer.
        # The superclass performs FSDP2 wrapping and optimizer initialization.
        return self.model

    def _build_optimizer(self, module):
        self.parameters = list(module.trainable_parameters())
        names = {id(parameter): name for name, parameter in module.named_parameters()}
        self.parameter_names = [names[id(parameter)] for parameter in self.parameters]
        return build_optimizer(self.parameters, self.optimizer_config)

    def forward_step(self, micro_batch, loss_function, forward_only):
        examples = get_non_tensor_data(micro_batch, "examples", None)
        selected = [examples[i] for i in micro_batch["sample_index"].tolist()]
        sample_count = get_non_tensor_data(micro_batch, "sample_count", None)
        context = (
            nullcontext()
            if self._autocast_dtype == torch.float32
            else torch.autocast("cuda", dtype=self._autocast_dtype)
        )
        with context:
            output = self.module(selected, sample_count)
            loss, metrics = loss_function(output, micro_batch, self.get_data_parallel_group())
        return loss, {"model_output": {}, "loss": output.loss.detach().item(), "metrics": metrics}

    def optimizer_step(self):
        norm = super().optimizer_step()
        # verl skips an invalid update; v1 also terminates the run in this case.
        if not math.isfinite(norm):
            raise FloatingPointError("non-finite pretraining gradient norm")
        return norm

    def canonical_state(self, value_network):
        # PEFT can toggle the visible FSDP parameter's requires_grad separately
        # from its sharded counterpart. The optimizer's initialization contract
        # determines what must be saved, not those transient flags.
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state = get_fsdp_full_state_dict(self.module)
        state = {name: state[name] for name in self.parameter_names}
        optimizer = get_optimizer_state_dict(self.module, self.optimizer, options=options)
        if dist.get_rank() != 0:
            return None

        def subtree(prefix):
            return {
                name[len(prefix) :]: value
                for name, value in state.items()
                if name.startswith(prefix)
            }

        model_state = {
            "backbone": {
                "input_projection": subtree("backbone.input_projection."),
                "memory_projection": subtree("backbone.memory_projection."),
                "reader_lora": get_peft_model_state_dict(
                    self.model.backbone.language_model,
                    state_dict=subtree("backbone.language_model."),
                    save_embedding_layers=False,
                ),
            },
            "writer": subtree("writer."),
            "value_network": {k: v.detach().cpu() for k, v in value_network.state_dict().items()},
        }
        name_to_index = {name: i for i, name in enumerate(self.parameter_names)}
        optimizer_state = {
            "state": {name_to_index[name]: value for name, value in optimizer["state"].items()},
            "param_groups": [
                {**group, "params": [name_to_index[name] for name in group["params"]]}
                for group in optimizer["param_groups"]
            ],
        }
        return model_state, optimizer_state

    def load_canonical_state(self, model_state, optimizer_state):
        state = {}
        backbone = model_state["backbone"]
        for component in ("input_projection", "memory_projection"):
            state.update({f"backbone.{component}.{k}": v for k, v in backbone[component].items()})
        state.update({f"writer.{k}": v for k, v in model_state["writer"].items()})
        prefix = "backbone.language_model."
        for name in self.parameter_names:
            if name.startswith(prefix):
                key = name[len(prefix) :].replace(".default.", ".")
                state[name] = backbone["reader_lora"][key]
        if set(state) != set(self.parameter_names):
            raise ValueError(
                "checkpoint must contain all and only trainable pretraining parameters"
            )
        # Frozen base weights are supplied by model construction, not the checkpoint.
        options = StateDictOptions(full_state_dict=True, strict=False)
        set_model_state_dict(self.module, state, options=options)
        named_optimizer = {
            "state": {self.parameter_names[i]: v for i, v in optimizer_state["state"].items()},
            "param_groups": [
                {**group, "params": [self.parameter_names[i] for i in group["params"]]}
                for group in optimizer_state["param_groups"]
            ],
        }
        set_optimizer_state_dict(self.module, self.optimizer, named_optimizer, options=options)
