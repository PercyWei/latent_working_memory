"""Pre-Accelerate training step from adb3b57 for paired regression validation."""

from typing import Any
import torch
import torch.distributed as dist

from latent_working_memory.v1.backbone import LatentMemoryBackbone
from latent_working_memory.v1.config import ExperimentConfig
from latent_working_memory.v1.distributed import synchronize_gradients
from latent_working_memory.v1.model import JointMemoryWriter
from latent_working_memory.v1.pretrain.sampling import PretrainExample
from latent_working_memory.v1.pretrain.training import pretrain_forward
from latent_working_memory.v1.training import precision_context


class LegacyPretrainTrainer:
    def __init__(
        self,
        config: ExperimentConfig,
        backbone: LatentMemoryBackbone,
        writer: JointMemoryWriter,
        device: torch.device,
    ) -> None:
        self.config, self.backbone, self.writer, self.device = config, backbone, writer, device
        self.parameters = list(backbone.trainable_parameters()) + list(writer.parameters())
        self.optimizer = torch.optim.AdamW(
            self.parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )

    def step(self, examples: list[PretrainExample]) -> dict[str, Any]:
        if not examples:
            raise ValueError("a training step requires examples")
        self.backbone.train()
        self.writer.train()
        self.optimizer.zero_grad(set_to_none=True)
        # Sorting changes only microbatch grouping, preserving the sampled document weights.
        ordered = sorted(
            examples,
            key=lambda e: max(e.input_length, len(e.lm.target_ids) if e.lm else 0) + e.capacity,
        )
        sample_count = round(sum(e.loss_weight for e in examples))
        distributed = dist.is_initialized()
        if distributed:
            ordered = ordered[dist.get_rank() :: dist.get_world_size()]
        records, loss_value = [], 0.0
        for start in range(0, len(ordered), self.config.batch_size):
            batch = ordered[start : start + self.config.batch_size]
            with precision_context(self.device):
                output = pretrain_forward(
                    self.config, self.backbone, self.writer, batch, sample_count
                )
                scaled_loss = output.loss
            scaled_loss.backward()
            loss_value += float(scaled_loss.detach())
            for example, ae, lm in zip(batch, output.ae, output.lm, strict=True):
                source = example.episode.sources[0]
                records.append(
                    {
                        "episode_id": example.episode.episode_id,
                        "document_id": source.document_id,
                        "boundary_variant": source.provenance["boundary_variant"],
                        "input_tokens": example.input_length,
                        "length_bucket": next(
                            (
                                b
                                for b in self.config.input_length_bounds
                                if example.input_length <= b
                            ),
                            self.config.max_input_tokens,
                        ),
                        "continuation_tokens": len(example.lm.target_ids) - 1 if example.lm else 0,
                        "capacity": example.capacity,
                        "effective_ratio": example.input_length / example.capacity,
                        "loss_weight": example.loss_weight,
                        "ae_nll": float(ae.mean_nll.detach()) if ae is not None else None,
                        "lm_nll": float(lm.mean_nll.detach()) if lm is not None else None,
                    }
                )
            del output, scaled_loss, ae, lm
        if distributed:
            synchronize_gradients(self.parameters)
            loss_tensor = torch.tensor(loss_value, device=self.device)
            dist.all_reduce(loss_tensor)
            loss_value = loss_tensor.item()
            rank_records = [None] * dist.get_world_size()
            dist.all_gather_object(rank_records, records)
            records = [row for rows in rank_records for row in rows]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters,
            self.config.gradient_clip,
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        return {
            "loss": loss_value,
            "gradient_norm": float(grad_norm),
            "samples": records,
            "input_length_bounds": sorted(
                {*self.config.input_length_bounds, self.config.max_input_tokens}
            ),
            "input_tokens": sum(e.input_length for e in examples),
            "sample_visits": sample_count,
            "capacity_reads": len(examples),
            "target_tokens": sum(
                len(e.ae.target_ids if e.ae else e.lm.target_ids) for e in examples
            ),
        }
