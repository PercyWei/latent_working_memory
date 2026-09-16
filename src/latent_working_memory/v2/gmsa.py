"""Adapted from Twilightaaa/GMSA 2da109e (MIT; see LICENSE.GMSA).

Preserves causal encoder LoRA -> group mean -> decoder-initialized LSA -> decoder.
Exposes pre-LSA memories; packs each decoder row before padding to keep positions invariant.
"""

import random

import torch
from peft import LoraConfig, get_peft_model
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModelForCausalLM

from latent_working_memory.v2.gmsa_config import GMSAConfig


def group_mean(hidden: Tensor, mask: Tensor, ratio: int) -> list[Tensor]:
    if type(ratio) is not int or ratio < 1:
        raise ValueError("ratio must be a positive integer")
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("expected hidden [batch, tokens, dim] and boolean mask [batch, tokens]")
    result = []
    for row, valid in zip(hidden, mask, strict=True):
        current = row[valid]
        if len(current) == 0:
            raise ValueError("cannot compress empty context")
        full = len(current) // ratio * ratio
        pooled = current[:full].reshape(-1, ratio, hidden.shape[-1]).mean(1)
        if full < len(current):
            pooled = torch.cat((pooled, current[full:].mean(0, keepdim=True)))
        result.append(pooled)
    return result


def padded_memories(memories: list[Tensor]) -> tuple[Tensor, Tensor]:
    if not memories or any(m.ndim != 2 or len(m) == 0 for m in memories):
        raise ValueError("memories must be non-empty matrices")
    padded = pad_sequence(memories, batch_first=True)
    lengths = torch.tensor([len(m) for m in memories], device=padded.device)
    mask = torch.arange(padded.shape[1], device=padded.device)[None] < lengths[:, None]
    return padded, mask


class GMSA(nn.Module):
    def __init__(self, config: GMSAConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.model_config = config
        base_config = AutoConfig.from_pretrained(
            config.model_name_or_path, revision=config.revision
        )
        if base_config.model_type not in {"llama", "qwen3"}:
            raise ValueError("v2 GMSA supports Llama and Qwen3 backbones")
        encoder_layers = (
            base_config.num_hidden_layers
            if config.encoder_layers is None
            else config.encoder_layers
        )
        if max(encoder_layers, config.alignment_layers) > base_config.num_hidden_layers:
            raise ValueError("encoder/LSA depth must not exceed the pretrained backbone")
        load_args = dict(
            revision=config.revision,
            torch_dtype=dtype,
            attn_implementation=config.attention_implementation,
        )
        encoder_config = AutoConfig.from_pretrained(
            config.model_name_or_path, revision=config.revision
        )
        encoder_config.num_hidden_layers = encoder_layers
        encoder = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path, config=encoder_config, **load_args
        )
        self.encoder = get_peft_model(
            encoder,
            LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules="all-linear",
            ),
        )
        self.decoder = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **load_args)
        fusion_config = AutoConfig.from_pretrained(
            config.model_name_or_path, revision=config.revision
        )
        fusion_config.num_hidden_layers = config.alignment_layers
        # Retain the pretrained final norm, but discard unused embedding/head parameters.
        fusion = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path, config=fusion_config, **load_args
        )
        self.alignment = fusion.model
        self.alignment.embed_tokens = None
        self.width = base_config.hidden_size
        self.max_positions = base_config.max_position_embeddings
        self.checkpointing = False
        self.stage = "autoencoding"
        self.set_stage(self.stage)

    def set_stage(self, stage: str):
        if stage not in {"autoencoding", "finetune", "dynamic"}:
            raise ValueError("stage must be autoencoding, finetune, or dynamic")
        self.stage = stage
        self.requires_grad_(False)
        if stage == "autoencoding":
            for name, parameter in self.encoder.named_parameters():
                parameter.requires_grad_("lora_" in name)
            self.alignment.layers.requires_grad_(True)
        elif stage == "finetune":
            self.decoder.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # Fixed representations must not drift because of encoder LoRA dropout.
        if self.stage != "autoencoding":
            self.encoder.eval()
            self.alignment.eval()
        if self.stage != "finetune" and not self.checkpointing:
            self.decoder.eval()
        return self

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # Frozen readers still need their activation graph to train the compressor/updater.
        kwargs = gradient_checkpointing_kwargs or {"use_reentrant": False}
        if kwargs.get("use_reentrant", False):
            raise ValueError("GMSA requires non-reentrant gradient checkpointing")
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
        self.alignment.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
        self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
        self.checkpointing = True
        # Transformers only performs activation recomputation while the decoder is in train mode.
        # Frozen parameters and a differentiable input are independent of this mode switch.
        self.train(self.training)

    def encode(self, context_ids: Tensor, context_mask: Tensor, ratio: int) -> list[Tensor]:
        if ratio not in self.model_config.compression_ratios:
            raise ValueError("ratio is not configured")
        if context_ids.shape != context_mask.shape or context_mask.dtype != torch.bool:
            raise ValueError("context IDs and boolean mask must align")
        if not bool(context_mask.any(1).all()) or context_ids.shape[1] > self.max_positions:
            raise ValueError("context must be non-empty and fit the encoder window")
        positions = context_mask.long().cumsum(1).sub(1).clamp_min(0)
        hidden = (
            self.encoder.get_base_model()
            .model(
                input_ids=context_ids,
                attention_mask=context_mask,
                position_ids=positions,
                use_cache=False,
                return_dict=True,
            )
            .last_hidden_state
        )
        return group_mean(hidden, context_mask, ratio)

    def align(self, memories: list[Tensor]) -> list[Tensor]:
        padded, mask = padded_memories(memories)
        if padded.shape[1] > self.max_positions or padded.shape[2] != self.width:
            raise ValueError("memory does not fit the LSA interface")
        hidden = self.alignment(
            inputs_embeds=padded, attention_mask=mask, use_cache=False, return_dict=True
        ).last_hidden_state
        return [row[: len(m)] for row, m in zip(hidden, memories, strict=True)]

    def reader_prefix(self, memories, prompt_ids, prompt_mask):
        aligned = self.align(memories)
        prompts = self.decoder.get_input_embeddings()(prompt_ids)
        if len(aligned) != len(prompts) or prompt_ids.shape != prompt_mask.shape:
            raise ValueError("memories and prompts must align")
        return [
            torch.cat((memory, prompt[mask.bool()]))
            for memory, prompt, mask in zip(aligned, prompts, prompt_mask, strict=True)
        ]

    def read(self, memories, prompt_ids, prompt_mask, labels):
        prefixes = self.reader_prefix(memories, prompt_ids, prompt_mask)
        targets = [row[row != -100] for row in labels]
        if len(targets) != len(prefixes) or any(len(t) == 0 for t in targets):
            raise ValueError("each memory requires a non-empty target")
        embedding = self.decoder.get_input_embeddings()
        rows = [
            torch.cat((prefix, embedding(target)))
            for prefix, target in zip(prefixes, targets, strict=True)
        ]
        if max(map(len, rows)) > self.max_positions:
            raise ValueError("memory + prompt + target exceeds decoder window")
        inputs, mask = padded_memories(rows)
        full_labels = pad_sequence(
            [
                torch.cat((target.new_full((len(prefix),), -100), target))
                for prefix, target in zip(prefixes, targets, strict=True)
            ],
            batch_first=True,
            padding_value=-100,
        )
        return self.decoder(
            inputs_embeds=inputs,
            attention_mask=mask,
            labels=full_labels,
            use_cache=False,
            return_dict=True,
        )

    def forward(self, context_ids, context_mask, prompt_ids, prompt_mask, labels, ratio=None):
        if ratio is None:
            if not self.training:
                raise ValueError("evaluation requires an explicit compression ratio")
            ratio = random.choice(self.model_config.compression_ratios)
            if torch.distributed.is_initialized():
                selected = context_ids.new_tensor(ratio)
                torch.distributed.broadcast(selected, src=0)
                ratio = selected.item()
        return self.read(
            self.encode(context_ids, context_mask, ratio), prompt_ids, prompt_mask, labels
        )

    @torch.no_grad()
    def generate(
        self,
        memories,
        prompt_ids,
        prompt_mask,
        max_new_tokens,
        eos_token_id,
        pad_token_id,
        repetition_penalty=1.3,
    ):
        if self.training:
            raise ValueError("call eval() before generation")
        rows = self.reader_prefix(memories, prompt_ids, prompt_mask)
        if max_new_tokens < 1 or max(map(len, rows)) + max_new_tokens > self.max_positions:
            raise ValueError("generation budget exceeds decoder window")
        # Left padding keeps the final position valid for every generation row.
        inputs = pad_sequence([row.flip(0) for row in rows], batch_first=True).flip(1)
        lengths = torch.tensor([len(row) for row in rows], device=inputs.device)
        mask = torch.arange(inputs.shape[1], device=inputs.device)[None] >= (
            inputs.shape[1] - lengths[:, None]
        )
        return self.decoder.generate(
            inputs_embeds=inputs,
            attention_mask=mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            use_cache=True,
            repetition_penalty=repetition_penalty,
        )
