"""单一共享基座：Encoder LoRA 写入，禁用 LoRA 读取，独立读写对齐。"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass

import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.tuners_utils import BaseTunerLayer
from torch import nn
from torch.nn.attention.varlen import varlen_attn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
from transformers import AttentionInterface, AutoConfig, AutoModelForCausalLM
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from latent_working_memory.v2.compression import SlotCompression


@dataclass(frozen=True)
class CodecConfig:
    model_name_or_path: str
    revision: str | None = None
    encoder_layers: int | None = None
    alignment_layers: int = 1
    lora_rank: int = 128
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    attention_implementation: str = "sdpa"
    compression: str = "mean"
    feature_layer: str = "last"
    spectral_bottleneck: int = 256
    gradient_checkpointing: bool = True
    lm_head_chunk_size: int = 256
    padding_free: bool = False

    def __post_init__(self):
        if self.padding_free and self.attention_implementation != "sdpa":
            raise ValueError("padding_free uses PyTorch varlen attention with the sdpa dense path")
        if self.feature_layer not in {"last", "mean"}:
            raise ValueError("feature_layer must be last or mean")
        if self.compression not in {"mean", "weighted", "spectral"}:
            raise ValueError("unknown compression method")
        if self.attention_implementation not in {"eager", "sdpa"}:
            raise ValueError("reconstruction uses eager or sdpa attention")
        for name in (
            "alignment_layers",
            "lora_rank",
            "lora_alpha",
            "spectral_bottleneck",
            "lm_head_chunk_size",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must lie in [0, 1)")


def varlen_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout=0.0,
    scaling=None,
    cu_seq_lens=None,
    max_length=None,
    **kwargs,
):
    # Generation uses a dense KV cache; training passes explicit packed boundaries.
    if cu_seq_lens is None:
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs
        )
    if dropout or getattr(module, "sliding_window", None) is not None:
        raise ValueError(
            "packed reconstruction requires zero attention dropout and full causal attention"
        )
    q, k, v = [x.squeeze(0).transpose(0, 1) for x in (query, key, value)]
    output = varlen_attn(
        q,
        k,
        v,
        cu_seq_lens,
        cu_seq_lens,
        max_length,
        max_length,
        scale=scaling,
        window_size=(-1, 0),
        enable_gqa=True,
    )
    return output.unsqueeze(0), None


AttentionInterface.register("lwm_varlen", varlen_attention_forward)


def alignment_from_backbone(transformer, layers):
    """Allocate only the selected pretrained blocks; no checkpoint reload or real embedding."""
    config = deepcopy(transformer.config)
    config.num_hidden_layers = layers
    with torch.device("meta"):
        alignment = type(transformer)(config)
    alignment.embed_tokens = None
    alignment.layers = nn.ModuleList([deepcopy(block) for block in transformer.layers[:layers]])
    alignment.norm = deepcopy(transformer.norm)
    alignment.rotary_emb = deepcopy(transformer.rotary_emb)
    return alignment


class MemoryCodec(nn.Module):
    def __init__(self, config: CodecConfig, dtype=torch.float32):
        super().__init__()
        self.config = config
        base_config = AutoConfig.from_pretrained(
            config.model_name_or_path, revision=config.revision
        )
        if base_config.model_type not in {"llama", "qwen3"}:
            raise ValueError("MemoryCodec supports Llama and Qwen3 backbones")
        if config.encoder_layers not in (None, base_config.num_hidden_layers):
            raise ValueError("shared encoder/decoder must use the full backbone")
        if config.alignment_layers > base_config.num_hidden_layers:
            raise ValueError("alignment depth cannot exceed the backbone")
        base = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            revision=config.revision,
            config=base_config,
            torch_dtype=dtype,
            attn_implementation=config.attention_implementation,
        )
        # Copy before injecting LoRA: alignment blocks have their own dense weights only.
        self.read_alignment = alignment_from_backbone(base.model, config.alignment_layers)
        self.write_alignment = deepcopy(self.read_alignment)
        if config.padding_free:
            base.set_attn_implementation("lwm_varlen")
        self.backbone = get_peft_model(
            base,
            LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules="all-linear",
            ),
            adapter_name="encoder",
        )
        self.backbone.add_adapter(
            "decoder",
            LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules="all-linear",
            ),
        )
        self.backbone.set_adapter("encoder")
        self._backbone_modules = tuple(self.backbone.modules())
        self._lora_layers = tuple(
            module for module in self._backbone_modules if isinstance(module, BaseTunerLayer)
        )
        self._adapter_parameters = tuple(
            p for name, p in self.backbone.named_parameters() if "lora_" in name
        )
        self.width = base_config.hidden_size
        self.max_positions = base_config.max_position_embeddings
        self.compression = SlotCompression(
            self.width, config.compression, config.spectral_bottleneck
        )
        for name, parameter in self.named_parameters():
            if (
                name.startswith(
                    ("read_alignment.layers.", "write_alignment.layers.", "compression.")
                )
                or "lora_" in name
            ):
                parameter.data = parameter.data.float()
        self.stage = "multiround"
        self.set_stage(self.stage)
        # Native per-layer checkpointing is intentionally off: its later replay could run
        # after an adapter switch. Whole encoder/read functions select adapters on every call.

    def set_stage(self, stage):
        if stage not in {"warmup", "multiround"}:
            raise ValueError("stage must be warmup or multiround")
        self.stage = stage
        self.requires_grad_(False)
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad_("lora_" in name and ".encoder." in name)
        self.compression.requires_grad_(True)
        self.read_alignment.layers.requires_grad_(True)
        self.write_alignment.layers.requires_grad_(stage == "multiround")
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.stage == "warmup":
            self.write_alignment.eval()
        return self

    @contextmanager
    def use_adapter(self, adapter, training=False):
        """Select a named LoRA (or None) without changing the training parameter contract."""
        previous = self.backbone.active_adapter
        modes = tuple(module.training for module in self._backbone_modules)
        disabled = tuple(layer.disable_adapters for layer in self._lora_layers)
        flags = tuple(p.requires_grad for p in self._adapter_parameters)
        try:
            if adapter is not None and adapter != previous:
                self.backbone.set_adapter(adapter)
            for layer in self._lora_layers:
                if layer.disable_adapters != (adapter is None):
                    layer.enable_adapters(adapter is not None)
            # PEFT's public toggles also change requires_grad. Restore the optimizer contract
            # before forward, including while the reader bypasses the encoder adapter.
            for parameter, flag in zip(self._adapter_parameters, flags, strict=True):
                parameter.requires_grad_(flag)
            # Qwen/Llama and their LoRA/dropout modules use Module's training flag. A flat
            # traversal avoids recursively visiting the same descendants on every read/replay.
            for module in self._backbone_modules:
                module.training = training
            yield self.backbone
        finally:
            if self.backbone.active_adapter != previous:
                self.backbone.set_adapter(previous)
            for layer, was_disabled in zip(self._lora_layers, disabled, strict=True):
                if layer.disable_adapters != was_disabled:
                    layer.enable_adapters(not was_disabled)
            for parameter, flag in zip(self._adapter_parameters, flags, strict=True):
                parameter.requires_grad_(flag)
            for module, mode in zip(self._backbone_modules, modes, strict=True):
                module.training = mode

    def initialize_write_alignment(self):
        self.write_alignment.load_state_dict(self.read_alignment.state_dict())

    def align(self, memory, writing=False):
        module = self.write_alignment if writing else self.read_alignment
        single = memory.ndim == 2
        memory = memory[None] if single else memory
        aligned = module(
            inputs_embeds=memory,
            attention_mask=torch.ones(memory.shape[:2], dtype=torch.bool, device=memory.device),
            use_cache=False,
            return_dict=True,
        ).last_hidden_state.to(self.backbone.get_input_embeddings().weight.dtype)
        return aligned[0] if single else aligned

    def _backbone_hidden(self, backbone, rows, output_hidden_states=False):
        lengths = [len(row) for row in rows]
        if max(lengths) > self.max_positions:
            raise ValueError("a sequence exceeds the backbone window")
        if self.config.padding_free:
            inputs = torch.cat(rows)[None]
            positions = torch.cat([torch.arange(n, device=inputs.device) for n in lengths])[None]
            cu_seq = torch.tensor([0, *lengths], device=inputs.device, dtype=torch.int32).cumsum(
                0, dtype=torch.int32
            )
            kwargs = dict(position_ids=positions, cu_seq_lens=cu_seq, max_length=max(lengths))
            mask = None
        else:
            inputs = pad_sequence(rows, batch_first=True)
            mask = torch.ones(inputs.shape[:2], dtype=torch.bool, device=inputs.device)
            kwargs = {}
        output = backbone.get_base_model().model(
            inputs_embeds=inputs,
            attention_mask=mask,
            use_cache=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        hidden = (
            torch.stack(output.hidden_states[1:]).mean(0)
            if output_hidden_states
            else output.last_hidden_state
        )
        if self.config.padding_free:
            return hidden[0].split(lengths)
        return tuple(row[:length] for row, length in zip(hidden, lengths, strict=True))

    def _encode(self, rows, training):
        with self.use_adapter("encoder", training=training) as backbone:
            return self._backbone_hidden(backbone, rows, self.config.feature_layer == "mean")

    def write(self, previous, token_ids, capacity):
        return self.write_batch(
            previous[None] if previous is not None else None, [token_ids], capacity
        )[0]

    def write_batch(self, previous, token_ids, capacity):
        embed = self.backbone.get_input_embeddings()
        rows = [embed(ids) for ids in token_ids]
        if previous is not None:
            aligned = self.align(previous, writing=True)
            rows = [torch.cat((memory, row)) for memory, row in zip(aligned, rows, strict=True)]
        if self.config.gradient_checkpointing and self.training:
            hidden = checkpoint(self._encode, rows, self.training, use_reentrant=False)
        else:
            hidden = self._encode(rows, self.training)
        return torch.stack([self.compression(row, capacity) for row in hidden])

    def _read_hidden(self, memory, input_ids):
        aligned = self.align(memory)
        embed = self.backbone.get_input_embeddings()
        rows = [torch.cat((m, embed(ids))) for m, ids in zip(aligned, input_ids, strict=True)]
        with self.use_adapter(None) as backbone:
            return self._backbone_hidden(backbone, rows)

    def _token_loss(self, hidden, targets):
        logits = self.backbone.get_output_embeddings()(hidden)
        return nn.functional.cross_entropy(logits.float(), targets, reduction="sum")

    def read_loss(self, memory, prompt_ids, target_ids):
        return self.read_loss_batch(memory[None], [prompt_ids], [target_ids])[0]

    def read_loss_batch(self, memory, prompt_ids, target_ids):
        inputs = [
            torch.cat((prompt, ids[:-1]))
            for prompt, ids in zip(prompt_ids, target_ids, strict=True)
        ]
        checkpointing = self.config.gradient_checkpointing and self.training
        if checkpointing:
            hidden = checkpoint(self._read_hidden, memory, inputs, use_reentrant=False)
        else:
            hidden = self._read_hidden(memory, inputs)
        losses = []
        chunk = self.config.lm_head_chunk_size
        for row, prompt, targets in zip(hidden, prompt_ids, target_ids, strict=True):
            # Last prompt state predicts target[0]; score no memory, prompt or padding.
            offset = memory.shape[1] + len(prompt) - 1
            row = row[offset : offset + len(targets)]
            pieces = []
            for start in range(0, len(targets), chunk):
                h, y = row[start : start + chunk], targets[start : start + chunk]
                # Separately checkpoint the vocabulary projection/CE: replaying the decoder
                # must not materialize all vocabulary logits for every target at once.
                loss = (
                    checkpoint(self._token_loss, h[: len(y)], y, use_reentrant=False)
                    if checkpointing
                    else self._token_loss(h[: len(y)], y)
                )
                pieces.append(loss)
            losses.append(torch.stack(pieces).sum() / len(targets))
        return torch.stack(losses)

    @torch.no_grad()
    def generate(self, memory, prompt_ids, max_new_tokens, eos_token_id, pad_token_id):
        prefix = torch.cat((self.align(memory), self.backbone.get_input_embeddings()(prompt_ids)))
        if len(prefix) + max_new_tokens > self.max_positions:
            raise ValueError("generation exceeds reader window")
        with self.use_adapter(None) as backbone:
            return backbone.generate(
                inputs_embeds=prefix[None],
                attention_mask=torch.ones(1, len(prefix), dtype=torch.long, device=prefix.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                use_cache=True,
            )[0]
