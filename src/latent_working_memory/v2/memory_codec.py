"""单一共享基座：Encoder LoRA 写入，禁用 LoRA 读取，独立读写对齐。"""

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModelForCausalLM

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

    def __post_init__(self):
        if self.feature_layer not in {"last", "mean"}:
            raise ValueError("feature_layer must be last or mean")
        if self.compression not in {"mean", "weighted", "spectral"}:
            raise ValueError("unknown compression method")
        if self.attention_implementation not in {"eager", "sdpa"}:
            raise ValueError("reconstruction uses eager or sdpa attention")
        for name in ("alignment_layers", "lora_rank", "lora_alpha", "spectral_bottleneck"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must lie in [0, 1)")


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
        was_training = self.backbone.training
        flags = [
            (p, p.requires_grad) for name, p in self.backbone.named_parameters() if "lora_" in name
        ]
        try:
            if adapter is not None:
                self.backbone.set_adapter(adapter)
                # PEFT set_adapter changes requires_grad; retain this stage's optimizer contract.
                for parameter, flag in flags:
                    parameter.requires_grad_(flag)
            self.backbone.train(training)
            context = self.backbone.disable_adapter() if adapter is None else nullcontext()
            with context:
                yield self.backbone
        finally:
            self.backbone.set_adapter(previous)
            for parameter, flag in flags:
                parameter.requires_grad_(flag)
            self.backbone.train(was_training)

    def initialize_write_alignment(self):
        self.write_alignment.load_state_dict(self.read_alignment.state_dict())

    def align(self, memory, writing=False):
        module = self.write_alignment if writing else self.read_alignment
        return module(
            inputs_embeds=memory[None], use_cache=False, return_dict=True
        ).last_hidden_state[0]

    def _encode(self, embeddings, training):
        with self.use_adapter("encoder", training=training) as backbone:
            output = backbone.get_base_model().model(
                inputs_embeds=embeddings[None],
                use_cache=False,
                output_hidden_states=self.config.feature_layer == "mean",
                return_dict=True,
            )
            if self.config.feature_layer == "mean":
                return torch.stack(output.hidden_states[1:]).mean(0)[0]
            return output.last_hidden_state[0]

    def write(self, previous, token_ids, capacity):
        embeddings = self.backbone.get_input_embeddings()(token_ids)
        if previous is not None:
            embeddings = torch.cat((self.align(previous, writing=True), embeddings))
        if len(embeddings) > self.max_positions:
            raise ValueError("joint writer input exceeds model window")
        if self.config.gradient_checkpointing and self.training:
            hidden = checkpoint(self._encode, embeddings, self.training, use_reentrant=False)
        else:
            hidden = self._encode(embeddings, self.training)
        return self.compression(hidden, capacity)

    def _read_loss(self, memory, prompt_ids, target_ids):
        aligned = self.align(memory)
        embed = self.backbone.get_input_embeddings()
        prefix = torch.cat((aligned, embed(prompt_ids)))
        inputs = torch.cat((prefix, embed(target_ids)))
        if len(inputs) > self.max_positions:
            raise ValueError("memory + prompt + target exceeds reader window")
        labels = torch.cat((target_ids.new_full((len(prefix),), -100), target_ids))
        # The reconstruction experiment uses the frozen base reader, with both LoRAs disabled.
        with self.use_adapter(None) as backbone:
            return backbone(
                inputs_embeds=inputs[None], labels=labels[None], use_cache=False, return_dict=True
            ).loss

    def read_loss(self, memory, prompt_ids, target_ids):
        if self.config.gradient_checkpointing and self.training:
            return checkpoint(self._read_loss, memory, prompt_ids, target_ids, use_reentrant=False)
        return self._read_loss(memory, prompt_ids, target_ids)

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
