from dataclasses import dataclass


@dataclass(frozen=True)
class GMSAConfig:
    model_name_or_path: str
    revision: str | None = None
    encoder_layers: int | None = 8
    alignment_layers: int = 1
    compression_ratios: tuple[int, ...] = (4, 8)
    lora_rank: int = 128
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    attention_implementation: str = "sdpa"

    def __post_init__(self):
        object.__setattr__(self, "compression_ratios", tuple(self.compression_ratios))
        if not self.model_name_or_path:
            raise ValueError("model_name_or_path must be non-empty")
        if self.encoder_layers is not None and (
            type(self.encoder_layers) is not int or self.encoder_layers <= 0
        ):
            raise ValueError("encoder_layers must be null (full backbone) or a positive integer")
        for name in ("alignment_layers", "lora_rank", "lora_alpha"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not self.compression_ratios
            or any(type(r) is not int or r < 1 for r in self.compression_ratios)
            or len(set(self.compression_ratios)) != len(self.compression_ratios)
        ):
            raise ValueError("compression_ratios must contain unique positive integers")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.attention_implementation not in {"eager", "sdpa", "flash_attention_2"}:
            raise ValueError("unsupported attention implementation")
