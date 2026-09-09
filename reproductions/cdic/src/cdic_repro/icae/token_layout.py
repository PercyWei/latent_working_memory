from __future__ import annotations

from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

ICAE_PAD_TOKEN = "<|icae_pad|>"
ICAE_CONTROL_TOKEN_COUNT = 3


@dataclass(frozen=True, slots=True)
class IcaeTokenLayout:

    tokenizer_vocabulary_size: int
    memory_size: int

    def __post_init__(self) -> None:
        if self.tokenizer_vocabulary_size < 1:
            raise ValueError("tokenizer_vocabulary_size must be positive")
        if self.memory_size < 1:
            raise ValueError("memory_size must be positive")

    @property
    def memory_token_start(self) -> int:
        return self.tokenizer_vocabulary_size

    @property
    def memory_token_end(self) -> int:
        return self.memory_token_start + self.memory_size

    @property
    def memory_token_ids(self) -> list[int]:
        return list(range(self.memory_token_start, self.memory_token_end))

    @property
    def ae_token_id(self) -> int:
        return self.memory_token_end

    @property
    def lm_token_id(self) -> int:
        return self.ae_token_id + 1

    @property
    def ft_token_id(self) -> int:
        return self.lm_token_id + 1

    @property
    def token_id_upper_bound(self) -> int:
        return self.tokenizer_vocabulary_size + self.memory_size + ICAE_CONTROL_TOKEN_COUNT


def prepare_icae_token_layout(
    tokenizer: PreTrainedTokenizerBase,
    memory_size: int,
) -> IcaeTokenLayout:
    """配置 tokenizer 的 pad token, 并据其实际 ID 构造 ICAE token 布局."""

    base_vocabulary_size = len(tokenizer)
    tokenizer.add_special_tokens({"pad_token": ICAE_PAD_TOKEN})
    if tokenizer.pad_token_id != base_vocabulary_size:
        raise ValueError("ICAE pad token must follow the base tokenizer vocabulary")
    if len(tokenizer) != base_vocabulary_size + 1:
        raise ValueError("ICAE pad token must add exactly one tokenizer entry")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define one eos_token_id")

    return IcaeTokenLayout(
        tokenizer_vocabulary_size=len(tokenizer),
        memory_size=memory_size,
    )
